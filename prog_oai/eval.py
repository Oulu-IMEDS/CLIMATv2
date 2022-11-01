import json

import coloredlogs
import hydra
import pandas as pd
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, balanced_accuracy_score, \
    mean_squared_error, accuracy_score, mean_absolute_error
from tqdm import tqdm

from common.ece import ECELoss, AdaptiveECELoss, ClasswiseECELoss
from common.itemloader import ItemLoader
from common.utils import proc_targets, calculate_metric, load_metadata, init_mean_std, update_max_grades
from prognosis import train
from prognosis.models import create_model
from prognosis.utils import *
import pickle

# from prognosis.train import main_loop


coloredlogs.install()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

task_names = ('pn',)
task2metrics = {'pr': ['f1', 'ba', 'auc', 'ap'], 'pn': ['ba', 'mse'], 'tkr': ['ba', 'auc', 'ap']}


def filter_most_by_pa(ds, df_most_ex, pas=['PA05', 'PA10', 'PA15']):
    std_rows = []
    for i, row in df_most_ex.iterrows():
        std_row = dict()
        std_row['ID'] = row['ID_ex'].split('_')[0]
        std_row['visit_id'] = int(row['visit'][1:])
        std_row['PA'] = row['PA']
        std_rows.append(std_row)
    df_most_pa = pd.DataFrame(std_rows)

    ds_most_filtered = pd.merge(ds, df_most_pa, how='left', on=['ID', 'visit_id'])
    if isinstance(pas, str):
        ds_most_filtered = ds_most_filtered[ds_most_filtered['PA'] == pas]
    elif isinstance(pas, list):
        ds_most_filtered = ds_most_filtered[ds_most_filtered['PA'].isin(pas)]
    else:
        raise ValueError(f'Not support type {type(ds_most_filtered)}.')

    return ds_most_filtered


def load_data(cfg, site):
    # Compute mean and std of OAI
    wdir = os.environ['PWD']
    # Load and split data
    if cfg.dataset == "oai":
        if cfg.test_mode == 'validation':
            split_data, _, _ = load_metadata(cfg, proc_targets=proc_targets, modes='train')
            _, meta_test = split_data[cfg.fold_index - 1]
        else:
            _, _, meta_test = load_metadata(cfg, proc_targets=proc_targets, modes='eval')

    elif cfg.dataset == "most":
        most_data_fullname = os.path.join(cfg.meta_root, cfg.most_meta_filename)
        meta_test = pd.read_csv(most_data_fullname)
        meta_test = meta_test.replace({np.nan, None})
        print(f'Original MOST test data: {len(meta_test.index)}')

        print(f'Insert dummy columns')
        for grading in cfg.parser.output:
            for i in range(1, cfg.seq_len + 1):
                if f'{grading}_{i}y' not in meta_test:
                    meta_test[f'{grading}_{i}y'] = [np.nan] * len(meta_test.index)

        meta_fu_fullname = os.path.join(cfg.meta_root, cfg.most_followup_meta_filename)
        df_most_fu = pd.read_csv(meta_fu_fullname, sep='/', header=None,
                                 names=["ID_ex", "visit", 'ex1', 'PA', 'ex2'])
        print(f'Before selecting PA10: {len(meta_test.index)}')
        meta_test = filter_most_by_pa(meta_test, df_most_fu, ['PA10'])
        print(f'After selecting PA10: {len(meta_test.index)}')

        print(f'Before filtering out records without image: {len(meta_test.index)}')
        meta_test = remove_empty_img_rows(cfg.root, meta_test)
        print(f'After filtering out records without image: {len(meta_test.index)}')

        print(f'Reading file {meta_fu_fullname}')

        meta_test = proc_targets(meta_test, dataset='most')
        print(f'Final test data: {len(meta_test.index)}')
    else:
        raise ValueError(f'Not support dataset {cfg.dataset}.')

    # Loaders
    if cfg.dataset == "mnist3x3":
        _mean = (128.,)
        _std = (128.,)
        loader = ItemLoader(
            meta_data=meta_test, root=cfg.root, batch_size=cfg.bs, num_workers=cfg.num_workers,
            transform=init_transforms_mnist(_mean, _std)['eval'], parser_kwargs=cfg.parser,
            parse_item_cb=parse_item_mnist, shuffle=False, drop_last=False)
    else:
        mean_, std_ = init_mean_std(cfg, wdir, None, parse_img)
        print(f'Mean: {mean_}\nStd: {std_}')

        # Only choose records with baseline + first follow up
        print(f'Before filtering: {len(meta_test.index)}')

        if isinstance(cfg.use_only_grading, int) and cfg.use_only_grading >= 0:
            meta_test = meta_test[meta_test[cfg.grading] == cfg.use_only_grading]
            print(f'Only select grading {cfg.grading} = {cfg.use_only_grading}... Remaining: {len(meta_test.index)}')

        if cfg.use_only_baseline:
            print(f'Only select baseline...')
            meta_test = meta_test[meta_test['visit'] == 0]
            print(f'Only select baseline... Remaining: {len(meta_test.index)}')

        print(f'After filtering: {len(meta_test.index)}')

        # Cast visit to int
        meta_test['visit'] = meta_test['visit'].astype(int)
        loader = ItemLoader(
            meta_data=meta_test, root=cfg.root, batch_size=cfg.bs, num_workers=cfg.num_workers,
            transform=init_transforms(mean_, std_)['eval'], parser_kwargs=cfg.parser,
            parse_item_cb=parse_item_progs, shuffle=False)
    return loader


def eval(pretrained_model, loader, cfg, device, store=True):
    model = create_model(cfg, device)

    if pretrained_model and not os.path.exists(pretrained_model):
        log.fatal(f'Cannot find pretrained model {pretrained_model}')
        assert False
    elif pretrained_model:
        log.info(f'Loading pretrained model {pretrained_model}')
        try:
            model.load_state_dict(torch.load(pretrained_model), strict=False)
        except ValueError:
            log.fatal(f'Failed loading {pretrained_model}')

    metrics, accumulated_metrics = train.main_loop(loader, 0, model, cfg, "test")

    if store:
        with open("eval_metrics.json", "w") as f:
            json.dump(metrics, f)

    return metrics, accumulated_metrics


def eval_full(pretrained_model, loader, cfg, device):
    stage = 'eval'
    model = create_model(cfg, device)
    model.eval()

    if pretrained_model and not os.path.exists(pretrained_model):
        log.fatal(f'Cannot find pretrained model {pretrained_model}')
        assert False
    elif pretrained_model:
        log.info(f'Loading pretrained model {pretrained_model}')
        try:
            model.load_state_dict(torch.load(pretrained_model), strict=True)
        except ValueError:
            log.fatal(f'Failed loading {pretrained_model}')

    accumulated_metrics = {'loss': [], 'l_KL': [], 'l_prog': [], 'l_prognosis': [], 'pr': None, 'pn': None, 'tkr': None}
    for task in task_names:
        accumulated_metrics[task] = {}
        accumulated_metrics[task]['prob_by'] = [[] for i in range(8)]
        accumulated_metrics[task]['pred_by'] = [[] for i in range(8)]
        accumulated_metrics[task]['label_by'] = [[] for i in range(8)]

    n_iters = len(loader)
    progress_bar = tqdm(range(n_iters), total=n_iters, desc=f"{stage}::")
    # final_kappa_pr_by = None
    final_metrics = {}

    for batch_i in progress_bar:
        batch = loader.sample(1)[0]

        # Input
        input = {}
        for in_key in batch['data']['input']:
            if isinstance(batch['data']['input'][in_key], torch.Tensor):
                input[in_key] = batch['data']['input'][in_key].to(device)
            else:
                input[in_key] = batch['data']['input'][in_key]

        for inp in input.values():
            if isinstance(inp, torch.Tensor):
                batch_size = inp.shape[0]
                break
            elif (isinstance(inp, tuple) or isinstance(inp, list)) and isinstance(inp[0], torch.Tensor):
                batch_size = inp[0].shape[0]
                break

        seq_len = batch['progs'].shape[1]
        input['label_len'] = torch.tensor([seq_len] * batch_size, dtype=torch.int32).to(device)

        # Target
        targets = {}
        targets['progs'] = batch['progs'].to(device)
        targets['prognosis'] = batch['prognosis'].to(device)
        targets['progs_mask'] = batch['progs_mask'].to(device)
        targets['prognosis_mask'] = batch['prognosis_mask'].to(device)

        # losses, outputs = model.generate(input, targets, n_iters=1)
        losses, outputs = model.fit(input, targets, batch_i, n_iters, 0, stage="eval")

        # Metrics
        display_metrics = {}
        for loss_name in losses:
            if losses[loss_name] is not None:
                accumulated_metrics[loss_name].append(losses[loss_name])
                display_metrics[loss_name] = f'{np.array(accumulated_metrics[loss_name]).mean():.03f}'

        metrics_by = {'pr': {}, 'pn': {}, 'tkr': {}}
        for task in task_names:
            metrics_by[task] = {}
            for _name in task2metrics[task]:
                metrics_by[task][_name] = {i: None for i in range(seq_len)}

        for t in range(0, seq_len):
            for task in task_names:
                if task == "pr":  # Progression
                    preds = np.where(outputs[task]['prob'][t] > 0.5, 1, 0).flatten()
                    probs = outputs[task]['prob'][t].flatten()
                    labels = outputs[task]['label'][t].flatten()

                    accumulated_metrics[task]['prob_by'][t].extend(list(probs))
                    accumulated_metrics[task]['pred_by'][t].extend(list(preds))
                    accumulated_metrics[task]['label_by'][t].extend(list(labels.astype(int)))
                elif task == "pn":  # Prognosis
                    preds = np.argmax(outputs[task]['prob'][t], axis=-1)
                    probs = outputs[task]['prob'][t]
                    labels = outputs[task]['label'][t].flatten()

                    accumulated_metrics[task]['prob_by'][t].extend(list(probs))
                    accumulated_metrics[task]['pred_by'][t].extend(list(preds))
                    accumulated_metrics[task]['label_by'][t].extend(list(labels.astype(int)))
                elif task == "tkr":  # TKR
                    preds = np.argmax(outputs['pn']['prob'][t], axis=-1).flatten()
                    probs = outputs['pn']['prob'][t]
                    probs = np.reshape(probs, (-1, 5))
                    labels = outputs['pn']['label'][t].flatten()

                    tkr_mask = labels == 4
                    preds = preds[tkr_mask]
                    probs = probs[tkr_mask, :]
                    labels = labels[tkr_mask]

                    accumulated_metrics[task]['prob_by'][t].extend(list(probs))
                    accumulated_metrics[task]['pred_by'][t].extend(list(preds))
                    accumulated_metrics[task]['label_by'][t].extend(list(labels.astype(int)))
                else:
                    raise ValueError(f'Not support task {task} for not being {task_names}.')

            # Progression
            metrics_by['pr']['ba'][t] = balanced_accuracy_score(accumulated_metrics['pr']['label_by'][t],
                                                                accumulated_metrics['pr']['pred_by'][t])
            metrics_by['pr']['f1'][t] = calculate_metric(f1_score, accumulated_metrics['pr']['label_by'][t],
                                                         accumulated_metrics['pr']['pred_by'][t])
            metrics_by['pr']['auc'][t] = calculate_metric(roc_auc_score,
                                                          accumulated_metrics['pr']['label_by'][t],
                                                          accumulated_metrics['pr']['prob_by'][t])
            metrics_by['pr']['ap'][t] = calculate_metric(average_precision_score,
                                                         accumulated_metrics['pr']['label_by'][t],
                                                         accumulated_metrics['pr']['prob_by'][t])
            # Prognosis
            # metrics_by['pn']['f1'][t] = calculate_metric(f1_score,
            #                                              accumulated_metrics['pn']['label_by'][t],
            #                                              accumulated_metrics['pn']['pred_by'][t])
            metrics_by['pn']['ba'][t] = calculate_metric(balanced_accuracy_score,
                                                         accumulated_metrics['pn']['label_by'][t],
                                                         accumulated_metrics['pn']['pred_by'][t])
            metrics_by['pn']['mse'][t] = calculate_metric(mean_squared_error,
                                                          accumulated_metrics['pn']['label_by'][t],
                                                          accumulated_metrics['pn']['pred_by'][t])
            # metrics_by['pn']['ka'][t] = calculate_metric(cohen_kappa_score,
            #                                              accumulated_metrics['pn']['label_by'][t],
            #                                              accumulated_metrics['pn']['pred_by'][t],
            #                                              weights='quadratic')
            # metrics_by['pn']['f1'][t] = calculate_metric(f1_score, accumulated_metrics['pn']['label_by'][t],
            #                                              accumulated_metrics['pn']['pred_by'][t])
            # TKR
            # metrics_by['tkr']['auc'][t] = calculate_metric(roc_auc_score,
            #                                              accumulated_metrics['tkr']['label_by'][t],
            #                                              accumulated_metrics['tkr']['prob_by'][t])
            # metrics_by['tkr']['ap'][t] = calculate_metric(average_precision_score,
            #                                               accumulated_metrics['tkr']['label_by'][t],
            #                                               accumulated_metrics['tkr']['prob_by'][t])

        display_metrics[f'pr:ba'] = "-".join(
            [f'{v:.03f}' if v is not None else "" for v in metrics_by['pr']['ba'].values()])
        display_metrics[f'pr:auc'] = "-".join(
            [f'{v:.03f}' if v is not None else "" for v in metrics_by['pr']['auc'].values()])
        display_metrics[f'pr:f1'] = "-".join(
            [f'{v:.03f}' if v is not None else "" for v in metrics_by['pr']['f1'].values()])
        display_metrics[f'pr:ap'] = "-".join(
            [f'{v:.03f}' if v is not None else "" for v in metrics_by['pr']['ap'].values()])
        display_metrics[f'pn:ba'] = "-".join(
            [f'{v:.03f}' if v is not None else "" for v in metrics_by['pn']['ba'].values()])
        display_metrics[f'pn:mse'] = "-".join(
            [f'{v:.03f}' if v is not None else "" for v in metrics_by['pn']['mse'].values()])
        # display_metrics[f'pn:ka'] = "-".join(
        #     [f'{v:.03f}' if v is not None else "" for v in metrics_by['pn']['ka'].values()])
        # display_metrics[f'pn:f1'] = "-".join(
        #     [f'{v:.03f}' if v is not None else "" for v in metrics_by['pn']['f1'].values()])
        # display_metrics[f'tkr:auc'] = "-".join(
        #     [f'{v:.03f}' if v is not None else "" for v in metrics_by['tkr']['auc'].values()])
        # display_metrics[f'tkr:ap'] = "-".join(
        #     [f'{v:.03f}' if v is not None else "" for v in metrics_by['tkr']['ap'].values()])

        progress_bar.set_postfix(display_metrics)

        # Last batch
        final_metrics = metrics_by

    metrics = {}
    for task in task_names:
        metrics[task] = {}
        for _name in task2metrics[task]:
            if _name in final_metrics[task]:
                metrics[task][_name] = list(final_metrics[task][_name].values())

    with open("eval_metrics.json", "w") as f:
        json.dump(metrics, f)
    return metrics


def eval_from_saved_folders(cfg, root_dir, device, patterns="bacc"):
    d = os.listdir(root_dir)
    collector = {}
    result_collector = {}

    trained_config = None
    for dir in d:
        args_fullname = os.path.join(root_dir, dir, "args.yaml")

        if os.path.isfile(args_fullname):
            config = OmegaConf.load(args_fullname)
        else:
            print(f'Not found {args_fullname}')
            config_fullname = os.path.join(root_dir, dir, ".hydra", "config.yaml")
            override_fullname = os.path.join(root_dir, dir, ".hydra", "overrides.yaml")

            config = OmegaConf.load(config_fullname)
            overriden_config = OmegaConf.load(override_fullname)

            if isinstance(overriden_config, ListConfig):
                or_config_names = ["site", "fold_index", "n_out_features", "parser"]
                for line in overriden_config:
                    k, v = line.split("=")
                    if k in or_config_names:
                        config[k] = v

        if cfg.seed != config["seed"]:
            print(f'[{cfg.seed}] Skip {dir}.')
            continue

        if "output" not in config["parser"]:
            config["parser"]["output"] = ['KL']

        if "grading" not in config["parser"]:
            config["parser"]["grading"] = config["grading"]

        or_config_names = ["root", "pkl_meta_filename", "meta_root", "bs", "num_workers", "dataset", "root",
                           "most_meta_filename", "oai_meta_filename", "multi_class_mode", "log_dir",
                           "use_y0_class_weights", "use_pn_class_weights", "use_pr_class_weights",
                           "use_only_grading", "use_only_baseline", "model_selection_mode", "save_attn",
                           "most_followup_meta_filename", "test_mode"]
        eval_config_names = ['root', 'patterns', 'n_resamplings']
        for k in or_config_names:
            config[k] = cfg[k]

        config.eval = {}
        for k in eval_config_names:
            config.eval[k] = cfg.eval[k]

        config.pkl_meta_filename = f"cv_split_5folds_{config.grading}_oai_evalsite_{config.site}_{config.seed}.pkl"
        config.skip_store = True

        update_max_grades(config)

        # print(config.pretty())
        if trained_config is None:
            trained_config = config

        print(config)

        print(f'Loading data site {config.site}...')
        loader = load_data(config, config.site)

        print(f'Finding pretrained model...')
        run_root = os.path.join(root_dir, dir)
        for r, d, f in os.walk(run_root):
            for filename in f:
                if isinstance(patterns, tuple) or isinstance(patterns, list):
                    matched = all([s in filename for s in patterns])
                else:
                    matched = patterns in filename

                if filename.endswith(".pth") and matched:
                    model_fullname = os.path.join(r, filename)

                    key = f'Site:{config.site}:{config.fold_index}'

                    print(f'{key}, model: {model_fullname}...')

                    metrics, accumulated_metrics = eval(model_fullname, loader, config, device, False)

                    result_collector = convert_metrics_to_dataframe(config, result_collector, accumulated_metrics)
                    collector[key] = metrics
                    continue

    print('Trained setting:')
    print(trained_config)
    output_fullname = os.path.abspath(cfg.eval.output)
    # with open(output_fullname[:-5] + f"_raw.pkl", "wb") as f:
    #     pickle.dump(result_collector, f)

    result_collector_agg, metrics_by = aggregate_dataframe(trained_config, result_collector)

    if cfg.save_predictions:
       with open(output_fullname[:-5] + f"_agg.pkl", "wb") as f:
          pickle.dump(result_collector_agg, f)
    for grading in trained_config.parser.output:
        output_fullname_grading = output_fullname[:-5] + "_" + grading + ".json"
        print(f'Saving output files to {output_fullname_grading}')
        with open(output_fullname_grading, "w") as f:
            json.dump(metrics_by[grading], f)


def get_sites(cfg, IDs):
    if cfg.dataset == "mnist3x3":
        enum_sites = ['Mnist']
        sites = ['Mnist'] * len(IDs)
    elif cfg.dataset == "oai":
        # enum_sites = ['C']  # ['A', 'B', 'C', 'D', 'E']
        sites = IDs.str[:1]
        enum_sites = sites.unique().tolist()
    elif cfg.dataset == "most":
        enum_sites = ['M']
        sites = ['M'] * len(IDs)
    else:
        raise ValueError(f'Not support dataset {cfg.dataset}')
    return sites, enum_sites


def aggregate_dataframe(cfg, result_collector):
    ece_criterion = ECELoss(normalized=True).cuda()
    adaece_criterion = AdaptiveECELoss(normalized=True).cuda()
    clsece_criterion = ClasswiseECELoss(normalized=True).cuda()
    # grading = cfg.grading
    metrics_grading_map = {}
    result_collector_agg_grading_map = {}
    for grading in cfg.parser.output:
        print(f'Aggregating {grading}...')
        result_collector_agg = {}

        metrics_by = {}

        metrics_by[grading] = {'ba': -1.0, 'ac': -1.0, 'mse': -1.0, 'mae': -1.0, 'ka': -1.0, 'mauc': -1.0, 'ece': -1.0,
                               'ada_ece': -1.0,
                               'cls_ece': -1.0}
        metrics_by['pn'] = {'ba': [-1.0] * (cfg.seq_len + 1),
                            'ac': [-1.0] * (cfg.seq_len + 1),
                            'mse': [-1.0] * (cfg.seq_len + 1),
                            'mae': [-1.0] * (cfg.seq_len + 1),
                            'mauc': [-1.0] * (cfg.seq_len + 1),
                            'ece': [-1.0] * (cfg.seq_len + 1),
                            'ada_ece': [-1.0] * (cfg.seq_len + 1),
                            'cls_ece': [-1.0] * (cfg.seq_len + 1)}

        # PN and PR
        task = 'pn'
        for t in range(cfg.seq_len + 1):
            if f'{grading}_{t}' in result_collector:

                metrics_by_site = {}
                for metric in metrics_by[task]:
                    metrics_by_site[metric] = []

                result_collector_agg[f'{grading}_{t}'] = \
                    result_collector[f'{grading}_{t}'].groupby('ID', as_index=False)[
                        f'{grading}_label', f'{grading}_prob'].apply(np.mean)

                result_collector_agg[f'{grading}_{t}']['Site'], list_sites = get_sites(cfg, result_collector_agg[
                    f'{grading}_{t}'].ID)

                for site in list_sites:
                    result_agg_site = result_collector_agg[f'{grading}_{t}'][
                        result_collector_agg[f'{grading}_{t}']['Site'] == site]
                    if len(result_agg_site.index) == 0:
                        print(f'Cannot find data of Side {site} for task {task} at follow-up {t}.')
                        continue
                    labels = result_agg_site[f'{grading}_label'].tolist()
                    probs = [v[0].tolist() for v in result_agg_site[f'{grading}_prob'].tolist()]
                    probs = np.stack(probs, 0)
                    preds = list(np.argmax(probs, -1))
                    if task == 'pn':
                        # Prognosis
                        metrics_by_site['ba'].append(calculate_metric(balanced_accuracy_score, labels, preds))
                        metrics_by_site['ac'].append(calculate_metric(accuracy_score, labels, preds))
                        metrics_by_site['mse'].append(calculate_metric(mean_squared_error, labels, preds))
                        metrics_by_site['mae'].append(calculate_metric(mean_absolute_error, labels, preds))
                        metrics_by_site['mauc'].append(calculate_metric(roc_auc_score, labels, probs,
                                                                        average='macro',
                                                                        labels=[i for i in range(cfg.n_pn_classes)],
                                                                        multi_class=cfg.multi_class_mode))
                        metrics_by_site['ece'].append(ece_criterion(probs, labels, return_tensor=False))
                        metrics_by_site['ada_ece'].append(adaece_criterion(probs, labels, return_tensor=False))
                        metrics_by_site['cls_ece'].append(clsece_criterion(probs, labels, return_tensor=False))
                    else:
                        raise ValueError(f'Not support {task}.')

                for metric in metrics_by[task]:
                    metrics_by[task][metric][t] = metrics_by_site[metric]
        metrics_grading_map[grading] = metrics_by
        result_collector_agg_grading_map[grading] = result_collector_agg

    return result_collector_agg_grading_map, metrics_grading_map


def convert_metrics_to_dataframe(cfg, result_collector, accumulated_metrics):
    grading = cfg.grading
    tasks = []
    if cfg.prognosis_coef > 0.0:
        tasks.append('pn')
    if cfg.diag_coef > 0.0:
        tasks.append(grading)

    for grading in cfg.parser.output:
        if f"{grading}:pn" in accumulated_metrics or ('pn' in accumulated_metrics and grading == 'KL'):
            if f"{grading}:pn" in accumulated_metrics:
                key1 = f"{grading}:pn"
            else:
                key1 = 'pn'
            if accumulated_metrics[key1] is not None and 'ID_by' in accumulated_metrics[key1] and \
                    np.array(accumulated_metrics[key1]['ID_by']).size > 0:
                seq_len = len(accumulated_metrics[key1]['label_by'])
                for t in range(seq_len):
                    df = pd.DataFrame()
                    num_samples = len(accumulated_metrics[key1]['label_by'][t])
                    df['ID'] = list(accumulated_metrics[key1]['ID_by'][t])
                    df['site'] = [cfg.seed] * len(df.index)
                    df[f'{grading}_label'] = list(accumulated_metrics[key1]['label_by'][t])
                    probs = np.concatenate(accumulated_metrics[key1]['softmax_by'][t], 0)
                    if num_samples > 0:
                        df[f'{grading}_prob'] = np.array_split(probs, num_samples, axis=0)
                        if f'{grading}_{t + 1}' not in result_collector:
                            result_collector[f'{grading}_{t + 1}'] = df
                        else:
                            result_collector[f'{grading}_{t + 1}'] = pd.concat(
                                (result_collector[f'{grading}_{t + 1}'], df),
                                ignore_index=True, sort=False)
        if grading in accumulated_metrics:
            key = grading
            if accumulated_metrics[key] is not None and 'ID' in accumulated_metrics[key] and \
                    np.array(accumulated_metrics[key]['ID']).size > 0:
                df = pd.DataFrame()
                num_samples = len(accumulated_metrics[key]['ID'])
                df['ID'] = list(accumulated_metrics[key]['ID'])
                df[f'{key}_label'] = list(accumulated_metrics[key]['label'])
                probs = np.concatenate(accumulated_metrics[key]['softmax'], 0)
                df[f'{key}_prob'] = np.array_split(probs, num_samples, axis=0)
                if num_samples > 0:
                    if key not in result_collector:
                        result_collector[f'{key}_0'] = df
                    else:
                        result_collector[f'{key}_0'] = pd.concat((result_collector[f'{key}_0'], df), ignore_index=True,
                                                                 sort=False)

    return result_collector


@hydra.main(config_path="configs/config_eval.yaml")
def main(cfg):
    wdir = os.environ['PWD']
    if not os.path.isabs(cfg.meta_root):
        cfg.meta_root = os.path.join(wdir, cfg.meta_root)

    if not os.path.isdir(cfg.snapshots):
        os.makedirs(cfg.snapshots)

    if cfg.eval.root:
        eval_from_saved_folders(cfg, cfg.eval.root, device, cfg.eval.patterns)
    else:
        loaders = load_data(cfg, cfg.site)
        eval(cfg.pretrained_model, loaders, cfg, device, True)


if __name__ == "__main__":
    main()
