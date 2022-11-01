import random

import coloredlogs
import hydra
import yaml
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, \
    mean_squared_error, cohen_kappa_score
from tqdm import tqdm

# from common import copy_src
from common.ece import ECELoss, AdaptiveECELoss, ClasswiseECELoss
from common.itemloader import ItemLoader
from common.utils import proc_targets, calculate_metric, load_metadata, init_mean_std, \
    store_model, count_parameters
from prognosis.models import create_model
from prognosis.utils import *

coloredlogs.install()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

task_names = ('pn', 'all',)
task2metrics = {'grading': ['ba', 'ka', 'mauc', 'ba.ka', 'ece', 'eca', 'ada_ece'],
                'pn': ['ba', 'mse', 'mauc', 'ece', 'eca', 'ada_ece', 'cls_ece', 'loss'],
                'all': ['loss']}

stored_models = {}


@hydra.main(config_path="configs/config_train.yaml")
def main(cfg):
    if int(cfg.seed) < 0:
        cfg.seed = random.randint(0, 1000000)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # update_max_grades(cfg)
    if not cfg.use_oarsi:
        cfg.parser.output = [cfg.grading]

    wdir = os.environ['PWD']
    if not os.path.isabs(cfg.meta_root):
        cfg.meta_root = os.path.join(wdir, cfg.meta_root)

    if not os.path.isdir(cfg.snapshots):
        os.makedirs(cfg.snapshots, exist_ok=True)

    print(OmegaConf.to_yaml(cfg))

    with open("args.yaml", "w") as f:
        yaml.dump(OmegaConf.to_container(cfg), f, default_flow_style=False)

    # Load and split data
    oai_site_folds, oai_meta, _ = load_metadata(cfg, proc_targets=proc_targets, modes='train')

    # Compute mean and std of OAI
    oai_mean, oai_std = init_mean_std(cfg, wdir, oai_meta, parse_img)
    print(f'Mean: {oai_mean}\nStd: {oai_std}')

    oai_meta.describe()
    # Copy src files
    # copy_src(wdir, __file__, dst_dir="src")

    df_train, df_val = oai_site_folds[cfg.fold_index - 1]

    loaders = dict()

    for stage, df in zip(['train', 'eval'], [df_train, df_val]):
        df['visit'] = df['visit'].astype(int)
        if stage == 'eval' and cfg.use_only_baseline:
            df = df[df['visit_id'] == 0]
        loaders[f'oai_{stage}'] = ItemLoader(
            meta_data=df, root=cfg.root, batch_size=cfg.bs, num_workers=cfg.num_workers,
            transform=init_transforms(oai_mean, oai_std)[stage], parser_kwargs=cfg.parser,
            parse_item_cb=parse_item_progs, shuffle=True if stage == "train" else False, drop_last=False)

    model = create_model(cfg, device)

    print(model)
    n_trainable_params, n_params = count_parameters(model)
    print(f'Trainable parameters: {n_trainable_params}/{n_params}')

    if cfg.pretrained_model and not os.path.exists(cfg.pretrained_model):
        log.fatal(f'Cannot find pretrained model {cfg.pretrained_model}')
        assert False
    elif cfg.pretrained_model:
        log.info(f'Loading pretrained model {cfg.pretrained_model}')
        try:
            if cfg.method_name == 'medbert_prog':
                check_point = torch.load(cfg.pretrained_model)
                model.medbert.load_state_dict(check_point['model_state_dict'], strict=True)
                print(f'Loaded pretrained MedBERT checkpoint {cfg.pretrained_model}.')
            else:
                model.load_state_dict(torch.load(cfg.pretrained_model), strict=True)
        except ValueError:
            log.fatal(f'Failed loading {cfg.pretrained_model}')

    if cfg.method_name == 'medbert_prog':
        model.freeze()

    for epoch_i in range(cfg.n_epochs):
        for stage in ["train", "eval"]:
            # for stage in ["eval"]:
            main_loop(loaders[f'oai_{stage}'], epoch_i, model, cfg, stage)


def whether_update_metrics(batch_i, n_iters):
    return batch_i % 10 == 0 or batch_i >= n_iters - 1


def check_y0_exists(cfg):
    return cfg.predict_y0 and cfg.diag_coef > 0


def cherry_picking(input, targets, outputs):
    import matplotlib.pyplot as plt
    seq_len = len(outputs['pn']['label'])
    labels = []
    preds = [np.argmax(outputs['KL']['prob'], -1)]
    kl_probs = outputs['KL']['prob']
    probs = []
    probs_by_fu = []
    for b in range(kl_probs.shape[0]):
        probs_by_fu.append(kl_probs[b, :])
    probs.append(probs_by_fu)

    mask = targets['prognosis_mask']
    for l in range(mask.shape[1]):
        ind = 0
        probs_by_fu = []
        preds_by_fu = []
        for b in range(mask.shape[0]):
            if mask[b, l]:
                probs_by_fu.append(outputs['pn']['prob'][l][ind, :])
                preds_by_fu.append(np.argmax(outputs['pn']['prob'][l][ind, :], -1))
                ind += 1
            else:
                probs_by_fu.append(None)
                preds_by_fu.append(1000)
        probs.append(probs_by_fu)
        preds.append(preds_by_fu)

    labels.append(targets['current_KL'].to('cpu').detach().numpy())
    # preds.append(np.argmax(outputs['KL']['prob'], -1))
    for i in range(seq_len):
        labels.append(torch.flatten(targets['prognosis'][:, i, :]).to('cpu').detach().numpy())
        # preds.append(np.argmax(outputs['pn']['prob'][i], -1).flatten())
    labels = np.stack(labels, 0)
    preds = np.stack(preds, 0)
    # probs = np.stack(probs, 1)

    abs_diff = np.count_nonzero(np.abs(labels - preds), 0)

    kl = [i for i in range(5)]

    for i in range(len(abs_diff)):
        if abs_diff[i] <= 9:
            plt.figure(figsize=(20, 4))
            plt.title('Ground truth')
            for k in range(seq_len + 1):

                plt.subplot(2, seq_len + 1, k + 1)
                plt.axis('off')
                if k == 0:

                    img = input['IMG'][i, 0, :, :].to('cpu').detach().numpy()
                    m = np.min(img)
                    M = np.max(img)
                    img = (img - m) / (M - m)
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    # if k > 0:
                    # img = ndimage.rotate(img, 45 * (labels[k, i] - labels[0, i]), reshape=False)
                    # plt.title(f'{int(labels[k, i])}')
                    # else:
                    #     img = cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0.0, 255.0, 0.0))
                    plt.title(f'[Baseline] {int(labels[k, i])}')
                    plt.imshow(img)
                else:
                    plt.title(f'{int(labels[k, i]) if labels[k, i] >= 0 else "Missing"}')

                if probs[k][i] is not None:
                    plt.subplot(2, seq_len + 1, k + seq_len + 2)
                    if k > 0:
                        plt.yticks([])
                    else:
                        plt.yticks([0, 0.5, 1.0])
                    plt.xticks([0, 1, 2, 3, 4])

                    plt.ylim([0, 1.0])
                    plt.bar(kl, probs[k][i])

            plt.show()
            # output_plot_filename = "./visualizations/"
            # plt.savefig(output_plot_filename, dpi=600, format="pdf")


def filter_metrics(cfg, metrics):
    global task_names, task2metrics
    filtered_metrics = metrics
    if cfg.dataset == "oai":
        task = 'pn'
        for grading in cfg.parser.output:
            # Remove missing/minor follow-ups in OAI
            if grading in ['KL', 'JSL', 'JSM']:
                followups_mask = [True, True, True, True, False, True, False, True]
            else:
                followups_mask = [True, True, True, True, False, False, False, False]

            for metric_name in filtered_metrics[f'{grading}:{task}']:
                _metric = filtered_metrics[f'{grading}:{task}'][metric_name]
                if isinstance(_metric, list) and len(_metric) == len(followups_mask):
                    filtered_metrics[f'{grading}:{task}'][metric_name] = np.array(_metric)[followups_mask].tolist()
    return filtered_metrics


def model_selection(cfg, filtered_metrics, model, epoch_i):
    global stored_models
    # y0
    if False:  # check_y0_exists(cfg):
        gradings = cfg.parser.output
        stored_models = store_model(
            epoch_i, gradings, ['ba', 'ka'], filtered_metrics, stored_models, model, cfg.snapshots, cond="max",
            mode="scalar")

    # Prognosis
    if cfg.prognosis_coef > 0:
        gradings = [f'{grading}:pn' for grading in cfg.parser.output]
        # stored_models = store_model(
        #    epoch_i, gradings, "mse", filtered_metrics, stored_models, model, cfg.snapshots, cond="min",
        #    mode=f"{cfg.model_selection_mode}_rev" if cfg.model_selection_mode == "beta" else cfg.model_selection_mode)
        stored_models = store_model(
            epoch_i, gradings, "ba", filtered_metrics, stored_models, model, cfg.snapshots, cond="max",
            mode=f"{cfg.model_selection_mode}_rev" if cfg.model_selection_mode == "beta" else cfg.model_selection_mode)
        stored_models = store_model(
            epoch_i, gradings, ["ba", "eca"], filtered_metrics, stored_models, model, cfg.snapshots, cond="max",
            mode=f"{cfg.model_selection_mode}_rev" if cfg.model_selection_mode == "beta" else cfg.model_selection_mode)
        # stored_models = store_model(
        #     epoch_i, ['all'], "loss", filtered_metrics, stored_models, model, cfg.snapshots, cond="min", mode="scalar")

    # stored_models = store_model(
    #     epoch_i, 'all', "loss", filtered_metrics, stored_models, model, cfg.snapshots, cond="min", mode="scalar")


def prepare_display_metrics(cfg, display_metrics, metrics_by, grading):
    if check_y0_exists(cfg):
        display_metrics[f'{grading}:ba'] = metrics_by[grading]['ba']
        if cfg.compute_all_metrics:
            display_metrics[f'{grading}:mauc'] = metrics_by[grading]['mauc']
            display_metrics[f'{grading}:mse'] = metrics_by[grading]['mse']
            display_metrics[f'{grading}:ka'] = metrics_by[grading]['ka']
            display_metrics[f'{cfg.grading}:ece'] = metrics_by[grading]['ece']
            display_metrics[f'{cfg.grading}:ada_ece'] = metrics_by[grading]['ada_ece']
            display_metrics[f'{cfg.grading}:cls_ece'] = metrics_by[grading]['cls_ece']

    if cfg.prognosis_coef:
        display_metrics[f'{grading}:pn:ba'] = "-".join(
            [f'{v:.03f}' if v is not None else "" for v in metrics_by[f'{grading}:pn']['ba'].values()])
        if cfg.compute_all_metrics:
            display_metrics[f'{grading}:pn:mse'] = "-".join(
                [f'{v:.03f}' if v is not None else "" for v in metrics_by[f'{grading}:pn']['mse'].values()])
            display_metrics[f'{grading}:pn:mauc'] = "-".join(
                [f'{v:.03f}' if v is not None else "" for v in metrics_by[f'{grading}:pn']['mauc'].values()])
            display_metrics[f'{grading}:pn:ece'] = "-".join(
                [f'{v:.03f}' if v is not None else "" for v in metrics_by[f'{grading}:pn']['ece'].values()])
            display_metrics[f'{grading}:pn:ada_ece'] = "-".join(
                [f'{v:.03f}' if v is not None else "" for v in metrics_by[f'{grading}:pn']['ada_ece'].values()])
            display_metrics[f'{grading}:pn:cls_ece'] = "-".join(
                [f'{v:.03f}' if v is not None else "" for v in metrics_by[f'{grading}:pn']['cls_ece'].values()])

    return display_metrics


def get_masked_IDs(cfg, batch, mask_name, t=None):
    IDs = batch['data']['input']['ID']
    if "classifier" in cfg.method_name and t is not None:
        t = cfg.target_time - 1

    if t is None:
        return [IDs[i] for i in range(len(IDs)) if batch[mask_name][i]]
    else:
        return [IDs[i] for i in range(len(IDs)) if batch[mask_name][i, t]]


def main_loop(loader, epoch_i, model, cfg, stage="train"):
    # global best_kappa_quad, saved_kappa_model_fullname
    global best_bacc, saved_bacc_model_fullname
    global best_f1, saved_f1_model_fullname
    global best_auc, saved_auc_model_fullname
    global best_ap, saved_ap_model_fullname
    global task_names, task2metrics
    global stored_models

    ece_criterion = ECELoss(normalized=True).cuda()
    adaece_criterion = AdaptiveECELoss(normalized=True).cuda()
    clsece_criterion = ClasswiseECELoss(normalized=True).cuda()

    gradings = cfg.parser.output

    n_iters = len(loader)
    progress_bar = tqdm(range(n_iters), total=n_iters, desc=f"{stage}::{epoch_i}")
    accumulated_metrics = {'ID': [], 'loss': [], f'l_{cfg.grading}': [], 'loss_pn': [], 'loss_y0': [],
                           'pn': None, cfg.grading: None, 'tkr': None}

    for grading in gradings:
        for task in task_names:
            accumulated_metrics[f'{grading}:{task}'] = {}
            accumulated_metrics[f'{grading}:{task}']['ID_by'] = [[] for i in range(cfg.seq_len)]
            accumulated_metrics[f'{grading}:{task}']['softmax_by'] = [[] for i in range(cfg.seq_len)]
            accumulated_metrics[f'{grading}:{task}']['prob_by'] = [[] for i in range(cfg.seq_len)]
            accumulated_metrics[f'{grading}:{task}']['pred_by'] = [[] for i in range(cfg.seq_len)]
            accumulated_metrics[f'{grading}:{task}']['label_by'] = [[] for i in range(cfg.seq_len)]

    if check_y0_exists(cfg):
        for grading in gradings:
            accumulated_metrics[grading] = {'ID': [], 'pred': [], 'label': [], 'softmax': [], 'prob': []}

    if stage == "eval":
        model.eval()
        if cfg.method_name == "gssm":
            model.rnn.eval()
    else:
        model.train()
        if cfg.method_name == "gssm":
            model.rnn.train()

    # final_kappa_pr_by = None
    final_metrics = {}

    for batch_i in progress_bar:
        batch = loader.sample(1)[0]

        input = {}
        target = {}
        IDs = batch['data']['input']['ID']
        for key in batch:
            if key != 'data':
                target[key] = batch[key]
            else:
                for in_key in batch['data']['input']:
                    if isinstance(batch['data']['input'][in_key], torch.Tensor):
                        input[in_key] = batch['data']['input'][in_key].to(device)
                    else:
                        input[in_key] = batch['data']['input'][in_key]

        batch_size = len(IDs)
        accumulated_metrics['ID'].extend(IDs)

        losses, outputs = model.fit(input, target, batch_i=batch_i, n_iters=n_iters, epoch_i=epoch_i,
                                    stage=stage)

        # cherry_picking(input, targets, outputs)
        # Metrics
        display_metrics = {}
        for loss_name in losses:
            if losses[loss_name] is not None:
                accumulated_metrics[loss_name].append(losses[loss_name])
                display_metrics[loss_name] = f'{np.array(accumulated_metrics[loss_name]).mean():.03f}'

        metrics_by = {'pn': {}, 'all': {}}

        for grading in gradings:
            metrics_by[grading] = {}
            for task in task_names:
                metrics_by[f'{grading}:{task}'] = {}
                for _name in task2metrics[task]:
                    metrics_by[f'{grading}:{task}'][_name] = {i: None for i in range(cfg.seq_len)}

        accumulated_metrics['loss_pn'].append(losses['loss_pn'])
        accumulated_metrics['loss_y0'].append(losses['loss_y0'])
        accumulated_metrics['loss'].append(losses['loss'])

        for grading in gradings:
            if grading == 'KL':
                n_pn_classes = 5
            else:
                n_pn_classes = 4
            for t in range(cfg.seq_len):
                for task in task_names:
                    if task == "pn" and f'{grading}:{task}' in outputs:  # Prognosis
                        labels = outputs[f'{grading}:{task}']['label'][t].flatten()
                        preds = np.argmax(outputs[f'{grading}:{task}']['prob'][t], axis=-1)
                        probs = outputs[f'{grading}:{task}']['prob'][t]

                        IDs_masked = get_masked_IDs(cfg, batch, f'prognosis_mask_{grading}', t + 1)
                        accumulated_metrics[f'{grading}:{task}']['ID_by'][t].extend(IDs_masked)
                        accumulated_metrics[f'{grading}:{task}']['softmax_by'][t].append(
                            outputs[f'{grading}:{task}']['prob'][t])
                        accumulated_metrics[f'{grading}:{task}']['pred_by'][t].extend(list(preds))
                        accumulated_metrics[f'{grading}:{task}']['prob_by'][t].extend(list(probs))
                        accumulated_metrics[f'{grading}:{task}']['label_by'][t].extend(list(labels.astype(int)))

                        if whether_update_metrics(batch_i, n_iters):
                            # Prognosis
                            metrics_by[f'{grading}:{task}']['ba'][t] = \
                                calculate_metric(balanced_accuracy_score,
                                                 accumulated_metrics[f'{grading}:{task}']['label_by'][t],
                                                 accumulated_metrics[f'{grading}:{task}']['pred_by'][t])
                            if cfg.compute_all_metrics:
                                metrics_by[f'{grading}:{task}']['mauc'][t] = \
                                    calculate_metric(roc_auc_score,
                                                     accumulated_metrics[f'{grading}:{task}']['label_by'][t],
                                                     accumulated_metrics[f'{grading}:{task}']['prob_by'][t],
                                                     average='macro',
                                                     labels=[i for i in range(n_pn_classes)],
                                                     multi_class=cfg.multi_class_mode)

                                metrics_by[f'{grading}:{task}']['mse'][t] = \
                                    calculate_metric(mean_squared_error,
                                                     accumulated_metrics[f'{grading}:{task}']['label_by'][t],
                                                     accumulated_metrics[f'{grading}:{task}']['pred_by'][t])

                            # ECE
                            if len(accumulated_metrics[f'{grading}:{task}']['label_by'][t]) > 0 \
                                    and cfg.compute_all_metrics:
                                pn_probs = torch.tensor(
                                    np.concatenate(accumulated_metrics[f'{grading}:{task}']['softmax_by'][t],
                                                   axis=0)).to(device)
                                pn_labels = torch.tensor(accumulated_metrics[f'{grading}:{task}']['label_by'][t]).to(
                                    device)
                                metrics_by[f'{grading}:pn']['ece'][t] = ece_criterion(pn_probs, pn_labels,
                                                                                      return_tensor=False)
                                metrics_by[f'{grading}:pn']['eca'][t] = 1.0 - metrics_by[f'{grading}:{task}']['ece'][t]
                                metrics_by[f'{grading}:pn']['ada_ece'][t] = adaece_criterion(pn_probs, pn_labels,
                                                                                             return_tensor=False)
                                metrics_by[f'{grading}:pn']['cls_ece'][t] = clsece_criterion(pn_probs, pn_labels,
                                                                                             return_tensor=False)

            # Current KL
            if check_y0_exists(cfg) and grading in outputs and outputs[grading] is not None and \
                    outputs[grading]['label'] is not None:
                IDs_masked = get_masked_IDs(cfg, batch, f'prognosis_mask_{grading}', 0)
                accumulated_metrics[grading]['ID'].extend(IDs_masked)
                accumulated_metrics[grading]['pred'].extend(list(np.argmax(outputs[grading]['prob'], axis=-1)))
                accumulated_metrics[grading]['label'].extend(list(outputs[grading]['label']))
                accumulated_metrics[grading]['softmax'].append(outputs[grading]['prob'])
                accumulated_metrics[grading]['prob'].extend(list(outputs[grading]['prob']))
                if whether_update_metrics(batch_i, n_iters):
                    metrics_by[grading]['ba'] = calculate_metric(balanced_accuracy_score,
                                                                 accumulated_metrics[grading]['label'],
                                                                 accumulated_metrics[grading]['pred'])
                    if cfg.compute_all_metrics:
                        metrics_by[grading]['mse'] = calculate_metric(mean_squared_error,
                                                                      accumulated_metrics[grading]['label'],
                                                                      accumulated_metrics[grading]['pred'])
                        metrics_by[grading]['ka'] = calculate_metric(cohen_kappa_score,
                                                                     accumulated_metrics[grading]['label'],
                                                                     accumulated_metrics[grading]['pred'],
                                                                     weights="quadratic")
                        metrics_by[grading]['mauc'] = calculate_metric(roc_auc_score,
                                                                       accumulated_metrics[grading]['label'],
                                                                       accumulated_metrics[grading]['prob'],
                                                                       average='macro',
                                                                       labels=[i for i in range(n_pn_classes)],
                                                                       multi_class=cfg.multi_class_mode)
                    if len(accumulated_metrics[grading]['label']) > 0 and cfg.compute_all_metrics:
                        grading_probs = torch.tensor(
                            np.concatenate(accumulated_metrics[grading]['softmax'], axis=0)).to(device)
                        grading_labels = torch.tensor(accumulated_metrics[grading]['label']).to(device)
                        metrics_by[grading]['ece'] = ece_criterion(grading_probs, grading_labels, return_tensor=False)
                        metrics_by[grading]['eca'] = 1.0 - metrics_by[grading]['ece']
                        metrics_by[grading]['ada_ece'] = adaece_criterion(grading_probs, grading_labels,
                                                                          return_tensor=False)
                        metrics_by[grading]['cls_ece'] = clsece_criterion(grading_probs, grading_labels,
                                                                          return_tensor=False)

            if whether_update_metrics(batch_i, n_iters):
                display_metrics = prepare_display_metrics(cfg, display_metrics, metrics_by, grading)
                progress_bar.set_postfix(display_metrics)

        # Last batch
        if batch_i >= n_iters - 1:
            final_metrics = metrics_by

    metrics = {'all': {}}
    for grading in gradings:
        metrics[grading] = {}

    task = 'pn'
    for _name in task2metrics[task]:
        list_metrics = []
        for grading in gradings:
            if f'{grading}:{task}' not in metrics:
                metrics[f'{grading}:{task}'] = {}
            if _name in final_metrics[f'{grading}:{task}']:
                list_values = list(final_metrics[f'{grading}:{task}'][_name].values())
                list_values = [v for v in list_values if v is not None]
                metrics[f'{grading}:{task}'][_name] = list_values
                list_metrics.extend(list_values)
            if _name in final_metrics[grading]:
                metrics[grading][_name] = final_metrics[grading][_name]
        if len(list_metrics) > 0:
            metrics['all'][_name] = np.array(list_metrics).mean()
        else:
            metrics['all'][_name] = 0.0

    # Losses
    metrics['all']['loss_pn'] = np.array(accumulated_metrics['loss_pn']).mean()
    metrics['all']['loss'] = np.array(accumulated_metrics['loss']).mean()

    # Store model
    if stage == "eval" and not cfg.skip_store:
        # Remove minor follow-ups
        if 'classifier' not in cfg.method_name:
            filtered_metrics = filter_metrics(cfg, metrics)
        else:
            filtered_metrics = metrics
        model_selection(cfg, filtered_metrics, model, epoch_i)

    return metrics, accumulated_metrics


if __name__ == "__main__":
    main()
