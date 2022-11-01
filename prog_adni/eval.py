import json
import logging as log
import os
import pickle

import coloredlogs
import hydra
import numpy as np
import pandas as pd
import torch
from common.itemloader import ItemLoader
from omegaconf import OmegaConf
import random
from omegaconf.listconfig import ListConfig
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score, mean_squared_error

from adni.models import create_model
from adni.train import main_loop
from common.adni.utils import init_transforms
from common.adni.utils import load_metadata, parse_item_progs
from common.ece import ECELoss, AdaptiveECELoss, ClasswiseECELoss
from common.utils import calculate_metric

coloredlogs.install()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

task_names = ('grading', 'pn')
task2metrics = {'grading': ['ba', 'ac', 'mauc'], 'pn': ['ba', 'ac', 'mauc']}

def load_data(cfg):
    wdir = os.environ['PWD']
    # Load and split data
    data_folds = load_metadata(cfg, img_root=cfg.root, meta_root=cfg.meta_root, meta_filename=cfg.meta_filename,
                              pkl_meta_filename=cfg.pkl_meta_filename, seq_len=cfg.seq_len, seed=cfg.seed, eval_only=True)

    _, test_data = data_folds[cfg.fold_index - 1]
    # Loaders
    if cfg.use_only_baseline:
        print(f'[INFO] Evaluating only initial data at baseline!')
        test_data = test_data[test_data['VISCODE'] == 'm00']

    loader = ItemLoader(
        meta_data=test_data, root=cfg.root, batch_size=cfg.bs, num_workers=cfg.num_workers,
        transform=init_transforms()["eval"], parser_kwargs=cfg.parser,
        parse_item_cb=parse_item_progs, shuffle=False, drop_last=False)
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

    metrics, accumulated_metrics = main_loop(loader, 0, model, cfg, "test")

    if store:
        with open("eval_metrics.json", "w") as f:
            json.dump(metrics, f)

    return metrics, accumulated_metrics


def eval_from_saved_folders(cfg, root_dir, device, patterns="bacc"):
    d = os.listdir(root_dir)
    collector = {}
    result_collector = {}
    for dir in d:
        args_fullname = os.path.join(root_dir, dir, "args.yaml")

        if os.path.isfile(args_fullname):
            config = OmegaConf.load(args_fullname)
        else:
            raise ValueError(f'Not found {args_fullname}')

        if cfg.seed != config["seed"]:
            print(f'[{cfg.seed}] Skip {dir}.')
            continue

        if "grading" not in config["parser"]:
            config["parser"]["grading"] = config["grading"]

        or_config_names = ["root", "pkl_meta_filename", "meta_root", "bs", "num_workers", "dataset", "root",
                           "meta_filename", "use_y0_class_weights", "use_pn_class_weights", "log_dir",
                           "use_only_grading", "use_only_baseline", "model_selection_mode", "save_attn", "save_preds", "display_ece"]
        eval_config_names = ['output', 'root', 'patterns', 'n_resamplings', 'mode']
        for k in or_config_names:
            config[k] = cfg[k]

        config.eval = {}
        for k in eval_config_names:
            config.eval[k] = cfg.eval[k]

        # config.pkl_meta_filename = f"test_{config.grading}_ADNI_D2.pkl"
        config.skip_store = True

        #print(config.pretty())
        print(OmegaConf.to_yaml(cfg))

        loader = load_data(config)

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

                    key = f'Fold {config.fold_index}'

                    print(f'{key}, model: {model_fullname}...')

                    metrics, accumulated_metrics = eval(model_fullname, loader, config, device, False)

                    result_collector = convert_metrics_to_dataframe(config, result_collector, accumulated_metrics)
                    collector[key] = metrics
                    continue

    output_fullname = os.path.abspath(cfg.eval.output)
    if config.save_preds:
        with open(output_fullname[:-4] + "_preds.pkl") as f:
            pickle.dump(result_collector, f, protocol=4)

    result_collector_agg, metrics_by = aggregate_dataframe(cfg, result_collector)

    print(f'Saving output files to {output_fullname}')
    with open(output_fullname, "w") as f:
        json.dump(metrics_by, f)


def get_sites(cfg, df):
    if cfg.dataset == "adni":
        if cfg.eval.mode == 'by_fold':
            enum_sites = list(df['fold_index'].unique())
            sites = df['fold_index'].tolist()
        elif cfg.eval.mode == 'by_seed':
            enum_sites = list(df['seed'].unique())
            sites = df['seed'].tolist()
        else:
            raise ValueError(f'Not support eval mode {cfg.eval.mode}.')
    else:
        raise ValueError(f'Not support dataset {cfg.dataset}')
    return sites, enum_sites


def aggregate_dataframe(cfg, result_collector):
    ece_criterion = ECELoss(normalized=True).cuda()
    adaece_criterion = AdaptiveECELoss(normalized=True).cuda()
    clsece_criterion = ClasswiseECELoss(normalized=True).cuda()
    grading = cfg.grading
    tasks = ['pn']
    result_collector_agg = {}

    metrics_by = {}

    metrics_by[grading] = {'ba': -1.0, 'ac': -1.0, 'mse': -1.0, 'mauc': -1.0, 'ece': -1.0, 'ada_ece': -1.0, 'cls_ece': -1.0}

    metrics_by['pn'] = {'ba': [-1.0] * cfg.seq_len,
                        'ac': [-1.0] * cfg.seq_len,
                        'mse': [-1.0] * cfg.seq_len,
                        'mauc': [-1.0] * cfg.seq_len,
                        'ece': [-1.0] * cfg.seq_len,
                        'ada_ece': [-1.0] * cfg.seq_len,
                        'cls_ece': [-1.0] * cfg.seq_len}

    if grading in result_collector:
        result_collector_agg[grading] = result_collector[grading].groupby('ID', as_index=False)[
            'seed', 'fold_index', f'{grading}_label', f'{grading}_prob'].apply(np.mean)

        result_collector_agg[grading]['Site'], list_sites = get_sites(cfg, result_collector_agg[grading])

        metrics_by_site = {'ba': [], 'ac': [], 'rmse': [], 'mauc': [], 'ece': [], 'ada_ece': [], 'cls_ece': []}
        for site in list_sites:
            result_agg_site = result_collector_agg[grading][result_collector_agg[grading]['Site'] == site]
            if len(result_agg_site.index) == 0:
                print(f'Cannot find data of Side {site}.')
                continue
            labels = result_agg_site[f'{grading}_label'].tolist()
            probs = [v[0].tolist() for v in result_agg_site[f'{grading}_prob'].tolist()]
            probs = np.stack(probs, 0)
            preds = list(np.argmax(probs, -1))

            metrics_by_site['ba'].append(calculate_metric(balanced_accuracy_score, labels, preds))
            metrics_by_site['ac'].append(calculate_metric(accuracy_score, labels, preds))
            metrics_by_site['mse'].append(calculate_metric(mean_squared_error, labels, preds))
            metrics_by_site['mauc'].append(calculate_metric(roc_auc_score, labels, probs, multi_class="ovo"))
            metrics_by_site['ece'].append(ece_criterion(probs, labels, return_tensor=False))
            metrics_by_site['ada_ece'].append(adaece_criterion(probs, labels, return_tensor=False))
            metrics_by_site['cls_ece'].append(clsece_criterion(probs, labels, return_tensor=False))
        for metric in metrics_by[grading]:
            metrics_by[grading][metric] = metrics_by_site[metric]

    # PN
    for task in tasks:
        for t in range(cfg.seq_len):
            if f'{task}_{t}' in result_collector:

                metrics_by_site = {}
                for metric in metrics_by[task]:
                    metrics_by_site[metric] = []

                result_collector_agg[f'{task}_{t}'] = result_collector[f'{task}_{t}'].groupby('ID', as_index=False)[
                    'seed', 'fold_index', f'{task}_label_{t}', f'{task}_prob_{t}'].apply(np.mean)

                result_collector_agg[f'{task}_{t}']['Site'], list_sites = get_sites(cfg, result_collector_agg[
                    f'{task}_{t}'])

                for site in list_sites:
                    result_agg_site = result_collector_agg[f'{task}_{t}'][
                        result_collector_agg[f'{task}_{t}']['Site'] == site]
                    if len(result_agg_site.index) == 0:
                        print(f'Cannot find data of Side {site} for task {task} at follow-up {t}.')
                        continue
                    labels = result_agg_site[f'{task}_label_{t}'].tolist()
                    probs = [v[0].tolist() for v in result_agg_site[f'{task}_prob_{t}'].tolist()]
                    probs = np.stack(probs, 0)
                    preds = list(np.argmax(probs, -1))

                    # Prognosis
                    metrics_by_site['ba'].append(calculate_metric(balanced_accuracy_score, labels, preds))
                    metrics_by_site['ac'].append(calculate_metric(accuracy_score, labels, preds))
                    metrics_by_site['mse'].append(calculate_metric(mean_squared_error, labels, preds))
                    metrics_by_site['mauc'].append(calculate_metric(roc_auc_score, labels, probs, multi_class="ovo"))
                    metrics_by_site['ece'].append(ece_criterion(probs, labels, return_tensor=False))
                    metrics_by_site['ada_ece'].append(adaece_criterion(probs, labels, return_tensor=False))
                    metrics_by_site['cls_ece'].append(clsece_criterion(probs, labels, return_tensor=False))

                for metric in metrics_by[task]:
                    metrics_by[task][metric][t] = metrics_by_site[metric]

    return result_collector_agg, metrics_by


def convert_metrics_to_dataframe(cfg, result_collector, accumulated_metrics):
    grading = cfg.grading
    tasks = [grading, 'pn']
    for key in accumulated_metrics:
        if key in tasks:
            if key != grading and \
                    accumulated_metrics[key] is not None and \
                    'ID_by' in accumulated_metrics[key] and \
                    np.array(accumulated_metrics[key]['ID_by']).size > 0:
                seq_len = len(accumulated_metrics[key]['label_by'])
                for t in range(seq_len):
                    df = pd.DataFrame()
                    #print(accumulated_metrics)
                    print(f't={t}')
                    for kkk in accumulated_metrics[key]:
                        print(f'kkk={kkk}, len={len(accumulated_metrics[key][kkk][t])}')
                    num_samples = len(accumulated_metrics[key]['ID_by'][t])
                    df['ID'] = list(accumulated_metrics[key]['ID_by'][t])
                    df['seed'] = [cfg.seed] * num_samples
                    df['fold_index'] = [cfg.fold_index] * num_samples
                    df[f'{key}_label_{t}'] = list(accumulated_metrics[key]['label_by'][t])
                    probs = np.concatenate(accumulated_metrics[key]['softmax_by'][t], 0)
                    if num_samples > 0:
                        df[f'{key}_prob_{t}'] = np.array_split(probs, num_samples, axis=0)
                        if f'{key}_{t}' not in result_collector:
                            result_collector[f'{key}_{t}'] = df
                        else:
                            result_collector[f'{key}_{t}'] = pd.concat((result_collector[f'{key}_{t}'], df),
                                                                       ignore_index=True, sort=False)
            elif accumulated_metrics[key] is not None and \
                    'ID_by' in accumulated_metrics[key] and \
                    np.array(accumulated_metrics[key]['ID_by']).size > 0:
                df = pd.DataFrame()
                num_samples = len(accumulated_metrics[key]['ID'])
                df['ID'] = list(accumulated_metrics[key]['ID'])
                df['seed'] = [cfg.seed] * num_samples
                df['fold_index'] = [cfg.fold_index] * num_samples
                df[f'{key}_label'] = list(accumulated_metrics[key]['label'])
                probs = np.concatenate(accumulated_metrics[key]['softmax'], 0)
                df[f'{key}_prob'] = np.array_split(probs, num_samples, axis=0)
                if num_samples > 0:
                    if key not in result_collector:
                        result_collector[key] = df
                    else:
                        result_collector[key] = pd.concat((result_collector[f'{key}'], df), ignore_index=True, sort=False)

    return result_collector

def fix_randomness(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@hydra.main(config_path="configs", config_name="config_eval")
def main(cfg):
    fix_randomness(cfg.seed)    

    wdir = os.environ['PWD']
    if not os.path.isabs(cfg.meta_root):
        cfg.meta_root = os.path.join(wdir, cfg.meta_root)

    if cfg.eval.root:
        eval_from_saved_folders(cfg, cfg.eval.root, device, cfg.eval.patterns)
    else:
        raise ValueError(f'Cannot find {cfg.eval.root}')


if __name__ == "__main__":
    main()
