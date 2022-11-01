import os
import random

import logging as log
import coloredlogs
import cv2
import hydra
import numpy as np
import torch
import yaml
from common.itemloader import ItemLoader
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score
from tqdm import tqdm

from adni.models import create_model
from common.adni.utils import init_transforms
from common.adni.utils import load_metadata, parse_item_progs, calculate_class_weights
# from common import copy_src
from common.ece import ECELoss, AdaptiveECELoss, ClasswiseECELoss
from common.utils import calculate_metric, store_model

coloredlogs.install()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

task_names = ('grading', 'pn', 'all')
task2metrics = {'grading': ['ba', 'ac', 'mauc', 'ba.ka', 'ece', 'eca', 'ada_ece', 'cls_ece', 'ba.eca'],
                'pn': ['ba', 'ac', 'mauc', 'ba.eca', 'ece', 'eca', 'ada_ece', 'cls_ece', 'loss'],
                'all': ['loss']}

stored_models = {}
for task in task_names:
    stored_models[task] = {}
    for _name in task2metrics[task]:
        if _name == "mse" or "loss" in _name or 'ece' in _name:
            stored_models[task][_name] = {'best': 1000000.0, "filename": ""}
        else:
            stored_models[task][_name] = {'best': -1, "filename": ""}


@hydra.main(config_path="configs/config_train.yaml")
def main(cfg):
    if int(cfg.seed) < 0:
        cfg.seed = random.randint(0, 1000000)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if isinstance(cfg.train_size_per_class, int):
        cfg.pkl_meta_filename = cfg.pkl_meta_filename[:-4] + f'_{cfg.train_size_per_class}perclass.pkl'

    wdir = os.environ['PWD']
    if not os.path.isabs(cfg.meta_root):
        cfg.meta_root = os.path.join(wdir, cfg.meta_root)

    if not os.path.isdir(cfg.snapshots):
        os.makedirs(cfg.snapshots, exist_ok=True)

    # print(cfg.pretty())
    print(OmegaConf.to_yaml(cfg))

    with open("args.yaml", "w") as f:
        yaml.dump(OmegaConf.to_container(cfg), f, default_flow_style=False)

    # Load and split data
    data_folds = load_metadata(cfg, img_root=cfg.root, meta_root=cfg.meta_root, meta_filename=cfg.meta_filename,
                               pkl_meta_filename=cfg.pkl_meta_filename, seq_len=cfg.seq_len, seed=cfg.seed)

    # Compute mean and std of OAI
    # oai_mean, oai_std = init_mean_std(cfg, wdir, oai_meta, parse_img)
    # print(f'Mean: {oai_mean}\nStd: {oai_std}')

    # oai_meta.describe()
    # Copy src files
    # copy_src(wdir, __file__, dst_dir="src")

    df_train, df_val = data_folds[cfg.fold_index - 1]
    # df_train = df_train[df_train['MRI_filename'].notnull()]
    # df_val = df_val[df_val['MRI_filename'].notnull()]
    print(f'df_train={df_train.describe()}')
    print(f'df_val={df_val.describe()}')
    print(f'Training data:\n{df_train["DXTARGET_0"].value_counts()}')
    print(f'Validation data:\n{df_val["DXTARGET_0"].value_counts()}')

    y0_weights, pn_weights = calculate_class_weights(df_train, cfg)

    loaders = dict()

    for stage, df in zip(['train', 'eval'], [df_train, df_val]):
        # TODO: Undo after debugging
        # if stage == 'train':
        #     df = df.iloc[:40, :]
        loaders[f'oai_{stage}'] = ItemLoader(
            meta_data=df, root=cfg.root, batch_size=cfg.bs, num_workers=cfg.num_workers,
            transform=init_transforms()[stage], parser_kwargs=cfg.parser,
            parse_item_cb=parse_item_progs, shuffle=True if stage == "train" else False, drop_last=False)

    # pn_weights = y0_weights = None
    model = create_model(cfg, device, pn_weights, y0_weights)
    # print(model)

    if cfg.pretrained_model and not os.path.exists(cfg.pretrained_model):
        log.fatal(f'Cannot find pretrained model {cfg.pretrained_model}')
        assert False
    elif cfg.pretrained_model:
        log.info(f'Loading pretrained model {cfg.pretrained_model}')
        try:
            model.load_state_dict(torch.load(cfg.pretrained_model), strict=True)
        except ValueError:
            log.fatal(f'Failed loading {cfg.pretrained_model}')

    for epoch_i in range(cfg.n_epochs):
        # TODO: Debug
        for stage in ["train", "eval"]:
            # for stage in ["eval"]:
            main_loop(loaders[f'oai_{stage}'], epoch_i, model, cfg, stage)


def whether_update_metrics(batch_i, n_iters):
    return batch_i % 10 == 0 or batch_i >= n_iters - 1


def check_y0_exists(cfg):
    return cfg.diag_coef > 0


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
        # Remove missing/minor follow-ups in OAI
        followups_mask = [True, True, True, False, False, True, False, True]
        for task in task_names:
            if task == 'grading':
                continue
            for metric_name in task2metrics[task]:
                _metric = filtered_metrics[task][metric_name]
                if isinstance(_metric, list):
                    filtered_metrics[task][metric_name] = np.array(_metric)[followups_mask].tolist()
    return filtered_metrics


def model_selection(cfg, filtered_metrics, model, epoch_i):
    global stored_models
    # y0
    if check_y0_exists(cfg):
        stored_models = store_model(
            epoch_i, 'grading', "mauc", filtered_metrics, stored_models, model, cfg.snapshots, cond="max",
            mode="scalar")
    # Prognosis
    if cfg.prognosis_coef > 0:
        # stored_models = store_model(
        #    epoch_i, 'pn', "mse", filtered_metrics, stored_models, model, cfg.snapshots, cond="min",
        #    mode=f"{cfg.model_selection_mode}_rev" if cfg.model_selection_mode == "beta" else cfg.model_selection_mode)
        stored_models = store_model(
            epoch_i, 'pn', "ba", filtered_metrics, stored_models, model, cfg.snapshots, cond="max",
            mode=f"{cfg.model_selection_mode}_rev" if cfg.model_selection_mode == "beta" else cfg.model_selection_mode)
        stored_models = store_model(
            epoch_i, 'pn', "mauc", filtered_metrics, stored_models, model, cfg.snapshots, cond="max",
            mode=f"{cfg.model_selection_mode}_rev" if cfg.model_selection_mode == "beta" else cfg.model_selection_mode)
        # stored_models = store_model(
        #    epoch_i, 'pn', "ba.eca", filtered_metrics, stored_models, model, cfg.snapshots, cond="max",
        #    mode=f"{cfg.model_selection_mode}_rev" if cfg.model_selection_mode == "beta" else cfg.model_selection_mode)
        # stored_models = store_model(
        #     epoch_i, 'pn', "loss", filtered_metrics, stored_models, model, cfg.snapshots, cond="min", mode="scalar")

    # stored_models = store_model(
    #     epoch_i, 'all', "loss", filtered_metrics, stored_models, model, cfg.snapshots, cond="min", mode="scalar")


def prepare_display_metrics(cfg, display_metrics, metrics_by):
    if check_y0_exists(cfg):
        display_metrics[f'{cfg.grading}:ba'] = metrics_by['grading']['ba']
        display_metrics[f'{cfg.grading}:ac'] = metrics_by['grading']['ac']
        display_metrics[f'{cfg.grading}:mauc'] = metrics_by['grading']['mauc']
        # display_metrics[f'{cfg.grading}:ka'] = metrics_by['grading']['ka']
        if cfg.display_ece:
            display_metrics[f'{cfg.grading}:ece'] = metrics_by['grading']['ece']
            display_metrics[f'{cfg.grading}:ada_ece'] = metrics_by['grading']['ada_ece']
            display_metrics[f'{cfg.grading}:cls_ece'] = metrics_by['grading']['cls_ece']

    if cfg.prognosis_coef:
        display_metrics[f'pn:ba'] = "-".join(
            [f'{v:.03f}' if v is not None else "" for v in metrics_by['pn']['ba'].values()])
        display_metrics[f'pn:ac'] = "-".join(
            [f'{v:.03f}' if v is not None else "" for v in metrics_by['pn']['ac'].values()])
        display_metrics[f'pn:mauc'] = "-".join(
            [f'{v:.03f}' if v is not None else "" for v in metrics_by['pn']['mauc'].values()])
        # display_metrics[f'pn:mse'] = "-".join(
        #     [f'{v:.03f}' if v is not None else "" for v in metrics_by['pn']['mse'].values()])
        if cfg.display_ece:
            display_metrics[f'pn:ece'] = "-".join(
                [f'{v:.03f}' if v is not None else "" for v in metrics_by['pn']['ece'].values()])
            display_metrics[f'pn:ada_ece'] = "-".join(
                [f'{v:.03f}' if v is not None else "" for v in metrics_by['pn']['ada_ece'].values()])
            display_metrics[f'pn:cls_ece'] = "-".join(
                [f'{v:.03f}' if v is not None else "" for v in metrics_by['pn']['cls_ece'].values()])

    return display_metrics


def get_masked_IDs(cfg, batch, mask_name, t=None):
    IDs = batch['data']['input']['ID']
    if "classifier" in cfg.method_name and t is not None:
        t = cfg.target_time - 1

    if t is None:
        return [IDs[i] for i in range(len(IDs)) if batch[mask_name][i, 0]]
    else:
        return [IDs[i] for i in range(len(IDs)) if batch[mask_name][i, t + 1]]


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

    n_iters = len(loader)
    progress_bar = tqdm(range(n_iters), total=n_iters, desc=f"{stage}::{epoch_i}")
    accumulated_metrics = {'ID': [], 'loss': [], f'l_{cfg.grading}': [], 'loss_pn': [], 'loss_y0': [],
                           'pn': None, cfg.grading: None}
    for task in task_names:
        accumulated_metrics[task] = {}
        accumulated_metrics[task]['ID_by'] = [[] for i in range(cfg.seq_len)]
        accumulated_metrics[task]['softmax_by'] = [[] for i in range(cfg.seq_len)]
        accumulated_metrics[task]['prob_by'] = [[] for i in range(cfg.seq_len)]
        accumulated_metrics[task]['pred_by'] = [[] for i in range(cfg.seq_len)]
        accumulated_metrics[task]['label_by'] = [[] for i in range(cfg.seq_len)]

    if cfg.predict_current_KL:
        accumulated_metrics[cfg.grading] = {'ID': [], 'pred': [], 'label': [], 'softmax': [], 'prob': []}

    if stage == "eval":
        model.eval()
    else:
        model.train()

    # final_kappa_pr_by = None
    final_metrics = {}

    for batch_i in progress_bar:
        batch = loader.sample(1)[0]

        IDs = batch['data']['input']['ID']
        accumulated_metrics['ID'].extend(IDs)
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

        out_seq_len = cfg.seq_len

        input['label_len'] = torch.tensor([out_seq_len] * batch_size, dtype=torch.int32).to(device)

        # Target
        targets = {}
        for k in batch:
            if "data" not in k and isinstance(batch[k], torch.Tensor):
                targets[k] = batch[k].to(device)

        losses, outputs = model.fit(input=input, target=targets, batch_i=batch_i, n_iters=n_iters, epoch_i=epoch_i,
                                    stage=stage)

        # cherry_picking(input, targets, outputs)
        # Metrics
        display_metrics = {}
        for loss_name in losses:
            if losses[loss_name] is not None:
                accumulated_metrics[loss_name].append(losses[loss_name])
                display_metrics[loss_name] = f'{np.array(accumulated_metrics[loss_name]).mean():.03f}'

        metrics_by = {'pn': {}, cfg.grading: {}, 'grading': {}}
        for task in task_names:
            metrics_by[task] = {}
            for _name in task2metrics[task]:
                if task != 'grading':
                    metrics_by[task][_name] = {i: None for i in range(out_seq_len)}
                else:
                    metrics_by[task][_name] = None

        accumulated_metrics['loss_pn'].append(losses['loss_pn'])
        accumulated_metrics['loss_y0'].append(losses['loss_y0'])
        accumulated_metrics['loss'].append(losses['loss'])

        for t in range(cfg.seq_len):
            for task in task_names:
                if task == "pn" and task in outputs:  # Prognosis
                    labels = outputs[task]['label'][t].flatten()

                    preds = np.argmax(outputs[task]['prob'][t], axis=-1)
                    probs = outputs[task]['prob'][t]
                    accumulated_metrics[task]['prob_by'][t].extend(list(probs))

                    IDs_masked = get_masked_IDs(cfg, batch, 'prognosis_mask_DXTARGET', t)
                    if len(IDs_masked) != outputs[task]['prob'][t].shape[0]:
                        print('Unmatched!')
                    accumulated_metrics[task]['ID_by'][t].extend(IDs_masked)
                    accumulated_metrics[task]['softmax_by'][t].append(outputs[task]['prob'][t])
                    accumulated_metrics[task]['pred_by'][t].extend(list(preds))
                    accumulated_metrics[task]['label_by'][t].extend(list(labels.astype(int)))
                elif task in task_names:
                    pass
                else:
                    raise ValueError(f'Not support task {task} for not being {task_names}.')

            if whether_update_metrics(batch_i, n_iters):
                # Prognosis
                metrics_by['pn']['ba'][t] = calculate_metric(balanced_accuracy_score,
                                                             accumulated_metrics['pn']['label_by'][t],
                                                             accumulated_metrics['pn']['pred_by'][t])
                metrics_by['pn']['ac'][t] = calculate_metric(accuracy_score,
                                                             accumulated_metrics['pn']['label_by'][t],
                                                             accumulated_metrics['pn']['pred_by'][t])
                metrics_by['pn']['mauc'][t] = calculate_metric(roc_auc_score,
                                                               accumulated_metrics['pn']['label_by'][t],
                                                               accumulated_metrics['pn']['prob_by'][t],
                                                               multi_class=cfg.rocauc_mode)
                # metrics_by['pn']['mse'][t] = calculate_metric(mean_squared_error,
                #                                               accumulated_metrics['pn']['label_by'][t],
                #                                               accumulated_metrics['pn']['pred_by'][t])

                if len(accumulated_metrics['pn']['label_by'][t]) > 0 and cfg.display_ece:
                    pn_probs = torch.tensor(np.concatenate(accumulated_metrics['pn']['softmax_by'][t], axis=0)).to(
                        device)
                    pn_labels = torch.tensor(accumulated_metrics['pn']['label_by'][t]).to(device)
                    metrics_by['pn']['ece'][t] = ece_criterion(pn_probs, pn_labels, return_tensor=False)
                    metrics_by['pn']['eca'][t] = 1.0 - metrics_by['pn']['ece'][t]
                    metrics_by['pn']['ada_ece'][t] = adaece_criterion(pn_probs, pn_labels, return_tensor=False)
                    metrics_by['pn']['cls_ece'][t] = clsece_criterion(pn_probs, pn_labels, return_tensor=False)

        # Current KL
        if check_y0_exists(cfg) and cfg.grading in outputs and outputs[cfg.grading] is not None and \
                outputs[cfg.grading]['label'] is not None:
            IDs_masked = get_masked_IDs(cfg, batch, f'prognosis_mask_{cfg.grading}')
            accumulated_metrics[cfg.grading]['ID'].extend(IDs_masked)
            accumulated_metrics[cfg.grading]['pred'].extend(list(np.argmax(outputs[cfg.grading]['prob'], axis=-1)))
            accumulated_metrics[cfg.grading]['label'].extend(list(outputs[cfg.grading]['label']))
            accumulated_metrics[cfg.grading]['softmax'].append(outputs[cfg.grading]['prob'])
            accumulated_metrics[cfg.grading]['prob'].extend(list(outputs[cfg.grading]['prob']))
            if whether_update_metrics(batch_i, n_iters):
                metrics_by['grading']['ba'] = calculate_metric(balanced_accuracy_score,
                                                               accumulated_metrics[cfg.grading]['label'],
                                                               accumulated_metrics[cfg.grading]['pred'])
                metrics_by['grading']['ac'] = calculate_metric(accuracy_score,
                                                               accumulated_metrics[cfg.grading]['label'],
                                                               accumulated_metrics[cfg.grading]['pred'])
                metrics_by['grading']['mauc'] = calculate_metric(roc_auc_score,
                                                                 accumulated_metrics[cfg.grading]['label'],
                                                                 accumulated_metrics[cfg.grading]['prob'],
                                                                 multi_class=cfg.rocauc_mode)
                if len(accumulated_metrics[cfg.grading]['label']) > 0 and cfg.display_ece:
                    grading_probs = torch.tensor(
                        np.concatenate(accumulated_metrics[cfg.grading]['softmax'], axis=0)).to(device)
                    grading_labels = torch.tensor(accumulated_metrics[cfg.grading]['label']).to(device)
                    metrics_by['grading']['ece'] = ece_criterion(grading_probs, grading_labels, return_tensor=False)
                    metrics_by['grading']['eca'] = 1.0 - metrics_by['grading']['ece']
                    metrics_by['grading']['ada_ece'] = adaece_criterion(grading_probs, grading_labels,
                                                                        return_tensor=False)
                    metrics_by['grading']['cls_ece'] = clsece_criterion(grading_probs, grading_labels,
                                                                        return_tensor=False)

        if whether_update_metrics(batch_i, n_iters):
            display_metrics = prepare_display_metrics(cfg, display_metrics, metrics_by)
            progress_bar.set_postfix(display_metrics)

        # Last batch
        if batch_i >= n_iters - 1:
            final_metrics = metrics_by

    metrics = {'all': {}}
    for task in task_names:
        metrics[task] = {}
        for _name in task2metrics[task]:
            if _name in final_metrics[task]:
                if task == 'grading':
                    metrics[task][_name] = final_metrics[task][_name]
                else:
                    metrics[task][_name] = list(final_metrics[task][_name].values())

    # Losses
    metrics['pn']['loss'] = np.array(accumulated_metrics['loss_pn']).mean()
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
