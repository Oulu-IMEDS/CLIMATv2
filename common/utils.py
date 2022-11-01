import copy
import json
import logging as log
import os
import pickle
import warnings
from random import random

import coloredlogs
import cv2
import numpy as np
import pandas as pd
import solt
import torch
from sas7bdat import SAS7BDAT
from scipy.special import softmax
from scipy.stats import beta
from solt import core as slc, transforms as slt, data as sld
from termcolor import colored
from torch.utils.data import DataLoader
from common.data import FoldSplit, DataFrameDataset
from tqdm import tqdm

coloredlogs.install()

MAX_GRADES = {'KL': 4, 'OSTL': 3, 'OSTM': 3, 'OSFL': 3, 'OSFM': 3, 'JSL': 3, 'JSM': 3, 'angle': 7}


def to_cpu(x: torch.Tensor or torch.cuda.FloatTensor, required_grad=False, use_numpy=True):
    x_cpu = x

    if isinstance(x, torch.Tensor):
        if x.is_cuda:
            if use_numpy:
                x_cpu = x.to('cpu').detach().numpy()
            elif required_grad:
                x_cpu = x.to('cpu')
            else:
                x_cpu = x.to('cpu').required_grad_(False)
        elif use_numpy:
            if x.requires_grad:
                x_cpu = x.detach().numpy()
            else:
                x_cpu = x.numpy()

    return x_cpu


class Compose(object):
    def __init__(self, transforms: list or tuple):
        self.__transforms = transforms

    def __call__(self, x):
        for trf in self.__transforms:
            x = trf(x)

        return x


def update_max_grades(cfg):
    global MAX_GRADES
    MAX_GRADES[cfg.grading] = cfg.n_pn_classes - 1


def count_tensor_elements(x):
    return np.array([l.numel() for l in x]).sum()


class ApplyTransform(object):
    """Applies a callable transform to certain objects in iterable using given indices.

    Parameters
    ----------
    transform: callable
        Callable transform to be applied
    idx: int or tuple or or list None
        Index or set of indices, where the transform will be applied.

    """

    def __init__(self, transform: callable, idx: int or tuple or list = 0):
        self.__transform: callable = transform
        if isinstance(idx, int):
            idx = (idx,)
        self.__idx: int or tuple or list = idx

    def __call__(self, items):
        """
        Applies a transform to the given sequence of elements.

        Uses the locations (indices) specified in the constructor.

        Parameters
        ----------
        items: tuple or list
            Set of items
        Returns
        -------
        result: tuple
            Transformed list

        """

        if self.__idx is None:
            return items

        if not isinstance(items, (tuple, list)):
            if isinstance(items, np.ndarray) or isinstance(items, torch.Tensor):
                items = (items,)
            else:
                raise TypeError

        idx = set(self.__idx)
        res = []
        for i, item in enumerate(items):
            if i in idx:
                res.append(self.__transform(item))
            else:
                res.append(copy.deepcopy(item))

        return tuple(res)


class Normalize(object):
    def __init__(self, mean, std):
        self.__mean = mean
        self.__std = std

    def __call__(self, tensor, inplace=True):
        if not inplace:
            tensor_trf = tensor.copy()
        else:
            tensor_trf = tensor

        if len(tensor_trf.size()) != 3:
            raise ValueError(f'Input tensor must have 3 dimensions (CxHxW), but found {len(tensor_trf)}')

        if tensor_trf.size(0) != len(self.__mean):
            raise ValueError(f'Incompatible number of channels. '
                             f'Mean has {len(self.__mean)} channels, tensor - {tensor_trf.size()}')

        if tensor_trf.size(0) != len(self.__std):
            raise ValueError(f'Incompatible number of channels. '
                             f'Std has {len(self.__mean)} channels, tensor - {tensor_trf.size()}')

        for channel in range(tensor_trf.size(0)):
            tensor_trf[channel, :, :] -= self.__mean[channel]
            tensor_trf[channel, :, :] /= self.__std[channel]

        return tensor_trf


def parse_item_img_prog(root, entry, trf, data_key, target_key):
    mean = [0.51109564, 0.51109564, 0.51109564]
    std = [0.28390905, 0.28390905, 0.28390905]
    img1_filename = f"{entry['ID']}_{entry['visit1']:02d}_{entry['Side']}.png"
    img2_filename = f"{entry['ID']}_{entry['visit2']:02d}_{entry['Side']}.png"
    img1_fullname = os.path.join(root, img1_filename)
    img2_fullname = os.path.join(root, img2_filename)
    img1 = cv2.imread(img1_fullname, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_fullname, cv2.IMREAD_GRAYSCALE)

    kl = int(entry['KL3'])
    progressor = int(entry['Progressor'])
    trf_img1 = trf({'image': img1}, normalize=True, mean=mean, std=std)['image']
    trf_img2 = trf({'image': img2}, normalize=True, mean=mean, std=std)['image']

    # Assume that d_fu = 12k
    dt1 = entry['d_fu3'] // 12
    dt2 = (entry['d_fu3'] - entry['d_fu2']) // 12
    max_dt = 9
    dt1_ts = torch.tensor([0] * max_dt, dtype=torch.float32)
    dt2_ts = torch.tensor([0] * max_dt, dtype=torch.float32)
    dt1_ts[dt1] = 1.0
    dt2_ts[dt2] = 1.0

    # KLs
    # kl1 = torch.tensor([0] * 5, dtype=torch.float32)
    # kl2 = torch.tensor([0] * 5, dtype=torch.float32)
    # kl1[int(entry['KL1'])] = 1
    # kl2[int(entry['KL2'])] = 1

    if random() > 0.5:
        input = {'img1': trf_img1, 'img2': trf_img2, 'dt1': dt1_ts, 'dt2': dt2_ts}
        kl1 = int(entry['KL1'])
        kl2 = int(entry['KL2'])
        order = 1
    else:
        input = {'img1': trf_img2, 'img2': trf_img1, 'dt1': dt2_ts, 'dt2': dt1_ts}
        kl1 = int(entry['KL2'])
        kl2 = int(entry['KL1'])
        order = 0

    return {'data': {'op': input}, 'order': order, 'future_KL': kl, 'future_prog': progressor, 'KL1': kl1, 'KL2': kl2}


def parse_item_pp_img_prog(root, entry, trf, data_key, target_key):
    output = parse_item_img_prog(root, entry, trf, data_key, target_key)

    input_rev = {'img1': output['data']['op']['img2'].clone(),
                 'img2': output['data']['op']['img1'].clone(),
                 'dt1': output['data']['op']['dt2'],
                 'dt2': output['data']['op']['dt1']}
    output['data_rev'] = {'op': input_rev}
    output['order_rev'] = 1 if output['order'] == 0 else 0

    return output


def init_transforms_019():
    train_trf = solt.Stream([
        slt.Pad(pad_to=(700, 700)),
        slt.Crop(crop_to=(700, 700), crop_mode='c'),
        slt.Resize((310, 310)),
        slt.Noise(p=0.5, gain_range=0.3),
        slt.Rotate(p=1, angle_range=(-10, 10)),
        slt.Crop(crop_to=(300, 300), crop_mode='r'),
        slt.GammaCorrection(p=0.5, gamma_range=(0.5, 1.5)),
        slt.CvtColor(mode='gs2rgb')
    ], interpolation='area', padding='z')

    test_trf = solt.Stream([
        slt.Pad(pad_to=(700, 700)),
        slt.Crop(crop_to=(700, 700), crop_mode='c'),
        slt.Resize((310, 310)),
        slt.Crop(crop_to=(300, 300), crop_mode='c'),
        slt.CvtColor(mode='gs2rgb')
    ], interpolation='area', padding='z')

    return {'train': train_trf, 'eval': test_trf}


def parse_class_prog(o, layer_ind):
    if isinstance(o, dict):
        x = to_cpu(o['prog'])
    elif isinstance(o, list) or isinstance(o, tuple):
        x = o[1][layer_ind]
    else:
        raise ValueError(f'Not support input type {type(o)}')

    x = to_cpu(x)
    if len(x.shape) == 2:
        classes = np.argmax(x, axis=1)
    elif len(x.shape) == 1:
        classes = x
    else:
        raise ValueError(f'Not support input with {len(x)}-dim.')
    return classes


def parse_target_bin_prog(o):
    x = to_cpu(o['prog'])
    if len(x.shape) == 2:
        classes = np.argmax(x, axis=1)
    elif len(x.shape) == 1:
        classes = x
    else:
        raise ValueError(f'Not support input with {len(x)}-dim.')

    bin_cls = []
    for c in classes:
        bin_cls.append(0 if c == 0 else 1)

    return bin_cls


def parse_output_bin_prog(o, layer_ind):
    x = to_cpu(o[1][layer_ind])
    x = softmax(x, axis=1)
    return np.sum(x[:, 1:], axis=1)


def parse_output_bin_kl(o, layer_ind):
    x = to_cpu(o[1][layer_ind])
    x = softmax(x, axis=1)
    return np.sum(x[:, 2:], axis=1)


class CustomParser(object):
    def __init__(self):
        super().__init__()

    def parse_class_kl0(o):
        return parse_class_kl(o, 0)

    def parse_class_kl1(o):
        return parse_class_kl(o, 1)

    def parse_class_kl2(o):
        return parse_class_kl(o, 2)

    def parse_class_kl3(o):
        return parse_class_kl(o, 3)

    def parse_class_kl4(o):
        return parse_class_kl(o, 4)

    def parse_class_prog0(o):
        return parse_class_prog(o, 0)

    def parse_class_prog1(o):
        return parse_class_prog(o, 1)

    def parse_class_prog2(o):
        return parse_class_prog(o, 2)

    def parse_class_prog3(o):
        return parse_class_prog(o, 3)

    def parse_class_prog4(o):
        return parse_class_prog(o, 4)

    def parse_output_bin_prog0(o):
        return parse_output_bin_prog(o, 0)

    def parse_output_bin_prog1(o):
        return parse_output_bin_prog(o, 1)

    def parse_output_bin_prog2(o):
        return parse_output_bin_prog(o, 2)

    def parse_output_bin_prog3(o):
        return parse_output_bin_prog(o, 3)

    def parse_output_bin_prog4(o):
        return parse_output_bin_prog(o, 4)

    def parse_output_bin_kl0(o):
        return parse_output_bin_kl(o, 0)

    def parse_output_bin_kl1(o):
        return parse_output_bin_kl(o, 1)

    def parse_output_bin_kl2(o):
        return parse_output_bin_kl(o, 2)

    def parse_output_bin_kl3(o):
        return parse_output_bin_kl(o, 3)

    def parse_output_bin_kl4(o):
        return parse_output_bin_kl(o, 4)


def parse_class_ouput_kl_op(o):
    kl = to_cpu(o['op']['future_KL'])
    kl = np.argmax(kl, axis=1)
    return kl


def parse_class_ouput_kl1_op(o):
    kl = to_cpu(o['op']['KL1'])
    kl = np.argmax(kl, axis=1)
    return kl


def parse_class_ouput_kl2_op(o):
    kl = to_cpu(o['op']['KL2'])
    kl = np.argmax(kl, axis=1)
    return kl


def parse_class_ouput_order_op(o):
    orders = to_cpu(o['op']['order'])
    order_cls = [1 if orders[i] > 0.5 else 0 for i in range(orders.shape[0])]
    return order_cls


def parse_class_ouput_prog_op(o):
    prog = to_cpu(o['op']['future_prog'])
    return np.argmax(prog, axis=1)
    # return parse_class_ouput_bin_prog_op(o)


def parse_class_ouput_bin_kl_op(o):
    kl = to_cpu(o['op']['future_KL'])
    kl = np.sum(kl[:, 2:], axis=1)
    oa = [1 if kl[i] > 0.5 else 0 for i in range(kl.shape[0])]
    return oa


def parse_class_ouput_bin_prog_op(o):
    x = to_cpu(o['op']['future_prog'])
    x = softmax(x, axis=1)
    return x[:, 1:].sum(axis=1)
    # prog = to_cpu(o['op']['future_prog'])
    # oa = [1 if prog[i] > 0.5 else 0 for i in range(prog.shape[0])]
    # return oa


def parse_class_target_kl_op(o):
    return o['future_KL']


def parse_class_target_kl1_op(o):
    return o['KL1']


def parse_class_target_kl2_op(o):
    return o['KL2']


def parse_class_target_order_op(o):
    return o['order']


def parse_class_target_prog_op(o):
    if 'future_prog' in o:
        return o['future_prog']
    return None


def parse_class_target_bin_kl_op(o):
    kl = to_cpu(o['future_KL'])
    oa = [1 if k > 1 else 0 for k in kl]
    return oa


def parse_class_target_bin_prog_op(o):
    # return o['future_prog']
    x = to_cpu(o['future_prog'])
    if len(x.shape) == 2:
        classes = np.argmax(x, axis=1)
    elif len(x.shape) == 1:
        classes = x
    else:
        raise ValueError(f'Not support input with {len(x)}-dim.')

    bin_cls = []
    for c in classes:
        bin_cls.append(0 if c == 0 else 1)

    return bin_cls


def parse_class_kl(o, layer_ind):
    if isinstance(o, dict):
        x = o['KL']
    elif isinstance(o, list) or isinstance(o, tuple):
        x = o[0][layer_ind]
        if x is None:
            return None
    else:
        raise ValueError(f'Not support input type {type(o)}')

    x = to_cpu(x)
    if len(x.shape) == 2:
        classes = np.argmax(x, axis=1)
    elif len(x.shape) == 1:
        classes = x
    else:
        raise ValueError(f'Not support input with {len(x)}-dim.')
    return classes


def read_sas7bdata_pd(fname):
    data = []
    with SAS7BDAT(fname) as f:
        for row in f:
            data.append(row)

    return pd.DataFrame(data[1:], columns=data[0])


def summarize_hiar_splitter(split_data, classes, target_col):
    for i, (train, val) in enumerate(split_data):
        num_train = np.array([len(l) for l in train]).sum()
        num_val = np.array([len(l) for l in val]).sum()
        log.info(f'Fold {i} has {num_train} and {num_val} training and validation samples.')
        for j, t in enumerate(train):
            t_n_per_cls = []
            for cls in classes:
                n_per_cls = len(t[t[target_col] == cls])
                t_n_per_cls.append(n_per_cls)
            log.info(f'--[Train] Chunk {j} has {t_n_per_cls} per class')

        for j, t in enumerate(val):
            t_n_per_cls = []
            for cls in classes:
                n_per_cls = len(t[t[target_col] == cls])
                t_n_per_cls.append(n_per_cls)
            log.info(f'--[Val] Chunk {j} has {t_n_per_cls} per class')


def summarize_splitter(split_data, classes, target_col):
    for i, (train, val) in enumerate(split_data):
        num_train = np.array([len(l) for l in train]).sum()
        num_val = np.array([len(l) for l in val]).sum()
        log.info(f'Fold {i} has {num_train} and {num_val} training and validation samples.')

        t_n_per_cls = []
        for cls in classes:
            n_per_cls = len(train[train[target_col] == cls])
            t_n_per_cls.append(n_per_cls)
        log.info(f'--[Train] {t_n_per_cls} per class')

        t_n_per_cls = []
        for cls in classes:
            n_per_cls = len(val[val[target_col] == cls])
            t_n_per_cls.append(n_per_cls)
        log.info(f'--[Val] {t_n_per_cls} per class')


def proc_targets(df, dataset='oai', *args, **kwargs):
    grading_types = ['KL', 'OSTL', 'OSTM', 'OSFL', 'OSFM', 'JSL', 'JSM']
    max_grades = {'KL': 4, 'OSTL': 3, 'OSTM': 3, 'OSFL': 3, 'OSFM': 3, 'JSL': 3, 'JSM': 3}

    if dataset in ['oai', 'most']:
        lower_y = 1
        upper_y = 9
    else:
        raise ValueError(f'Not support dataset {dataset}!')

    dt = []
    for i in range(lower_y, upper_y):
        dt_key = f'DT_{i}y'
        if dt_key not in df.columns:
            df[dt_key] = float(i)
        df[dt_key] = df[dt_key].astype(float)
        dt.append(df[dt_key].tolist())
    dt = np.array(dt)
    dt = np.transpose(dt, (1, 0))

    df[f'DT'] = list(dt)

    for grading in grading_types:
        for i in range(lower_y, upper_y):
            grade_key = f'{grading}_{i}y'

            if grade_key not in df.columns:
                df[grade_key] = np.nan

            df[grade_key] = df[grade_key].astype(float)

        prognosis = []

        for i in range(lower_y, upper_y):
            grade_key = f'{grading}_{i}y'
            prognosis.append(df[grade_key].tolist())

        # Prognosis
        # Valid KL: 0 - 4
        # 0,1 -> 0
        # 2 -> 1
        # 3 -> 2
        # 4 -> 3
        # TKR -> 4
        prognosis = np.array(prognosis)
        prognosis = np.transpose(prognosis, (1, 0))
        prognosis = np.nan_to_num(prognosis, nan=-1)

        # Mask for missing values:
        # 1: not missing
        # 0: missing
        prognosis_mask = copy.deepcopy(prognosis)
        prognosis_mask[prognosis_mask != -1] = 1
        prognosis_mask[prognosis_mask == -1] = 0

        df[f'prognosis_{grading}'] = list(prognosis)
        df[f'prognosis_mask_{grading}'] = list(prognosis_mask)

    return df


def calculate_class_weights(oai_meta, cfg):
    grading = cfg.grading
    # all_stats = None
    all_stats = {'y0': [], 'pn': [], 'pr': []}
    max_grade = MAX_GRADES[cfg.grading]
    for i in range(1, cfg.seq_len + 1):
        pn_counts = []
        pr_counts = []
        for v in range(0, max_grade + 1):
            count = len(oai_meta[oai_meta[f'{grading}_{i}y'] == v].index)
            pn_counts.append(count)
        for v in range(0, 2):
            count = len(oai_meta[oai_meta[f'Progressor_{grading}_{i}y'] == v].index)
            pr_counts.append(count)

        all_stats['pn'].append(pn_counts)
        all_stats['pr'].append(pr_counts)

    y0_counts = []
    for v in range(0, max_grade + 1):
        count = len(oai_meta[oai_meta[f'{grading}'] == v].index)
        y0_counts.append(count)

    all_stats['y0'] = np.array([y0_counts])
    all_stats['y0'] = all_stats['y0'] / all_stats['y0'].sum(axis=1, keepdims=True)

    all_stats['pn'] = np.array(all_stats['pn'])
    all_stats['pn'] = all_stats['pn'] / all_stats['pn'].sum(axis=1, keepdims=True)

    all_stats['pr'] = np.array(all_stats['pr'])
    all_stats['pr'] = all_stats['pr'] / all_stats['pr'].sum(axis=1, keepdims=True)

    all_stats['y0'] = swap_weights(all_stats['y0'])[0, :]
    all_stats['pn'] = swap_weights(all_stats['pn'])
    all_stats['pr'] = swap_weights(all_stats['pr'])
    return all_stats['y0'], all_stats['pn'], all_stats['pr']


def swap_weights(input):
    n_classes = input.shape[1]
    sort_idx = np.argsort(input, axis=1)
    swap_weights = np.zeros((sort_idx.shape), dtype=np.float)
    for r in range(sort_idx.shape[0]):
        for c in range(n_classes):
            _c = sort_idx[r, c]
            _c_f = n_classes - 1 - _c
            ind1 = sort_idx[r, _c]
            ind2 = sort_idx[r, _c_f]
            swap_weights[r, ind1] = input[r, ind2]
            swap_weights[r, ind2] = input[r, ind1]
    return swap_weights


def calculate_metric(metric_func, y_true, y_pred, **kwargs):
    result = None
    if len(y_pred) == len(y_true) and len(y_pred) > 0:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = metric_func(y_true, y_pred, **kwargs)
        except ValueError:
            pass
    return result


def load_metadata(cfg, proc_targets=None, modes=('train', 'eval')):
    if isinstance(modes, str):
        modes = [modes]

    if cfg.dataset != 'oai':
        raise ValueError(f'This function only support OAI, not {cfg.dataset}')

    img_root = cfg.root
    meta_root = cfg.meta_root
    pkl_meta_filename = cfg.pkl_meta_filename
    site = cfg.site

    oai_filename = cfg.oai_meta_filename
    target_col = cfg.grading

    pkl_meta_fullname = os.path.join(meta_root, pkl_meta_filename)
    oai_filename = os.path.join(meta_root, oai_filename)

    # Eval mode
    oai_meta_test = None
    if "eval" in modes:
        pkl_meta_oai_site_test_fullname = os.path.join(meta_root, f"OAI_site_{site}.pkl")
        if os.path.isfile(pkl_meta_oai_site_test_fullname):
            log.info(f'Reading OAI meta file {pkl_meta_oai_site_test_fullname}')
            with open(pkl_meta_oai_site_test_fullname, 'rb') as f:
                oai_meta_test = pickle.load(f)
                oai_meta_test['ID'] = oai_meta_test['ID'].astype(str)
                print(f'Loaded OAI entries: {len(oai_meta_test.index)}')
        else:
            log.info(f'Cannot find OAI meta file {pkl_meta_fullname}. Creating new file...')
            oai_meta_all = pd.read_csv(oai_filename)
            print(f'Loaded OAI entries: {len(oai_meta_all.index)}')

            oai_meta_test = oai_meta_all[oai_meta_all['V00SITE'] == site]
            oai_meta_test['ID'] = oai_meta_test['ID'].astype(str)
            if proc_targets is not None:
                print(f'Original OAI test data: {len(oai_meta_test.index)}')
                oai_meta_test = remove_empty_img_rows(img_root, oai_meta_test)
                print(f'After removing entries without image, OAI test data: {len(oai_meta_test.index)}')
                oai_meta_test = proc_targets(oai_meta_test, dataset='oai')

            print(f'Write test file {pkl_meta_oai_site_test_fullname}')
            with open(pkl_meta_oai_site_test_fullname, 'wb') as f:
                pickle.dump(oai_meta_test, f, 4)

    split_data = None
    oai_meta = None
    # Train mode
    if 'train' in modes:
        if os.path.isfile(pkl_meta_fullname):
            log.info(f'Read meta file {pkl_meta_fullname}')
            with open(pkl_meta_fullname, 'rb') as f:
                loaded_data = pickle.load(f)
                split_data = loaded_data['oai_site_folds']
                oai_meta = loaded_data['oai_site_train']
        else:
            log.info(f'Cannot find OAI meta file {pkl_meta_fullname}. Creating new file...')
            oai_meta_all = pd.read_csv(oai_filename)

            print(f'Loaded OAI entries: {len(oai_meta_all.index)}')

            # Training-validation data
            sites = oai_meta_all['V00SITE'].unique()
            log.info(f'Sites are {sites}, and test site is {site}')
            oai_meta = oai_meta_all[oai_meta_all['V00SITE'] != site]

            oai_meta['ID'] = oai_meta['ID'].astype(str)
            oai_meta = oai_meta.reset_index(drop=True)

            if proc_targets is not None:
                print(f'Original OAI training data: {len(oai_meta.index)}')
                oai_meta = remove_empty_img_rows(img_root, oai_meta)
                print(f'Filtered OAI training data: {len(oai_meta.index)}')
                oai_meta = proc_targets(oai_meta, dataset='oai')

            splitter = FoldSplit(oai_meta, n_folds=5, target_col=target_col, group_col='ID')

            # split_data = tuple([(train, val) for (train, val) in splitter])
            split_data = post_process_data(splitter, proc_targets)
            # summarize_splitter(split_data, classes, target_col)

            loaded_data = {'oai_site_folds': split_data, 'oai_site_train': oai_meta}
            log.info(f'Save metadata to {pkl_meta_fullname}.')

            print(f'Write file {pkl_meta_fullname}')
            with open(pkl_meta_fullname, 'wb') as f:
                pickle.dump(loaded_data, f, protocol=4)

    return split_data, oai_meta, oai_meta_test

def count_parameters(model):
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_params = sum(p.numel() for p in model.parameters())
    return n_trainable_params, n_params

def filter_df_by_targets(df, cfg):
    target_col = cfg.grading
    targets_list = [float(t) for t in range(cfg.n_pn_classes)]

    all_targets = list(df[target_col].unique())
    for k in range(1, cfg.seq_len + 1):
        if f'{target_col}_{k}y' in df:
            all_targets += list(df[f'{target_col}_{k}y'].unique())
    all_targets = [t for t in all_targets if str(t) != 'nan']
    all_targets = set(all_targets)

    excluded_targets = list(all_targets - set(targets_list))
    excluded_targets_w_max = excluded_targets + [max(targets_list)]
    print(f'Exclude targets {excluded_targets} from test data')
    replace_map = {}
    df = df[~df[target_col].isin(excluded_targets_w_max)]

    for k in range(1, cfg.seq_len + 1):
        replace_map[f'{target_col}_{k}y'] = {t: np.nan for t in excluded_targets}
    df.replace(replace_map, inplace=True)

    return df


def init_mean_std(cfg, output_dir, ds, parse_item_img_prog):
    if os.path.isfile(os.path.join(output_dir, 'mean_std.npy')):
        tmp = np.load(os.path.join(output_dir, 'mean_std.npy'))
        mean_vector, std_vector = tmp
    else:
        std_size_trf = Compose([
            img_labels2solt,
            slc.Stream([
                slt.ResizeTransform((300, 300)),
            ]),
            unpack_solt_data
        ])

        dataset = DataFrameDataset(cfg.root, ds, parse_item_img_prog, transform=std_size_trf)
        tmp_loader = DataLoader(dataset, batch_size=cfg.bs, num_workers=cfg.num_workers)
        mean_vector = None
        std_vector = None
        print(colored('==> ', 'green') + 'Calculating mean and std')
        for batch in tqdm(tmp_loader, total=len(tmp_loader)):
            imgs = batch['data']
            if mean_vector is None:
                mean_vector = np.zeros(imgs.size(1))
                std_vector = np.zeros(imgs.size(1))
            for j in range(mean_vector.shape[0]):
                mean_vector[j] += imgs[:, j, :, :].mean()
                std_vector[j] += imgs[:, j, :, :].std()

        mean_vector /= (len(tmp_loader))
        std_vector /= (len(tmp_loader))
        mean_std_fullname = os.path.join(output_dir, f'mean_std.npy')
        print(f'Save mean_std to {os.path.abspath(mean_std_fullname)}')
        np.save(os.path.join(output_dir, f'mean_std.npy'),
                [mean_vector.astype(np.float32), std_vector.astype(np.float32)])

    return mean_vector, std_vector


def store_model(epoch_i, gradings, metric_names, metrics, stored_models, model, saved_dir, cond="max", mode="avg"):
    if isinstance(metric_names, str):
        metric_names = [metric_names]
    if isinstance(gradings, str):
        gradings = [gradings]

    metric_values = []
    for name in metric_names:
        for grading in gradings:
            if name in metrics[grading]:
                metric_values.append(np.array(metrics[grading][name]))

    if len(metric_values) > 1:
        cur_metric = np.nanmean(np.array([v for v in np.concatenate(metric_values,0) if v is not None]))
    else:
        cur_metric = np.nanmean(np.array([v for v in metric_values if v is not None]))

    task_code = ".".join(gradings)
    if task_code not in stored_models:
        stored_models[task_code] = {}
    metric_code = ".".join(metric_names)
    if metric_code not in stored_models[task_code]:
        if cond == "max":
            stored_models[task_code][metric_code] = {'best': -1, "filename": ""}
        else:
            stored_models[task_code][metric_code] = {'best': 1e10, "filename": ""}

    if check_cond(cur_metric, stored_models[task_code][metric_code]['best'], cond):
        print(f'[{epoch_i}] Improve {metric_code} from {stored_models[task_code][metric_code]["best"]} to {cur_metric}.')
        stored_models[task_code][metric_code]['best'] = cur_metric

        # Remove prev model
        prev_model_fullname = os.path.join(saved_dir, stored_models[task_code][metric_code]['filename'])
        if os.path.isfile(prev_model_fullname):
            os.remove(prev_model_fullname)
        if os.path.isfile(prev_model_fullname[:-4] + ".json"):
            os.remove(prev_model_fullname[:-4] + ".json")

        # Save improved model
        stored_models[task_code][metric_code]['filename'] = \
            "model_" + f"{epoch_i:03d}" + "_" + ".".join(gradings) + "_" + mode + "_" + metric_code + "_" + f'{cur_metric:.03f}' + ".pth"
        saved_model_fullname = os.path.join(saved_dir, stored_models[task_code][metric_code]['filename'])
        saved_log_fullname = saved_model_fullname[:-4] + ".json"
        torch.save(model.state_dict(), saved_model_fullname)
        print(f'Saved best model to {saved_model_fullname}.')

        results = {mode: cur_metric}
        for metric_name in metric_names:
            for grading in gradings:
                if metric_name in metrics[grading]:
                    results[f'{grading}:{metric_name}'] = metrics[grading][metric_name]
        with open(saved_log_fullname, "w") as f:
            json.dump(results, f)

    return stored_models

def get_output_map(output_heads_per_grading, gradings):
    output_map = {}
    for g_i, grading in enumerate(gradings):
        for i in range(output_heads_per_grading):
            output_map[f"{grading}_{i}"] = g_i * output_heads_per_grading + i
    return output_map

def check_missing_data(x):
    return (isinstance(x, str) and "missing" in x) or not x or x != x


def parse_img(root, entry, trf, **kwargs):
    img_filename = f"{entry['ID']}_00_{entry['Side']}.png"
    img_fullname = os.path.join(root, img_filename)
    img = cv2.imread(img_fullname, cv2.IMREAD_GRAYSCALE)
    img = trf(img)

    return {'data': img}


def img_labels2solt(inp):
    img = inp
    return sld.DataContainer((img), fmt='I')


def unpack_solt_data(dc: sld.DataContainer):
    img = dc.data[0]
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img


def init_transforms(mean=(0.51109564, 0.51109564, 0.51109564), std=(0.28390905, 0.28390905, 0.28390905), n_channels=3):
    if n_channels == 3:
        norm_mean_std = Normalize(mean, std)
    else:
        raise ValueError("Not support channels of {}".format(n_channels))

    train_trf = Compose([
        img_labels2solt,
        slc.Stream([
            slt.CropTransform(crop_size=(700, 700), crop_mode='c'),
            slt.ResizeTransform((280, 280)),
            slt.ImageAdditiveGaussianNoise(p=0.5, gain_range=0.3),
            slt.RandomRotate(p=1, rotation_range=(-10, 10)),
            slt.CropTransform(crop_size=(256, 256), crop_mode='r'),
            slt.ImageGammaCorrection(p=0.5, gamma_range=(0.5, 1.5)),
        ], interpolation='area', padding='z'),
        unpack_solt_data,
        ApplyTransform(norm_mean_std)
    ])

    test_trf = Compose([
        img_labels2solt,
        slc.Stream([
            slt.PadTransform(pad_to=(700, 700)),
            slt.CropTransform(crop_size=(700, 700), crop_mode='c'),
            slt.ResizeTransform((280, 280)),
            slt.CropTransform(crop_size=(256, 256), crop_mode='c'),
        ], interpolation='area'),
        unpack_solt_data,
        ApplyTransform(norm_mean_std)
    ])

    return {'train': train_trf, 'eval': test_trf}


def post_process_data(splitter, proc_targets):
    split_data = []
    for fold_i, (train, val) in enumerate(splitter):
        if proc_targets is not None:
            train = proc_targets(train)
            val = proc_targets(val)

        split_data.append((train, val))
    split_data = tuple(split_data)
    return split_data


def remove_empty_img_rows(root, df):
    rows = []
    for ind, row in df.iterrows():
        img_filename = create_img_name(row)
        img_fullname = os.path.join(root, img_filename)
        if os.path.isfile(img_fullname):
            rows.append(row)
        else:
            print(f'Not found {img_fullname}.')
    return pd.DataFrame(rows)


def create_img_name(entry):
    img_filename = f"{entry['ID']}_{int(entry['visit']):02d}_{entry['Side']}.png"
    return img_filename


def merge_train_eval_dfs(train_df, eval_df):
    len_train = len(train_df.index)
    len_eval = len(eval_df.index)
    train_df['stage'] = ['train'] * len_train
    eval_df['stage'] = ['eval'] * len_eval
    df = pd.concat((train_df, eval_df))
    return df


def check_cond(a, b, cond):
    if a is None or b is None:
        return False
    elif cond == "max":
        return a > b
    elif cond == "min":
        return a < b
    else:
        raise ValueError(f'Not support cond "{cond}".')
