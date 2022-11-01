import logging as log
import os

import cv2
import numpy as np
import pandas as pd
import torch
from collagen import Compose, ApplyTransform, Normalize
from solt import core as slc, transforms as slt, data as sld


def parse_metadata(entry, metadata):
    n_segments = 4
    mean_bmi = 28
    mean_age = 60
    mean_womac = 8.78
    bmi_ranges = [0, 18.5, 25, 30, float("Inf")]

    input = {}

    # AGE
    age_stats = [45, 60, 90 + 1]
    d_age = (age_stats[-1] - age_stats[0]) / n_segments
    age_ranges = [age_stats[0] + i * d_age for i in range(n_segments + 1)]
    if "AGE" in metadata:
        age_level = None
        if check_missing_data(entry['AGE']):
            input["AGE_mask"] = torch.tensor(False)
            age = [0.0] * n_segments
            input["AGE"] = torch.tensor(age)
        else:
            input["AGE_mask"] = torch.tensor(True)
            age_ori = float(entry['AGE'])
            for i in range(len(age_ranges) - 1):
                if age_ranges[i] <= age_ori < age_ranges[i + 1]:
                    age_level = torch.tensor(i)
                    break

            age = [0.0] * n_segments
            age[age_level] = 1.0
            input["AGE"] = torch.tensor(age)

    # SEX
    if "SEX" in metadata:
        if check_missing_data(entry['SEX']):
            input["SEX_mask"] = torch.tensor(False)
            sex = [0.0] * 2
        else:
            input["SEX_mask"] = torch.tensor(True)
            if (isinstance(entry['SEX'], str) and "Female" in entry['SEX']) or (
                    isinstance(entry['SEX'], int) and entry['SEX'] == 0):
                sex = [0.0, 1.0]
            else:
                sex = [1.0, 0.0]
        input["SEX"] = torch.tensor(sex)

    # BMI
    if "BMI" in metadata:
        if check_missing_data(entry['BMI']):
            bmi_ori = mean_bmi
        else:
            bmi_ori = float(entry['BMI'])
        bmi_level = None
        for bmi_i in range(len(bmi_ranges) - 1):
            if bmi_ranges[bmi_i] <= bmi_ori < bmi_ranges[bmi_i + 1]:
                bmi_level = torch.tensor(bmi_i)
                break
        if bmi_level is None:
            input["BMI_mask"] = torch.tensor(False)
            input["BMI"] = torch.tensor([0.0] * 4, dtype=torch.float32)
        else:
            input["BMI_mask"] = torch.tensor(True)
            bmi = [0.0] * 4
            bmi[bmi_level] = 1.0
            input["BMI"] = torch.tensor(bmi)

    # INJURY
    if "INJ" in metadata:
        if check_missing_data(entry['INJ']):
            input['INJ_mask'] = torch.tensor(False)
            inj = [0.0] * 2
        else:
            input['INJ_mask'] = torch.tensor(True)
            inj = [1.0, 0.0] if (isinstance(entry['INJ'], str) and "0" in entry['INJ']) or entry['INJ'] == 0 else [0.0,
                                                                                                                   1.0]
        input["INJ"] = torch.tensor(inj)

    # SURGERY
    if "SURG" in metadata:
        if check_missing_data(entry['SURG']):
            input["SURG_mask"] = torch.tensor(False)
            surg = [0.0] * 2
        else:
            input["SURG_mask"] = torch.tensor(True)
            surg = [1.0, 0.0] if (isinstance(entry['SURG'], str) and "0" in entry['SURG']) or entry['SURG'] == 0 else [
                0.0, 1.0]
        input["SURG"] = torch.tensor(surg)

    # WOMAC
    womac_stats = [0, 9, 85 + 1]
    d_womac = (womac_stats[-1] - womac_stats[0]) / n_segments
    womac_ranges = [womac_stats[0] + i * d_womac for i in range(n_segments + 1)]
    if "WOMAC" in metadata:
        womac_level = 0
        if check_missing_data(entry['WOMAC']):
            input['WOMAC_mask'] = torch.tensor(False)
            womac_ori = womac_stats[1]
        else:
            input['WOMAC_mask'] = torch.tensor(True)
            womac_ori = float(entry['WOMAC'])
        for i in range(len(womac_ranges) - 1):
            if womac_ranges[i] <= womac_ori < womac_ranges[i + 1]:
                womac_level = i
                break
        if womac_level is None:
            raise ValueError(f'Cannot find WOMAC level of {womac_ori}.')
        womac = [0.0] * n_segments
        womac[womac_level] = 1.0
        input["WOMAC"] = torch.tensor(womac)

    return input


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


def summarize_hiar_splitter(split_data, classes, target_col):
    print(f'Classes: {classes}')
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
    print(f'Classes: {classes}')
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


def check_cond(a, b, cond):
    if a is None or b is None:
        return False
    elif cond == "max":
        return a > b
    elif cond == "min":
        return a < b
    else:
        raise ValueError(f'Not support cond "{cond}".')


def parse_item_progs(root, entry, trf, **kwargs):
    grading = kwargs['grading']

    sample = {}
    if 'V00SITE' in entry:
        input = {'ID': f"{entry['V00SITE']}_{entry['ID']}_{entry['Side']}_{entry['visit_id']}"}
    else:
        input = {'ID': f"{entry['ID']}_{entry['Side']}_{entry['visit_id']}"}

    if "IMG" in kwargs["metadata"]:
        img_filename = f"{entry['ID']}_{int(entry['visit']):02d}_{entry['Side']}.png"
        img_fullname = os.path.join(root, img_filename)

        img = cv2.imread(img_fullname, cv2.IMREAD_GRAYSCALE)
        if entry['Side'] == 'R':
            img = img[:, ::-1]

        if img is None:
            print(f'{img_fullname}')
        trf_img = trf((img,))[0]

        input['IMG'] = trf_img

    meta = parse_metadata(entry, kwargs["metadata"])
    input.update(meta)

    sample['data'] = {'input': input}

    gradings = kwargs['output']

    for grading in gradings:
        has_cur_grading = entry[grading] is not None and entry[grading] == entry[grading]
        cur_grading = torch.tensor(int(entry[grading])) if has_cur_grading else torch.tensor(-1)
        sample[f'prognosis_{grading}'] = torch.tensor(np.concatenate(([cur_grading], entry[f'prognosis_{grading}']), 0))
        sample[f'prognosis_mask_{grading}'] = torch.tensor(
            np.concatenate(([has_cur_grading], entry[f'prognosis_mask_{grading}']), 0), dtype=torch.bool)

    return sample
