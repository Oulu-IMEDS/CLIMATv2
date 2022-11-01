import pandas as pd
import numpy as np
import pandas as pd
from tqdm import tqdm

from common.adni.utils import INPUT_COLS, TARGET_COLS, NAME2DX, MRI_BIOMARKERS


def standardize_metadata(df, out_filename, dt=6):
    n_max_fu = 10
    targets = ["DXTARGET", "ADAS13", "Ventricles"]
    # avail_cols = df.columns.tolist()
    # avail_cols = set.intersection(set(cols), set(INPUT_COLS + TARGET_COLS + MRI_BIOMARKERS))
    # df = df[avail_cols]
    df.replace({'DX': NAME2DX}, inplace=True)
    df.replace({'VISCODE': {'bl': 'm00', 'm0': 'm00', 'y1': 'm12'}}, inplace=True)
    df['DXTARGET'] = df['DX']
    df['EXAMDATE'] = pd.to_datetime(df['EXAMDATE'], format='%Y-%m-%d')
    if 'EXAMDATE_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16' in df:
        df['MRI_DATE'] = pd.to_datetime(df['EXAMDATE_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16'], format='%Y-%m-%d',
                                        errors='coerce')
        df['MRI_DATE'] = df['MRI_DATE'].dt.strftime('%Y-%m')
    if 'EXAMDATE_BAIPETNMRC_09_12_16' in df:
        df['DATE_FDGPET'] = pd.to_datetime(df['EXAMDATE_BAIPETNMRC_09_12_16'], format='%Y-%m-%d', errors='coerce')
        df['DATE_FDGPET'] = df['DATE_FDGPET'].dt.strftime('%Y-%m')

    df.replace('Unknown', np.nan, inplace=True)
    # if 'DXCHANGE' in df:
    #     df['DXCHANGE'] = df['DXCHANGE'].sub(1)

    df['VISIT'] = df['VISCODE'].str[1:].astype(int)
    df['ID'] = df[['PTID', 'VISCODE']].agg("_".join, axis=1)
    rids = df['RID'].unique()

    data = []

    for rid in tqdm(rids, total=len(rids), desc="Processing"):
        fus = df[df['RID'] == rid]
        fus = fus.sort_values(by='VISIT')
        visits = fus['VISIT'].tolist()
        v_n = visits[-1]
        n_visits = len(visits)
        for i_0, v_0 in enumerate(visits):
            rec = fus[fus['VISIT'] == v_0]

            if len(rec) < 1:
                continue
            elif len(rec) > 1:
                raise ValueError(f'Found duplicate records with ID={rid}, visit={v_0}.')

            rec = rec.to_dict('r')[0]
            rec['AGE'] += (dt / 12.0) * (v_0 // dt)
            n_fus = (v_n - v_0) / dt
            for v_i in range(v_0 + dt, v_n + dt, dt):
                fu_i = (v_i - v_0) // dt
                if fu_i > n_max_fu:
                    break
                if v_i in visits:
                    for target in targets:
                        rec[f'{target}_{fu_i}'] = fus.loc[fus['VISIT'] == v_i, target].item()
            data.append(rec)

    # data = pd.concat(data, axis=0)
    data = pd.DataFrame(data)
    name_map = {target: f'{target}_0' for target in ["DXTARGET", "ADAS13", "Ventricles"]}
    data = data.rename(columns=name_map)
    data.sort_index(axis=1, inplace=True)
    data.to_csv(out_filename, index=None)
    return data


def generate_neuroimaging_data(fullname="/home/hoang/data/ANDI/ADNIMERGE_FDG.csv",
                               out_fullname="/home/hoang/data/ANDI/adni_fdgpet_prognosis.csv"):
    dt = 6
    df = pd.read_csv(fullname)
    return standardize_metadata(df, out_fullname)


if __name__ == "__main__":
    generate_neuroimaging_data("/home/hoang/data/ANDI/ADNIMERGE_FDG.csv",
                               "/home/hoang/data/ANDI/adni_fdgpet_prognosis.csv")
