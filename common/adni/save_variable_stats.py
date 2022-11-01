import os
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from common.adni.utils import INPUT_COLS, TARGET_COLS, NAME2DX, NUMERICAL_COLS, MRI_BIOMARKERS, ECOG_COLS, RAVLT_COLS

def get_stats(df):
    stats = {}
    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    for col in NUMERICAL_COLS + ECOG_COLS + RAVLT_COLS: # + MRI_BIOMARKERS
        print(col)
        if df[col].dtype == object:
            df.replace({col: {r'^<(\d+)': r'\1'}}, regex=True, inplace=True)
            df[col] = df[col].astype(float)
        data = df[col].to_numpy()
        stats[col] = {'mean': np.nanmean(data), 'std': np.nanstd(data)}

    with open("num_var_stats.pkl", "wb") as f:
        pickle.dump(stats, f, protocol=4)
    return stats

if __name__ == "__main__":
    # fullname = "/home/hoang/data/ANDI/tadpole_challenge/TADPOLE_D1_prognosis.csv"
    fullname = "/home/hoang/data/ANDI/ADNIMERGE_FDG.csv"
    ds = pd.read_csv(fullname)
    ds.replace(r'^\s*$', np.nan, regex=True, inplace=True)

    for col in NUMERICAL_COLS:
        if ds[col].dtype == object:
            ds.replace({col: {r'^<(\d+)': r'\1'}}, regex=True, inplace=True)
            ds.replace({col: {r'^>(\d+)': r'\1'}}, regex=True, inplace=True)
        ds[col] = ds[col].astype(float)
    stats = get_stats(ds)
    print(stats)
