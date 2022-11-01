import pandas as pd
import numpy as np
import pickle
import yaml

if __name__ == "__main__":
    meta_fullname = "/home/hoang/workspace/OAPOP/adni/Metadata/cv_split_10folds_DXTARGET_fdg_12345.pkl"
    config_fullname = "/home/hoang/workspace/OAPOP/adni/configs/data/img:1_fdg_meta:all.yaml"

    with open(meta_fullname, "rb") as f:
        data = pickle.load(f)

    with open(config_fullname, "rt") as f:
        config = yaml.load(f)

    cols = config['parser']['metadata']
    cols.remove('IMG')

    df_all = pd.concat(data[0], axis=0)
    print(f'Num of unique patients (only baseline): {len(df_all.loc[df_all["VISIT"] == 0, "RID"].unique())}')
    print(f'Num of unique patients: {len(df_all["RID"].unique())}')

    df_dict = {}

    df_dict['Train'] = data[0][0][data[0][0]['VISIT'] == 0]
    df_dict['with FUBA'] = data[0][0]
    df_dict['Validation with FUBA'] = data[0][1]
    df_dict['Test'] = data[0][1][data[0][1]['VISIT'] == 0]

    final_df = []
    for ind_stage, stage in enumerate(df_dict):
        row = {'Stage': stage}
        df = df_dict[stage]
        df_input = df[cols]
        df_mask_target = df['prognosis_mask_DXTARGET']
        n_missing = df_input.isna().sum().sum()
        n_total = len(df_input.index) * len(cols)
        missing_rate = n_missing / n_total
        row['Patients'] = len(df.loc[df['VISIT'] == 0, 'RID'].unique())
        row['Missing input'] = f'{missing_rate:.01f}'

        mask_target = np.stack(df_mask_target.tolist(), 0)
        for y in range(1, 6):
            row[y] = int(np.sum(mask_target[:, y]))

        final_df.append(row)

    final_df = pd.DataFrame(final_df)
    final_df.set_index('Stage', inplace=True)
    print(final_df.to_latex(index=True, multirow=True, multicolumn=True, escape=False, bold_rows=False))