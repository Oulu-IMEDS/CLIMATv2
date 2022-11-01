import os

import nibabel as nib
import pandas as pd
from tqdm import tqdm


def group_patient_visit(df):
    # IDs = df['PTID'].unique().tolist()
    # projects = ['axial', 'sagittal', 'coronal']
    # rows = []
    # for ID in IDs:
    #     row = {}
    #     missing_data = False
    #     for i, project in enumerate(projects):
    #         entry = df[(df['PTID'] == ID) & (df['Project'] == project)]
    #         if len(entry) < 1:
    #             print(f'Cannot find ID {ID} with projection {project}. Skip!')
    #             missing_data = True
    #             break
    #         entry = entry.to_dict()
    #         if i == 0:
    #             row = {'PTID': entry['PTID'], 'Group': entry['Group']}
    #         row[f'MRI_filename_{project}'] = entry['MRI_filename']
    #         row[f'MRI_ID_{project}'] = entry['MRI_ID']
    #     if not missing_data:
    #         rows.append(row)
    # new_df = pd.DataFrame(rows)

    df = df.set_index(['PTID', 'Visit'])
    df = df.groupby(['PTID', 'Visit'], as_index=True).transform(
        lambda x: '|'.join(x) if len(x.unique()) > 1 else x.iloc[0])
    df = df.reset_index()
    rows = []
    for ind, row in df.iterrows():
        projects = row['Project'].split("|")
        mri_ids = row['MRI_ID'].split("|")
        mri_filenames = row['MRI_filename'].split("|")
        if len(projects) == 3:
            for i, project in enumerate(projects):
                row[f'MRI_filename_{project}'] = mri_filenames[i]
                row[f'MRI_ID_{project}'] = mri_ids[i]
            rows.append(row)

    new_df = pd.DataFrame(rows)
    new_df = new_df.drop(columns=['Type', 'Format', 'Project', 'MRI_filename', 'MRI_ID', 'Sex', 'Modality'])
    return new_df


def get_reordered_spacing(img_nii, proj_name):
    spacing = img_nii.header.get_zooms()

    if proj_name == 'axial' or proj_name is None:
        return spacing[0], spacing[1], spacing[2]
    elif proj_name == 'sagittal':
        return spacing[1], spacing[2], spacing[0]
    else:
        return spacing[0], spacing[2], spacing[1]


def check_valid_spacing(img_nii, std_spacing, proj_name=None, eps=0.01):
    proj_name = 'axial' if proj_name is None else proj_name
    reordered_spacing = get_reordered_spacing(img_nii, proj_name)
    return abs(reordered_spacing[0] - std_spacing[0]) < eps and abs(reordered_spacing[1] - std_spacing[1]) < eps


def create_df(image_root, fullname, out_fullname, prefix, std_spacing, force=True):
    ext = 'VXSTD.nii'
    spatial_size = 256
    if not os.path.isfile(out_fullname) or force:
        df = pd.read_csv(fullname)
        n = len(image_root)
        fullnames = []
        for r, d, f in os.walk(image_root):
            for filename in f:
                fullname = os.path.join(r, filename)
                if fullname.endswith(ext):
                    fullnames.append(fullname)

        fullnames.sort(reverse=True)
        rows = []
        for ind, row in tqdm(df.iterrows(), total=len(df.index), desc="Processing: "):
            row = row.to_dict()
            matched_filename = None
            remove_items = []
            valid = True
            for fullname in fullnames:
                if row['Subject'] in fullname and row['Image Data ID'] in fullname:
                    filename = fullname[n:]
                    filename = filename[1:] if filename.startswith('/') else filename
                    matched_filename = filename
                    remove_items.append(fullname)
                    img_nii = nib.load(fullname)
                    # proj_name = check_projection(img_nii, spatial_size)
                    if not check_valid_spacing(img_nii, std_spacing=std_spacing, proj_name=None):
                        valid = False
                        raise ValueError(f'Found invalid spacing {img_nii.header.get_zooms()}')
                    # row['Project'] = proj_name

            for item in remove_items:
                fullnames.remove(item)

            row[f'{prefix}_filename'] = matched_filename

            rows.append(row)

        updated_df = pd.DataFrame(rows)
        updated_df[f'{prefix}_DATE'] = pd.to_datetime(updated_df['Acq Date'], format='%m/%d/%Y', errors='coerce')
        updated_df[f'{prefix}_exam_year'] = updated_df[f'{prefix}_DATE'].dt.strftime('%Y').astype(int)
        updated_df[f'{prefix}_exam_month'] = updated_df[f'{prefix}_DATE'].dt.strftime('%m').astype(int)
        updated_df.rename(columns={'Subject': 'PTID', 'Image Data ID': f'{prefix}_ID', 'Description': f'{prefix}_desc'},
                          inplace=True)
        # updated_df = updated_df[[f'{prefix}_ID', 'PTID', f'{prefix}_DATE' ,f'{prefix}_desc', f'{prefix}_filename']]
        # updated_df = group_patient_visit(updated_df)
        print(f'Saving file to {out_fullname}')
        updated_df.to_csv(out_fullname, index=None)

    else:
        updated_df = pd.read_csv(out_fullname)

    updated_df = updated_df[
        ['PTID', f'{prefix}_ID', f'{prefix}_desc', f'{prefix}_filename', f'{prefix}_DATE', f'{prefix}_exam_year',
         f'{prefix}_exam_month']]

    return updated_df, [f'{prefix}_ID', f'{prefix}_desc', f'{prefix}_filename', f'{prefix}_DATE', f'{prefix}_exam_year',
                        f'{prefix}_exam_month']


if __name__ == "__main__":
    # root = "/home/hoang/data/ANDI/tadpole_challenge"
    output_dir = "/home/hoang/data/ANDI/"
    fdgpet_image_root = "/home/hoang/data/ANDI/FDG_PET_CoregAvg"
    fdgpet_fullname = os.path.join(fdgpet_image_root, "FDG_PET_CoregAvg_7_05_2021.csv")
    fdgpet_out_fullname = fdgpet_fullname[:-4] + "_path.csv"

    av45pet_image_root = "/home/hoang/data/ANDI/AV45_PET_coreg_avg"
    av45pet_fullname = os.path.join(av45pet_image_root, "AV45_PET_coreg_avg_7_05_2021.csv")
    av45pet_out_fullname = av45pet_fullname[:-4] + "_path.csv"

    day_thresh = 40
    std_spacing = (1.5, 1.5, 1.5)
    force = False
    fdgpet_updated_df, fdgpet_cols = create_df(fdgpet_image_root, fdgpet_fullname, fdgpet_out_fullname, 'FDGPET',
                                               force=force, std_spacing=std_spacing)
    av45pet_updated_df, av45pet_cols = create_df(av45pet_image_root, av45pet_fullname, av45pet_out_fullname, 'AV45PET',
                                                 force=force, std_spacing=std_spacing)

    merge_fullname = "/home/hoang/data/ANDI/ADNIMERGE.csv"
    merge_df = pd.read_csv(merge_fullname)

    merge_df.replace({'VISCODE': {'bl': 'm00', 'm0': 'm00', 'y1': 'm12'}}, inplace=True)
    merge_df = merge_df[~merge_df['EXAMDATE'].isnull()]
    merge_df_date = pd.to_datetime(merge_df['EXAMDATE'], format='%Y-%m-%d', errors='coerce')
    merge_df['exam_year'] = merge_df_date.dt.strftime('%Y').astype(int)
    merge_df['exam_month'] = merge_df_date.dt.strftime('%m').astype(int)

    # MERGE with same month and year
    # df_fdg = pd.merge(merge_df, fdgpet_updated_df, how="left", left_on=["PTID", "exam_year", "exam_month"],
    #                   right_on=["PTID", "FDGPET_exam_year", "FDGPET_exam_month"])
    # df_fdg_av45 = pd.merge(df_fdg, av45pet_updated_df, how="left", left_on=["PTID", "exam_year", "exam_month"],
    #                        right_on=["PTID", "AV45PET_exam_year", "AV45PET_exam_month"])
    df_fdg = pd.merge(merge_df, fdgpet_updated_df, how="left", left_on=["PTID", "exam_year", "exam_month"],
                      right_on=["PTID", "FDGPET_exam_year", "FDGPET_exam_month"])
    df_av45 = pd.merge(merge_df, av45pet_updated_df, how="left", left_on=["PTID", "exam_year", "exam_month"],
                       right_on=["PTID", "AV45PET_exam_year", "AV45PET_exam_month"])
    df_fdg_av45 = pd.merge(df_fdg, av45pet_updated_df, how="left", left_on=["PTID", "exam_year", "exam_month"],
                           right_on=["PTID", "AV45PET_exam_year", "AV45PET_exam_month"])

    # MERGE with less than 30 days difference
    # df_fdg = pd.merge(merge_df, fdgpet_updated_df, how="left", on=["PTID"])
    # df_fdg[['EXAMDATE', 'FDGPET_DATE']] = df_fdg[['EXAMDATE', 'FDGPET_DATE']].apply(pd.to_datetime)
    # df_fdg['FDG_EXAMDATE_DIFF'] = (df_fdg['EXAMDATE'] - df_fdg['FDGPET_DATE']).dt.days.abs()
    # df_fdg = df_fdg[df_fdg['FDG_EXAMDATE_DIFF'] < day_thresh]
    # df_fdg_av45 = pd.merge(df_fdg, av45pet_updated_df, how="left", on=["PTID"])
    # df_fdg_av45[['EXAMDATE', 'AV45PET_DATE']] = df_fdg_av45[['EXAMDATE', 'AV45PET_DATE']].apply(pd.to_datetime)
    # df_fdg_av45['AV45PET_EXAMDATE_DIFF'] = (df_fdg_av45['EXAMDATE'] - df_fdg_av45['AV45PET_DATE']).dt.days.abs()
    # df_fdg_av45 = df_fdg_av45[(df_fdg_av45['FDG_EXAMDATE_DIFF'] < day_thresh) | (df_fdg_av45['AV45PET_EXAMDATE_DIFF'] < day_thresh)]
    # df_fdg_av45.loc[df_fdg_av45['AV45PET_EXAMDATE_DIFF'] >= day_thresh, av45pet_cols + ['AV45PET_EXAMDATE_DIFF']] = None

    # MERGE with less than days_thresh difference
    # df_fdg_av45 = pd.merge(fdgpet_updated_df, av45pet_updated_df, how="inner", on=["PTID"])
    # df_fdg_av45 = pd.merge(merge_df, df_fdg_av45, how="left", on=["PTID"])
    # df_fdg_av45[['EXAMDATE', 'AV45PET_DATE', 'FDGPET_DATE']] = df_fdg_av45[['EXAMDATE', 'AV45PET_DATE', 'FDGPET_DATE']].apply(pd.to_datetime)
    # df_fdg_av45['FDG_EXAMDATE_DIFF'] = (df_fdg_av45['EXAMDATE'] - df_fdg_av45['FDGPET_DATE']).dt.days.abs()
    # df_fdg_av45['AV45PET_EXAMDATE_DIFF'] = (df_fdg_av45['EXAMDATE'] - df_fdg_av45['AV45PET_DATE']).dt.days.abs()
    # df_fdg_av45 = df_fdg_av45[(~df_fdg_av45['FDGPET_filename'].isnull()) & (~df_fdg_av45['AV45PET_filename'].isnull())]
    # df_fdg_av45 = df_fdg_av45[
    #     (df_fdg_av45['FDG_EXAMDATE_DIFF'] < day_thresh) | (df_fdg_av45['AV45PET_EXAMDATE_DIFF'] < day_thresh)]
    # df_fdg_av45.loc[df_fdg_av45['FDG_EXAMDATE_DIFF'] >= day_thresh, fdgpet_cols + ['FDG_EXAMDATE_DIFF']] = None
    # df_fdg_av45.loc[df_fdg_av45['AV45PET_EXAMDATE_DIFF'] >= day_thresh, av45pet_cols + ['AV45PET_EXAMDATE_DIFF']] = None

    df_fdg.drop_duplicates(subset=['PTID', 'VISCODE'], keep='last', inplace=True)
    df_fdg.to_csv(os.path.join(output_dir, "ADNIMERGE_FDG.csv"), index=None)

    df_av45.drop_duplicates(subset=['PTID', 'VISCODE'], keep='last', inplace=True)
    df_av45.to_csv(os.path.join(output_dir, "ADNIMERGE_AV45.csv"), index=None)

    df_fdg_av45.drop_duplicates(subset=['PTID', 'VISCODE'], keep='last', inplace=True)
    df_fdg_av45.to_csv(os.path.join(output_dir, "ADNIMERGE_FDG_AV45.csv"), index=None)

    n_fdg = len(df_fdg_av45[~df_fdg_av45['FDGPET_filename'].isnull()])

    n_av45 = len(df_fdg_av45[~df_fdg_av45['AV45PET_filename'].isnull()])

    n_fdg_av45 = len(
        df_fdg_av45[(~df_fdg_av45['AV45PET_filename'].isnull()) & (~df_fdg_av45['FDGPET_filename'].isnull())])


    print(f'Num of FDG: {n_fdg}\nNum of AV45: {n_av45}\nNum of both: {n_fdg_av45}')
