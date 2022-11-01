import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

from common.utils import read_sas7bdata_pd

sides = [None, 'R', 'L']


def build_single_img_based_multi_progression_meta(oai_src_dir, only_baseline=False, use_sas=False):
    # visits =     ['00', '12', '24', '36', '72', '96']
    # exam_codes = ['00', '01', '03', '05', '08', '10']

    visits = ['00', '12', '24', '36', '48', '72', '96']
    exam_codes = ['00', '01', '03', '05', '06', '08', '10']

    # visits =     ['48', '72', '96']
    # exam_codes = ['06', '08', '10']

    non_progressor_code = len(visits)

    grading_types = ['KL', 'OSTL', 'OSTM', 'OSFL', 'OSFM', 'JSL', 'JSM']
    max_grades = {'KL': 4, 'OSTL': 3, 'OSTM': 3, 'OSFL': 3, 'OSFM': 3, 'JSL': 3, 'JSM': 3}
    include_TKR = True
    grad_files = []
    if not os.path.isfile("oai_master_df.csv") or not os.path.isfile("oai_grad_files.pkl"):

        for i, visit in enumerate(visits):
            print(f'==> Reading OAI {visit} visit')

            if use_sas:
                meta = read_sas7bdata_pd(os.path.join(oai_src_dir,
                                                      'Semi-Quant Scoring_SAS',
                                                      f'kxr_sq_bu{exam_codes[i]}.sas7bdat'))
            #     data_clinical = read_sas7bdata_pd(os.path.join(oai_src_dir, f'allclinical{exam_codes[i]}.sas7bdat'))
            else:
                meta = pd.read_csv(os.path.join(oai_src_dir,
                                                'Semi-Quant Scoring_ASCII',
                                                f'kxr_sq_bu{exam_codes[i]}.txt'), sep='|')
            #     # data_clinical = pd.read_csv(os.path.join(oai_src_dir, 'AllClinical_ASCII', f'AllClinical{exam_codes[i]}.txt'), sep='|')
            data_clinical = build_clinical(oai_src_dir, visit_id=exam_codes[i], use_sas=True)

            meta['ID'] = meta['ID'].astype(str)
            meta.replace({'SIDE': {1: sides[1], 2: sides[2]}}, inplace=True)
            data_clinical['ID'] = data_clinical['ID'].astype(str)
            data_clinical.rename(columns={'Side': 'SIDE'}, inplace=True)

            meta = meta.merge(data_clinical, on=['ID', 'SIDE'], how='left')
            # Dropping the data from multiple projects
            meta.drop_duplicates(subset=['ID', 'SIDE'], inplace=True)
            meta.fillna(-1, inplace=True)
            for c in meta.columns:
                meta[c.upper()] = meta[c]
            # Removing the TKR and KL4 at the baseline
            if i == 0:
                meta = meta[meta[f'V{exam_codes[i]}XRKL'] != -1]
                # meta = meta[meta[f'V{exam_codes[i]}XRKL'] < 4]
            # meta = meta[meta[f'V{exam_codes[i]}XRKL'] <= max_grades['KL']]

            meta['KL'] = meta[f'V{exam_codes[i]}XRKL']
            meta['OSTL'] = meta[f'V{exam_codes[i]}XROSTL']
            meta['OSTM'] = meta[f'V{exam_codes[i]}XROSTM']

            meta['OSFL'] = meta[f'V{exam_codes[i]}XROSFL']
            meta['OSFM'] = meta[f'V{exam_codes[i]}XROSFM']

            meta['JSL'] = meta[f'V{exam_codes[i]}XRJSL']
            meta['JSM'] = meta[f'V{exam_codes[i]}XRJSM']

            # Merge levels 0 into 1 in every grading
            # Let TKR be the highest level
            for grading in grading_types:
                meta[grading] = meta[grading].round(decimals=0)
                if grading == "KL":
                    # Remove invalid records that are out of upper bound
                    # meta = meta[meta[grading] <= max_grades[grading]]
                    meta.loc[meta[grading] > max_grades[grading], grading] = None

                    # Set TKR the highest level
                    if include_TKR:
                        meta.loc[meta[grading] == -1, grading] = max_grades[grading] + 1
                        max_kl = max_grades[grading] + 1
                    else:
                        max_kl = max_grades[grading]

                    # Remove invalid records that are out of lower bound
                    # meta = meta[meta[grading] >= 0]
                    meta.loc[meta[grading] < 0, grading] = None

                    for v in range(max_kl):
                        meta = meta.replace({grading: {v + 1: v}})
                else:
                    meta.loc[meta[grading] < 0, grading] = None

            if i == 0:
                _pain_key = f'P01'
            else:
                _pain_key = f'V{exam_codes[i]}'

            # Add visit columns
            meta['visit'] = int(visit)
            meta['visit_id'] = int(exam_codes[i])

            grad_files.append(
                meta[['ID', 'SIDE', 'KL', 'visit', 'visit_id', 'OSTL', 'OSTM', 'OSFL', 'OSFM', 'JSL', 'JSM',
                      'AGE', 'SEX', 'BMI', 'INJ', 'SURG', 'WOMAC', 'V00SITE']])

        id_set_last_fu = set(grad_files[-1].ID.values.astype(int).tolist())  # Subjects present at all FU

        master_df = pd.concat(grad_files)

        with open("oai_grad_files.pkl", "wb") as f:
            pickle.dump(grad_files, f, protocol=4)
        master_df.to_csv("oai_master_df.csv", index=None)
    else:
        print(f'Loading oai_grad_files.pkl')
        with open("oai_grad_files.pkl", "rb") as f:
            grad_files = pickle.load(f)
        master_df = pd.read_csv("oai_master_df.csv")

    master_df['ID'] = master_df['ID'].astype(str)
    # Get baseline and follow-ups
    # for follow_up_id in range(0, len(KL_files)):
    #     KL_files[follow_up_id] = KL_files[follow_up_id].set_index(['ID', 'SIDE'])

    # master_df = master_df.set_index(['ID', 'SIDE'])

    # looking for progressors
    identified_prog = set()

    fus_df = []

    n_bs_knees = grad_files[0].shape[0]

    for bs_knee_id, knee in tqdm(grad_files[0].iterrows(), total=n_bs_knees, desc='Processing OAI:'):
        if int(knee.ID) in identified_prog:
            if identified_prog[int(knee.ID)] == sides[int(knee.SIDE)]:
                continue

        participant = master_df[(master_df['ID'] == knee.ID) & (master_df['SIDE'] == knee.SIDE)]
        participant.sort_values(by=['visit_id'])

        assert len(participant) > 0

        n_follow_ups = len(participant.index)

        for fu1_id in range(n_follow_ups - 1):
            valid_gradings1 = get_valid_gradings_1st_fu(participant.iloc[fu1_id], grading_types, max_grades)

            fu1 = participant.iloc[fu1_id]  # .add_suffix('1')

            # Condition if forced to use baseline only
            if only_baseline:
                fu1_cond = fu1['visit'] == 0
            else:
                fu1_cond = True

            if len(valid_gradings1) == 0:
                print(f'No valid gradings in FU1.')
                continue

            if not fu1_cond:
                print(f'Skip as the initial data is not a baseline.')
                continue

            start_visit = fu1['visit']

            fus = fu1

            valids = []
            for grade_id, grading in enumerate(grading_types):
                prev_grade = fu1[f'{grading}']

                reach_max = False
                valid_by_grading = True
                fus_by_grading = None
                for fu2_id in range(fu1_id + 1, n_follow_ups):
                    fu2 = participant.iloc[fu2_id].add_suffix('2')
                    dt12_y = int((fu2['visit2'] - start_visit) // 12)

                    if not check_valid_grading(fu2[f'{grading}2']):
                        continue

                    # Time points after a time point with max grade must be graded the max value
                    if reach_max or prev_grade == max_grades[grading]:
                        reach_max = True
                        fu2[f'{grading}2'] = max_grades[grading]

                    valid_bl = check_progression_validity(fu1[f'{grading}'], fu2[f'{grading}2'])
                    valid_prev = check_progression_validity(prev_grade, fu2[f'{grading}2'])
                    if not valid_bl or not valid_prev:
                        print(
                            f'[Invalid] {grading}, BL: {fu1[grading]}, prev: {prev_grade}, cur: {fu2[f"{grading}2"]}.')
                        valid_by_grading = False
                        break
                    if prev_grade == 1.0 and fu2[f"{grading}2"] == 0.0:
                        print(f'{grading}, BL: {fu1[grading]}, prev: {prev_grade}, cur: {fu2[f"{grading}2"]}.')

                    new_cols = {}
                    new_cols[f'DT_{dt12_y}y'] = (fu2['visit2'] - start_visit) / 12.0
                    new_cols[f'{grading}_{dt12_y}y'] = fu2[f'{grading}2']
                    new_cols = pd.Series(new_cols)

                    if fus_by_grading is None:
                        fus_by_grading = new_cols
                    else:
                        fus_by_grading = pd.concat([fus_by_grading, new_cols])

                    if check_valid_grading(fu2[f'{grading}2']):
                        prev_grade = fu2[f'{grading}2']

                if not valid_by_grading:
                    fus_by_grading = None

                # End fu2 loop
                if fus_by_grading is not None:
                    fus = pd.concat([fus, fus_by_grading])
                valids.append(valid_by_grading)

            # End grading loop
            if any(valids):
                fus_df.append(fus.to_dict())

    print(f'\nConverting to dataframes...')
    fus_df = pd.DataFrame(fus_df)

    # Remove redundant columns
    # Rename ID and Side
    fus_df = fus_df.rename(columns={'SIDE': 'Side'})
    if 'IDs' in fus_df:
        fus_df = fus_df.drop(columns=['ID2'])
    if 'SIDE2' in fus_df:
        fus_df = fus_df.drop(columns=['SIDE2'])

    fus_df = fus_df.astype({'ID': str})

    return fus_df


def check_valid_grading(v):
    return v is not None and not np.isnan(v)


def build_clinical(oai_src_dir, use_sas=False, visit_id='00'):
    if use_sas:
        data_enrollees = read_sas7bdata_pd(os.path.join(oai_src_dir, 'AllClinical_SAS', 'enrollees.sas7bdat'))
        data_clinical = read_sas7bdata_pd(
            os.path.join(oai_src_dir, 'AllClinical_SAS', f'allclinical{visit_id}.sas7bdat'))
    else:
        data_enrollees = pd.read_csv(os.path.join(oai_src_dir, 'General_ASCII', 'Enrollees.txt'), sep='|')
        data_clinical = pd.read_csv(os.path.join(oai_src_dir, 'AllClinical_ASCII', f'AllClinical{visit_id}.txt'),
                                    sep='|')

    clinical_data_oai = data_clinical.merge(data_enrollees, on='ID')

    clinical_data_oai = clinical_data_oai.replace({'0: No': 0, '1: Yes': 1})

    if visit_id == '00':
        AGE_col = 'V00AGE'
        BMI_col = 'P01BMI'
        HEIGHT_col = 'P01HEIGHT'
        WEIGHT_col = 'P01WEIGHT'
        INJL_col = 'P01INJL'
        INJR_col = 'P01INJR'
        SURGL_col = 'P01KSURGL'
        SURGR_col = 'P01KSURGR'
        WOMACL_col = 'V00WOMTSL'
        WOMACR_col = 'V00WOMTSR'
    else:
        AGE_col = f'V{visit_id}AGE'
        BMI_col = f'V{visit_id}BMI'
        HEIGHT_col = f'V{visit_id}HEIGHT'
        WEIGHT_col = f'V{visit_id}WEIGHT'
        INJL_col = f'V{visit_id}INJL12'
        INJR_col = f'V{visit_id}INJR12'
        SURGL_col = f'V{visit_id}KSRGL12'
        SURGR_col = f'V{visit_id}KSRGR12'
        WOMACL_col = f'V{visit_id}WOMTSL'
        WOMACR_col = f'V{visit_id}WOMTSR'

    # Age, Sex, BMI
    clinical_data_oai['SEX'] = clinical_data_oai['P02SEX']
    clinical_data_oai['AGE'] = clinical_data_oai[AGE_col]
    clinical_data_oai['BMI'] = clinical_data_oai[BMI_col]

    # clinical_data_oai['HEIGHT'] = clinical_data_oai[HEIGHT_col]
    # clinical_data_oai['WEIGHT'] = clinical_data_oai[WEIGHT_col]

    clinical_data_oai_left = clinical_data_oai.copy()
    clinical_data_oai_right = clinical_data_oai.copy()

    # Making side-wise metadata
    clinical_data_oai_left['Side'] = 'L'
    clinical_data_oai_right['Side'] = 'R'

    # Injury (ever had)
    clinical_data_oai_left['INJ'] = clinical_data_oai_left[INJL_col]
    clinical_data_oai_right['INJ'] = clinical_data_oai_right[INJR_col]

    # Surgery (ever had)
    clinical_data_oai_left['SURG'] = clinical_data_oai_left[SURGL_col]
    clinical_data_oai_right['SURG'] = clinical_data_oai_right[SURGR_col]

    # Total WOMAC score
    clinical_data_oai_left['WOMAC'] = clinical_data_oai_left[WOMACL_col]
    clinical_data_oai_right['WOMAC'] = clinical_data_oai_right[WOMACR_col]

    clinical_data_oai_left['V00SITE'] = clinical_data_oai['V00SITE']
    clinical_data_oai_right['V00SITE'] = clinical_data_oai['V00SITE']

    clinical_data_oai = pd.concat((clinical_data_oai_left, clinical_data_oai_right))
    clinical_data_oai.ID = clinical_data_oai.ID.values.astype(str)

    for col in ['BMI', 'INJ', 'SURG', 'WOMAC']:
        clinical_data_oai.loc[clinical_data_oai[col].isin(['.: Missing Form/Incomplete Workbook'])] = None
        clinical_data_oai.loc[clinical_data_oai[col] < 0, col] = None
    return clinical_data_oai[['ID', 'Side', 'AGE', 'SEX', 'BMI', 'INJ', 'SURG', 'WOMAC', 'V00SITE']]


def get_valid_gradings_1st_fu(r, grading_types, max_grades):
    valid_gradings = []
    for grading in grading_types:
        if r[grading] is None or np.isnan(r[grading]) or 0 <= r[grading] < max_grades[grading]:
            valid_gradings.append(grading)
        else:
            # print(f'Init input violation: grading {grading} = {r[grading]}.')
            pass
    return valid_gradings


def check_progression_validity(g1, g2):
    if g1 is None or g2 is None or np.isnan(g1) or np.isnan(g2):
        valid = True
    elif g1 <= g2:
        valid = True
    else:
        valid = False
    return valid
