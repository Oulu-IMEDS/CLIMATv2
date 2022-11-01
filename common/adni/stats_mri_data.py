import os
import re
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from common.adni.utils import check_projection
from common.adni.update_mri_metadata import check_valid_spacing, get_reordered_spacing
from nibabel.processing import conform


if __name__ == "__main__":
    # root = "/home/hoang/data/ANDI/MRI_T1_3plan_all"
    # fullname = os.path.join(root, "MRI_T1_3plan_all_6_10_2021.csv")
    # root = "/home/hoang/data/ANDI/AV45_PET_coreg_avg"
    # fullname = os.path.join(root, "AV45_PET_coreg_avg_7_05_2021.csv")
    # out_fullname = "/home/hoang/data/ANDI/AV45_PET_coreg_avg/AV45_PET_coreg_avg_7_05_2021_path.csv"
    # image_root = "/home/hoang/data/ANDI/MRI_T1_3plan_all/ADNI/"
    image_root = "/home/hoang/data/ANDI/AV45_PET_coreg_avg"
    # image_root = "/home/hoang/data/ANDI/FDG_PET_CoregAvg"

    spacing_stats = []
    nslices_stats = []
    size_stats = []
    force = True

    suffix = 'VXSTD.nii'
    mean = 0
    std = 0

    rows = []
    count = 0
    for r, d, f in os.walk(image_root):
        for filename in f:
            fullname = os.path.join(r, filename)
            if fullname.endswith(suffix):
                try:
                    row = {'filename': fullname}
                    img_nii = nib.load(fullname)
                    img = img_nii.get_fdata()

                    mean += img.mean()
                    std += img.std()
                    count += 1

                    # projection = check_projection(img, 256)
                    # spacing = get_reordered_spacing(img_nii, projection)
                    # row['x'] = spacing[0]
                    # row['y'] = spacing[1]
                    # row['z'] = spacing[2]
                    # if check_valid_spacing(img_nii, projection):
                    #     print(f'Found {fullname} with invalid spacing {img_nii.header.get_zooms()}')
                    spacing_stats.append(img_nii.header.get_zooms())
                    nslices_stats.append(min(img.shape[:3]))
                    size_stats.append(max(img.shape[:3]))
                    rows.append(row)
                except:
                    print(f'Cannot open {fullname}')

    mean /= count
    std /= count

    print(f'Mean: {mean}, std: {std}')

    df = pd.DataFrame(rows)
    # df.to_csv('./mri_spacing.csv', index=None)

    nslices_stats = np.array(nslices_stats)
    size_stats = np.array(size_stats)
    min_num_slices = np.min(nslices_stats)
    max_num_slices = np.max(nslices_stats)
    print(nslices_stats)
    # plt.hist(nslices_stats, 50)
    # plt.show()
    min_size = np.min(size_stats)
    max_size = np.max(size_stats)
    print(f'Min size: {min_size}, max size: {max_size}')
    print(f'Min num of slices: {min_num_slices}, max num of slices: {max_num_slices}')
    spacing_stats = np.array(spacing_stats)
    print(f'x-spacing: {set(list(spacing_stats[:, 0]))},\nMean: {spacing_stats[:, 0].mean()}')
    print(f'y-spacing: {set(list(spacing_stats[:, 1]))},\nMean: {spacing_stats[:, 1].mean()}')
    print(f'z-spacing: {set(list(spacing_stats[:, 2]))},\nMean: {spacing_stats[:, 2].mean()}')
    # df = pd.DataFrame(rows)
    # df['x'] = list(spacing_stats[:, 0])
    # df['y'] = list(spacing_stats[:, 1])
    # df['z'] = list(spacing_stats[:, 2])
    # df.to_csv('./mri_spacing.csv', index=None)
    # plt.hist(spacing_stats[:, 0], 50)
    # plt.show()
    # plt.hist(spacing_stats[:, 1], 50)
    # plt.show()
    # plt.hist(spacing_stats[:, 2], 50)
    # plt.show()
