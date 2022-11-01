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
    image_root = "/home/hoang/data/ANDI/FDG_PET_CoregAvg"

    spacing_stats = []
    nslices_stats = []
    size_stats = []
    force = True
    std_shape = (160, 160, 160)
    voxel_size = (1.5, 1.5, 1.5)
    suffix = "_VXSTD.nii"
    for r, d, f in os.walk(image_root):
        for filename in f:
            fullname = os.path.join(r, filename)
            if fullname.endswith(".nii") and not fullname.endswith(suffix):
                try:
                    img_nii = nib.load(fullname)
                    img_nii = nib.funcs.squeeze_image(img_nii)
                    img_nii = conform(img_nii, out_shape=std_shape, voxel_size=voxel_size)
                    out_fullname = fullname[:-4] + suffix
                    if out_fullname != fullname:
                        nib.loadsave.save(img_nii, out_fullname)
                    else:
                        print(f'WARN: Overwriting file.')
                except:
                    print(f'Corruped file {filename}')