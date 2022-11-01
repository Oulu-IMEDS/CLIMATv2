import pandas as pd
import numpy as np
import argparse
import time
import os


if __name__ == "__main__":

    # search for all subjects who have an MRI scan.
    #
    root = '/home/hoang/data/ANDI/tadpole_challenge/'
    d12Df = pd.read_csv(os.path.join(root, 'TADPOLE_D1_D2.csv'))

    print(d12Df.loc[:,'IMAGEUID_UCSFFSX_11_02_15_UCSFFSX51_08_01_16'])
    imageIDs = pd.to_numeric(d12Df.loc[:,'IMAGEUID_UCSFFSX_11_02_15_UCSFFSX51_08_01_16'], errors='coerce')

    print(np.sum(np.isnan(imageIDs)))
    print(d12Df.loc[:, 'Ventricles'])
    print(imageIDs.shape)
    dropMask = np.logical_and(np.logical_not(np.isnan(imageIDs)), np.logical_not(np.isnan(d12Df.loc[:, 'Ventricles'])))
    imageIDs2 = imageIDs[np.logical_not(np.isnan(imageIDs))] # with MRI scan
    imageIDs3 = imageIDs[dropMask] # with MRI scan and ventricle passing QC

    print(f'Num of MRI scans: {imageIDs2.shape}') # with MRI scan
    print(f'Num of MRI scans and ventricle passing QC: {imageIDs3.shape}') # with MRI scan and ventricle passing QC

    # print the imageUIDs to be downloaded from LONI

    for i in range(imageIDs3.shape[0]):
      print(int(imageIDs3.iloc[i]),end=',')

    print('\n\n Search for all these IDs on LONI, add them to collection, then download then as NIFTI. ')
