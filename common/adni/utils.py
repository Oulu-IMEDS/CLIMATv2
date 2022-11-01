import copy
import os
import pickle
import re

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import omegaconf
import pandas as pd
import torch
import torchio as tio
from collagen.data import FoldSplit
from solt import data as sld

from common.utils import swap_weights

MARRY = {'Married': 0, 'Divorced': 1, 'Never married': 2, 'Widowed': 3}
RACE = {'White': 0, 'More than one': 1, 'Black': 2, 'Asian': 3, 'Am Indian/Alaskan': 4, 'Hawaiian/Other PI': 5}
ETHIC = {'Not Hisp/Latino': 0, 'Hisp/Latino': 1}
SEX = {'Male': 0, 'Female': 1}

NAME2DX = {'NL': 0, 'MCI to NL': 0, 'Dementia to NL': 0,
           'MCI': 1, 'NL to MCI': 1, 'Dementia to MCI': 1,
           'Dementia': 2, 'MCI to Dementia': 2, 'NL to Dementia': 2,
           'CN': 0}
NAME2DXCHANGE = {'NL': 0, 'MCI': 1, 'Dementia': 2,
                 'NL to MCI': 3, 'MCI to Dementia': 4, 'NL to Dementia': 5,
                 'MCI to NL': 6, 'Dementia to MCI': 7, 'Dementia to NL': 8}

MRI_BIOMARKERS = ['ST101SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST102CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST102SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST102TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST102TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST103CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST103SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST103TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST103TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST104CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST104SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST104TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST104TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST105CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST105SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST105TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST105TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST106CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST106SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST106TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST106TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST107CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST107SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST107TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST107TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST108CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST108SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST108TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST108TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST109CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST109SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST109TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST109TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST10CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST110CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST110SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST110TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST110TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST111CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST111SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST111TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST111TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST112SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST113CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST113SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST113TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST113TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST114CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST114SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST114TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST114TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST115CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST115SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST115TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST115TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST116CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST116SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST116TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST116TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST117CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST117SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST117TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST117TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST118CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST118SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST118TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST118TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST119CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST119SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST119TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST119TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST11SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST120SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST121CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST121SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST121TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST121TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST123CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST123SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST123TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST123TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST124SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST125SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST127SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST128SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST129CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST129SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST129TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST129TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST12SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST130CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST130SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST130TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST130TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST13CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST13SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST13TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST13TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST14CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST14SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST14TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST14TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST15CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST15SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST15TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST15TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST16SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST17SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST18SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST19SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST1SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST20SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST21SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST23CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST23SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST23TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST23TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST24CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST24SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST24TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST24TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST25CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST25SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST25TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST25TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST26CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST26SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST26TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST26TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST27SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST29SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST2SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST30SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST31CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST31SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST31TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST31TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST32CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST32SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST32TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST32TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST34CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST34SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST34TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST34TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST35CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST35SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST35TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST35TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST36CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST36SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST36TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST36TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST37SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST38CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST38SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST38TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST38TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST39CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST39SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST39TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST39TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST3SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST40CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST40SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST40TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST40TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST42SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST43CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST43SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST43TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST43TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST44CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST44SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST44TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST44TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST45CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST45SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST45TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST45TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST46CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST46SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST46TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST46TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST47CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST47SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST47TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST47TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST48CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST48SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST48TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST48TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST49CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST49SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST49TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST49TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST4SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST50CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST50SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST50TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST50TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST51CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST51SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST51TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST51TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST52CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST52SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST52TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST52TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST53SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST54CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST54SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST54TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST54TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST55CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST55SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST55TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST55TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST56CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST56SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST56TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST56TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST57CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST57SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST57TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST57TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST58CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST58SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST58TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST58TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST59CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST59SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST59TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST59TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST5SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST60CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST60SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST60TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST60TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST61SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST62CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST62SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST62TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST62TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST64CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST64SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST64TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST64TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST65SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST66SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST68SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST69SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST6SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST70SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST71SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST72CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST72SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST72TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST72TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST73CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST73SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST73TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST73TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST74CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST74SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST74TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST74TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST75SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST76SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST77SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST78SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST79SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST7SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST80SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST82CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST82SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST82TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST82TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST83CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST83SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST83TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST83TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST84CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST84SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST84TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST84TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST85CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST85SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST85TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST85TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST86SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST88SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST89SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST8SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST90CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST90SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST90TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST90TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST91CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST91SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST91TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST91TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST93CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST93SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST93TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST93TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST94CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST94SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST94TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST94TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST95CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST95SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST95TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST95TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST96SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST97CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST97SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST97TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST97TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST98CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST98SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST98TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST98TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST99CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST99SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST99TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                  'ST99TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST9SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16']

ECOG_COLS = ["EcogPtMem", "EcogPtLang", "EcogPtVisspat", "EcogPtPlan", "EcogPtOrgan", "EcogPtDivatt", "EcogPtTotal",
             "EcogSPMem", "EcogSPLang", "EcogSPVisspat", "EcogSPPlan", "EcogSPOrgan", "EcogSPDivatt", "EcogSPTotal"]

RAVLT_COLS = ["RAVLT_immediate", "RAVLT_learning", "RAVLT_forgetting", "RAVLT_perc_forgetting"]

CATEGORICAL_COLS = ['PTMARRY', 'PTRACCAT', 'PTGENDER', 'PTETHCAT']

IMAGING_COLS = ["FDG", "AV45", "Hippocampus", "WholeBrain", "Entorhinal", "Fusiform", "MidTemp", "ICV"]

NUMERICAL_COLS = ["PTEDUCAT", "CDRSB", "ADAS11", "MMSE", "FDG", "AV45", "ABETA",
                  "TAU", "PTAU", "APOE4", "AGE", "MOCA", "FAQ"] + IMAGING_COLS + ECOG_COLS + RAVLT_COLS

INPUT_COLS = ["RID", "PTID", "VISCODE", "EXAMDATE", "EXAMDATE_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16",
              "EXAMDATE_BAIPETNMRC_09_12_16"] + NUMERICAL_COLS + CATEGORICAL_COLS + MRI_BIOMARKERS

TARGET_COLS = ["DX", "ADAS13", "Ventricles"]

CATEGORY2CLASSNUM = {'PTMARRY': len(MARRY), 'PTRACCAT': len(RACE), 'PTGENDER': len(SEX), 'PTETHCAT': len(ETHIC)}

ANDI_STATS = {'PTEDUCAT': {'mean': 16.080601092896174, 'std': 2.7812653323981786},
              'CDRSB': {'mean': 2.0778595439265923, 'std': 2.8608800605250813},
              'ADAS11': {'mean': 10.978956457153469, 'std': 8.427395391255095},
              'MMSE': {'mean': 26.754575707154743, 'std': 3.9046620231606877},
              'FDG': {'mean': 1.205519013104013, 'std': 0.15897461584143763},
              'AV45': {'mean': 1.1896393719806766, 'std': 0.23153835156677274},
              'ABETA': {'mean': 957.5605063291141, 'std': 460.34889029374267},
              'TAU': {'mean': 293.18391561181437, 'std': 131.26801315886678},
              'PTAU': {'mean': 28.080776699029126, 'std': 14.444288974864556},
              'APOE4': {'mean': 0.519850383883457, 'std': 0.6494580344346474},
              'AGE': {'mean': 73.40147747982297, 'std': 7.009761188339303},
              'MOCA': {'mean': 23.305212014134277, 'std': 4.71146721997978},
              'ICV': {'mean': 1533433.5336497505, 'std': 164879.36544152227},
              'FAQ': {'mean': 5.292045041772612, 'std': 7.860126092900822},
              'Ventricles': {'mean': 42102.86017004578, 'std': 23210.83833331294},
              'Hippocampus': {'mean': 6693.833215447729, 'std': 1241.5608765386853},
              'WholeBrain': {'mean': 1012221.810259658, 'std': 111656.97621662005},
              'Entorhinal': {'mean': 3448.046553536567, 'std': 810.4115619671255},
              'Fusiform': {'mean': 17167.849827301397, 'std': 2810.187620409941},
              'MidTemp': {'mean': 19226.781198378136, 'std': 3131.9215423698915},
              'EcogPtMem': {'mean': 2.0516510867709816, 'std': 0.7390529051501148},
              'EcogPtLang': {'mean': 1.718833252067294, 'std': 0.6444962550095585},
              'EcogPtVisspat': {'mean': 1.3903625244674724, 'std': 0.54970113537349},
              'EcogPtPlan': {'mean': 1.4108301143020432, 'std': 0.5657352555768607},
              'EcogPtOrgan': {'mean': 1.527731052554957, 'std': 0.634450454165273},
              'EcogPtDivatt': {'mean': 1.82578151227919, 'std': 0.7626452115828485},
              'EcogPtTotal': {'mean': 1.6755577738213931, 'std': 0.5552456721101502},
              'EcogSPMem': {'mean': 2.1068210103005502, 'std': 0.9977728055462852},
              'EcogSPLang': {'mean': 1.6974968659241507, 'std': 0.8317339477493499},
              'EcogSPVisspat': {'mean': 1.5810584960346072, 'std': 0.8531298831479815},
              'EcogSPPlan': {'mean': 1.6808382060796347, 'std': 0.9149035905290949},
              'EcogSPOrgan': {'mean': 1.7552584644857059, 'std': 0.9739040385453893},
              'EcogSPDivatt': {'mean': 1.9680304112961622, 'std': 1.007808430904522},
              'EcogSPTotal': {'mean': 1.8020756534010725, 'std': 0.855051062100388},
              'RAVLT_immediate': {'mean': 35.42464215548695, 'std': 13.784074370955514},
              'RAVLT_learning': {'mean': 4.172684752104771, 'std': 2.835318212889977},
              'RAVLT_forgetting': {'mean': 4.1777840322731965, 'std': 2.6711444355967933},
              'RAVLT_perc_forgetting': {'mean': 57.40668932080106, 'std': 43.72792724993297}}


def normalize_variables(df):
    for col in NUMERICAL_COLS + MRI_BIOMARKERS:
        if col in df and col in ANDI_STATS:
            print(
                f'z-score normalize column ({col}) with mean {ANDI_STATS[col]["mean"]}, std {ANDI_STATS[col]["std"]}.')
            df[col] = (df[col] - ANDI_STATS[col]['mean']) / ANDI_STATS[col]['std']
    return df


def load_metadata(cfg, meta_root, meta_filename, pkl_meta_filename, seed, seq_len=5, eval_only=False, *args, **kwargs):
    pkl_meta_fullname = os.path.join(meta_root, pkl_meta_filename)
    if not os.path.isfile(pkl_meta_fullname):
        meta_fullname = os.path.join(meta_root, meta_filename)
        ds = pd.read_csv(meta_fullname)

        # Preprocess numerical data
        ds.replace(r'^\s*$', np.nan, regex=True, inplace=True)

        for col in NUMERICAL_COLS:
            if ds[col].dtype == object:
                ds.replace({col: {r'^<(\d+)': r'\1'}}, regex=True, inplace=True)
                ds.replace({col: {r'^>(\d+)': r'\1'}}, regex=True, inplace=True)
            ds[col] = ds[col].astype(float)

        if 'fdg' in cfg.parser.input and 'av45' in cfg.parser.input:
            ds = ds[(~ds['FDGPET_filename'].isnull()) & (~ds['AV45PET_filename'].isnull())]
        elif 'fdg' in cfg.parser.input:
            ds = ds[~ds['FDGPET_filename'].isnull()]
        elif 'av45' in cfg.parser.input:
            ds = ds[~ds['AV45PET_filename'].isnull()]
        else:
            raise ValueError(f'Not support input {cfg.parser.input}.')
        ds = normalize_variables(ds)
        for target in ['DXTARGET', 'ADAS13', 'Ventricles']:
            prognosis_target = []
            n_rows = len(ds.index)

            # ds[f'prognosis_{target}'] = [-1] * n_rows
            # ds[f'prognosis_mask_{target}'] = [None] * n_rows

            for i in range(0, seq_len + 1):
                prognosis_target.append(ds[f'{target}_{i}'].tolist())

            prognosis_target = np.array(prognosis_target)
            prognosis_target = np.transpose(prognosis_target, (1, 0))
            prognosis_target[np.isnan(prognosis_target)] = -1

            prognosis_mask = copy.deepcopy(prognosis_target)
            prognosis_mask[prognosis_mask != -1] = True
            prognosis_mask[prognosis_mask == -1] = False

            ds[f'prognosis_{target}'] = list(prognosis_target)
            ds[f'prognosis_mask_{target}'] = list(prognosis_mask)

        # Only keep rows with MRI images
        # ds = ds[ds['MRI_filename'].notnull()]

        # Handle nan targets
        # all_targets = ds.loc[: , ["DXTARGET_0", 'DXTARGET_1', 'DXTARGET_2', 'DXTARGET_3', 'DXTARGET_4', 'DXTARGET_5']].copy()
        # ds['DXTARGET_mean'] = all_targets.mean(skipna=True, axis=1).round(0)
        #
        # ds = ds[ds['DXTARGET_mean'].notnull()]
        # ds['DXTARGET_mean'] = ds['DXTARGET_mean'].astype(dtype='int32')

        splits = FoldSplit(ds, n_folds=cfg.n_folds, target_col='DXTARGET_0', group_col='PTID', random_state=seed)
        split_data = []
        for fold_i, (train, val) in enumerate(splits):
            # train = train.drop(columns=['DXTARGET_mean'])
            # val = val.drop(columns=['DXTARGET_mean'])
            split_data.append((train, val))
        split_data = tuple(split_data)

        print(f'Writing metadata {pkl_meta_fullname}')
        with open(pkl_meta_fullname, "wb") as f:
            pickle.dump(split_data, f, protocol=4)
    else:
        print(f'Loading metadata {pkl_meta_fullname}')
        with open(pkl_meta_fullname, "rb") as f:
            split_data = pickle.load(f)
    return split_data


def remove_date_from_name(x):
    m = re.search('^(.+)_\d\d_\d\d_\d\d$', x)
    if m is not None:
        o = m.group(1)
    else:
        o = x
    return o


def remove_suffix_from_mri_biomarkers(x):
    suffix = "_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16"
    l = len(suffix)
    return x[:-l]


def is_nan(x):
    return isinstance(x, float) and not np.isnan(x)


def parse_categorical_metadata(entry, metadata):
    input = {}
    # ['PTMARRY', 'PTRACCAT', 'PTGENDER', 'PTETHCAT']

    if 'PTMARRY' in metadata:
        marriage = [0.0] * len(MARRY)
        if is_nan(entry['PTMARRY']):
            marriage_value = int(MARRY[entry['PTMARRY']])
            marriage[marriage_value] = 1.0
            input['PTMARRY_mask'] = torch.tensor(1.0, dtype=torch.float32)
        else:
            input['PTMARRY_mask'] = torch.tensor(0.0, dtype=torch.float32)
        input['PTMARRY'] = torch.tensor(marriage)

    if 'PTRACCAT' in metadata:
        race = [0.0] * len(RACE)
        if is_nan(entry['PTRACCAT']):
            race_value = int(RACE[entry['PTRACCAT']])
            race[race_value] = 1.0
            input['PTRACCAT_mask'] = torch.tensor(1.0, dtype=torch.float32)
        else:
            input['PTRACCAT_mask'] = torch.tensor(0.0, dtype=torch.float32)
        input['PTRACCAT'] = torch.tensor(race)

    if 'PTGENDER' in metadata:
        sex = [0.0] * len(SEX)
        if is_nan(entry['PTGENDER']):
            sex_value = int(SEX[entry['PTGENDER']])
            sex[sex_value] = 1.0
            input['PTGENDER_mask'] = torch.tensor(1.0, dtype=torch.float32)
        else:
            input['PTGENDER_mask'] = torch.tensor(0.0, dtype=torch.float32)
        input['PTGENDER'] = torch.tensor(sex)

    if 'PTETHCAT' in metadata:
        ethic = [0.0] * len(ETHIC)
        if is_nan(entry['PTETHCAT']):
            ethic_value = int(ETHIC[entry['PTETHCAT']])
            ethic[ethic_value] = 1.0
            input['PTETHCAT_mask'] = torch.tensor(1.0, dtype=torch.float32)
        else:
            input['PTETHCAT_mask'] = torch.tensor(0.0, dtype=torch.float32)
        input['PTETHCAT'] = torch.tensor(ethic)

    return input


def parse_numerical_metadata(entry, metadata):
    input = {}
    for col in NUMERICAL_COLS:
        item_name = remove_date_from_name(col)
        if item_name in metadata:
            # print(item_name)
            if entry[col] and entry[col] == entry[col]:
                input[f'{item_name}_mask'] = torch.tensor(1.0, dtype=torch.float32)
                input[item_name] = torch.tensor(entry[col]).unsqueeze(-1).type(torch.float32)
            else:
                input[f'{item_name}_mask'] = torch.tensor(0.0, dtype=torch.float32)
                input[item_name] = torch.tensor([-1.0], dtype=torch.float32)

    if 'MRI_BIOMARKERS' in metadata:
        for col in MRI_BIOMARKERS:
            item_name = remove_suffix_from_mri_biomarkers(col)
            if entry[col] and entry[col] == entry[col]:
                input[f'{item_name}_mask'] = torch.tensor(1.0, dtype=torch.float32)
                input[item_name] = torch.tensor(entry[col]).unsqueeze(-1).type(torch.float32)
            else:
                input[f'{item_name}_mask'] = torch.tensor(0.0, dtype=torch.float32)
                input[item_name] = torch.tensor([-1.0], dtype=torch.float32)

    if 'Ecog' in metadata:
        for col in ECOG_COLS:
            if entry[col] and entry[col] == entry[col]:
                input[f'{col}_mask'] = torch.tensor(1.0, dtype=torch.float32)
                input[col] = torch.tensor(entry[col]).unsqueeze(-1).type(torch.float32)
            else:
                input[f'{col}_mask'] = torch.tensor(0.0, dtype=torch.float32)
                input[col] = torch.tensor([-1.0], dtype=torch.float32)

    return input


def check_empty(x):
    return x and x == x


def standardize_nslices(x, std_sz=256, std_nslices=5):
    projection = check_projection(x, std_sz)

    if projection == 'axial':
        nslice = x.shape[2]
    elif projection == 'sagittal':
        nslice = x.shape[0]
    else:
        nslice = x.shape[1]

    if nslice < std_nslices:
        pad1 = (std_nslices - nslice) // 2
        pad2 = std_nslices - pad1
        if projection == 'axial':
            x = np.pad(x, ((0, 0), (0, 0), (pad1, pad2)))
        elif projection == 'sagittal':
            x = np.pad(x, ((pad1, pad2), (0, 0), (0, 0)))
        else:
            x = np.pad(x, ((0, 0), (pad1, pad2), (0, 0)))
    else:
        left = (nslice - std_nslices) // 2
        right = left + std_nslices
        if projection == 'axial':
            x = x[:, :, left:right]
        elif projection == 'sagittal':
            x = np.transpose(x[left:right, :, :], (1, 2, 0))
        else:
            x = np.transpose(x[:, left:right, :], (0, 2, 1))

    if x.shape[-1] != std_nslices:
        raise ValueError(f'Standardize wrongly. Get {x.shape[-1]} instead of {std_nslices}.')

    return x


def parse_item_progs(root, entry, trf, **kwargs):
    input = {'ID': entry['ID']}
    cm = plt.get_cmap('jet')
    if "IMG" in kwargs["metadata"]:
        if not isinstance(root, dict) and not isinstance(root, omegaconf.DictConfig):
            raise ValueError(f'Root must be a dict. But found {root}.')

        imgs = []
        if 'fdg' in kwargs['input'] and root['dir_fdg']:
            fullname = os.path.join(root['path'], root['dir_fdg'], entry['FDGPET_filename'])
            img = nib.load(fullname).get_fdata()
            # m = np.min(img)
            # M = np.max(img)
            # img = 255.0 * (img - m) / (M - m)
            mean = 1385.473
            std = 3435.718
            img = (img - mean) / std
            ts = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
            imgs.append(ts)
        if 'av45' in kwargs['input'] and root['dir_av45']:
            fullname = os.path.join(root['path'], root['dir_av45'], entry['AV45PET_filename'])
            img = nib.load(fullname).get_fdata()
            # m = np.min(img)
            # M = np.max(img)
            # img = 255.0 * (img - m) / (M - m)
            mean = 24911.911
            std = 52297.735
            img = (img - mean) / std
            ts = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
            imgs.append(ts)

        if len(imgs) > 1:
            input['IMG'] = torch.cat(imgs, 0)
            input['IMG_mask'] = torch.tensor(True)
        elif len(imgs) == 1:
            input['IMG'] = imgs[0]
            input['IMG_mask'] = torch.tensor(True)
        else:
            zeros = torch.zeros((1, 160, 160, 160), dtype=torch.float32)
            input['IMG'] = [zeros, zeros]
            input['IMG_mask'] = torch.tensor(False)

    output = {}

    targets = ["DXTARGET", "ADAS13", "Ventricles"]
    # target2gradingnum = {dx : 3, dxchange: 9}
    for target in targets:
        # has_grading = entry[target] is not None and entry[target] == entry[target]
        output[f"prognosis_{target}"] = torch.tensor(entry[f"prognosis_{target}"].astype(int))
        output[f"prognosis_mask_{target}"] = entry[f"prognosis_mask_{target}"] == 1

    cate_meta = parse_categorical_metadata(entry, kwargs["metadata"])
    nume_meta = parse_numerical_metadata(entry, kwargs["metadata"])
    input.update(cate_meta)
    input.update(nume_meta)

    output['data'] = {'input': input}

    return output


def calculate_class_weights(df, cfg):
    grading = cfg.grading
    # all_stats = None
    all_stats = {'y0': [], 'pn': []}
    max_grade = 2
    for i in range(1, cfg.seq_len + 1):
        pn_counts = []
        for v in range(0, max_grade + 1):
            count = len(df[df[f'{grading}_{i}'] == v].index)
            pn_counts.append(count)

        all_stats['pn'].append(pn_counts)

    y0_counts = []
    for v in range(0, max_grade + 1):
        count = len(df[df[f'{grading}_0'] == v].index)
        y0_counts.append(count)

    all_stats['y0'] = np.array([y0_counts])
    all_stats['y0'] = all_stats['y0'] / all_stats['y0'].sum(axis=1, keepdims=True)

    all_stats['pn'] = np.array(all_stats['pn'])
    all_stats['pn'] = all_stats['pn'] / all_stats['pn'].sum(axis=1, keepdims=True)

    all_stats['y0'] = swap_weights(all_stats['y0'])[0, :]
    all_stats['pn'] = swap_weights(all_stats['pn'])
    return all_stats['y0'], all_stats['pn']


def img_labels2solt(inp):
    img = inp
    return sld.DataContainer((img), fmt='I')


def unpack_solt_data(dc: sld.DataContainer):
    img = dc.data[0]
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img


def init_transforms():
    train_trf = tio.transforms.Compose((
        tio.transforms.RandomAffine(degrees=(0, 0, 20), translation=20, center='image', default_pad_value='minimum'),
        tio.transforms.RandomElasticDeformation(),
        tio.transforms.Gamma(0.3),
        tio.transforms.RandomNoise(std=(0, 0.25))
    ))

    test_trf = tio.transforms.Compose(())

    return {'train': train_trf, 'eval': test_trf}


if __name__ == "__main__":
    fullname = "/home/hoang/data/ANDI/tadpole_challenge/TADPOLE_D1_prognosis_br.csv"
    df = pd.read_csv(fullname)
    print(f"PTMARRY:\n{df['PTMARRY'].unique()}")
    print(f"PTRACCAT:\n{df['PTRACCAT'].unique()}")
    print(f"PTGENDER:\n{df['PTGENDER'].unique()}")
    print(f"PTETHCAT:\n{df['PTETHCAT'].unique()}")
    print(f"PTGENDER:\n{df['PTGENDER'].unique()}")

    MARRY = {'Married': 0, 'Divorced': 1, 'Never married': 2, 'Widowed': 3}
    RACE = {'White': 0, 'More than one': 1, 'Black': 2, 'Asian': 3, 'Am Indian/Alaskan': 4, 'Hawaiian/Other PI': 5}
    ETHIC = {'Not Hisp/Latino': 0, 'Hisp/Latino': 1}
    SEX = {'Male': 0, 'Female': 1}


def check_projection(x, s):
    if x.shape[0] == s and x.shape[1] == s:
        return 'axial'
    elif x.shape[1] == s and x.shape[2] == s:
        return 'sagittal'
    elif x.shape[0] == s and x.shape[2] == s:
        return 'coronal'
