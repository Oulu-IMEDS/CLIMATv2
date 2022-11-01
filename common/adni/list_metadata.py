import pandas as pd
import re
import numpy as np

if __name__ == "__main__":
    fullname1 = "/home/hoang/data/ANDI/Metadata/UCSFFSL51ALL_08_01_16.csv"
    fullname2 = "/home/hoang/data/ANDI/tadpole_challenge/TADPOLE_D1_D2.csv"
    df1 = pd.read_csv(fullname1)
    df2 = pd.read_csv(fullname2)

    suffix = "_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16"

    l1 = [c + suffix for c in df1.columns.tolist() if c.startswith("ST") and c != "STATUS"]
    print(l1)
    print(len(l1))

    l2 = df2.columns.tolist()
    # print(l2)
    l = [m for m in l2 if m.startswith("ST") and not m.startswith("STATUS") and "UCSFFSL" in m]

    print(l)
    print(len(l))

    l_final = []
    for ll in l:
        has_data = any([isinstance(x, str) and bool(x and x.strip()) for x in df2[ll].tolist()])
        if has_data:
            l_final.append(ll)

    print(l_final)
    print(len(l_final))

    # l = list(set(l1) & set(l2))
    # for m in l1:
    #     if m not in l2:
    #         print(f'Cannot found {m}')
    #     else:
    #         print(f'Found {m}')