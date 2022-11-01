import argparse
import os

import pandas as pd

from common import most
from common import oai
from common.oai import build_jsw

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--oai_meta',
                        default='/home/hoang/data/X-Ray_Image_Assessments_SAS')
    parser.add_argument('--imgs_dir', default='/home/hoang/data/MOST_OAI_00_0_2')
    parser.add_argument('--save_meta', default='./Metadata/in1_outseq_gradingmulti_inter_all_new')
    parser.add_argument('--grad_type', default='multi')
    parser.add_argument('--suffix', default='')
    args = parser.parse_args()

    print(args)

    if args.grad_type != 'multi':
        raise ValueError(f'Not support grade type {args.grad_type}.')

    if not os.path.isdir(args.save_meta):
        os.makedirs(args.save_meta)

    oai_progression_filename = f'OAI_progression{args.suffix}_site.csv'
    oai_participants_filename = f'OAI_participants{args.suffix}.csv'
    force_oai = False

    os.makedirs(args.save_meta, exist_ok=True)

    # OAI
    if not os.path.isfile(os.path.join(args.save_meta, oai_progression_filename)) or force_oai:
        oai_meta = oai.build_single_img_based_multi_progression_meta(args.oai_meta)
        print(f'Saving OAI meta file to {oai_progression_filename}')
        oai_meta.to_csv(os.path.join(args.save_meta, oai_progression_filename), index=None)
    else:
        oai_meta = pd.read_csv(os.path.join(args.save_meta, oai_progression_filename))
        print('OAI progression metadata exists!')
        oai_meta = oai_meta.reindex(sorted(oai_meta.columns), axis=1)
