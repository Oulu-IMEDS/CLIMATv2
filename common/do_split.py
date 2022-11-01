import random

import hydra
import yaml
from omegaconf import OmegaConf
import pickle
from common.utils import proc_targets, load_metadata
from prognosis.utils import *


@hydra.main(config_path="../prognosis/configs/config_train.yaml")
def main(cfg):
    wdir = os.environ['PWD']
    if not os.path.isabs(cfg.meta_root):
        cfg.meta_root = os.path.join(wdir, cfg.meta_root)

    print(cfg.pretty())

    train = True
    eval = False

    cfg.meta_root = '/home/hoang/workspace/OAPOP/common/Metadata/in1_outseq_gradingmulti_inter_all_new'
    cfg.oai_meta_filename = 'OAI_progression_site.csv'
    # Load and split data
    if train:
        for i, seed in enumerate([12345, 19196, 37035, 49762, 52804]):
            # seed = random.randint(0, 100000)
            print(f'>>> Random seed {i}: {seed}')
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            for site in ["C"]:
                for grading in ['JSL', 'JSM', 'OSFL', "OSFM", 'OSTL', 'OSTM']:
                    cfg.pkl_meta_filename = f"cv_split_5folds_{grading}_oai_evalsite_{site}_{seed}.pkl"
                    cfg.site = site
                    load_metadata(cfg=cfg, proc_targets=proc_targets, modes='train')

    if eval:
        cfg.most_meta_filename = 'MOST_progression_all.csv'
        for site in ["A", "B", "C", "D", "E"]:
            cfg.site = site
            load_metadata(cfg=cfg, proc_targets=proc_targets, modes='eval')


if __name__ == "__main__":
    main()
