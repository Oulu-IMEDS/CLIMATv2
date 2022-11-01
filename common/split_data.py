import dill as pickle
import pandas as pd
from sklearn import model_selection
from sklearn.utils import resample
import numpy as np
import logging, coloredlogs

from collagen.data import Splitter

coloredlogs.install()


class HiarFoldSplit(Splitter):
    def __init__(self, ds: pd.DataFrame, train_size_per_class: int, n_folds: int = 5, n_subfolds: int = 3,
                 random_state: int or None = None, test_ratio: float = 0.25, target_col: str = 'target',
                 group_col: str or None = None, shuffle: bool = True, separate_targets=False, chunk_ratios=None,
                 train_size=None):

        super().__init__()

        self._test_ratio = test_ratio

        # Master split into n_folds
        if group_col is None:
            master_splitter = model_selection.StratifiedKFold(n_splits=n_folds, random_state=random_state)
            splitter_res = master_splitter.split(ds, ds[target_col])

        else:
            master_splitter = model_selection.GroupKFold(n_splits=n_folds)
            splitter_res = master_splitter.split(ds, ds[target_col], groups=ds[group_col])

        targets = ds[target_col].unique()

        self._train_sizes = None
        self._chunk_ratios = None
        if chunk_ratios is not None:
            if len(chunk_ratios) != n_subfolds:
                logging.fatal(f'Num of chunk ratios ({chunk_ratios}) and subfolds ({n_subfolds}) must match!')
                assert False
            if train_size is None:
                logging.fatal(f'Must input `total_train_size` when using `chunk_ratios`.')
                assert False
            chunk_ratios = np.array(chunk_ratios)
            self._chunk_ratios = chunk_ratios / chunk_ratios.sum()
            self._train_sizes = [int(train_size * r) for r in self._chunk_ratios]

        self.__ds_chunks = []
        self.__cv_folds_idx = []
        for fold_id, (t_idx, v_idx) in enumerate(splitter_res):
            train_ds = ds.iloc[t_idx]
            val_ds = ds.iloc[v_idx]
            if group_col is None:
                sub_splitter = model_selection.StratifiedKFold(n_splits=n_subfolds, random_state=random_state)
                train_ss_res = sub_splitter.split(train_ds, train_ds[target_col])
                val_ss_res = sub_splitter.split(val_ds, val_ds[target_col])
            else:
                sub_splitter = model_selection.GroupKFold(n_splits=n_subfolds)
                train_ss_res = sub_splitter.split(train_ds, train_ds[target_col], groups=train_ds[group_col])
                val_ss_res = sub_splitter.split(val_ds, val_ds[target_col], groups=val_ds[group_col])

            train_sub_ds = [train_ds.iloc[t_sub_idx] for _, t_sub_idx in train_ss_res]
            val_sub_ds = [val_ds.iloc[v_sub_idx] for _, v_sub_idx in val_ss_res]

            if train_size_per_class is not None or self._train_sizes is not None:
                if train_size_per_class < 1:
                    logging.fatal(
                        f'Num of training samples per class must be larger than 0, but found {train_size_per_class}')
                    assert False
                selected_tr_ds_per_target = []
                selected_tr_idx_per_target = []
                selected_va_ds_per_target = pd.DataFrame(columns=val_ds.columns)
                selected_va_idx_per_target = []
                for sf_id, t_sub_ds in enumerate(train_sub_ds):
                    v_sub_ds = val_sub_ds[sf_id]
                    if self._train_sizes is not None:
                        train_size_per_class = int(1.0 * self._train_sizes[sf_id] / len(targets))

                    selected_tr_sub_ds_per_target = []
                    selected_tr_sub_idx_per_target = []
                    selected_va_sub_ds_per_target = []
                    selected_va_sub_idx_per_target = []
                    for target_id, target in enumerate(targets):
                        t_sub_ds_per_target = t_sub_ds[t_sub_ds[target_col] == target]
                        v_sub_ds_per_target = v_sub_ds[v_sub_ds[target_col] == target]

                        _ds, _idx, is_replace = self._sample(t_sub_ds_per_target, train_size_per_class, random_state)
                        selected_tr_sub_ds_per_target.append(_ds)
                        if separate_targets:
                            selected_tr_sub_idx_per_target.append(_idx)
                        else:
                            selected_tr_sub_idx_per_target.extend(_idx)

                        if is_replace:
                            logging.info(
                                f'Training fold {fold_id}, sub-fold {sf_id}, target {target} using resampling with replacement.')

                        _ds, _idx, is_replace = self._sample(v_sub_ds_per_target,
                                                             int(train_size_per_class * test_ratio), random_state)
                        selected_va_sub_ds_per_target.append(_ds)
                        if separate_targets:
                            selected_va_sub_idx_per_target.append(_idx)
                        else:
                            selected_va_sub_idx_per_target.extend(_idx)
                        if is_replace:
                            logging.info(
                                f'Validation fold {fold_id}, sub-fold {sf_id}, target {target} using resampling with replacement.')

                    if not separate_targets:
                        selected_tr_sub_ds_per_target = pd.concat(selected_tr_sub_ds_per_target)
                        selected_va_sub_ds_per_target = pd.concat(selected_va_sub_ds_per_target)

                    selected_tr_ds_per_target.append(selected_tr_sub_ds_per_target)
                    selected_va_ds_per_target = pd.concat((selected_va_ds_per_target, selected_va_sub_ds_per_target))
                    selected_tr_idx_per_target.append(selected_tr_sub_idx_per_target)
                    selected_va_idx_per_target.extend(selected_va_sub_idx_per_target)

                # Replicate validation set to n_subfolds times
                selected_va_ds_per_target = [selected_va_ds_per_target] * n_subfolds
                selected_va_idx_per_target = [selected_va_idx_per_target] * n_subfolds
            else:
                logging.error(f'Only support same amounts among classes.')
                assert False

            self.__cv_folds_idx.append((selected_tr_idx_per_target, selected_va_idx_per_target))
            self.__ds_chunks.append((selected_tr_ds_per_target, selected_va_ds_per_target))

        self.__folds_iter = iter(self.__ds_chunks)

    def _sample(self, sub_ds_per_target, size_per_class, random_state):
        is_replace = len(sub_ds_per_target.index) < size_per_class
        if is_replace:
            logging.warning(f'Need {size_per_class}, but got {len(sub_ds_per_target.index)}')
        t_sub_idx_per_target = sub_ds_per_target.index
        _idx = resample(t_sub_idx_per_target, n_samples=size_per_class, replace=is_replace, random_state=random_state)
        return sub_ds_per_target.loc[_idx], _idx, is_replace

    def dump(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.__ds_chunks, f, pickle.HIGHEST_PROTOCOL)

    def __next__(self):
        return next(self.__folds_iter)

    def __iter__(self):
        return self

    def stats(self):
        for i, (tr, val) in enumerate(self.__cv_folds_idx):
            logging.info(f'Fold {i}')
            for j, sub_tr in enumerate(tr):
                logging.info(f'[Train] Subfold {j} has {len(sub_tr)}')
            for j, sub_va in enumerate(val):
                logging.info(f'[Val] Subfold {j} has {len(sub_va)}')

    def fold(self, i):
        return self.__ds_chunks[i]

    def fold_idx(self, i):
        return self.__cv_folds_idx[i]

    def n_folds(self):
        return len(self.__ds_chunks)
