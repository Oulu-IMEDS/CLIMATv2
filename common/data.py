import pickle
import numpy as np
import torch
import pandas as pd
from sklearn import model_selection
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

try:  # Handling API difference between pytorch 1.1 and 1.2
    from torch.utils.data.dataloader import default_collate
except ImportError:
    from torch.utils.data._utils.collate import default_collate


class Splitter(object):
    def __init__(self):
        self.__ds_chunks = None
        self.__folds_iter = None
        pass

    def __next__(self):
        if self.__folds_iter is None:
            raise NotImplementedError
        else:
            next(self.__folds_iter)

    def __iter__(self):
        if self.__ds_chunks is None:
            raise NotImplementedError
        else:
            return self

    def dump(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.__ds_chunks, f, pickle.HIGHEST_PROTOCOL)

    def load(self, filename):
        with open(filename, "rb") as f:
            self.__ds_chunks = pickle.load(f)
            self.__folds_iter = iter(self.__ds_chunks)


class FoldSplit(Splitter):
    def __init__(self, ds: pd.DataFrame, n_folds: int = 5, target_col: str = 'target',
                 group_col: str or None = None, random_state: int or None = None):
        super().__init__()
        if group_col is None:
            splitter = model_selection.StratifiedKFold(n_splits=n_folds, random_state=random_state)
            split_iter = splitter.split(ds, ds[target_col])
        else:
            splitter = model_selection.GroupKFold(n_splits=n_folds)
            split_iter = splitter.split(ds, ds[target_col], groups=ds[group_col])

        self.__cv_folds_idx = [(train_idx, val_idx) for (train_idx, val_idx) in split_iter]
        self.__ds_chunks = [(ds.iloc[split[0]], ds.iloc[split[1]]) for split in self.__cv_folds_idx]
        self.__folds_iter = iter(self.__ds_chunks)

    def __next__(self):
        return next(self.__folds_iter)

    def __iter__(self):
        return self

    def dump(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.__ds_chunks, f, pickle.HIGHEST_PROTOCOL)

    def fold(self, i):
        return self.__ds_chunks[i]

    def n_folds(self):
        return len(self.__cv_folds_idx)

    def fold_idx(self, i):
        return self.__cv_folds_idx[i]


class DataFrameDataset(Dataset):
    """Dataset based on ``pandas.DataFrame``.

    Parameters
    ----------
    root : str
        Path to root directory of input data.
    meta_data : pandas.DataFrame
        Meta data of data and labels.
    parse_item_cb : callable
        Callback function to parse each row of :attr:`meta_data`.
    transform : callable, optional
        Transformation applied to row of :attr:`meta_data` (the default is None).
    parser_kwargs: dict
        Dict of args for :attr:`parse_item_cb` (the default is None, )

    Raises
    ------
    TypeError
        `root` must be `str`.
    TypeError
        `meta_data` must be `pandas.DataFrame`.

    """

    def __init__(self, root: str, meta_data: pd.DataFrame, parse_item_cb: callable, transform: callable or None = None,
                 parser_kwargs: dict or None = {'data_key': 'data', 'target_key': 'target'}):
        if not isinstance(root, str):
            raise TypeError("`root` must be `str`")
        if not isinstance(meta_data, pd.DataFrame):
            raise TypeError("`meta_data` must be `pandas.DataFrame`, but found {}".format(type(meta_data)))
        self.root = root
        self.meta_data = meta_data
        self.parse_item_cb = parse_item_cb
        self.transform = transform
        self.parser_kwargs = parser_kwargs if parser_kwargs is not None else self._default_parser_args()

    def _default_parser_args(self):
        return {'data_key': 'data', 'target_key': 'target'}

    @property
    def data_key(self):
        return self.__data_key

    @property
    def target_key(self):
        return self.__target_key

    def __getitem__(self, index):
        """Get ``index``-th parsed item of :attr:`meta_data`.

        Parameters
        ----------
        index : int
            Index of row.

        Returns
        -------
        entry : dict
            Dictionary of `index`-th parsed item.
        """
        entry = self.meta_data.iloc[index]
        entry = self.parse_item_cb(self.root, entry, self.transform, **self.parser_kwargs)
        if not isinstance(entry, dict):
            raise TypeError("Output of `parse_item_cb` must be `dict`, but found {}".format(type(entry)))
        return entry

    def __len__(self):
        """Get length of `meta_data`.
        """
        return len(self.meta_data.index)
