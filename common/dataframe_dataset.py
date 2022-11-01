import pandas as pd
from torch.utils.data import Dataset


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
        # if not isinstance(root, str):
        #     raise TypeError("`root` must be `str`")
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
