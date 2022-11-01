import numpy as np
import pandas as pd
import torch
try:  # Handling API difference between pytorch 1.1 and 1.2
    from torch.utils.data.dataloader import default_collate
except ImportError:
    from torch.utils.data._utils.collate import default_collate

from common.dataframe_dataset import DataFrameDataset
from torch.utils.data.sampler import Sampler

class ItemLoader(object):
    """Combines DataFrameDataset and DataLoader, and provides single- or multi-process iterators over the dataset.

    Parameters
    ----------
    meta_data : pandas.DataFrame or None
        Meta data of data and labels.
    parse_item_cb : callable or None
        Parses each row of :attr:meta_data`.
    batch_size : int, optional
        How many data per batch to load. (the default is 1)
    root : str
        Path to root directory of input data. Default is None (empty)
    num_workers : int, optional
        How many subprocesses to use for data loading. If equals to 0,
        the data will be loaded in the main process. (the default is 0)
    shuffle : bool, optional
        Set to ``True`` to have the data reshuffled at every epoch. (the default is False)
    pin_memory : bool, optional
        If ``True``, the data loader will copy tensors into CUDA pinned memory
        before returning them. (the default is False)
    collate_fn : callable, optional
        Merges a list of samples to form a mini-batch. (the default is None)
    transform : callable, optional
        Transforms row of :attr:`meta_data`. (the default is None)
    sampler : Sampler, optional
        Defines the strategy to draw samples from the dataset. If specified,
        ``shuffle`` must be False. (the default is None)
    batch_sampler : callable, optional
        Like sampler, but returns a batch of indices at a time.
        Mutually exclusive with :attr:`batch_size`, :attr:`shuffle`,
        :attr:`sampler`, and :attr:`drop_last`. (the default is None)
    drop_last : bool, optional
        Set to ``True`` to drop the last incomplete batch,
        if the dataset size is not divisible by the batch size. If ``False`` and
        the size of dataset is not divisible by the batch size, then the last batch
        will be smaller. (the default is False)
    timeout : int, optional
        If positive, the timeout value for collecting a batch from workers.
        If ``0``, ignores ``timeout`` notion. Must be non-negative. (the default is 0)
    """

    def __init__(self, meta_data: pd.DataFrame or None = None,
                 parse_item_cb: callable or None = None, parser_kwargs: dict or None = None,
                 root: str or None = None, batch_size: int = 1,
                 num_workers: int = 0, shuffle: bool = False, pin_memory: bool = False,
                 collate_fn: callable = default_collate, transform: callable or None = None,
                 sampler: Sampler or None = None,
                 batch_sampler=None, drop_last: bool = False, timeout: int = 0, name: str = "",
                 worker_init_fn=None):
        if root is None:
            root = ''

        self.__name = name

        self.__root = root
        self.__shuffle = shuffle
        self.__sampler = sampler
        self.__batch_sampler = batch_sampler
        self.__num_workers = num_workers
        self.__collate_fn = collate_fn
        self.__pin_memory = pin_memory
        self.__timeout = timeout
        self.__worker_init_fn = worker_init_fn

        self.__transform = transform
        self.drop_last: bool = drop_last
        self.batch_size: int = batch_size
        self.__iter_loader = None
        self.parse_item = parse_item_cb
        self.parser_kwargs = parser_kwargs

        self.update_dataset(meta_data)

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, input):
        self.__name = input

    @staticmethod
    def _worker_init(wid):
        np.random.seed(np.uint32(torch.initial_seed() + wid))

    @property
    def dataset(self):
        return self.__dataset

    @property
    def meta_data(self):
        return self.__meta_data

    def update_dataset(self, meta_data):
        self.__meta_data = meta_data
        if self.__meta_data is None:
            self.__dataset = None
        else:
            self.__dataset = DataFrameDataset(self.__root, meta_data=self.__meta_data, parser_kwargs=self.parser_kwargs,
                                              parse_item_cb=self.parse_item, transform=self.__transform)

        if self.__dataset is None:
            self.__data_loader = None
        else:
            self.__data_loader = torch.utils.data.DataLoader(dataset=self.__dataset,
                                                             batch_size=self.batch_size,
                                                             shuffle=self.__shuffle,
                                                             sampler=self.__sampler,
                                                             batch_sampler=self.__batch_sampler,
                                                             num_workers=self.__num_workers,
                                                             collate_fn=self.__collate_fn,
                                                             pin_memory=self.__pin_memory,
                                                             drop_last=self.drop_last,
                                                             timeout=self.__timeout,
                                                             worker_init_fn=self.__worker_init_fn)

    @property
    def transform(self):
        return self.__transform

    def __len__(self):
        """ Get length of the dataloader.
        """
        return len(self.__data_loader)

    def sample(self, k=1):
        """Samples one or more mini-batches.

        Parameters
        ----------
        k : int, optional
            The number of batches to sample. (the default is 1)

        Returns
        -------
        samples : list
            List of sampled batches.
        """
        samples = []
        for i in range(k):
            try:
                if self.__iter_loader is None:
                    self.__iter_loader = iter(self.__data_loader)
                batch = next(self.__iter_loader)
            except StopIteration:
                del self.__iter_loader
                self.__iter_loader = iter(self.__data_loader)
                batch = next(self.__iter_loader)

            batch['name'] = self.__name
            samples.append(batch)

        return samples

    def set_epoch(self, epoch):
        pass