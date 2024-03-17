import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Callable, Tuple, List
from .hmsio import HMSDataProvider


class HMSSplitDataset(Dataset):
    """Class used to split dataset into train/validation"""
    def __init__(self, base_dataset, start=0, stop=-1):
        super().__init__()
        self.base_dataset = base_dataset
        self.start = start
        self.stop = stop if stop > 0 else len(base_dataset)

    def __len__(self):
        return self.stop - self.start

    def __getitem__(self, idx):
        return self.base_dataset[self.start + idx]


class HMSDataset(Dataset):
    """Base class for handling HMS competition"""
    chunk_size = 10000

    def __init__(self, data_provider: HMSDataProvider, transform: Callable = None, shuffle=True, seed=42):
        """
            data_provider: the source of data items
            transform: additional augmentations
            shuffle: whether to shuffle the order of items
            seed: the seed number used for RNG
        """

        super().__init__()
        self.transform = transform
        self.load(data_provider)
        self.length = len(data_provider)

        self.shuffle = np.arange(len(data_provider))
        if shuffle:
            rng = np.random.default_rng(seed=seed)
            rng.shuffle(self.shuffle)
        self.shuffle = self.shuffle.tolist()

    def load(self, data_provider: HMSDataProvider):
        sg, eeg, labels = [], [], []
        for n in range(len(data_provider)):
            dt = data_provider[n]
            sg.append(dt.sg)
            eeg.append(dt.eeg)
            labels.append(dt.label)

        self.sg = torch.from_numpy(np.array(sg))
        self.eeg = torch.from_numpy(np.array(eeg)[:, None, ...])
        self.labels = np.array(labels)
        self.labels = self.labels / self.labels.sum(axis=1).reshape((-1, 1))

    def train_test_split(self, train_size=0.8, test_size=0.2):
        thresh = int(len(self) * train_size / (train_size + test_size))
        train_ds = HMSSplitDataset(self, 0, thresh)
        test_ds = HMSSplitDataset(self, thresh)
        return train_ds, test_ds

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Tuple[Tuple, np.ndarray]:
        index = self.shuffle[index]
        sg = self.sg[index]
        eeg = self.eeg[index]

        if self.transform:
            sg, eeg = self.transform(sg, eeg)

        return (sg, eeg), self.labels[index]


class HMSSeparateDataset(HMSDataset):
    """Dataset class for handling HMS competition with separate load of SG and EEG data"""

    def __init__(self, transform: Callable = None, shuffle=True, seed=42):
        """
            transform: additional augmentations
            shuffle: whether to shuffle the order of items
            seed: the seed number used for RNG
        """
        self.transform = transform
        self._shuffle = shuffle
        self._seed = seed

    def __init_shuffle(self):
        self.shuffle = np.arange(self.length)
        if self._shuffle:
            rng = np.random.default_rng(seed=self._seed)
            rng.shuffle(self.shuffle)
        self.shuffle = self.shuffle.tolist()

    def _data_load(self, data_provider: HMSDataProvider, sg_eeg: str = 'sg'):
        self.length = len(data_provider)
        data = np.zeros((self.length, *getattr(data_provider[0], sg_eeg).shape), dtype=np.float32)
        labels = np.zeros((self.length, 6), dtype=np.float32)

        for n in range(len(data_provider)):
            if n % 1000 == 0:
                print(n)
            dt = data_provider[n]
            data[n], labels[n] = getattr(dt, sg_eeg), dt.label

        labels = np.array(labels)
        labels = labels / labels.sum(axis=1).reshape((-1, 1))

        self.__init_shuffle()

        return data, labels

    def sg_load(self, data_provider: HMSDataProvider):
        self.sg, self.labels = self._data_load(data_provider, 'sg')

    def eeg_load(self, data_provider: HMSDataProvider):
        self.eeg, self.labels = self._data_load(data_provider, 'eeg')


class HMSIndexedDataset(Dataset):
    """Class used to select a part of the base dataset specified by indices array"""

    def __init__(self, base_dataset: Dataset, indices: List):
        super().__init__()
        self.base_dataset = base_dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.base_dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)
