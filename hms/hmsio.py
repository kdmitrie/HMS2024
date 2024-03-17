import numpy as np
import pandas as pd

from typing import Tuple, List, Union
from dataclasses import dataclass
from pathlib import Path
import pickle

from .pipeline import HMSItem, HMSProcessor, HMSDataProvider

SG_FS: int = 1
EEG_FS: int = 200


class HMSReader(HMSDataProvider):
    """Class for reading the train and test data"""
    sg_fs: int = SG_FS
    eeg_fs: int = EEG_FS
    eeg_len: int = 10000
    sg_len: int = 300

    def __init__(self, csv_file: str, sg_path: str, eeg_path: str, limit: int = None):
        """
        csv_file: the dataframe
        sg_path: the folder containing spectrogram files in parquet format
        eeg_path: the folder containing eeg files in parquet format
        limit: manually limit the number of records
        """
        self.df = pd.read_csv(csv_file)
        self.eeg_path = eeg_path
        self.sg_path = sg_path
        self.limit = limit
        self.process = []

        if 'eeg_label_offset_seconds' in self.df.columns:
            self._get_item_data = self._get_item_data_train
        else:
            self._get_item_data = lambda item: (0, 0, np.array([np.nan]*6))

    def __len__(self) -> int:
        return len(self.df) if self.limit is None else self.limit

    def _get_item_data_train(self, item) -> Tuple[int, int, np.ndarray]:
        eeg_start = int(self.eeg_fs * item.eeg_label_offset_seconds)
        sg_start = int(item.spectrogram_label_offset_seconds / 2)
        return sg_start, eeg_start, item.iloc[-6:].to_numpy().astype(int)

    def __getitem__(self, selection: Union[int, slice]) -> Union[HMSItem, List[HMSItem]]:
        if isinstance(selection, int):
            return self._get_one_item(self.df.iloc[selection, :])
        return [self._get_one_item(item) for _, item in self.df.iloc[selection, :].iterrows()]

    def _get_one_item(self, item: object) -> HMSItem:
        """Reads the data from parquet files"""
        sg_start, eeg_start, label = self._get_item_data(item)

        sg = pd.read_parquet(self.sg_path % item.spectrogram_id).iloc[sg_start:sg_start + self.sg_len, 1:]
        sg = sg.to_numpy().reshape((-1, 4, 100))
        sg = np.moveaxis(sg, 1, 0)
        eeg = pd.read_parquet(self.eeg_path % item.eeg_id).iloc[eeg_start:eeg_start + self.eeg_len].to_numpy()

        item = HMSItem(sg=sg, eeg=eeg, label=label, sg_fs=self.sg_fs, eeg_fs=self.eeg_fs)
        return self._process_one_item(item)


@dataclass
class HMSSave:
    """Class for saving the data"""
    source: HMSReader = None
    processor: HMSProcessor = None
    sg_path: str = './output/sg'
    eeg_path: str = './output/eeg'
    chunk_size: int = 1000
    verbose: bool = True

    def print(self, s: str) -> None:
        if self.verbose:
            print(s)

    def save(self) -> None:
        Path(self.sg_path).mkdir(parents=True, exist_ok=True)
        Path(self.eeg_path).mkdir(parents=True, exist_ok=True)

        if self.source is None:
            raise ValueError('Source must not be None')

        labels = []
        for n in range(0, len(self.source), self.chunk_size):
            m = min(n + self.chunk_size, len(self.source))
            self.print(f'Items {n} - {m}')

            self.print(f'\tloading')
            items = self.source[n:m]

            if self.processor is not None:
                self.print(f'\tprocessing')
                items = self.processor(items)

            self.print(f'\tsaving sg')
            with open(f'{self.sg_path}/sg_{n}.pkl', 'wb') as f:
                data = [item.sg for item in items]
                pickle.dump(data, f)

            self.print(f'\tsaving eeg')
            with open(f'{self.eeg_path}/eeg_{n}.pkl', 'wb') as f:
                data = [item.eeg for item in items]
                pickle.dump(data, f)

            labels += [item.label for item in items]

        data = {'eeg_fs': items[0].eeg_fs, 'sg_fs': items[0].sg_fs, 'labels': labels}
        with open(f'{self.eeg_path}/labels.pkl', 'wb') as f:
            pickle.dump(data, f)
        with open(f'{self.sg_path}/labels.pkl', 'wb') as f:
            pickle.dump(data, f)


@dataclass
class HMSLoad(HMSDataProvider):
    """Class for loading the data"""
    sg_path: str = './output/sg'
    eeg_path: str = './output/eeg'
    verbose: bool = True

    def __post_init__(self):
        self.eeg_fs = EEG_FS
        self.sg_fs = SG_FS
        self.sg, self.eeg, self.labels = [], [], []

    def load(self) -> None:
        with open(f'{self.eeg_path}/labels.pkl', 'rb') as f:
            data = pickle.load(f)
        self.eeg_fs = data['eeg_fs']
        self.sg_fs = data['sg_fs']
        self.labels = data['labels']

        self.sg = []
        self.eeg = []
        n = 0
        while (n < len(self.labels)) and ((self.limit is None) or (n < self.limit)):
            self.print(f'Loading {n}')
            with open(f'{self.sg_path}/sg_{n}.pkl', 'rb') as f:
                self.sg += pickle.load(f)

            with open(f'{self.eeg_path}/eeg_{n}.pkl', 'rb') as f:
                data = pickle.load(f)
                self.eeg += data
                n += len(data)

    def __len__(self) -> int:
        return len(self.labels) if self.limit is None else self.limit

    def __getitem__(self, selection: Union[int, slice]) -> Union[HMSItem, List[HMSItem]]:
        if isinstance(selection, int) or isinstance(selection, np.int32):
            return self._get_one_item(selection)
        if isinstance(selection, list):
            return [self._get_one_item(item_id) for item_id in selection]
        return [self._get_one_item(item_id) for item_id in range(len(self.labels))[selection]]

    def _get_one_item(self, item_id: int) -> HMSItem:
        """Reads the data from memory"""
        label = self.labels[item_id]
        if label is not np.nan:
            label = label.astype(int)
        item = HMSItem(sg=self.sg[item_id], eeg=self.eeg[item_id], label=label, sg_fs=self.sg_fs,
                       eeg_fs=self.eeg_fs)
        return self._process_one_item(item)


@dataclass
class HMSSeparateLoad(HMSLoad):
    """Class for loading the data"""
    df_path: str = './train.csv'

    def load(self) -> None:
        self.target = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
        self.load_labels()
        if self.sg_path:
            self.load_sg()
        if self.eeg_path:
            self.load_eeg()

    def load_labels(self) -> None:
        train_df = pd.read_csv(self.df_path)
        aux1 = train_df.groupby(['eeg_id', 'spectrogram_id'])[self.target].agg('sum')
        aux2 = train_df.groupby(['eeg_id', 'spectrogram_id'])['patient_id'].agg('first')
        self.labels = aux1.join(aux2).reset_index()
        if self.limit:
            self.labels = self.labels[:self.limit]
        self.groups = self.labels.patient_id

    def load_sg(self, get_item: bool = True) -> None:
        self.sg = np.load(self.sg_path, allow_pickle=True)
        if get_item:
            self.sg = self.sg.item()

    def load_eeg(self, get_item: bool = False) -> None:
        self.eeg = np.load(self.eeg_path, allow_pickle=True)
        if get_item:
            self.sg = self.sg.item()

    def _get_one_item(self, item_id: int) -> HMSItem:
        """Reads the data from memory"""
        item = self.labels.iloc[item_id]
        label = item[self.target].to_numpy()

        eeg = self.eeg[item_id] if self.eeg_path else np.zeros((1, 1, 1)).astype(np.single)
        sg = self.sg[item['spectrogram_id']] if self.sg_path else np.zeros((1, 1, 1)).astype(np.single)

        if label is not np.nan:
            label = label.astype(int)

        item = HMSItem(sg=sg, eeg=eeg, label=label, sg_fs=self.sg_fs, eeg_fs=self.eeg_fs)

        return self._process_one_item(item)
