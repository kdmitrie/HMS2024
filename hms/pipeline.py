import numpy as np
from typing import List, Union, Callable
from dataclasses import dataclass, field
from scipy import signal


@dataclass
class HMSItem:
    """Class for storing individual data item"""
    sg: np.ndarray
    eeg: np.ndarray
    label: np.ndarray

    sg_fs: int = 1
    eeg_fs: int = 1

    def __repr__(self) -> str:
        desc = [f'SG: array({self.sg.shape})',
                f'EEG: array({self.eeg.shape})',
                self.sg.__repr__(),
                self.eeg.__repr__(),
                ]
        return '\n'.join(desc)


@dataclass
class HMSDataProvider:
    limit: int = None
    verbose: bool = True
    process: List[Callable] = field(default_factory=list)


    def __len__(self) -> int:
        return 0

    def __getitem__(self, item) -> HMSItem:
        return HMSItem(sg=np.array([]), eeg=np.array([]), label=np.array([]))

    def print(self, s: str) -> None:
        if self.verbose:
            print(s)

    def add_processing(self, process: Callable) -> None:
        self.process.append(process)

    def _process_one_item(self, item: HMSItem) -> HMSItem:
        for proc in self.process:
            item = proc(item)
        return item

@dataclass
class HMSProcessor:
    def __call__(self, data: Union[HMSItem, List[HMSItem], HMSDataProvider]) -> Union[HMSItem, List[HMSItem], HMSDataProvider]:
        if isinstance(data, HMSItem):
            return self.process(data)

        if isinstance(data, HMSDataProvider):
            data.add_processing(self.process)
            return data

        return [self.process(item) for item in data]

    def process(self, item: HMSItem) -> HMSItem:
        return item


@dataclass
class HMSPipeline(HMSProcessor):
    def __init__(self, *args):
        self.pipeline = args

    def process(self, item: HMSItem) -> HMSItem:
        for proc in self.pipeline:
            item = proc(item)
        return item


@dataclass
class HMSFillna(HMSProcessor):
    def process(self, item: HMSItem) -> HMSItem:
        np.nan_to_num(item.eeg, copy=False)
        np.nan_to_num(item.sg, copy=False)
        return item


@dataclass
class HMSDecimate(HMSProcessor):
    q: int = 1
    def process(self, item: HMSItem) -> HMSItem:
        item.eeg = signal.decimate(item.eeg, self.q, axis=0)
        item.eeg_fs = item.eeg_fs // self.q
        return item


@dataclass
class HMSSelect(HMSProcessor):
    sg_t: float = 10.0
    eeg_t: float = 10.0

    def process(self, item: HMSItem) -> HMSItem:
        sg_center = item.sg.shape[1] // 2
        t_sg2 = int(self.sg_t * item.sg_fs / 2)
        item.sg = item.sg[:, sg_center - t_sg2: sg_center + t_sg2, :]

        eeg_center = item.eeg.shape[0] // 2
        t_eeg2 = int(self.eeg_t * item.eeg_fs / 2)
        item.eeg = item.eeg[eeg_center - t_eeg2: eeg_center + t_eeg2, :]

        return item


@dataclass
class HMSStandardScaler(HMSProcessor):
    def process(self, item: HMSItem) -> HMSItem:
        item.sg = (item.sg - item.sg.mean()) / item.sg.std()
        item.eeg = (item.eeg - item.eeg.mean()) / item.eeg.std()
        return item


@dataclass
class HMSLogSG(HMSProcessor):
    def process(self, item: HMSItem) -> HMSItem:
        item.sg = np.log(np.clip(item.sg, np.exp(-6), np.exp(10)))
        return item