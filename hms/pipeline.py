import numpy as np
from typing import List, Union, Callable
from dataclasses import dataclass, field
from scipy import signal
from .mneproc import MNEPreprocessor


@dataclass
class HMSItem:
    """Class for storing individual data item"""
    sg: np.ndarray
    eeg: np.ndarray
    label: np.ndarray

    sg_fs: float = 0.5
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
    def __call__(self, data: Union[HMSItem, List[HMSItem], HMSDataProvider]) \
            -> Union[HMSItem, List[HMSItem], HMSDataProvider]:
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
    epsilon: float = 1e-6

    def process(self, item: HMSItem) -> HMSItem:
        item.sg = (item.sg - np.nanmean(item.sg)) / (np.nanstd(item.sg) + self.epsilon)
        item.eeg = (item.eeg - np.nanmean(item.eeg)) / (np.nanstd(item.eeg) + self.epsilon)
        return item


@dataclass
class HMSLogSG(HMSProcessor):
    def process(self, item: HMSItem) -> HMSItem:
        item.sg = np.log(np.clip(item.sg, np.exp(-6), np.exp(10)))
        return item


@dataclass
class HMSStack(HMSProcessor):
    def process(self, item: HMSItem) -> HMSItem:
        item.sg = np.moveaxis(item.sg, 0, 1)
        item.sg = np.reshape(item.sg, (item.sg.shape[0], item.sg.shape[1] * item.sg.shape[2]))
        return item


@dataclass
class HMSCreateSGFromEEG(HMSProcessor):
    def __init__(self):
        self.proc = MNEPreprocessor()

    def process(self, item: HMSItem) -> HMSItem:
        raw = self.proc.load_numpy(item.eeg)
        self.proc.process(raw)
        item.eeg = self.proc.spectrogram_chains(raw)
        item.eeg = np.moveaxis(item.eeg, (0, 1, 2), (2, 1, 0))
        return item


@dataclass
class HMSCropPad(HMSProcessor):
    w: int = 256
    h: int = 128

    def process(self, item: HMSItem) -> HMSItem:
        item.sg = self.__crop_pad(item.sg)
        item.eeg = self.__crop_pad(item.eeg)
        return item

    def __crop_pad(self, data: np.ndarray) -> np.ndarray:
        w_offset = (data.shape[1] - self.w) // 2
        if w_offset > 0:
            data = data[:, w_offset: w_offset + self.w, :]
        else:
            data = np.pad(data, ((0, 0), (-w_offset, -w_offset), (0, 0)), constant_values=1)

        h_offset = (data.shape[2] - self.h) // 2
        if h_offset > 0:
            data = data[:, :, h_offset: h_offset + self.h]
        else:
            data = np.pad(data, ((0, 0), (0, 0), (-h_offset, -h_offset)), constant_values=1)

        return data


@dataclass
class HMSImage(HMSProcessor):
    def process(self, item: HMSItem) -> HMSItem:
        item.sg = np.concatenate([ch_img[None, ...] for ch_img in item.sg], axis=2)
        item.eeg = np.concatenate([ch_img[None, ...] for ch_img in item.eeg], axis=2)
        return item
