import os
import gc
import sys
import math
import time
import random
import datetime as dt
import numpy as np
import pandas as pd

from glob import glob
from pathlib import Path
from typing import Dict, List, Union
from scipy.signal import butter, lfilter, freqz
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset

sys.path.append("/kaggle/input/kaggle-kl-div")
from kaggle_kl_div import score

import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

"PRETTY_NAME=\"\K([^\"]*)"
print(f"BUILD_DATE={os.environ['BUILD_DATE']}, CONTAINER_NAME={os.environ['CONTAINER_NAME']}")

try:
    print(
        f"PyTorch Version:{torch.__version__}, CUDA is available:{torch.cuda.is_available()}, Version CUDA:{torch.version.cuda}"
    )
    print(
        f"Device Capability:{torch.cuda.get_device_capability()}, {torch.cuda.get_arch_list()}"
    )
    print(
        f"CuDNN Enabled:{torch.backends.cudnn.enabled}, Version:{torch.backends.cudnn.version()}"
    )
except Exception:
    pass


# %% [code] {"execution":{"iopub.status.busy":"2024-04-02T20:40:03.613460Z","iopub.execute_input":"2024-04-02T20:40:03.613919Z","iopub.status.idle":"2024-04-02T20:40:03.629775Z","shell.execute_reply.started":"2024-04-02T20:40:03.613878Z","shell.execute_reply":"2024-04-02T20:40:03.628537Z"}}
class APP:
    jupyter = "ipykernel" in globals()
    if not jupyter:
        try:
            if "IPython" in globals().get("__doc__", ""):
                jupyter = True
        except Exception as inst:
            print(inst)

    kaggle = os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "") != ""
    local = os.environ.get("DOCKER_USING", "") == "LOCAL"
    date_time_start = dt.datetime.now()
    dt_start_ymd_hms = date_time_start.strftime("%Y.%m.%d_%H-%M-%S")

    file_run_path = ""
    if jupyter:
        try:
            file_run_path = Path(globals().get("__vsc_ipynb_file__", ""))
        except Exception as inst:
            print(inst)

    else:
        try:
            file_run_path = Path(__file__)
        except Exception as inst:
            print(inst)

    file_run_name = file_run_path.stem
    path_app = file_run_path.parent
    path_run = Path(os.getcwd())
    path_out = (
        Path("/kaggle/working")
        if kaggle
        else file_run_path / f"{file_run_name}_{dt_start_ymd_hms}"
    )


print(f"jupyter:{APP.jupyter}, kaggle:{APP.kaggle}, local:{APP.local}")
print(APP.file_run_path)
print(APP.path_out)


# %% [code] {"execution":{"iopub.status.busy":"2024-04-02T20:40:03.631573Z","iopub.execute_input":"2024-04-02T20:40:03.631970Z","iopub.status.idle":"2024-04-02T20:40:03.647301Z","shell.execute_reply.started":"2024-04-02T20:40:03.631938Z","shell.execute_reply":"2024-04-02T20:40:03.645971Z"}}
class CFG:
    VERSION = 93

    model_name = "resnet1d_gru"

    seed = 2024
    batch_size = 32
    num_workers = 0

    fixed_kernel_size = 5
    # kernels = [3, 5, 7, 9]
    # linear_layer_features = 424
    kernels = [3, 5, 7, 9, 11]
    # linear_layer_features = 448  # Full Signal = 10_000
    # linear_layer_features = 352  # Half Signal = 5_000
    linear_layer_features = 304  # 1/5  Signal = 2_000

    seq_length = 50  # Second's
    sampling_rate = 200  # Hz
    nsamples = seq_length * sampling_rate  # Число семплов
    out_samples = nsamples // 5

    # bandpass_filter = {"low": 0.5, "high": 20, "order": 2}
    # rand_filter = {"probab": 0.1, "low": 10, "high": 20, "band": 1.0, "order": 2}
    freq_channels = []  # [(8.0, 12.0)]; [(0.5, 4.5)]
    filter_order = 2
    random_close_zone = 0.0  # 0.2

    target_cols = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]

    # target_preds = [x + "_pred" for x in target_cols]
    # label_to_num = {"Seizure": 0, "LPD": 1, "GPD": 2, "LRDA": 3, "GRDA": 4, "Other": 5}
    # num_to_label = {v: k for k, v in label_to_num.items()}

    map_features = [
        ("Fp1", "T3"),
        ("T3", "O1"),
        ("Fp1", "C3"),
        ("C3", "O1"),
        ("Fp2", "C4"),
        ("C4", "O2"),
        ("Fp2", "T4"),
        ("T4", "O2"),
        # ('Fz', 'Cz'), ('Cz', 'Pz'),
        # 'F7', 'F3', 'Cz', 'P4', 'T6'
        # 'F8', 'F4', 'Cz', 'P3', 'T5'
        ('F7', 'F3'),
        ('F3', 'Cz'),
        ('Cz', 'P4'),
        ('P4', 'T6'),
        ('F8', 'F4'),
        ('F4', 'Cz'),
        ('Cz', 'P3'),
        ('P3', 'T5'),
    ]

    eeg_features = ["Fp1", "T3", "C3", "O1", "Fp2", "C4", "T4", "O2",  # 'Fz', 'Cz', 'Pz'
                    'F3', 'P3', 'F7', 'T5', 'Fz', 'Cz', 'Pz', 'F4', 'P4', 'F8', 'T6']  # , 'EKG'
    feature_to_index = {x: y for x, y in zip(eeg_features, range(len(eeg_features)))}
    simple_features = []  # 'Fz', 'Cz', 'Pz', 'EKG'

    # eeg_features = [row for row in feature_to_index]
    # eeg_feat_size = len(eeg_features)

    n_map_features = len(map_features)
    in_channels = n_map_features + n_map_features * len(freq_channels) + len(simple_features)
    target_size = len(target_cols)

    PATH = "/kaggle/input/hms-harmful-brain-activity-classification/"
    test_eeg = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
    test_csv = "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"


# %% [code] {"execution":{"iopub.status.busy":"2024-04-02T20:40:03.650631Z","iopub.execute_input":"2024-04-02T20:40:03.651614Z","iopub.status.idle":"2024-04-02T20:40:03.663377Z","shell.execute_reply.started":"2024-04-02T20:40:03.651573Z","shell.execute_reply":"2024-04-02T20:40:03.662168Z"}}
koef_1 = 1.0
model_weights = [
    {
        'bandpass_filter': {'low': 0.5, 'high': 20, 'order': 2},
        'file_data':
            [
                {'koef': koef_1,
                 'file_mask': "/kaggle/input/hms-resnet1d-gru-weights-v102-3/pop_1_weight_oof/*_best.pth"},
                # {'koef':koef_1, 'file_mask':"/kaggle/input/hms-resnet1d-gru-weights-v102-3/pop_2_weight_oof/*_best.pth"},
                # {'koef':koef_1, 'file_mask':"/kaggle/input/hms-resnet1d-gru-weights-v102-3/pop_3_weight_oof/*_best.pth"},
            ]
    },
]


# %% [code] {"execution":{"iopub.status.busy":"2024-04-02T20:40:03.665349Z","iopub.execute_input":"2024-04-02T20:40:03.666531Z","iopub.status.idle":"2024-04-02T20:40:03.690407Z","shell.execute_reply.started":"2024-04-02T20:40:03.666464Z","shell.execute_reply":"2024-04-02T20:40:03.689110Z"}}
def init_logger(log_file="./test.log"):
    from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler

    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "%s (remain %s)" % (asMinutes(s), asMinutes(rs))


def quantize_data(data, classes):
    mu_x = mu_law_encoding(data, classes)
    return mu_x  # quantized


def mu_law_encoding(data, mu):
    mu_x = np.sign(data) * np.log(1 + mu * np.abs(data)) / np.log(mu + 1)
    return mu_x


def mu_law_expansion(data, mu):
    s = np.sign(data) * (np.exp(np.abs(data) * np.log(mu + 1)) - 1) / mu
    return s


def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype="band")


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_lowpass_filter(
        data, cutoff_freq=20, sampling_rate=CFG.sampling_rate, order=4
):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    filtered_data = lfilter(b, a, data, axis=0)
    return filtered_data


def denoise_filter(x):
    # Частота дискретизации и желаемые частоты среза (в Гц).
    # Отфильтруйте шумный сигнал
    y = butter_bandpass_filter(x, CFG.lowcut, CFG.highcut, CFG.sampling_rate, order=6)
    y = (y + np.roll(y, -1) + np.roll(y, -2) + np.roll(y, -3)) / 4
    y = y[0:-1:4]
    return y


# %% [code] {"execution":{"iopub.status.busy":"2024-04-02T20:40:03.692724Z","iopub.execute_input":"2024-04-02T20:40:03.693754Z","iopub.status.idle":"2024-04-02T20:40:03.712053Z","shell.execute_reply.started":"2024-04-02T20:40:03.693706Z","shell.execute_reply":"2024-04-02T20:40:03.710845Z"}}
def eeg_from_parquet(
        parquet_path: str, display: bool = False, seq_length=CFG.seq_length
) -> np.ndarray:
    """
    Эта функция читает файл паркета и извлекает средние 50 секунд показаний. Затем он заполняет значения NaN
    со средним значением (игнорируя NaN).
        :param parquet_path: путь к файлу паркета.
        :param display: отображать графики ЭЭГ или нет.
        :return data: np.array формы (time_steps, eeg_features) -> (10_000, 8)
    """

    # Вырезаем среднюю 50 секундную часть
    eeg = pd.read_parquet(parquet_path, columns=CFG.eeg_features)
    rows = len(eeg)

    # начало смещения данных, чтобы забрать середину
    offset = (rows - CFG.nsamples) // 2

    # средние 50 секунд, имеет одинаковое количество показаний слева и справа
    eeg = eeg.iloc[offset: offset + CFG.nsamples]

    if display:
        plt.figure(figsize=(10, 5))
        offset = 0

    # Конвертировать в numpy

    # создать заполнитель той же формы с нулями
    data = np.zeros((CFG.nsamples, len(CFG.eeg_features)))

    for index, feature in enumerate(CFG.eeg_features):
        x = eeg[feature].values.astype("float32")  # конвертировать в float32

        # Вычисляет среднее арифметическое вдоль указанной оси, игнорируя NaN.
        mean = np.nanmean(x)
        nan_percentage = np.isnan(x).mean()  # percentage of NaN values in feature

        # Заполнение значения Nan
        # Поэлементная проверка на NaN и возврат результата в виде логического массива.
        if nan_percentage < 1:  # если некоторые значения равны Nan, но не все
            x = np.nan_to_num(x, nan=mean)
        else:  # если все значения — Nan
            x[:] = 0
        data[:, index] = x

        if display:
            if index != 0:
                offset += x.max()
            plt.plot(range(CFG.nsamples), x - offset, label=feature)
            offset -= x.min()

    if display:
        plt.legend()
        name = parquet_path.split("/")[-1].split(".")[0]
        plt.yticks([])
        plt.title(f"EEG {name}", size=16)
        plt.show()
    return data


# %% [code] {"execution":{"iopub.status.busy":"2024-04-02T20:40:03.714298Z","iopub.execute_input":"2024-04-02T20:40:03.715385Z","iopub.status.idle":"2024-04-02T20:40:03.748374Z","shell.execute_reply.started":"2024-04-02T20:40:03.715329Z","shell.execute_reply":"2024-04-02T20:40:03.747051Z"}}
class EEGDataset(Dataset):
    def __init__(
            self,
            df: pd.DataFrame,
            batch_size: int,
            eegs: Dict[int, np.ndarray],
            mode: str = "train",
            downsample: int = None,
            bandpass_filter: Dict[str, Union[int, float]] = None,
            rand_filter: Dict[str, Union[int, float]] = None,
    ):
        self.df = df
        self.batch_size = batch_size
        self.mode = mode
        self.eegs = eegs
        self.downsample = downsample
        self.bandpass_filter = bandpass_filter
        self.rand_filter = rand_filter

    def __len__(self):
        """
        Length of dataset.
        """
        # Обозначает количество пакетов за эпоху
        return len(self.df)

    def __getitem__(self, index):
        """
        Get one item.
        """
        # Сгенерировать один пакет данных
        X, y_prob = self.__data_generation(index)
        if self.downsample is not None:
            X = X[:: self.downsample, :]
        output = {
            "eeg": torch.tensor(X, dtype=torch.float32),
            "labels": torch.tensor(y_prob, dtype=torch.float32),
        }
        return output

    def __data_generation(self, index):
        # Генерирует данные, содержащие образцы размера партии
        X = np.zeros(
            (CFG.out_samples, CFG.in_channels), dtype="float32"
        )  # Size=(10000, 14)

        row = self.df.iloc[index]  # Строка Pandas
        data = self.eegs[row.eeg_id]  # Size=(10000, 8)
        if CFG.nsamples != CFG.out_samples:
            if self.mode != "train":
                offset = (CFG.nsamples - CFG.out_samples) // 2
            else:
                # offset = random.randint(0, CFG.nsamples - CFG.out_samples)
                offset = ((CFG.nsamples - CFG.out_samples) * random.randint(0, 1000)) // 1000
            data = data[offset:offset + CFG.out_samples, :]

        for i, (feat_a, feat_b) in enumerate(CFG.map_features):
            if self.mode == "train" and CFG.random_close_zone > 0 and random.uniform(0.0, 1.0) <= CFG.random_close_zone:
                continue

            diff_feat = (
                    data[:, CFG.feature_to_index[feat_a]]
                    - data[:, CFG.feature_to_index[feat_b]]
            )  # Size=(10000,)

            if not self.bandpass_filter is None:
                diff_feat = butter_bandpass_filter(
                    diff_feat,
                    self.bandpass_filter["low"],
                    self.bandpass_filter["high"],
                    CFG.sampling_rate,
                    order=self.bandpass_filter["order"],
                )

            if (
                    self.mode == "train"
                    and not self.rand_filter is None
                    and random.uniform(0.0, 1.0) <= self.rand_filter["probab"]
            ):
                lowcut = random.randint(
                    self.rand_filter["low"], self.rand_filter["high"]
                )
                highcut = lowcut + self.rand_filter["band"]
                diff_feat = butter_bandpass_filter(
                    diff_feat,
                    lowcut,
                    highcut,
                    CFG.sampling_rate,
                    order=self.rand_filter["order"],
                )

            X[:, i] = diff_feat

        n = CFG.n_map_features
        if len(CFG.freq_channels) > 0:
            for i in range(CFG.n_map_features):
                diff_feat = X[:, i]
                for j, (lowcut, highcut) in enumerate(CFG.freq_channels):
                    band_feat = butter_bandpass_filter(
                        diff_feat, lowcut, highcut, CFG.sampling_rate, order=CFG.filter_order,  # 6
                    )
                    X[:, n] = band_feat
                    n += 1

        for spml_feat in CFG.simple_features:
            feat_val = data[:, CFG.feature_to_index[spml_feat]]

            if not self.bandpass_filter is None:
                feat_val = butter_bandpass_filter(
                    feat_val,
                    self.bandpass_filter["low"],
                    self.bandpass_filter["high"],
                    CFG.sampling_rate,
                    order=self.bandpass_filter["order"],
                )

            if (
                    self.mode == "train"
                    and not self.rand_filter is None
                    and random.uniform(0.0, 1.0) <= self.rand_filter["probab"]
            ):
                lowcut = random.randint(
                    self.rand_filter["low"], self.rand_filter["high"]
                )
                highcut = lowcut + self.rand_filter["band"]
                feat_val = butter_bandpass_filter(
                    feat_val,
                    lowcut,
                    highcut,
                    CFG.sampling_rate,
                    order=self.rand_filter["order"],
                )

            X[:, n] = feat_val
            n += 1

        # Обрезать края превышающие значения [-1024, 1024]
        X = np.clip(X, -1024, 1024)

        # Замените NaN нулем и разделить все на 32
        X = np.nan_to_num(X, nan=0) / 32.0

        # обрезать полосовым фильтром верхнюю границу в 20 Hz.
        X = butter_lowpass_filter(X, order=CFG.filter_order)  # 4

        y_prob = np.zeros(CFG.target_size, dtype="float32")  # Size=(6,)
        if self.mode != "test":
            y_prob = row[CFG.target_cols].values.astype(np.float32)

        return X, y_prob


# %% [code] {"execution":{"iopub.status.busy":"2024-04-02T20:40:03.750607Z","iopub.execute_input":"2024-04-02T20:40:03.751047Z","iopub.status.idle":"2024-04-02T20:40:03.767903Z","shell.execute_reply.started":"2024-04-02T20:40:03.751009Z","shell.execute_reply":"2024-04-02T20:40:03.766545Z"}}
device


# %% [code] {"execution":{"iopub.status.busy":"2024-04-02T20:40:03.769722Z","iopub.execute_input":"2024-04-02T20:40:03.770268Z","iopub.status.idle":"2024-04-02T20:40:03.805944Z","shell.execute_reply.started":"2024-04-02T20:40:03.770121Z","shell.execute_reply":"2024-04-02T20:40:03.804705Z"}}
class ResNet_1D_Block(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            downsampling,
            dilation=1,
            groups=1,
            dropout=0.0,
    ):
        super(ResNet_1D_Block, self).__init__()

        self.bn1 = nn.BatchNorm1d(num_features=in_channels)
        # self.relu = nn.ReLU(inplace=False)
        # self.relu_1 = nn.PReLU()
        # self.relu_2 = nn.PReLU()
        self.relu_1 = nn.Hardswish()
        self.relu_2 = nn.Hardswish()

        self.dropout = nn.Dropout(p=dropout, inplace=False)
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )

        self.bn2 = nn.BatchNorm1d(num_features=out_channels)
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )

        self.maxpool = nn.MaxPool1d(
            kernel_size=2,
            stride=2,
            padding=0,
            dilation=dilation,
        )
        self.downsampling = downsampling

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu_1(out)
        out = self.dropout(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu_2(out)
        out = self.dropout(out)
        out = self.conv2(out)

        out = self.maxpool(out)
        identity = self.downsampling(x)

        out += identity
        return out


class EEGNet(nn.Module):
    def __init__(
            self,
            kernels,
            in_channels,
            fixed_kernel_size,
            num_classes,
            linear_layer_features,
            dilation=1,
            groups=1,
    ):
        super(EEGNet, self).__init__()
        self.kernels = kernels
        self.planes = 24
        self.parallel_conv = nn.ModuleList()
        self.in_channels = in_channels

        for i, kernel_size in enumerate(list(self.kernels)):
            sep_conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=self.planes,
                kernel_size=(kernel_size),
                stride=1,
                padding=0,
                dilation=dilation,
                groups=groups,
                bias=False,
            )
            self.parallel_conv.append(sep_conv)

        self.bn1 = nn.BatchNorm1d(num_features=self.planes)
        # self.relu = nn.ReLU(inplace=False)
        # self.relu_1 = nn.ReLU()
        # self.relu_2 = nn.ReLU()
        self.relu_1 = nn.SiLU()
        self.relu_2 = nn.SiLU()

        self.conv1 = nn.Conv1d(
            in_channels=self.planes,
            out_channels=self.planes,
            kernel_size=fixed_kernel_size,
            stride=2,
            padding=2,
            dilation=dilation,
            groups=groups,
            bias=False,
        )

        self.block = self._make_resnet_layer(
            kernel_size=fixed_kernel_size,
            stride=1,
            dilation=dilation,
            groups=groups,
            padding=fixed_kernel_size // 2,
        )
        self.bn2 = nn.BatchNorm1d(num_features=self.planes)
        self.avgpool = nn.AvgPool1d(kernel_size=6, stride=6, padding=2)

        self.rnn = nn.GRU(
            input_size=self.in_channels,
            hidden_size=128,
            num_layers=1,
            bidirectional=True,
            # dropout=0.2,
        )

        self.fc = nn.Linear(in_features=linear_layer_features, out_features=num_classes)

    def _make_resnet_layer(
            self,
            kernel_size,
            stride,
            dilation=1,
            groups=1,
            blocks=9,
            padding=0,
            dropout=0.0,
    ):
        layers = []
        downsample = None
        base_width = self.planes

        for i in range(blocks):
            downsampling = nn.Sequential(
                nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
            )
            layers.append(
                ResNet_1D_Block(
                    in_channels=self.planes,
                    out_channels=self.planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    downsampling=downsampling,
                    dilation=dilation,
                    groups=groups,
                    dropout=dropout,
                )
            )
        return nn.Sequential(*layers)

    def extract_features(self, x):
        x = x.permute(0, 2, 1)
        out_sep = []

        for i in range(len(self.kernels)):
            sep = self.parallel_conv[i](x)
            out_sep.append(sep)

        out = torch.cat(out_sep, dim=2)
        out = self.bn1(out)
        out = self.relu_1(out)
        out = self.conv1(out)

        out = self.block(out)
        out = self.bn2(out)
        out = self.relu_2(out)
        out = self.avgpool(out)

        out = out.reshape(out.shape[0], -1)
        rnn_out, _ = self.rnn(x.permute(0, 2, 1))
        new_rnn_h = rnn_out[:, -1, :]  # <~~

        new_out = torch.cat([out, new_rnn_h], dim=1)
        return new_out

    def forward(self, x):
        new_out = self.extract_features(x)
        result = self.fc(new_out)
        return result


# %% [code] {"execution":{"iopub.status.busy":"2024-04-02T20:40:03.807524Z","iopub.execute_input":"2024-04-02T20:40:03.807878Z","iopub.status.idle":"2024-04-02T20:40:03.823803Z","shell.execute_reply.started":"2024-04-02T20:40:03.807848Z","shell.execute_reply":"2024-04-02T20:40:03.822466Z"}}
def inference_function(test_loader, model, device):
    model.eval()  # set model in evaluation mode
    softmax = nn.Softmax(dim=1)
    prediction_dict = {}
    preds = []
    with tqdm(test_loader, unit="test_batch", desc="Inference") as tqdm_test_loader:
        for step, batch in enumerate(tqdm_test_loader):
            X = batch.pop("eeg").to(device)  # send inputs to `device`
            batch_size = X.size(0)
            with torch.no_grad():
                y_preds = model(X)  # forward propagation pass
            y_preds = softmax(y_preds)
            preds.append(y_preds.to("cpu").numpy())  # save predictions

    prediction_dict["predictions"] = np.concatenate(
        preds
    )  # np.array() of shape (fold_size, target_cols)
    return prediction_dict


# %% [code] {"execution":{"iopub.status.busy":"2024-04-02T20:40:03.825412Z","iopub.execute_input":"2024-04-02T20:40:03.826192Z","iopub.status.idle":"2024-04-02T20:40:03.852616Z","shell.execute_reply.started":"2024-04-02T20:40:03.826146Z","shell.execute_reply":"2024-04-02T20:40:03.851435Z"}}
test_df = pd.read_csv(CFG.test_csv)
print(f"Test dataframe shape is: {test_df.shape}")
test_df.head()

# %% [code] {"execution":{"iopub.status.busy":"2024-04-02T20:40:03.854197Z","iopub.execute_input":"2024-04-02T20:40:03.854601Z","iopub.status.idle":"2024-04-02T20:40:04.672424Z","shell.execute_reply.started":"2024-04-02T20:40:03.854568Z","shell.execute_reply":"2024-04-02T20:40:04.671093Z"}}
test_eeg_parquet_paths = glob(CFG.test_eeg + "*.parquet")
test_eeg_df = pd.read_parquet(test_eeg_parquet_paths[0])
test_eeg_features = test_eeg_df.columns
print(f"There are {len(test_eeg_features)} raw eeg features")
print(list(test_eeg_features))
del test_eeg_df
_ = gc.collect()

# %%time
all_eegs = {}
eeg_ids = test_df.eeg_id.unique()
for i, eeg_id in tqdm(enumerate(eeg_ids)):
    # Save EEG to Python dictionary of numpy arrays
    eeg_path = CFG.test_eeg + str(eeg_id) + ".parquet"
    data = eeg_from_parquet(eeg_path)
    all_eegs[eeg_id] = data

# %% [markdown]
# ## Inference

# %% [code] {"execution":{"iopub.status.busy":"2024-04-02T20:40:04.674244Z","iopub.execute_input":"2024-04-02T20:40:04.674679Z","iopub.status.idle":"2024-04-02T20:40:08.380786Z","shell.execute_reply.started":"2024-04-02T20:40:04.674625Z","shell.execute_reply":"2024-04-02T20:40:08.379554Z"}}
koef_sum = 0
koef_count = 0
predictions = []
files = []

for model_block in model_weights:
    test_dataset = EEGDataset(
        df=test_df,
        batch_size=CFG.batch_size,
        mode="test",
        eegs=all_eegs,
        bandpass_filter=model_block['bandpass_filter']
    )

    if len(predictions) == 0:
        output = test_dataset[0]
        X = output["eeg"]
        print(f"X shape: {X.shape}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = EEGNet(
        kernels=CFG.kernels,
        in_channels=CFG.in_channels,
        fixed_kernel_size=CFG.fixed_kernel_size,
        num_classes=CFG.target_size,
        linear_layer_features=CFG.linear_layer_features,
    )

    for file_line in model_block['file_data']:
        koef = file_line['koef']
        for weight_model_file in glob(file_line['file_mask']):
            files.append(weight_model_file)
            checkpoint = torch.load(weight_model_file, map_location=device)
            model.load_state_dict(checkpoint["model"])
            model.to(device)
            prediction_dict = inference_function(test_loader, model, device)
            predict = prediction_dict["predictions"]
            predict *= koef
            koef_sum += koef
            koef_count += 1
            predictions.append(predict)
            torch.cuda.empty_cache()
            gc.collect()

predictions = np.array(predictions)
koef_sum /= koef_count
predictions /= koef_sum
predictions = np.mean(predictions, axis=0)

# %% [code] {"execution":{"iopub.status.busy":"2024-04-02T20:40:08.382722Z","iopub.execute_input":"2024-04-02T20:40:08.383119Z","iopub.status.idle":"2024-04-02T20:40:08.390659Z","shell.execute_reply.started":"2024-04-02T20:40:08.383083Z","shell.execute_reply":"2024-04-02T20:40:08.389638Z"}}
print(koef_count, koef_sum)

# %% [code] {"execution":{"iopub.status.busy":"2024-04-02T20:46:33.011666Z","iopub.execute_input":"2024-04-02T20:46:33.012109Z","iopub.status.idle":"2024-04-02T20:46:33.039244Z","shell.execute_reply.started":"2024-04-02T20:46:33.012075Z","shell.execute_reply":"2024-04-02T20:46:33.037921Z"}}
sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})
sub[CFG.target_cols] = predictions

#sub.to_csv(f"submission_model4.csv", index=False)
print(f"Submission shape: {sub.shape}")

predss_4 = sub[['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']].to_numpy()

sub.head()

columns = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
df = pd.read_csv('/kaggle/input/hms-harmful-brain-activity-classification/test.csv')
df.loc[:len(predss_4) - 1, columns] = predss_4

df[['eeg_id'] + columns].to_csv('submission_4.csv', index=False)