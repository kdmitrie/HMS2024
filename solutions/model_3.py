import gc
import os
import random
import warnings
import numpy as np
import pandas as pd
from IPython.display import display

import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

from scipy import signal

warnings.filterwarnings('ignore', category=Warning)
gc.collect()


# %% [code] {"execution":{"iopub.status.busy":"2024-04-02T20:39:53.063666Z","iopub.execute_input":"2024-04-02T20:39:53.064131Z","iopub.status.idle":"2024-04-02T20:39:53.078452Z","shell.execute_reply.started":"2024-04-02T20:39:53.064093Z","shell.execute_reply":"2024-04-02T20:39:53.077534Z"}}
class Config:
    seed = 3131
    image_transform = transforms.Resize((512, 512))
    num_folds = 5
    dataset_wide_mean = -0.2972692229201065  # From Train notebook
    dataset_wide_std = 2.5997336315611026  # From Train notebook
    ownspec_mean = 7.29084372799223e-05  # From Train spectrograms notebook
    ownspec_std = 4.510082606216031  # From Train spectrograms notebook


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


set_seed(Config.seed)

# %% [code] {"execution":{"iopub.status.busy":"2024-04-02T20:39:53.079885Z","iopub.execute_input":"2024-04-02T20:39:53.080415Z","iopub.status.idle":"2024-04-02T20:39:53.772997Z","shell.execute_reply.started":"2024-04-02T20:39:53.080382Z","shell.execute_reply":"2024-04-02T20:39:53.771533Z"}}
test_df = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/test.csv")
submission = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv")

submission = submission.merge(test_df, on='eeg_id', how='left')
submission['path_spec'] = submission['spectrogram_id'].apply(
    lambda x: f"/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/{x}.parquet")
submission['path_eeg'] = submission['eeg_id'].apply(
    lambda x: f"/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/{x}.parquet")

display(submission)

gc.collect()

# %% [code] {"execution":{"iopub.status.busy":"2024-04-02T20:39:53.774783Z","iopub.execute_input":"2024-04-02T20:39:53.775140Z","iopub.status.idle":"2024-04-02T20:39:57.782787Z","shell.execute_reply.started":"2024-04-02T20:39:53.775109Z","shell.execute_reply":"2024-04-02T20:39:57.781330Z"}}
models = []

# Load in original EfficientnetB0 model
for i in range(Config.num_folds):
    model_effnet_b0 = timm.create_model('efficientnet_b0', pretrained=False, num_classes=6, in_chans=1)
    model_effnet_b0.load_state_dict(torch.load(f'/kaggle/input/hms-train-efficientnetb0/efficientnet_b0_fold{i}.pth',
                                               map_location=torch.device('cpu')))
    models.append(model_effnet_b0)

models_datawide = []
# Load in hyperparameter optimized EfficientnetB1
for i in range(Config.num_folds):
    model_effnet_b1 = timm.create_model('efficientnet_b1', pretrained=False, num_classes=6, in_chans=1)
    model_effnet_b1.load_state_dict(
        torch.load(f'/kaggle/input/train/efficientnet_b1_fold{i}.pth', map_location=torch.device('cpu')))
    models_datawide.append(model_effnet_b1)

models_ownspec = []
# Load in EfficientnetB1 with new spectrograms
for i in range(Config.num_folds):
    model_effnet_b1 = timm.create_model('efficientnet_b1', pretrained=False, num_classes=6, in_chans=1)
    model_effnet_b1.load_state_dict(torch.load(
        f'/kaggle/input/efficientnet-b1-ownspectrograms/efficientnet_b1_fold{i}_datawide_CosineAnnealingLR_0.001_False.pth',
        map_location=torch.device('cpu')))
    models_ownspec.append(model_effnet_b1)

gc.collect()

# %% [code] {"execution":{"iopub.status.busy":"2024-04-02T20:39:57.784930Z","iopub.execute_input":"2024-04-02T20:39:57.785783Z","iopub.status.idle":"2024-04-02T20:40:02.428632Z","shell.execute_reply.started":"2024-04-02T20:39:57.785732Z","shell.execute_reply":"2024-04-02T20:40:02.427517Z"}}
test_predictions = []


def create_spectrogram(data):
    """This function will create a spectrogram based on EEG-data"""
    nperseg = 150  # Length of each segment
    noverlap = 128  # Overlap between segments
    NFFT = max(256, 2 ** int(np.ceil(np.log2(nperseg))))

    # LL Spec = ( spec(Fp1 - F7) + spec(F7 - T3) + spec(T3 - T5) + spec(T5 - O1) )/4
    freqs, t, spectrum_LL1 = signal.spectrogram(data['Fp1'] - data['F7'], nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    freqs, t, spectrum_LL2 = signal.spectrogram(data['F7'] - data['T3'], nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    freqs, t, spectrum_LL3 = signal.spectrogram(data['T3'] - data['T5'], nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    freqs, t, spectrum_LL4 = signal.spectrogram(data['T5'] - data['O1'], nfft=NFFT, noverlap=noverlap, nperseg=nperseg)

    LL = (spectrum_LL1 + spectrum_LL2 + spectrum_LL3 + spectrum_LL4) / 4

    # LP Spec = ( spec(Fp1 - F3) + spec(F3 - C3) + spec(C3 - P3) + spec(P3 - O1) )/4
    freqs, t, spectrum_LP1 = signal.spectrogram(data['Fp1'] - data['F3'], nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    freqs, t, spectrum_LP2 = signal.spectrogram(data['F3'] - data['C3'], nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    freqs, t, spectrum_LP3 = signal.spectrogram(data['C3'] - data['P3'], nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    freqs, t, spectrum_LP4 = signal.spectrogram(data['P3'] - data['O1'], nfft=NFFT, noverlap=noverlap, nperseg=nperseg)

    LP = (spectrum_LP1 + spectrum_LP2 + spectrum_LP3 + spectrum_LP4) / 4

    # RP Spec = ( spec(Fp2 - F4) + spec(F4 - C4) + spec(C4 - P4) + spec(P4 - O2) )/4
    freqs, t, spectrum_RP1 = signal.spectrogram(data['Fp2'] - data['F4'], nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    freqs, t, spectrum_RP2 = signal.spectrogram(data['F4'] - data['C4'], nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    freqs, t, spectrum_RP3 = signal.spectrogram(data['C4'] - data['P4'], nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    freqs, t, spectrum_RP4 = signal.spectrogram(data['P4'] - data['O2'], nfft=NFFT, noverlap=noverlap, nperseg=nperseg)

    RP = (spectrum_RP1 + spectrum_RP2 + spectrum_RP3 + spectrum_RP4) / 4

    # RL Spec = ( spec(Fp2 - F8) + spec(F8 - T4) + spec(T4 - T6) + spec(T6 - O2) )/4
    freqs, t, spectrum_RL1 = signal.spectrogram(data['Fp2'] - data['F8'], nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    freqs, t, spectrum_RL2 = signal.spectrogram(data['F8'] - data['T4'], nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    freqs, t, spectrum_RL3 = signal.spectrogram(data['T4'] - data['T6'], nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    freqs, t, spectrum_RL4 = signal.spectrogram(data['T6'] - data['O2'], nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    RL = (spectrum_RL1 + spectrum_RL2 + spectrum_RL3 + spectrum_RL4) / 4
    spectogram = np.concatenate((LL, LP, RP, RL), axis=0)
    return spectogram


def preprocess_ownspec(path_to_parquet):
    """The data will be processed from EEG to spectrogramdata"""
    data = pd.read_parquet(path_to_parquet)
    data = create_spectrogram(data)
    mask = np.isnan(data)
    data[mask] = -1
    data = np.clip(data, np.exp(-6), np.exp(10))
    data = np.log(data)

    return data


def preprocess(path_to_parquet):
    data = pd.read_parquet(path_to_parquet)
    data = data.fillna(-1).values[:, 1:].T
    data = np.clip(data, np.exp(-6), np.exp(10))
    data = np.log(data)

    return data


def normalize_datawide(data_point):
    """The spectrogram data will be normalized data wide."""
    eps = 1e-6

    data_point = (data_point - Config.dataset_wide_mean) / (Config.dataset_wide_std + eps)

    data_tensor = torch.unsqueeze(torch.Tensor(data_point), dim=0)
    data_point = Config.image_transform(data_tensor)

    return data_point


def normalize_datawide_ownspec(data):
    """The new spectrogram data will be normalized data wide."""
    eps = 1e-6

    data = (data - Config.ownspec_mean) / (Config.ownspec_std + eps)
    data_tensor = torch.unsqueeze(torch.Tensor(data), dim=0)
    data = Config.image_transform(data_tensor)

    return data


def normalize_instance_wise(data_point):
    """The spectrogram data will be normalized instance wise."""
    eps = 1e-6

    data_mean = data_point.mean(axis=(0, 1))
    data_std = data_point.std(axis=(0, 1))
    data_point = (data_point - data_mean) / (data_std + eps)

    data_tensor = torch.unsqueeze(torch.Tensor(data_point), dim=0)
    data_point = Config.image_transform(data_tensor)

    return data_point


# Loop over samples
for index in submission.index:
    test_predictions_per_model = []

    preprocessed_data = preprocess(submission.iloc[index]['path_spec'])
    preprocessed_data_ownspec = preprocess_ownspec(submission.iloc[index]['path_eeg'])

    # Predict based on original EfficientnetB0 models.
    for i in range(len(models)):
        models[i].eval()

        current_parquet_data = normalize_instance_wise(preprocessed_data).unsqueeze(0)

        with torch.no_grad():
            model_output = models[i](current_parquet_data)
            current_model_prediction = F.softmax(model_output)[0].detach().cpu().numpy()

        test_predictions_per_model.append(current_model_prediction)

    # Predict based on hyperparameter optimized EffcientnetB1.
    for i in range(len(models_datawide)):
        models_datawide[i].eval()

        current_parquet_data = normalize_datawide(preprocessed_data).unsqueeze(0)

        with torch.no_grad():
            model_output = models_datawide[i](current_parquet_data)
            current_model_prediction = F.softmax(model_output)[0].detach().cpu().numpy()

        test_predictions_per_model.append(current_model_prediction)

    # Predict based on EfficientnetB1 model with new spectrograms.
    for i in range(len(models_ownspec)):
        models_ownspec[i].eval()

        current_parquet_data = normalize_datawide_ownspec(preprocessed_data_ownspec).unsqueeze(0)

        with torch.no_grad():
            model_output = models_ownspec[i](current_parquet_data)
            current_model_prediction = F.softmax(model_output)[0].detach().cpu().numpy()

        test_predictions_per_model.append(current_model_prediction)

    # The mean of all models is taken.
    ensemble_prediction = np.mean(test_predictions_per_model, axis=0)

    test_predictions.append(ensemble_prediction)

test_predictions = np.array(test_predictions)

gc.collect()

# %% [code] {"execution":{"iopub.status.busy":"2024-04-02T20:40:02.430028Z","iopub.execute_input":"2024-04-02T20:40:02.430456Z","iopub.status.idle":"2024-04-02T20:40:02.439009Z","shell.execute_reply.started":"2024-04-02T20:40:02.430421Z","shell.execute_reply":"2024-04-02T20:40:02.437766Z"}}
predss_3 = test_predictions
predss_3

columns = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
df = pd.read_csv('/kaggle/input/hms-harmful-brain-activity-classification/test.csv')
df.loc[:len(test_predictions) - 1, columns] = test_predictions

df[['eeg_id'] + columns].to_csv('submission_3.csv', index=False)