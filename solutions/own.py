import sys
import torch
import sys
import numpy as np
import pickle
import pandas as pd
import mne
import timm

sys.path.append('/kaggle/input/hms-lib')
mne.set_log_level(False)

from HMS2024.hms.pipeline import HMSPipeline, HMSFillna, HMSLogSG, HMSStandardScaler, HMSCreateSGFromEEG, HMSCropPad, HMSImage
from HMS2024.hms.hmsio import HMSReader
from HMS2024.hms.dataset import HMSDataset
from HMS2024.models.lognet import *


SG = '/kaggle/input/hms-harmful-brain-activity-classification/%s_spectrograms/%%i.parquet'
EEG = '/kaggle/input/hms-harmful-brain-activity-classification/%s_eegs/%%i.parquet'
CSV = '/kaggle/input/hms-harmful-brain-activity-classification/%s.csv'

folds = 5
branch = 'test'
batch_size = 32
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

m = int(sys.argv[1])

with open('/kaggle/input/hms-models-collection/models.pkl', 'rb') as f:
    models = pickle.load(f)

model = models[m]
print(f'Model {m+1}')

with open(f'/kaggle/input/hms-models-collection/model{m + 1}/pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

reader = HMSReader(CSV % branch, SG % branch, EEG % branch)
data_submit = HMSDataset(data_provider=pipeline(reader), shuffle=False)
loader = torch.utils.data.DataLoader(data_submit, batch_size=batch_size, shuffle=False)

all_probs = []
for fold in range(folds):
    nn_model = torch.load(f'/kaggle/input/hms-models-collection/model{m + 1}/pytorch_fold_{fold}.trch', map_location=device)
    nn_model.eval()
    nn_model.device = device
    with torch.no_grad():
        log_probs = [nn_model(x).detach().cpu() for x, y in loader]

    log_probs = torch.concatenate(log_probs)
    probs = torch.exp(log_probs)
    probs = probs.detach().cpu().numpy()

    all_probs.append(probs)

probs = np.mean(np.stack(all_probs), axis=0)

columns = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']

df = pd.read_csv(CSV % 'test')
df.loc[:len(probs)-1, columns] = probs

df[['eeg_id'] + columns].to_csv(f'submission_own_{m}.csv', index=False)

predss_own = df[columns].to_numpy()

print(df.head())
