import librosa
import os, random
import tensorflow
import tensorflow as tf
import albumentations as albu
import pandas as pd, numpy as np
from scipy.signal import butter, lfilter
import tensorflow.keras.backend as K, gc
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Dense, Multiply, Add, Conv1D, Concatenate, LayerNormalization

LOAD_BACKBONE_FROM = '/kaggle/input/efficientnetb-tf-keras/EfficientNetB2.h5'
LOAD_MODELS_FROM = '/kaggle/input/features-head-starter-models'
MODEL = {'K+E+KE': 52}
for DATA_TYPE in MODEL: pass
TARGETS = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
FEATS2 = ['Fp1', 'T3', 'C3', 'O1', 'Fp2', 'C4', 'T4', 'O2']
FEAT2IDX = {x: y for x, y in zip(FEATS2, range(len(FEATS2)))}
FEATS = [['Fp1', 'F7', 'T3', 'T5', 'O1'],
         ['Fp1', 'F3', 'C3', 'P3', 'O1'],
         ['Fp2', 'F8', 'T4', 'T6', 'O2'],
         ['Fp2', 'F4', 'C4', 'P4', 'O2']]


class DataGenerator():
    'Generates data for Keras'

    def __init__(self, data, specs=None, eeg_specs=None, raw_eegs=None, augment=False, mode='train',
                 data_type=DATA_TYPE):
        self.augment = augment
        self.mode = mode
        self.data_type = data_type
        self.data = self.build_data(data.copy())
        self.specs = specs
        self.eeg_specs = eeg_specs
        self.raw_eegs = raw_eegs
        self.on_epoch_end()

    def build_data(self, data):
        if self.data_type in ['K+E']:
            data_dup = pd.concat([data] * 2, ignore_index=True)
            data_dup.loc[:len(data), 'data_type'] = 'K'
            data_dup.loc[len(data):, 'data_type'] = 'E'
            data = data_dup
        elif self.data_type in ['K+E+KE']:
            data_trp = pd.concat([data] * 3, ignore_index=True)
            data_trp.loc[:len(data), 'data_type'] = 'K'
            data_trp.loc[len(data):len(data) * 2, 'data_type'] = 'E'
            data_trp.loc[len(data) * 2:, 'data_type'] = 'KE'
            data = data_trp
        else:
            data['data_type'] = self.data_type
        return data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        X, y = self.data_generation(index)
        if self.augment: X = self.augmentation(X)
        return X, y

    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

            if i == self.__len__() - 1:
                self.on_epoch_end()

    def on_epoch_end(self):
        if self.mode == 'train':
            self.data = self.data.sample(frac=1).reset_index(drop=True)

    def data_generation(self, index):
        row = self.data.iloc[index]
        if row.data_type == 'KE':
            X, y = self.generate_all_specs(index)
        elif row.data_type in ['K', 'E']:
            X, y = self.generate_specs(index)
        elif row.data_type == 'R':
            X, y = self.generate_raw(index)
        elif row.data_type in ['ER', 'KR']:
            X1, y = self.generate_specs(index)
            X2, y = self.generate_raw(index)
            X = (X1, X2)
        elif row.data_type in ['KER']:
            X1, y = self.generate_all_specs(index)
            X2, y = self.generate_raw(index)
            X = (X1, X2)
        return X, y

    def generate_all_specs(self, index):
        X = np.zeros((512, 512, 3), dtype='float32')
        y = np.zeros((6,), dtype='float32')

        row = self.data.iloc[index]
        if self.mode == 'test':
            offset = 0
        else:
            offset = int(row.offset / 2)

        eeg = self.eeg_specs[row.eeg_id]
        spec = self.specs[row.spec_id]

        imgs = [spec[offset:offset + 300, k * 100:(k + 1) * 100].T for k in [0, 2, 1, 3]]  # to match kaggle with eeg
        img = np.stack(imgs, axis=-1)
        # LOG TRANSFORM SPECTROGRAM
        img = np.clip(img, np.exp(-4), np.exp(8))
        img = np.log(img)

        # STANDARDIZE PER IMAGE
        img = np.nan_to_num(img, nan=0.0)

        mn = img.flatten().min()
        mx = img.flatten().max()
        ep = 1e-5
        img = 255 * (img - mn) / (mx - mn + ep)

        X[0_0 + 56:100 + 56, :256, 0] = img[:, 22:-22, 0]  # LL_k
        X[100 + 56:200 + 56, :256, 0] = img[:, 22:-22, 2]  # RL_k
        X[0_0 + 56:100 + 56, :256, 1] = img[:, 22:-22, 1]  # LP_k
        X[100 + 56:200 + 56, :256, 1] = img[:, 22:-22, 3]  # RP_k
        X[0_0 + 56:100 + 56, :256, 2] = img[:, 22:-22, 2]  # RL_k
        X[100 + 56:200 + 56, :256, 2] = img[:, 22:-22, 1]  # LP_k

        X[0_0 + 56:100 + 56, 256:, 0] = img[:, 22:-22, 0]  # LL_k
        X[100 + 56:200 + 56, 256:, 0] = img[:, 22:-22, 2]  # RL_k
        X[0_0 + 56:100 + 56, 256:, 1] = img[:, 22:-22, 1]  # LP_k
        X[100 + 56:200 + 56, 256:, 1] = img[:, 22:-22, 3]  # RP_K

        # EEG
        img = eeg
        mn = img.flatten().min()
        mx = img.flatten().max()
        ep = 1e-5
        img = 255 * (img - mn) / (mx - mn + ep)
        X[200 + 56:300 + 56, :256, 0] = img[:, 22:-22, 0]  # LL_e
        X[300 + 56:400 + 56, :256, 0] = img[:, 22:-22, 2]  # RL_e
        X[200 + 56:300 + 56, :256, 1] = img[:, 22:-22, 1]  # LP_e
        X[300 + 56:400 + 56, :256, 1] = img[:, 22:-22, 3]  # RP_e
        X[200 + 56:300 + 56, :256, 2] = img[:, 22:-22, 2]  # RL_e
        X[300 + 56:400 + 56, :256, 2] = img[:, 22:-22, 1]  # LP_e

        X[200 + 56:300 + 56, 256:, 0] = img[:, 22:-22, 0]  # LL_e
        X[300 + 56:400 + 56, 256:, 0] = img[:, 22:-22, 2]  # RL_e
        X[200 + 56:300 + 56, 256:, 1] = img[:, 22:-22, 1]  # LP_e
        X[300 + 56:400 + 56, 256:, 1] = img[:, 22:-22, 3]  # RP_e

        if self.mode != 'test':
            y[:] = row[TARGETS]

        return X, y

    def generate_specs(self, index):
        X = np.zeros((512, 512, 3), dtype='float32')
        y = np.zeros((6,), dtype='float32')

        row = self.data.iloc[index]
        if self.mode == 'test':
            offset = 0
        else:
            offset = int(row.offset / 2)

        if row.data_type in ['E', 'ER']:
            img = self.eeg_specs[row.eeg_id]
        elif row.data_type in ['K', 'KR']:
            spec = self.specs[row.spec_id]
            imgs = [spec[offset:offset + 300, k * 100:(k + 1) * 100].T for k in
                    [0, 2, 1, 3]]  # to match kaggle with eeg
            img = np.stack(imgs, axis=-1)
            # LOG TRANSFORM SPECTROGRAM
            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)

            # STANDARDIZE PER IMAGE
            img = np.nan_to_num(img, nan=0.0)

        mn = img.flatten().min()
        mx = img.flatten().max()
        ep = 1e-5
        img = 255 * (img - mn) / (mx - mn + ep)

        X[0_0 + 56:100 + 56, :256, 0] = img[:, 22:-22, 0]
        X[100 + 56:200 + 56, :256, 0] = img[:, 22:-22, 2]
        X[0_0 + 56:100 + 56, :256, 1] = img[:, 22:-22, 1]
        X[100 + 56:200 + 56, :256, 1] = img[:, 22:-22, 3]
        X[0_0 + 56:100 + 56, :256, 2] = img[:, 22:-22, 2]
        X[100 + 56:200 + 56, :256, 2] = img[:, 22:-22, 1]

        X[0_0 + 56:100 + 56, 256:, 0] = img[:, 22:-22, 0]
        X[100 + 56:200 + 56, 256:, 0] = img[:, 22:-22, 1]
        X[0_0 + 56:100 + 56, 256:, 1] = img[:, 22:-22, 2]
        X[100 + 56:200 + 56, 256:, 1] = img[:, 22:-22, 3]

        X[200 + 56:300 + 56, :256, 0] = img[:, 22:-22, 0]
        X[300 + 56:400 + 56, :256, 0] = img[:, 22:-22, 1]
        X[200 + 56:300 + 56, :256, 1] = img[:, 22:-22, 2]
        X[300 + 56:400 + 56, :256, 1] = img[:, 22:-22, 3]
        X[200 + 56:300 + 56, :256, 2] = img[:, 22:-22, 3]
        X[300 + 56:400 + 56, :256, 2] = img[:, 22:-22, 2]

        X[200 + 56:300 + 56, 256:, 0] = img[:, 22:-22, 0]
        X[300 + 56:400 + 56, 256:, 0] = img[:, 22:-22, 2]
        X[200 + 56:300 + 56, 256:, 1] = img[:, 22:-22, 1]
        X[300 + 56:400 + 56, 256:, 1] = img[:, 22:-22, 3]

        if self.mode != 'test':
            y[:] = row[TARGETS]

        return X, y

    def generate_raw(self, index):
        if USE_PROCESSED and self.mode != 'test':
            X = np.zeros((2_000, 8), dtype='float32')
            y = np.zeros((6,), dtype='float32')
            row = self.data.iloc[index]
            X = self.raw_eegs[row.eeg_id]
            y[:] = row[TARGETS]
            return X, y

        X = np.zeros((10_000, 8), dtype='float32')
        y = np.zeros((6,), dtype='float32')

        row = self.data.iloc[index]
        eeg = self.raw_eegs[row.eeg_id]

        # FEATURE ENGINEER
        X[:, 0] = eeg[:, FEAT2IDX['Fp1']] - eeg[:, FEAT2IDX['T3']]
        X[:, 1] = eeg[:, FEAT2IDX['T3']] - eeg[:, FEAT2IDX['O1']]

        X[:, 2] = eeg[:, FEAT2IDX['Fp1']] - eeg[:, FEAT2IDX['C3']]
        X[:, 3] = eeg[:, FEAT2IDX['C3']] - eeg[:, FEAT2IDX['O1']]

        X[:, 4] = eeg[:, FEAT2IDX['Fp2']] - eeg[:, FEAT2IDX['C4']]
        X[:, 5] = eeg[:, FEAT2IDX['C4']] - eeg[:, FEAT2IDX['O2']]

        X[:, 6] = eeg[:, FEAT2IDX['Fp2']] - eeg[:, FEAT2IDX['T4']]
        X[:, 7] = eeg[:, FEAT2IDX['T4']] - eeg[:, FEAT2IDX['O2']]

        # STANDARDIZE
        X = np.clip(X, -1024, 1024)
        X = np.nan_to_num(X, nan=0) / 32.0

        # BUTTER LOW-PASS FILTER
        X = self.butter_lowpass_filter(X)
        # Downsample
        X = X[::5, :]

        if self.mode != 'test':
            y[:] = row[TARGETS]

        return X, y

    def butter_lowpass_filter(self, data, cutoff_freq=20, sampling_rate=200, order=4):
        nyquist = 0.5 * sampling_rate
        normal_cutoff = cutoff_freq / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        filtered_data = lfilter(b, a, data, axis=0)
        return filtered_data

    def resize(self, img, size):
        composition = albu.Compose([
            albu.Resize(size[0], size[1])
        ])
        return composition(image=img)['image']

    def augmentation(self, img):
        composition = albu.Compose([
            albu.HorizontalFlip(p=0.4)
        ])
        return composition(image=img)['image']


def spectrogram_from_eeg(parquet_path):
    # LOAD MIDDLE 50 SECONDS OF EEG SERIES
    eeg = pd.read_parquet(parquet_path)
    middle = (len(eeg) - 10_000) // 2
    eeg = eeg.iloc[middle:middle + 10_000]

    # VARIABLE TO HOLD SPECTROGRAM
    img = np.zeros((100, 300, 4), dtype='float32')

    for k in range(4):
        COLS = FEATS[k]

        for kk in range(4):
            # FILL NANS
            x1 = eeg[COLS[kk]].values
            x2 = eeg[COLS[kk + 1]].values
            m = np.nanmean(x1)
            if np.isnan(x1).mean() < 1:
                x1 = np.nan_to_num(x1, nan=m)
            else:
                x1[:] = 0
            m = np.nanmean(x2)
            if np.isnan(x2).mean() < 1:
                x2 = np.nan_to_num(x2, nan=m)
            else:
                x2[:] = 0

            # COMPUTE PAIR DIFFERENCES
            x = x1 - x2

            # RAW SPECTROGRAM
            mel_spec = librosa.feature.melspectrogram(y=x, sr=200, hop_length=len(x) // 300,
                                                      n_fft=1024, n_mels=100, fmin=0, fmax=20, win_length=128)

            # LOG TRANSFORM
            width = (mel_spec.shape[1] // 30) * 30
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[:, :width]
            img[:, :, k] += mel_spec_db

        # AVERAGE THE 4 MONTAGE DIFFERENCES
        img[:, :, k] /= 4.0

    return img


def eeg_from_parquet(parquet_path):
    eeg = pd.read_parquet(parquet_path, columns=FEATS2)
    rows = len(eeg)
    offset = (rows - 10_000) // 2
    eeg = eeg.iloc[offset:offset + 10_000]
    data = np.zeros((10_000, len(FEATS2)))
    for j, col in enumerate(FEATS2):

        # FILL NAN
        x = eeg[col].values.astype('float32')
        m = np.nanmean(x)
        if np.isnan(x).mean() < 1:
            x = np.nan_to_num(x, nan=m)
        else:
            x[:] = 0

        data[:, j] = x

    return data


def build_spec_model(hybrid=False):
    inp = tf.keras.layers.Input((512, 512, 3))
    base_model = load_model(f'{LOAD_BACKBONE_FROM}')
    x = base_model(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    if not hybrid:
        x = tf.keras.layers.Dense(6, activation='softmax', dtype='float32')(x)
    model = tf.keras.Model(inputs=inp, outputs=x)
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss = tf.keras.losses.KLDivergence()
    model.compile(loss=loss, optimizer=opt)
    return model


def dataset(data, mode='train', batch_size=32, data_type=DATA_TYPE,
            augment=False, specs=None, eeg_specs=None, raw_eegs=None):
    gen = DataGenerator(data, mode=mode, data_type=data_type, augment=augment,
                        specs=specs, eeg_specs=eeg_specs, raw_eegs=raw_eegs)
    inp = tf.TensorSpec(shape=(512, 512, 3), dtype=tf.float32)
    output_signature = (inp, tf.TensorSpec(shape=(6,), dtype=tf.float32))
    dataset = tf.data.Dataset.from_generator(generator=gen, output_signature=output_signature).batch(
        batch_size)
    return dataset


def predict(models, params, fold, models_path=None):
    preds = []
    if models_path is None: models_path = LOAD_MODELS_FROM
    model = build_spec_model()
    for data_type in models:
        data = params['data']
        ver = models[data_type]
        ds = dataset(data_type=data_type, **params)
        model.load_weights(f'{models_path}/model_{data_type}_{ver}_{fold}.weights.h5')
        pred = model.predict(ds)
        if data_type in ['K+E+KE']:
            pred = (pred[:len(data)] + pred[len(data):len(data) * 2] + pred[len(data) * 2:]) / 3
        preds.append(pred)
    pred = np.mean(preds, axis=0)
    del model
    gc.collect()
    return pred


# %% [markdown]
# ## Load Test Data

# %% [code] {"execution":{"iopub.status.busy":"2024-04-02T20:38:43.573720Z","iopub.execute_input":"2024-04-02T20:38:43.574513Z","iopub.status.idle":"2024-04-02T20:38:43.995097Z","shell.execute_reply.started":"2024-04-02T20:38:43.574461Z","shell.execute_reply":"2024-04-02T20:38:43.993460Z"}}
# READ ALL SPECTROGRAMS
test = pd.read_csv('/kaggle/input/hms-harmful-brain-activity-classification/test.csv')
PATH2 = '/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms'
files2 = os.listdir(PATH2)
spectrograms2 = {}
for i, f in enumerate(files2):
    if i % 100 == 0: print(i, ', ', end='')
    tmp = pd.read_parquet(f'{PATH2}/{f}')
    name = int(f.split('.')[0])
    spectrograms2[name] = tmp.iloc[:, 1:].values

# RENAME FOR DATA GENERATOR
test = test.rename({'spectrogram_id': 'spec_id'}, axis=1)

# READ ALL EEG SPECTROGRAMS
PATH2 = '/kaggle/input/hms-harmful-brain-activity-classification/test_eegs'
EEG_IDS2 = test.eeg_id.unique()
all_eegs2 = {}
for i, eeg_id in enumerate(EEG_IDS2):
    # CREATE SPECTROGRAM FROM EEG PARQUET
    img = spectrogram_from_eeg(f'{PATH2}/{eeg_id}.parquet')
    all_eegs2[eeg_id] = img

# READ ALL RAW EEG SIGNALS
all_raw_eegs2 = {}
for i, eeg_id in enumerate(EEG_IDS2):
    # SAVE EEG TO PYTHON DICTIONARY OF NUMPY ARRAYS
    data = eeg_from_parquet(f'{PATH2}/{eeg_id}.parquet')
    all_raw_eegs2[eeg_id] = data

# %% [markdown]
# ## Predict

# %% [code] {"execution":{"iopub.status.busy":"2024-04-02T20:38:44.005378Z","iopub.execute_input":"2024-04-02T20:38:44.011006Z","iopub.status.idle":"2024-04-02T20:39:52.388749Z","shell.execute_reply.started":"2024-04-02T20:38:44.010911Z","shell.execute_reply":"2024-04-02T20:39:52.387508Z"}}
preds = []
params = {'data': test, 'mode': 'test', 'specs': spectrograms2, 'eeg_specs': all_eegs2, 'raw_eegs': all_raw_eegs2}

for i in range(5):
    print(f'Fold {i + 1}')
    pred = predict(MODEL, params, i)
    preds.append(pred)

pred = np.mean(preds, axis=0)

# %% [code] {"execution":{"iopub.status.busy":"2024-04-02T20:39:52.390489Z","iopub.execute_input":"2024-04-02T20:39:52.391351Z","iopub.status.idle":"2024-04-02T20:39:52.399053Z","shell.execute_reply.started":"2024-04-02T20:39:52.391313Z","shell.execute_reply":"2024-04-02T20:39:52.397560Z"}}
predss_2 = pred
predss_2

columns = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
df = pd.read_csv('/kaggle/input/hms-harmful-brain-activity-classification/test.csv')
df.loc[:len(pred) - 1, columns] = pred

df[['eeg_id'] + columns].to_csv('submission_2.csv', index=False)