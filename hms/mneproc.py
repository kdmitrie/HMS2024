import pandas as pd
import numpy as np
import scipy
import mne

from sklearn.linear_model import Ridge
import librosa


class MNEPreprocessor:
    channel_groups = [[0, 11], [4, 5, 6], [1, 2, 3], [8, 9, 10], [12, 13, 14], [15, 16, 17], [7, 18]]
    channel_indices = [[0, 4, 5, 6, 7], [0, 1, 2, 3, 7], [11, 15, 16, 17, 18], [11, 12, 13, 14, 18]]

    def __init__(self, csv_path: str = None, eeg_path: str = None, n_components: int = 19, sfreq: int = 200):
        self.df = None if csv_path is None else pd.read_csv(csv_path)
        self.eeg_path = eeg_path
        self.ica = mne.preprocessing.ICA(n_components=n_components, random_state=42)

        self.ch_names = ['Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'Fz', 'Cz',
                         'Pz', 'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2', 'EKG', 'MEANEEG']
        self.ch_types = ['eeg'] * 19 + ['ecg'] * 2
        self.mne_info = mne.create_info(self.ch_names, sfreq, self.ch_types)
        mne.set_log_level(False)

    def load(self, item_id: int) -> mne.io.RawArray:
        if self.df is None:
            raise Exception('Dataframe not loaded')
        print(f'Load item #{item_id}')

        item = self.df.iloc[item_id]
        eeg_start = int(200 * item.eeg_label_offset_seconds)
        eeg = pd.read_parquet(self.eeg_path % item.eeg_id).iloc[eeg_start:eeg_start + 10000].reset_index()

        eeg = eeg.fillna(0)
        eeg.interpolate('linear', inplace=True)

        eeg['MEANEEG'] = eeg.iloc[:, 0:19].mean(axis=1)

        raw = mne.io.RawArray(eeg[self.ch_names].to_numpy().T, self.mne_info)
        raw.set_montage(mne.channels.make_standard_montage('standard_1020'))
        return raw

    def load_numpy(self, eeg: np.ndarray) -> mne.io.RawArray:
        meaneeg = np.mean(eeg, axis=1).reshape((-1, 1))
        eeg = np.concatenate((eeg, meaneeg), axis=1)

        raw = mne.io.RawArray(eeg.T, self.mne_info)
        raw.set_montage(mne.channels.make_standard_montage('standard_1020'))
        return raw

    @staticmethod
    def notch_filter(raw):
        psd = raw.compute_psd(fmax=100).get_data().sum(axis=0)

        logpsd = np.log(psd)
        logpsd_filt = scipy.signal.medfilt(logpsd, kernel_size=15)

        psd_diff = (logpsd - logpsd_filt)[100:400]
        psd_diff *= (psd_diff > max(0.5, np.std(psd_diff)))

        peaks_indices = scipy.signal.argrelextrema(psd_diff, np.greater)[0]

        freqs = [50, 60]
        for idx_grp in ((21, ), (63, 64, 65), (104, 105)):
            freqs += [(idx_grp[0] + 100) / 10.24 for idx in peaks_indices if idx in idx_grp]

        if len(freqs):
            print(f'\tFilter {freqs}')
            raw.notch_filter(freqs=freqs)

    def ecg_filter(self, raw):
        signal, average_pulse, disorder, ecg_events = self.__get_median_signal(raw, 'EKG')
        if ((disorder < 0.5) & (average_pulse > 35) & (average_pulse < 120)) or self.__is_ecg_corr_test(signal):
            # Good signal in EKG channel or correlation with etalon is high enough
            # We don't need to use MEAN channel then
            # First variant: using projection
            projs_ecg, _ = mne.preprocessing.compute_proj_ecg(raw, ch_name='EKG', n_eeg=1, reject=None)
            raw.add_proj(projs_ecg)
            # Second variant: using Ridge
            # self.__filter_signal_out(raw, signal, ecg_events)
            print('\tProjection based on EKG channel')
        else:
            # There's no signal in EKG chanel. We proceed with MEAN channel
            signal, average_pulse, disorder, ecg_events = self.__get_median_signal(raw, 'MEANEEG')
            if (average_pulse > 35) & (average_pulse < 120) & self.__is_ecg_corr_test(signal):
                # Mean channel contains signal
                # projs_ecg, _ =
                # mne.preprocessing.compute_proj_ecg(raw, ch_name='MEANEEG', n_eeg=n_eeg, reject=None, average=True)
                # raw.add_proj(projs_ecg)
                # Second variant: using Ridge
                self.__filter_signal_out(raw, signal, ecg_events)
                print('\tRidge based on MEANEEG channel')

    def eog_filter(self, raw):
        pass

    @staticmethod
    def __is_ecg_corr_test(signal, threshold=0.45):
        etalon = np.array([0.01198414, 0.01650751, 0.02251491, 0.02034455, 0.02078915,
                           0.01342222, 0.01757721, 0.01447701, 0.01569031, 0.00615449,
                           0.01630615, 0.01049864, 0.02659327, 0.01738447, 0.02292795,
                           0.0235114, 0.02530844, 0.02944412, 0.02692926, 0.03449348,
                           0.03293149, 0.04015765, 0.03912913, 0.04271125, 0.04108822,
                           0.05639384, 0.06663575, 0.0724415, 0.0715837, 0.08172248,
                           0.08495083, 0.08096384, 0.09680349, 0.1008174, 0.09134829,
                           0.11695058, 0.12228841, 0.11792368, 0.11723233, 0.13126412,
                           0.13099752, 0.12684739, 0.12724912, 0.14059116, 0.1389369,
                           0.14040862, 0.1375785, 0.15396994, 0.14902669, 0.16242614,
                           0.16392812, 0.15797727, 0.14972603, 0.15001038, 0.15653723,
                           0.14260301, 0.14974949, 0.16056372, 0.15133567, 0.14711525,
                           0.12549062, 0.10201, 0.08780026, 0.07845474, 0.07731387,
                           0.0876969, 0.08506632, 0.06840363, 0.04926966, 0.00533796,
                           -0.03807991, -0.06337805, -0.07291452, -0.04549089, -0.01277484,
                           0.0342271, 0.06106231, 0.06414278, 0.06441666, 0.05380654,
                           0.05201256, 0.10131522, 0.17909248, 0.25510424, 0.32564932,
                           0.34569502, 0.30403617, 0.23698051, 0.16294642, 0.12649605,
                           0.16298021, 0.26724774, 0.39886117, 0.45625734, 0.37875092,
                           0.09781716, -0.35703856, -0.9283634,  -1.5145929,  -1.9564718,
                           -2.1166213,  -1.9383564,  -1.4688296,  -0.82506067, -0.16937098,
                           0.3633424, 0.6887187, 0.7861108, 0.6793752, 0.4723144,
                           0.27672613, 0.13484937, 0.10322595, 0.15814756, 0.2348561,
                           0.3122613, 0.3306772, 0.2927965, 0.21753958, 0.1470311,
                           0.10745118, 0.0937568, 0.11452933, 0.15882938, 0.1824322,
                           0.17167017, 0.15484385, 0.09731194, 0.0593446, 0.02862671,
                           0.00779806, 0.00520544, 0.02598119, 0.01592493, 0.00377435,
                           -0.03732133, -0.07271291, -0.11800172, -0.16048467, -0.17587438,
                           -0.19520837, -0.19096963, -0.20627919, -0.22692238, -0.2473586,
                           -0.27966788, -0.3038011,  -0.32113287, -0.33063385, -0.32219964,
                           -0.32297856, -0.3185449,  -0.29508018, -0.28927943, -0.2963511,
                           -0.28586987, -0.25626022, -0.24873321, -0.21978779, -0.19830255,
                           -0.17597672, -0.15425229, -0.13469057, -0.11030912, -0.09690917,
                           -0.07748462, -0.05792586, -0.03809271, -0.01419599, 0.00336543,
                           0.00842424, 0.02865912, 0.03102778, 0.05141386, 0.05061839,
                           0.05767964, 0.0606734, 0.06780827, 0.0666832, 0.06668562,
                           0.08277746, 0.0687783, 0.07603061, 0.08948676, 0.08083023,
                           0.09215994, 0.09230389, 0.10183779, 0.10660672, 0.09427073,
                           0.11143697, 0.1033275, 0.10022035, 0.09948925, 0.11144029,
                           0.12270326, 0.11381514, 0.11932542, 0.11410742, 0.10787952])
        level = abs(np.sum(signal * etalon) / np.sqrt(np.sum(np.power(signal, 2)) * np.sum(np.power(etalon, 2))))
        return level > threshold

    @staticmethod
    def __get_median_signal(raw, ch_name):
        ecg_events, ch_ecg, average_pulse = mne.preprocessing.find_ecg_events(raw, ch_name=ch_name, tstart=0.5)
        x = raw.get_data(ch_name)[0]
        signal = []
        for ev in ecg_events:
            c = ev[0]
            if (c > 100) & (c < 9900):
                xx = x[c - 100: c + 100]
                xx -= np.mean(xx)
                signal.append(xx)
        signal = np.array(signal)
        signal = np.median(signal, axis=0)

        if (average_pulse == 0) or len(ecg_events) < 3:
            return signal, average_pulse, 2., ecg_events

        disorder = np.mean(np.std(signal, axis=0)) / np.std(signal)
        signal /= np.std(signal)

        return signal, average_pulse, disorder, ecg_events

    @staticmethod
    def __filter_signal_out(raw, signal, ecg_events):
        a = np.zeros((10000, len(ecg_events)))
        for n, ev in enumerate(ecg_events):
            c = ev[0]
            t1 = max(0, c - 100)
            t2 = min(10000, c + 100)
            a[t1: t2, n] = signal[100 + t1 - c: len(signal) + t2 - c - 100]

        eeg_data = raw.get_data('eeg')
        eeg_data_filtered = raw.copy().filter(l_freq=2.5, h_freq=20, picks='eeg').get_data('eeg')

        for ch in range(eeg_data.shape[0]):
            reg = Ridge(alpha=0.01)
            reg.fit(a, eeg_data_filtered[ch])
            ecg_data = reg.predict(a)
            raw._data[ch] = eeg_data[ch] - ecg_data

    def process(self, raw):
        self.notch_filter(raw)
        raw.filter(l_freq=0.5, h_freq=20, picks=['eeg', 'ecg'])
        self.ecg_filter(raw)
        # raw.resample(sfreq=40)

    @staticmethod
    def get_spectrogram(raw):
        sfreq = raw.info['sfreq']
        x = raw.get_data('eeg')
        pad_width = int((2 ** np.ceil(np.log(x.shape[1]) / np.log(2)) - x.shape[1]) / 2)
        x = np.pad(x, ((0, 0), (pad_width, pad_width)))
        f, t, sg = scipy.signal.stft(x, fs=sfreq, nperseg=2 * sfreq)
        return sg

    def spectrogram_chains(self, raw):
        eeg = raw.get_data('eeg')
        sg = np.zeros((100, 300, 4), dtype='float32')

        for k in range(4):
            for kk in range(4):
                # Spectrogram
                x = eeg[self.channel_indices[k][kk]] - eeg[self.channel_indices[k][kk + 1]]
                mel_spec = librosa.feature.melspectrogram(y=x, sr=raw.info['sfreq'], hop_length=len(x) // 300,
                                                          n_fft=1024, n_mels=100, fmin=0, fmax=20, win_length=128)

                # LOG TRANSFORM
                width = (mel_spec.shape[1] // 30) * 30
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[:, :width]
                sg[:, :, k] += mel_spec_db

            # AVERAGE THE 4 MONTAGE DIFFERENCES
            sg[:, :, k] /= 4.0
        return sg

    def group_sg_channels(self, sg):
        # FP: 0, 11
        # LL: 4, 5, 6
        # LP: 1, 2, 3
        # ZZ: 8, 9, 10
        # RP: 12, 13, 14
        # RL: 15, 16, 17
        # OO: 7, 18
        return np.array([sg[grp].mean(axis=0) for grp in self.channel_groups])
