# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this open-source project.


import librosa
import numpy as np


class FeatureExtractor:
    @staticmethod
    def get_melspectrogram(audio, sample_rate):
        melspe = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
        melspe_db = librosa.power_to_db(melspe, ref=np.max)
        # print(f'{melspe_db.shape} -> melspe_db')
        return melspe_db

    @staticmethod
    def get_hpss(audio):
        audio_harmonic, audio_percussive = librosa.effects.hpss(audio)
        print(f"{audio_percussive.shape} -> audio_percussive")
        return audio_harmonic, audio_percussive

    @staticmethod
    def get_mfcc(melspe_db):
        mfcc = librosa.feature.mfcc(S=melspe_db)
        print(f"{mfcc.shape} -> mfcc")
        return mfcc

    @staticmethod
    def get_mfcc_delta(mfcc):
        mfcc_delta = librosa.feature.delta(mfcc, width=3)
        print(f"{mfcc_delta.shape} -> mfcc_delta")
        return mfcc_delta

    @staticmethod
    def get_mfcc_delta2(mfcc):
        mfcc_delta_delta = librosa.feature.delta(mfcc, width=3, order=2)
        print(f"{mfcc_delta_delta.shape} -> mfcc_delta_delta")
        return mfcc_delta_delta

    @staticmethod
    def get_harmonic_melspe_db(audio_harmonic, sr):
        harmonic_melspe = librosa.feature.melspectrogram(audio_harmonic, sr=sr)
        harmonic_melspe_db = librosa.power_to_db(harmonic_melspe, ref=np.max)
        print(f"{harmonic_melspe_db.shape} -> harmonic_melspe_db")
        return harmonic_melspe_db

    @staticmethod
    def get_percussive_melspe_db(audio_percussive, sr):
        percussive_melspe = librosa.feature.melspectrogram(audio_percussive, sr=sr)
        percussive_melspe_db = librosa.power_to_db(percussive_melspe, ref=np.max)
        print(f"{percussive_melspe_db.shape} -> percussive_melspe_db")
        return percussive_melspe_db

    @staticmethod
    def get_chroma_cqt(audio_harmonic, sr, octave=7):
        chroma_cqt_harmonic = librosa.feature.chroma_cqt(
            y=audio_harmonic, sr=sr, n_octaves=octave
        )
        print(f"{chroma_cqt_harmonic.shape} -> chroma_cqt_harmonic")
        return chroma_cqt_harmonic

    @staticmethod
    def get_chroma_stft(audio_harmonic, sr):
        chroma_stft_harmonic = librosa.feature.chroma_stft(y=audio_harmonic, sr=sr)
        print(f"{chroma_stft_harmonic.shape} -> chroma_stft_harmonic")
        return chroma_stft_harmonic

    @staticmethod
    def get_tonnetz(audio_harmonic, sr):
        tonnetz = librosa.feature.tonnetz(y=audio_harmonic, sr=sr)
        print(f"{tonnetz.shape} -> tonnetz")
        return tonnetz

    @staticmethod
    def get_onset_strength(audio_percussive, sr):
        onset_env = librosa.onset.onset_strength(
            y=audio_percussive, aggregate=np.median, sr=sr
        )
        print(f"{onset_env.reshape(1, -1).shape} -> onset_env")
        return onset_env

    @staticmethod
    def get_tempogram(onset_env, sr):
        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
        print(f"{tempogram.shape} -> tempogram")
        return tempogram

    @staticmethod
    def get_onset_beat(onset_env, sr):
        onset_tempo, onset_beats = librosa.beat.beat_track(
            onset_envelope=onset_env, sr=sr
        )
        peaks = librosa.util.peak_pick(
            onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.5, wait=10
        )
        beats_one_hot = np.zeros(len(onset_env))
        peaks_one_hot = np.zeros(len(onset_env))
        for idx in onset_beats:
            beats_one_hot[idx] = 1
        for idx in peaks:
            peaks_one_hot[idx] = 1

        beats_one_hot = beats_one_hot.reshape(1, -1)
        peaks_one_hot = peaks_one_hot.reshape(1, -1)

        print(f"{beats_one_hot.shape} -> beats_feature")
        return beats_one_hot, peaks_one_hot
