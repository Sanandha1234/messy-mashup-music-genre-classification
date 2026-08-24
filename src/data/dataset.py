import os
import random

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    """
    Dataset for robust music genre classification.

    During training, stems are randomly selected from songs
    belonging to the same genre and mixed together. ESC-50
    environmental noise can also be added as augmentation.
    """

    def __init__(
        self,
        dataframe,
        stems_path,
        noise_files=None,
        sample_rate=22050,
        duration=30,
        n_mels=128,
        n_fft=2048,
        hop_length=512,
        train=True,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.stems_path = stems_path
        self.noise_files = noise_files or []

        self.sample_rate = sample_rate
        self.duration = duration
        self.samples = sample_rate * duration

        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.train = train

        self.genres = sorted(self.df["genre"].unique())
        self.label2idx = {
            genre: idx for idx, genre in enumerate(self.genres)
        }

        self.genre_groups = {
            genre: group
            for genre, group in self.df.groupby("genre")
        }

    def __len__(self):
        return len(self.df)

    def load_audio(self, path):
        """Load an audio file and resample it."""
        try:
            audio, _ = librosa.load(
                path,
                sr=self.sample_rate,
                mono=True,
            )
            return audio
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return np.zeros(self.samples, dtype=np.float32)

    def random_stem_mix(self, genre):
        """Mix stems randomly selected from songs of the same genre."""

        group = self.genre_groups[genre]

        if len(group) >= 4:
            selected_rows = group.sample(
                n=4,
                replace=False,
            )
        else:
            selected_rows = group.sample(
                n=4,
                replace=True,
            )

        stem_names = [
            "bass.wav",
            "drums.wav",
            "other.wav",
            "vocals.wav",
        ]

        signals = []

        for (_, row), stem_name in zip(
            selected_rows.iterrows(),
            stem_names,
        ):
            song_folder = row["song_folder"]

            if not os.path.isabs(song_folder):
                song_folder = os.path.join(
                    self.stems_path,
                    genre,
                    song_folder,
                )

            stem_path = os.path.join(
                song_folder,
                stem_name,
            )

            signals.append(self.load_audio(stem_path))

        min_length = min(len(signal) for signal in signals)

        signals = [
            signal[:min_length]
            for signal in signals
        ]

        mixture = np.sum(signals, axis=0)

        max_value = np.max(np.abs(mixture))

        if max_value > 0:
            mixture = mixture / max_value

        return mixture.astype(np.float32)

    def add_noise(self, signal):
        """Add environmental noise from ESC-50."""

        if not self.noise_files:
            return signal

        noise_path = random.choice(self.noise_files)

        noise = self.load_audio(noise_path)

        if len(noise) < len(signal):
            noise = np.pad(
                noise,
                (0, len(signal) - len(noise)),
            )
        else:
            start = random.randint(
                0,
                len(noise) - len(signal),
            )
            noise = noise[
                start:start + len(signal)
            ]

        noise_level = random.uniform(0.05, 0.15)

        noisy_signal = signal + noise_level * noise

        max_value = np.max(np.abs(noisy_signal))

        if max_value > 1:
            noisy_signal = (
                noisy_signal / max_value * 0.95
            )

        return noisy_signal.astype(np.float32)

    def crop_or_pad(self, signal):
        """Crop or zero-pad audio to a fixed duration."""

        if len(signal) > self.samples:
            if self.train:
                start = random.randint(
                    0,
                    len(signal) - self.samples,
                )
            else:
                start = (
                    len(signal) - self.samples
                ) // 2

            signal = signal[
                start:start + self.samples
            ]

        elif len(signal) < self.samples:
            signal = np.pad(
                signal,
                (0, self.samples - len(signal)),
            )

        return signal.astype(np.float32)

    def to_mel_spectrogram(self, signal):
        """Convert an audio waveform to a normalized Mel-spectrogram."""

        mel = librosa.feature.melspectrogram(
            y=signal,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )

        mel_db = librosa.power_to_db(
            mel,
            ref=np.max,
        )

        mel_db = (
            mel_db - mel_db.mean()
        ) / (mel_db.std() + 1e-6)

        return mel_db.astype(np.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        genre = row["genre"]

        signal = self.random_stem_mix(genre)

        if self.train:
            signal = self.add_noise(signal)

        signal = self.crop_or_pad(signal)

        mel = self.to_mel_spectrogram(signal)

        mel = torch.tensor(
            mel,
            dtype=torch.float32,
        ).unsqueeze(0)

        label = torch.tensor(
            self.label2idx[genre],
            dtype=torch.long,
        )

        return mel, label