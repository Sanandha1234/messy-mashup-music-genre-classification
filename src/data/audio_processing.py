import random

import librosa
import numpy as np


def load_audio(file_path, target_sr=22050):
    """Load an audio file at the target sample rate."""
    try:
        audio, _ = librosa.load(file_path, sr=target_sr, mono=True)
        return audio
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return np.zeros(target_sr * 30)


def estimate_tempo(audio, sr=22050):
    """Estimate the tempo of an audio signal."""
    try:
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)

        if isinstance(tempo, np.ndarray):
            tempo = tempo[0] if len(tempo) > 0 else 120

        return float(tempo) if tempo > 0 else 120.0

    except Exception:
        return 120.0


def apply_tempo_adjustment(audio, target_tempo, current_tempo):
    """Adjust audio tempo when the difference is significant."""
    target_tempo = float(target_tempo)
    current_tempo = float(current_tempo)

    if abs(current_tempo - target_tempo) < 5:
        return audio

    rate = target_tempo / current_tempo
    rate = np.clip(rate, 0.7, 1.3)

    try:
        return librosa.effects.time_stretch(audio, rate=rate)
    except Exception:
        return audio


def add_esc50_noise(
    audio,
    noise_files,
    num_noises=(1, 2),
    snr_range=(8, 25),
    sr=22050,
):
    """Add environmental noise from ESC-50 at a random SNR."""
    if not noise_files or random.random() > 0.5:
        return audio

    noisy_audio = audio.copy()
    audio_len = len(audio)

    num_to_add = random.randint(*num_noises)

    for _ in range(num_to_add):
        noise_path = random.choice(noise_files)
        noise, _ = librosa.load(noise_path, sr=sr, mono=True)

        if len(noise) < audio_len:
            start_pos = random.randint(0, audio_len - len(noise))
            noise_padded = np.zeros(audio_len)
            noise_padded[
                start_pos:start_pos + len(noise)
            ] = noise
        else:
            start_pos = random.randint(0, len(noise) - audio_len)
            noise_padded = noise[start_pos:start_pos + audio_len]

        snr_db = random.uniform(*snr_range)

        signal_power = np.mean(audio ** 2)
        noise_power = np.mean(noise_padded ** 2)

        if noise_power > 0:
            scale = np.sqrt(
                signal_power /
                (noise_power * (10 ** (snr_db / 10)))
            )
            noisy_audio += scale * noise_padded

    max_val = np.max(np.abs(noisy_audio))

    if max_val > 1.0:
        noisy_audio = noisy_audio / max_val * 0.95

    return noisy_audio