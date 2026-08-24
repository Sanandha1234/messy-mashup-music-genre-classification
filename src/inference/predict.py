import argparse

import librosa
import numpy as np
import torch

from src.models.cnn import CNN


GENRES = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]

SR = 22050
DURATION = 30
SAMPLES = SR * DURATION
N_MELS = 128


def load_audio(file_path):
    """Load and normalize an audio file."""

    audio, _ = librosa.load(
        file_path,
        sr=SR,
        mono=True,
    )

    if len(audio) > SAMPLES:
        audio = audio[:SAMPLES]
    else:
        audio = np.pad(
            audio,
            (0, SAMPLES - len(audio)),
        )

    return audio


def audio_to_mel(audio):
    """Convert audio to a normalized Mel-spectrogram."""

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SR,
        n_mels=N_MELS,
        n_fft=2048,
        hop_length=512,
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max,
    )

    mel_norm = (
        (mel_db - mel_db.mean())
        / (mel_db.std() + 1e-8)
    )

    return torch.tensor(
        mel_norm,
        dtype=torch.float32,
    ).unsqueeze(0).unsqueeze(0)


def predict(
    audio_path,
    checkpoint_path,
    device=None,
):
    """Predict the genre of an audio file."""

    device = device or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    device = torch.device(device)

    model = CNN(num_classes=len(GENRES))
    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device,
        )
    )

    model.to(device)
    model.eval()

    audio = load_audio(audio_path)
    features = audio_to_mel(audio).to(device)

    with torch.no_grad():
        logits = model(features)
        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        prediction = probabilities.argmax(
            dim=1
        ).item()

    return {
        "genre": GENRES[prediction],
        "confidence": float(
            probabilities[0, prediction].item()
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Predict the genre of an audio file."
    )

    parser.add_argument(
        "--audio",
        required=True,
        help="Path to the audio file.",
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the trained CNN checkpoint.",
    )

    args = parser.parse_args()

    result = predict(
        args.audio,
        args.checkpoint,
    )

    print(f"Predicted genre: {result['genre']}")
    print(
        f"Confidence: {result['confidence']:.4f}"
    )


if __name__ == "__main__":
    main()