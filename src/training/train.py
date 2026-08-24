import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import AudioDataset
from src.models.cnn import CNN
from src.models.crnn import CRNN
from src.training.train_utils import train_one_epoch, evaluate


def build_model(model_name, num_classes):
    """Create the requested model."""

    if model_name == "cnn":
        return CNN(num_classes=num_classes)

    if model_name == "crnn":
        return CRNN(num_classes=num_classes)

    raise ValueError(f"Unsupported model: {model_name}")


def train(
    model_name="cnn",
    num_classes=10,
    epochs=15,
    batch_size=16,
    learning_rate=1e-3,
    device=None,
):
    """Train a CNN or CRNN model."""

    device = device or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    device = torch.device(device)

    print(f"Using device: {device}")
    print(f"Model: {model_name}")

    # Dataset construction is kept separate so that the
    # Kaggle-specific paths do not become part of the model code.
    raise NotImplementedError(
        "Connect AudioDataset to your local train/validation "
        "dataframes before running this script."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train music genre classification models."
    )

    parser.add_argument(
        "--model",
        choices=["cnn", "crnn"],
        default="cnn",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )

    args = parser.parse_args()

    train(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()