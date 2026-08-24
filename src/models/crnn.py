import torch
import torch.nn as nn


class CRNN(nn.Module):
    """CNN + GRU model for temporal audio classification."""

    def __init__(self, num_classes=10):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.gru = nn.GRU(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.cnn(x)

        # Average over the frequency dimension.
        x = x.mean(dim=2)

        # Convert to (batch, time, features).
        x = x.transpose(1, 2)

        x, _ = self.gru(x)

        # Use the final temporal representation.
        x = x[:, -1, :]

        return self.classifier(x)