import torch
import torch.nn as nn

from transformers import (
    ASTForAudioClassification,
    WavLMForSequenceClassification,
    HubertModel,
)


class ASTClassifier(nn.Module):
    """AST-based audio classifier."""

    def __init__(self, num_classes=10):
        super().__init__()

        self.model = ASTForAudioClassification.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

    def forward(self, x):
        return self.model(x).logits


class WavLMClassifier(nn.Module):
    """WavLM-based audio classifier."""

    def __init__(self, num_classes=10):
        super().__init__()

        self.wavlm = WavLMForSequenceClassification.from_pretrained(
            "microsoft/wavlm-base-plus",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

    def forward(self, x):
        return self.wavlm(x).logits


class HubertClassifier(nn.Module):
    """HuBERT-based audio classifier."""

    def __init__(self, num_classes=10):
        super().__init__()

        self.hubert = HubertModel.from_pretrained(
            "facebook/hubert-base-ls960"
        )

        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, num_classes)

        # Freeze the first two encoder layers.
        for param in self.hubert.encoder.layers[:2].parameters():
            param.requires_grad = False

    def forward(self, x):
        outputs = self.hubert(x)

        hidden_states = outputs.last_hidden_state

        pooled = hidden_states.mean(dim=1)

        return self.classifier(
            self.dropout(pooled)
        )