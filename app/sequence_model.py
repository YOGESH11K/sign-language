"""Shared PyTorch architecture for dynamic (sequence) sign classification.

Defined here so training, evaluation and the real-time predictor build exactly
the same model.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn

    _TORCH_OK = True
except Exception:  # pragma: no cover - torch may be unavailable
    _TORCH_OK = False


class SequenceClassifier(nn.Module):
    """Bidirectional GRU over the frame window + masked-mean pooling + MLP head."""

    def __init__(self, input_dim: int, n_classes: int, hidden: int = 128):
        if not _TORCH_OK:
            raise ImportError("PyTorch is required for the sequence model")
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        out, _ = self.gru(x)
        m = mask.unsqueeze(-1)
        pooled = (out * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return self.head(pooled)


def torch_available() -> bool:
    return _TORCH_OK