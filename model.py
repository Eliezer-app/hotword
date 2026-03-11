"""
Wake word classifier head on top of Google's speech embedding.

The embedding model (frozen ONNX) produces 96-dim features per frame.
For 2s audio we get 16 frames -> (16, 96) input.

Architecture: conv1d for local temporal patterns + attention pooling + classifier.
"""

import torch
import torch.nn as nn


class WakeWordClassifier(nn.Module):
    """Conv1D + attention pooling classifier."""

    def __init__(self, n_frames=16, embedding_dim=96, hidden=64, dropout=0.3):
        super().__init__()
        self.n_frames = n_frames
        self.embedding_dim = embedding_dim
        self.conv = nn.Sequential(
            nn.Conv1d(embedding_dim, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pool_attn = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.Tanh(),
            nn.Linear(hidden // 2, 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (batch, n_frames, embedding_dim)
        c = self.conv(x.transpose(1, 2)).transpose(1, 2)  # (B, T, hidden)
        weights = torch.softmax(self.pool_attn(c), dim=1)
        pooled = (c * weights).sum(dim=1)
        return self.classifier(pooled).squeeze(-1)


def export_classifier_onnx(model, output_path, n_frames=16, embedding_dim=96):
    """Export classifier head to ONNX."""
    model.eval()
    dummy = torch.randn(1, n_frames, embedding_dim)
    torch.onnx.export(
        model, dummy, output_path,
        input_names=["embeddings"],
        output_names=["probability"],
        dynamic_axes={"embeddings": {0: "batch"}},
        opset_version=18,
    )
    print(f"Classifier ONNX exported to {output_path}")
