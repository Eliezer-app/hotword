"""
Wake word classifier head on top of Google's speech embedding.

The embedding model (frozen ONNX) produces 96-dim features per frame.
For 2s audio we get 16 frames -> 16x96 = 1536 features.
The classifier is a small MLP trained on these embeddings.
"""

import torch
import torch.nn as nn


class WakeWordClassifier(nn.Module):
    """Small MLP classifier on top of frozen speech embeddings."""

    def __init__(self, n_frames=16, embedding_dim=96, hidden=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(n_frames * embedding_dim),
            nn.Linear(n_frames * embedding_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


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
