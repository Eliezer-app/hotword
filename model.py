"""
Wake word classifier head on top of Google's speech embedding.

The embedding model (frozen ONNX) produces 96-dim features per frame.
For 2s audio we get 16 frames -> (16, 96) input.

Architecture: positional encoding + self-attention + classifier.
The model learns the temporal pattern of the wake word — which frames
carry which phonemes, and whether the full sequence is present.
"""

import torch
import torch.nn as nn


class WakeWordClassifier(nn.Module):
    """Temporal classifier with positional encoding and self-attention."""

    def __init__(self, n_frames=16, embedding_dim=96, hidden=128, dropout=0.3):
        super().__init__()
        self.n_frames = n_frames
        self.embedding_dim = embedding_dim

        # Positional encoding — so the model knows frame position
        self.pos_embed = nn.Parameter(torch.randn(1, n_frames, embedding_dim) * 0.02)

        # Self-attention: frames attend to each other
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=4, dropout=dropout, batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(embedding_dim)

        # Pooling attention: which frames matter for classification
        self.pool_attn = nn.Sequential(
            nn.Linear(embedding_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

        # Classifier on pooled representation
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (batch, n_frames, embedding_dim)
        x = x + self.pos_embed

        # Self-attention with residual
        attn_out, _ = self.self_attn(x, x, x)
        x = self.attn_norm(x + attn_out)

        # Attention pooling
        weights = torch.softmax(self.pool_attn(x), dim=1)
        pooled = (x * weights).sum(dim=1)  # (batch, embedding_dim)

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
