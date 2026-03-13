"""
Generate embedding_data.json for the embeddings visualizer.

Embeds samples, runs them through the trained classifier to extract
conv features and attention weights.

Usage: python viz_embeddings.py [--max-pos 10] [--max-neg 10]
"""

import argparse
import glob
import json
from pathlib import Path

import librosa
import numpy as np
import torch
import yaml

from augment import pad_or_trim
from embedding import EmbeddingExtractor
from model import WakeWordClassifier

_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(_DIR / "output" / "embedding_data.json"))
    parser.add_argument("--max-pos", type=int, default=10)
    parser.add_argument("--max-neg", type=int, default=10)
    args = parser.parse_args()

    # Load config and model
    with open(_DIR / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    mc = cfg["model"]

    model = WakeWordClassifier(
        n_frames=16, embedding_dim=96,
        hidden=mc.get("hidden", 64), dropout=mc.get("dropout", 0.3),
    )
    model.load_state_dict(torch.load(_DIR / "output" / "best_model.pt", weights_only=True))
    model.eval()

    extractor = EmbeddingExtractor()

    sources = [
        ("test_data/positive", "real_pos", args.max_pos),
        ("test_data/negative", "user_neg", args.max_neg),
    ]

    all_samples = []
    for directory, label, max_files in sources:
        path = _DIR / directory
        if not path.exists():
            continue
        files = sorted(glob.glob(str(path / "*.wav")))
        if max_files > 0:
            files = files[:max_files]

        for f in files:
            audio, _ = librosa.load(f, sr=16000)
            audio = pad_or_trim(audio, 32000)
            emb = extractor.extract_fixed(audio, 16000, n_frames=16)

            # Run through model, extract intermediates
            x = torch.from_numpy(emb).unsqueeze(0)  # (1, 16, 96)
            with torch.no_grad():
                conv_out = model.conv(x.transpose(1, 2)).transpose(1, 2)  # (1, 16, hidden)
                attn_logits = model.pool_attn(conv_out)  # (1, 16, 1)
                attn_weights = torch.softmax(attn_logits, dim=1).squeeze()  # (16,)
                prob = model(x).item()

            all_samples.append({
                "label": label,
                "name": Path(f).name,
                "prob": round(prob, 4),
                "raw": emb.tolist(),
                "conv": conv_out.squeeze(0).numpy().tolist(),  # (16, hidden)
                "attn": attn_weights.numpy().tolist(),  # (16,)
            })
        print(f"  {directory}: {len(files)} samples")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_samples, f)
    print(f"\nWrote {len(all_samples)} samples to {args.output}")


if __name__ == "__main__":
    main()
