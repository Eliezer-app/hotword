"""
Evaluate wake word model against test recordings.

Usage: python eval.py [--dir test_data] [--threshold 0.85]
"""

import argparse
import glob

import librosa
import numpy as np
import onnxruntime as ort

from augment import pad_or_trim
from embedding import EmbeddingExtractor


class Evaluator:
    def __init__(self, classifier_path="output/classifier.onnx"):
        self.extractor = EmbeddingExtractor()
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.classifier = ort.InferenceSession(classifier_path, opts)

    def predict(self, audio_float, sr=16000, n_frames=16):
        """Run full pipeline on float32 audio, return confidence.
        Center-pads to 2s for embedding."""
        embed_samples = int(2.0 * sr)
        audio = pad_or_trim(audio_float, embed_samples)
        emb = self.extractor.extract_fixed(audio, sr, n_frames)
        return self.classifier.run(None, {'embeddings': emb[None, :]})[0].item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="test_data")
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()

    ev = Evaluator()
    tp = fp = fn = tn = 0

    for label, pattern in [("POS", f"{args.dir}/positive/*.wav"), ("NEG", f"{args.dir}/negative/*.wav")]:
        files = sorted(glob.glob(pattern))
        for f in files:
            audio, _ = librosa.load(f, sr=16000)
            prob = ev.predict(audio)
            hit = prob > args.threshold
            tag = "DETECT" if hit else "reject"
            print(f"  {label} {f:30s}  conf={prob:.3f}  {tag}")

            if label == "POS" and hit: tp += 1
            elif label == "POS" and not hit: fn += 1
            elif label == "NEG" and hit: fp += 1
            else: tn += 1

    total = tp + fp + fn + tn
    if total:
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        print(f"\n  TP={tp} FP={fp} FN={fn} TN={tn}")
        print(f"  Precision={prec:.3f} Recall={rec:.3f} F1={f1:.3f}")


if __name__ == "__main__":
    main()
