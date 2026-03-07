"""
Evaluate wake word model against test recordings.

Usage: python eval.py [--dir test_data] [--threshold 0.85]
"""

import argparse
import glob

import librosa
import numpy as np
import onnxruntime as ort

from embedding import EmbeddingExtractor


class Evaluator:
    def __init__(self, classifier_path="output/classifier.onnx"):
        self.extractor = EmbeddingExtractor()
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.classifier = ort.InferenceSession(classifier_path, opts)

    def predict(self, audio_float, sr=16000, n_frames=16):
        """Run full pipeline on float32 audio, return max confidence across sliding windows."""
        window_samples = int(2.0 * sr)

        best = 0.0
        step = int(0.08 * sr)
        for start in range(0, max(len(audio_float) - window_samples + 1, 1), step):
            clip = audio_float[start:start + window_samples]
            if len(clip) < window_samples:
                clip = np.pad(clip, (0, window_samples - len(clip)))

            emb = self.extractor.extract_fixed(clip, sr, n_frames)
            prob = self.classifier.run(None, {'embeddings': emb[None, :]})[0].item()
            best = max(best, prob)

        return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="test_data")
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()

    ev = Evaluator()
    tp = fp = fn = tn = 0

    for label, pattern in [("POS", f"{args.dir}/pos*_16k*.wav"), ("NEG", f"{args.dir}/neg*_16k*.wav")]:
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
