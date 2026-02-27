"""
Wake word detection daemon.

Uses Google's pretrained speech embedding + trained classifier.
Continuously streams from the microphone and prints to stdout on detection.

Usage: python detect.py [--config output/config.yaml] [--debug]
"""

import argparse
import signal
import sys
import time

import numpy as np
import onnxruntime as ort
import sounddevice as sd
import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


class Detector:
    def __init__(self, melspec_path="models/melspectrogram.onnx",
                 embed_path="models/embedding_model.onnx",
                 classifier_path="output/classifier.onnx"):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.melspec = ort.InferenceSession(melspec_path, opts)
        self.embed = ort.InferenceSession(embed_path, opts)
        self.classifier = ort.InferenceSession(classifier_path, opts)

    def predict(self, audio_float, n_frames=16):
        """Run full pipeline on float32 audio buffer, return probability."""
        audio_int16 = (audio_float * 32767).clip(-32768, 32767).astype(np.int16)

        # Melspectrogram
        x = audio_int16.astype(np.float32)[None, :]
        mel = self.melspec.run(None, {'input': x})[0]
        mel = np.squeeze(mel) / 10 + 2

        # Embeddings: slide 76-frame windows, step 8
        windows = []
        for i in range(0, mel.shape[0], 8):
            w = mel[i:i+76]
            if w.shape[0] == 76:
                windows.append(w)
        if not windows:
            return 0.0

        batch = np.array(windows, dtype=np.float32)[:, :, :, None]
        emb = self.embed.run(None, {'input_1': batch})[0].squeeze(axis=(1, 2))

        # Pad/trim to n_frames
        if emb.shape[0] >= n_frames:
            emb = emb[:n_frames]
        else:
            emb = np.vstack([emb, np.zeros((n_frames - emb.shape[0], 96), dtype=np.float32)])

        # Classify
        prob = self.classifier.run(None, {'embeddings': emb[None, :]})[0].item()
        return prob


def main():
    parser = argparse.ArgumentParser(description="Wake word detector")
    parser.add_argument("--config", default="output/config.yaml")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ac = cfg["audio"]
    dc = cfg["detection"]

    detector = Detector()

    sr = ac["sample_rate"]
    window_samples = int(ac["window_sec"] * sr)
    step_samples = int(dc["step_ms"] / 1000 * sr)

    audio_buffer = np.zeros(window_samples, dtype=np.float32)
    consecutive = 0
    last_trigger = 0.0

    signal.signal(signal.SIGINT, lambda *_: (print("\nStopped."), sys.exit(0)))

    print(f"Listening for wake word (threshold={dc['threshold']}, "
          f"smoothing={dc['smoothing_window']}, step={dc['step_ms']}ms)")
    print("Press Ctrl+C to stop.\n")

    try:
        with sd.InputStream(samplerate=sr, channels=1, dtype="float32") as stream:
            while True:
                chunk, _ = stream.read(step_samples)
                chunk = chunk[:, 0]

                audio_buffer = np.roll(audio_buffer, -len(chunk))
                audio_buffer[-len(chunk):] = chunk

                prob = detector.predict(audio_buffer)

                if args.debug:
                    bar = "#" * int(prob * 40)
                    print(f"\r  conf: {prob:.3f} [{bar:<40s}]", end="", flush=True)

                if prob > dc["threshold"]:
                    consecutive += 1
                else:
                    consecutive = 0

                now = time.time()
                if (consecutive >= dc["smoothing_window"]
                        and now - last_trigger > dc["cooldown_sec"]):
                    print(f"DETECTED (confidence: {prob:.3f})")
                    consecutive = 0
                    last_trigger = now
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
