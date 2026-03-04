"""
Wake word detection daemon.

Uses Google's pretrained speech embedding + trained classifier.
Continuously streams from the microphone and prints to stdout on detection.

Usage:
  python detect.py [--debug]                       # mic (default)
  python detect.py --audio-source /tmp/audio.sock   # unix socket (16kHz mono float32 PCM)
"""

import argparse
import signal
import socket
import sys
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort
import yaml

_DIR = Path(__file__).resolve().parent


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


class Detector:
    def __init__(self, melspec_path=None, embed_path=None, classifier_path=None):
        melspec_path = melspec_path or _DIR / "models" / "melspectrogram.onnx"
        embed_path = embed_path or _DIR / "models" / "embedding_model.onnx"
        classifier_path = classifier_path or _DIR / "output" / "classifier.onnx"
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


def audio_from_mic(sr, step_samples):
    """Yield audio chunks from the microphone."""
    import sounddevice as sd
    with sd.InputStream(samplerate=sr, channels=1, dtype="float32") as stream:
        while True:
            chunk, _ = stream.read(step_samples)
            yield chunk[:, 0]


def audio_from_socket(sock_path, step_samples):
    """Yield audio chunks from a unix socket (float32 PCM)."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(sock_path)
    step_bytes = step_samples * 4  # float32 = 4 bytes
    buf = b""
    try:
        while True:
            while len(buf) < step_bytes:
                data = sock.recv(step_bytes - len(buf))
                if not data:
                    return
                buf += data
            yield np.frombuffer(buf[:step_bytes], dtype=np.float32)
            buf = buf[step_bytes:]
    finally:
        sock.close()


class Recorder:
    """Records audio for hotword hits and near-misses.

    Saves the 2s audio window at two moments:
    - hit: when detection fires
    - near: when score peaked above near_threshold but detection didn't fire

    Uses a simple state machine: idle → tracking → (save or discard) → idle
    After saving (hit or near), suppresses for suppress_sec to avoid duplicates.
    """

    def __init__(self, rec_dir, near_threshold, sr, suppress_sec=4.0):
        self.rec_dir = Path(rec_dir)
        self.rec_dir.mkdir(exist_ok=True)
        self.near_threshold = near_threshold
        self.sr = sr
        self.suppress_sec = suppress_sec
        self.suppress_until = 0
        self.peak = 0.0
        self.peak_audio = None

    def update(self, prob, audio_buffer):
        """Call every step with current score and audio."""
        now = time.time()
        if now < self.suppress_until:
            return

        if prob > self.near_threshold:
            if prob > self.peak:
                self.peak = prob
                self.peak_audio = audio_buffer.copy()
        elif self.peak > 0:
            # Score dropped — save peak as near-miss
            self._save("near", self.peak, self.peak_audio)
            self._suppress()

    def mark_hit(self, prob, audio_buffer):
        """Call when detection fires."""
        self._save("hit", prob, audio_buffer)
        self._suppress()

    def _suppress(self):
        self.suppress_until = time.time() + self.suppress_sec
        self.peak = 0.0
        self.peak_audio = None

    def _save(self, label, prob, audio):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.rec_dir / f"{label}_{prob:.3f}_{ts}.wav"
        audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
        with wave.open(str(path), 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sr)
            wf.writeframes(audio_int16.tobytes())
        self.hit = False


def main():
    parser = argparse.ArgumentParser(description="Wake word detector")
    parser.add_argument("--config", default=str(_DIR / "config.yaml"))
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--record", action="store_true",
                        help="Save WAV files for hits and near-misses")
    parser.add_argument("--audio-source", default="mic",
                        help="'mic' (default) or path to unix socket")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ac = cfg["audio"]
    dc = cfg["detection"]

    detector = Detector()
    print("Hotword: ready", file=sys.stderr, flush=True)

    sr = ac["sample_rate"]
    window_samples = int(ac["window_sec"] * sr)
    step_samples = int(dc["step_ms"] / 1000 * sr)

    audio_buffer = np.zeros(window_samples, dtype=np.float32)
    consecutive = 0
    armed = True
    cooldown_until = 0

    recorder = None
    if args.record:
        near_threshold = dc["threshold"] * 0.6
        recorder = Recorder(_DIR / "recordings", near_threshold, sr)
        print(f"Recording to {recorder.rec_dir} (hits + near>{near_threshold:.2f})",
              file=sys.stderr, flush=True)

    running = True
    def handle_signal(*_):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    source = args.audio_source
    if source == "mic":
        print(f"Audio: microphone")
        chunks = audio_from_mic(sr, step_samples)
    else:
        print(f"Audio: {source}")
        chunks = audio_from_socket(source, step_samples)

    print(f"Listening for wake word (threshold={dc['threshold']}, "
          f"smoothing={dc['smoothing_window']}, step={dc['step_ms']}ms)")
    print("Press Ctrl+C to stop.\n")

    for chunk in chunks:
        if not running:
            break
        audio_buffer = np.roll(audio_buffer, -len(chunk))
        audio_buffer[-len(chunk):] = chunk

        prob = detector.predict(audio_buffer)

        if args.debug:
            bar = "#" * int(prob * 40)
            print(f"\r  conf: {prob:.3f} [{bar:<40s}]", end="", flush=True)

        now = time.time()
        if now < cooldown_until:
            continue

        if recorder:
            recorder.update(prob, audio_buffer)

        if prob > dc["threshold"]:
            consecutive += 1
        else:
            consecutive = 0
            armed = True

        if consecutive >= dc["smoothing_window"] and armed:
            print(f"DETECTED (confidence: {prob:.3f})", flush=True)
            if recorder:
                recorder.mark_hit(prob, audio_buffer)
            consecutive = 0
            armed = False
            cooldown_until = now + dc.get("cooldown_sec", 2.0)

    print("\nStopped.")


if __name__ == "__main__":
    main()
