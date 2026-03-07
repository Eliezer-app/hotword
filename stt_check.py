"""
Transcribe wake word samples using pywhispercpp for quality checking.

Requires the stt venv (pywhispercpp). Auto-detects ../stt/.venv or uses current python.

Usage: python stt_check.py [files or directories...]

Examples:
    python stt_check.py train_data/positive/pos1_16k.wav
    python stt_check.py train_data/negative/
    python stt_check.py train_data/positive/ train_data/negative/
"""

import os
import subprocess
import sys
import wave
from pathlib import Path

_DIR = Path(__file__).resolve().parent
_STT_PYTHON = _DIR.parent / "stt" / ".venv" / "bin" / "python"

if _STT_PYTHON.exists() and Path(sys.executable).resolve() != _STT_PYTHON.resolve():
    os.execv(str(_STT_PYTHON), [str(_STT_PYTHON)] + sys.argv)

import numpy as np
from pywhispercpp.model import Model

INITIAL_PROMPT = (
    "Hey Eliezer. Hello Eliezer. Eliezer Yudkowsky is an AI researcher. "
    "Hey everyone. Hey Google. Hey Siri. Hey Alexa. "
    "Eliezer, hi. Good morning Eliezer. Yo Eliezer. "
    "Hey Elizabeth. Hey Oliver. Hey Alicia. "
    "The name is Eliezer, sometimes spelled Eliezer. "
)


def load_wav(path):
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return audio, sr


def main():
    args = sys.argv[1:] or ["train_data/positive/", "train_data/negative/"]

    files = []
    for arg in args:
        p = Path(arg)
        if p.is_dir():
            files.extend(sorted(p.glob("*.wav")))
        elif p.is_file():
            files.append(p)
        else:
            print(f"Not found: {arg}", file=sys.stderr)

    if not files:
        print("No WAV files found.", file=sys.stderr)
        sys.exit(1)

    model = Model("base.en")

    for f in files:
        audio, sr = load_wav(f)
        dur = len(audio) / sr
        segments = model.transcribe(audio, initial_prompt=INITIAL_PROMPT)
        text = " ".join(s.text.strip() for s in segments).strip()
        print(f"  {str(f):50s} ({dur:.2f}s)  {text}")


if __name__ == "__main__":
    main()
