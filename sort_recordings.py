"""
Sort recorded hotword audio into training data.

Plays each file from recordings/, single keypress to classify:
  [p] positive (wake word) → train_data/wake<N>_16k.wav
  [n] negative (not wake word) → train_data/neg<N>_16k.wav
  [r] replay
  [s] skip
  [d] delete
  [q] quit

Usage: python sort_recordings.py
"""

import shutil
import subprocess
import sys
import termios
import tty
from pathlib import Path

_DIR = Path(__file__).resolve().parent
REC_DIR = _DIR / "recordings"
POS_DIR = _DIR / "train_data" / "positive"
NEG_DIR = _DIR / "train_data" / "negative"


def getch():
    """Read a single keypress."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1).lower()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def play(path):
    """Play a WAV file using afplay (macOS) or aplay (Linux)."""
    if sys.platform == "darwin":
        subprocess.run(["afplay", str(path)], check=False)
    else:
        subprocess.run(["aplay", "-q", str(path)], check=False)


def main():
    if not REC_DIR.exists():
        print("No recordings/ directory found.")
        return

    files = sorted(REC_DIR.glob("*.wav"))
    if not files:
        print("No recordings to sort.")
        return

    POS_DIR.mkdir(parents=True, exist_ok=True)
    NEG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{len(files)} recordings. [p]ositive [n]egative [r]eplay [s]kip [d]elete [q]uit\n")

    for f in files:
        print(f"  {f.name}", end="", flush=True)
        play(f)

        while True:
            ch = getch()
            if ch == "r":
                play(f)
                continue
            if ch == "p":
                shutil.move(str(f), str(POS_DIR / f.name))
                print(f" → positive")
                break
            elif ch == "n":
                shutil.move(str(f), str(NEG_DIR / f.name))
                print(f" → negative")
                break
            elif ch == "s":
                print(" — skipped")
                break
            elif ch == "d":
                f.unlink()
                print(" — deleted")
                break
            elif ch == "q":
                print("\nQuit.")
                return

    print("\nDone.")


if __name__ == "__main__":
    main()
