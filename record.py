"""
Record wake word samples.

Usage: python record.py [--output-dir samples] [--prefix wake] [--duration 2.5]

Records one sample at a time. Press Enter to start, Enter to stop (or waits
for max duration). Saves as 16kHz mono WAV.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf


def get_next_index(output_dir, prefix):
    """Find the next available index for naming."""
    existing = sorted(output_dir.glob(f"{prefix}*_16k.wav"))
    if not existing:
        return 1
    # Parse the highest number from existing files
    nums = []
    for f in existing:
        name = f.stem.replace("_16k", "")
        try:
            nums.append(int(name.replace(prefix, "")))
        except ValueError:
            pass
    return max(nums, default=0) + 1


def record_sample(sr, duration):
    """Record for a fixed duration."""
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    return audio[:, 0]


def main():
    parser = argparse.ArgumentParser(description="Record wake word samples")
    parser.add_argument("--output-dir", default=".", help="Output directory")
    parser.add_argument("--prefix", default="wake", help="Filename prefix")
    parser.add_argument("--duration", type=float, default=3.0, help="Max seconds per recording")
    parser.add_argument("--sr", type=int, default=16000, help="Sample rate")
    parser.add_argument("-n", type=int, default=0, help="Number of samples to record (0 = until quit)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    idx = get_next_index(output_dir, args.prefix)
    count = 0

    print(f"Recording wake word samples ({args.sr}Hz, max {args.duration}s each)")
    print(f"Saving to: {output_dir}/")
    print()

    while True:
        if args.n > 0 and count >= args.n:
            break

        print(f"  [{idx}] Press Enter to START recording (or 'q' to quit): ", end="", flush=True)
        line = input()
        if line.strip().lower() == "q":
            break

        print(f"  [{idx}] Recording for {args.duration}s...", flush=True)
        audio = record_sample(args.sr, args.duration)

        if audio is None or len(audio) < args.sr * 0.3:
            print("  Too short, skipping.")
            continue

        filename = output_dir / f"{args.prefix}{idx}_16k.wav"
        sf.write(str(filename), audio, args.sr)
        duration = len(audio) / args.sr
        print(f"  Saved {filename} ({duration:.2f}s)")

        # Playback
        print(f"  Playing back...", end="", flush=True)
        sd.play(audio, args.sr)
        sd.wait()
        print(" done.")
        print()

        idx += 1
        count += 1

    print(f"\nRecorded {count} sample(s).")


if __name__ == "__main__":
    main()
