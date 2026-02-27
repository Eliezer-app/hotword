"""
Audio augmentation pipeline for wake word training.

Fast numpy-based augmentations (no librosa phase vocoder).
"""

import numpy as np
from scipy.signal import resample_poly
from scipy.interpolate import interp1d


def time_stretch(audio, rate):
    """Stretch/compress time via linear interpolation. Fast."""
    indices = np.arange(0, len(audio), rate)
    indices = indices[indices < len(audio) - 1]
    interp = interp1d(np.arange(len(audio)), audio)
    return interp(indices).astype(np.float32)


def pitch_shift(audio, semitones):
    """Shift pitch by resampling. Fast approximation."""
    ratio = 2.0 ** (semitones / 12.0)
    # Resample to change pitch, then stretch back to original length
    up = max(int(round(ratio * 100)), 1)
    down = 100
    shifted = resample_poly(audio, up, down).astype(np.float32)
    # Stretch back to original length
    if len(shifted) == len(audio):
        return shifted
    indices = np.linspace(0, len(shifted) - 1, len(audio))
    interp = interp1d(np.arange(len(shifted)), shifted)
    return interp(indices).astype(np.float32)


def add_noise(audio, snr_db):
    """Add Gaussian noise at a given SNR."""
    signal_power = np.mean(audio ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.randn(len(audio)) * np.sqrt(max(noise_power, 1e-10))
    return (audio + noise).astype(np.float32)


def random_gain(audio, min_db=-6, max_db=6):
    """Apply random gain in dB."""
    db = np.random.uniform(min_db, max_db)
    return audio * (10 ** (db / 20))


def time_shift(audio, max_frac=0.1):
    """Shift audio left/right, zero-pad the gap."""
    shift = int(np.random.uniform(-max_frac, max_frac) * len(audio))
    result = np.zeros_like(audio)
    if shift > 0:
        result[shift:] = audio[:-shift]
    elif shift < 0:
        result[:shift] = audio[-shift:]
    else:
        result = audio.copy()
    return result


def speed_perturb(audio, factor):
    """Change speed (pitch + tempo together). Very fast."""
    indices = np.linspace(0, len(audio) - 1, int(len(audio) / factor))
    indices = np.clip(indices, 0, len(audio) - 1)
    interp = interp1d(np.arange(len(audio)), audio)
    return interp(indices).astype(np.float32)


def augment_one(audio, sr):
    """Apply a random combination of augmentations to one sample."""
    aug = audio.copy()

    if np.random.random() < 0.5:
        rate = np.random.uniform(0.9, 1.1)
        aug = time_stretch(aug, rate)

    if np.random.random() < 0.5:
        semitones = np.random.uniform(-2, 2)
        aug = pitch_shift(aug, semitones)

    if np.random.random() < 0.3:
        factor = np.random.uniform(0.9, 1.1)
        aug = speed_perturb(aug, factor)

    if np.random.random() < 0.7:
        snr = np.random.uniform(10, 30)
        aug = add_noise(aug, snr)

    if np.random.random() < 0.7:
        aug = random_gain(aug, min_db=-6, max_db=6)

    if np.random.random() < 0.3:
        aug = time_shift(aug, max_frac=0.1)

    return aug


def pad_or_trim(audio, target_len):
    """Pad with zeros or trim to exact length."""
    if len(audio) >= target_len:
        return audio[:target_len]
    return np.pad(audio, (0, target_len - len(audio)))
