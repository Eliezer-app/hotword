"""
Wake word training pipeline using Google's speech embedding model.

Uses frozen pretrained embedding (from openWakeWord) as feature extractor,
trains a small MLP classifier on top.

Usage: python train.py [--config config.yaml]
"""

import argparse
import glob
import tarfile
import urllib.request
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

from augment import augment_one, pad_or_trim
from model import WakeWordClassifier, export_classifier_onnx


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


# --- Embedding extraction using Google's pretrained models ---

class EmbeddingExtractor:
    """Extract speech embeddings using Google's pretrained ONNX models."""

    def __init__(self, melspec_path="models/melspectrogram.onnx",
                 embedding_path="models/embedding_model.onnx"):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.melspec_sess = ort.InferenceSession(melspec_path, opts)
        self.embed_sess = ort.InferenceSession(embedding_path, opts)

    def extract(self, audio_int16):
        """Extract embeddings from int16 audio.

        Args:
            audio_int16: int16 PCM audio array, 16kHz

        Returns:
            np.ndarray of shape (n_frames, 96)
        """
        # Compute melspectrogram
        x = audio_int16.astype(np.float32)[None, :]  # (1, samples)
        mel = self.melspec_sess.run(None, {'input': x})[0]  # (1, 1, time, 32)
        mel = np.squeeze(mel)  # (time, 32)
        mel = mel / 10 + 2  # transform to match Google's TF implementation

        # Slide 76-frame windows, step 8
        windows = []
        for i in range(0, mel.shape[0], 8):
            window = mel[i:i+76]
            if window.shape[0] == 76:
                windows.append(window)

        if not windows:
            return np.zeros((1, 96), dtype=np.float32)

        batch = np.array(windows, dtype=np.float32)[:, :, :, None]  # (N, 76, 32, 1)
        embeddings = self.embed_sess.run(None, {'input_1': batch})[0]  # (N, 1, 1, 96)
        embeddings = embeddings.squeeze(axis=(1, 2))  # (N, 96)
        return embeddings

    def extract_fixed(self, audio_int16, n_frames=16):
        """Extract embeddings padded/trimmed to fixed frame count."""
        emb = self.extract(audio_int16)
        if emb.shape[0] >= n_frames:
            return emb[:n_frames]
        else:
            pad = np.zeros((n_frames - emb.shape[0], 96), dtype=np.float32)
            return np.vstack([emb, pad])


def float_to_int16(audio_float):
    """Convert float32 audio [-1, 1] to int16."""
    return (audio_float * 32767).clip(-32768, 32767).astype(np.int16)


# --- Data loading ---

def load_wavs(pattern, sr=16000, window_sec=2.0):
    """Load WAV files matching pattern, return list of (path, audio_float)."""
    window_samples = int(window_sec * sr)
    samples = []
    for path in sorted(glob.glob(pattern)):
        audio, _ = librosa.load(path, sr=sr)
        audio = pad_or_trim(audio, window_samples)
        samples.append((path, audio))
    return samples


def embed_samples(extractor, samples, cfg, n_aug=0, n_frames=16):
    """Compute embeddings for samples, with optional augmentation."""
    sr = cfg["audio"]["sample_rate"]
    window_samples = int(cfg["audio"]["window_sec"] * sr)
    embeddings = []

    for path, audio in samples:
        # Original
        embeddings.append(extractor.extract_fixed(float_to_int16(audio), n_frames))
        # Augmented
        for i in range(n_aug):
            aug = augment_one(audio, sr)
            aug = pad_or_trim(aug, window_samples)
            embeddings.append(extractor.extract_fixed(float_to_int16(aug), n_frames))
            if (i + 1) % 200 == 0:
                print(f"  {path}: {i + 1}/{n_aug} augmentations")

    return np.array(embeddings)


# --- Negative data ---

_last_pct = -1

def _progress_hook(block_num, block_size, total_size):
    global _last_pct
    downloaded = block_num * block_size
    pct = min(100, downloaded * 100 // total_size)
    if pct != _last_pct:
        _last_pct = pct
        mb = downloaded / 1e6
        total_mb = total_size / 1e6
        print(f"\r  {mb:.0f}/{total_mb:.0f} MB ({pct}%)", end="", flush=True)


def prepare_negatives(extractor, cfg, n_frames=16):
    """Prepare negative embeddings from LibriSpeech + user recordings."""
    tc = cfg["training"]
    ac = cfg["audio"]
    data_dir = Path(tc["data_dir"])
    cache_file = data_dir / "negative_embeddings.npy"

    if cache_file.exists():
        print(f"Loading cached negative embeddings from {cache_file}")
        libri_emb = np.load(cache_file)
    else:
        data_dir.mkdir(parents=True, exist_ok=True)

        # Download LibriSpeech
        tar_path = data_dir / "dev-clean.tar.gz"
        if not tar_path.exists():
            print(f"Downloading LibriSpeech dev-clean...")
            urllib.request.urlretrieve(tc["negative_data_url"], tar_path, _progress_hook)
            print()

        extract_dir = data_dir / "LibriSpeech"
        if not extract_dir.exists():
            print("Extracting...")
            with tarfile.open(tar_path) as tar:
                tar.extractall(data_dir)

        flac_files = sorted(glob.glob(str(data_dir / "LibriSpeech" / "dev-clean" / "**" / "*.flac"), recursive=True))
        print(f"Found {len(flac_files)} FLAC files")

        sr = ac["sample_rate"]
        window_samples = int(ac["window_sec"] * sr)
        max_samples = tc["negative_max_samples"]
        embeddings = []

        for fpath in flac_files:
            if len(embeddings) >= max_samples:
                break
            audio, _ = librosa.load(fpath, sr=sr)
            for start in range(0, len(audio) - window_samples, window_samples):
                if len(embeddings) >= max_samples:
                    break
                clip = audio[start:start + window_samples]
                emb = extractor.extract_fixed(float_to_int16(clip), n_frames)
                embeddings.append(emb)

            if len(embeddings) % 200 == 0 and len(embeddings) > 0:
                print(f"  {len(embeddings)}/{max_samples} clips embedded")

        # Silence embeddings
        print("Adding silence embeddings...")
        for _ in range(200):
            silence = np.zeros(window_samples, dtype=np.float32)
            embeddings.append(extractor.extract_fixed(float_to_int16(silence), n_frames))

        libri_emb = np.array(embeddings)
        np.save(cache_file, libri_emb)
        print(f"Cached {len(libri_emb)} negative embeddings")

    # User negatives (never cached — user can add new ones)
    neg_glob = tc.get("negative_glob", "neg*_16k.wav")
    user_negs = load_wavs(neg_glob, ac["sample_rate"], ac["window_sec"])
    if user_negs:
        print(f"Embedding {len(user_negs)} user negatives with 50x augmentation...")
        user_emb = embed_samples(extractor, user_negs, cfg, n_aug=50, n_frames=n_frames)
        print(f"User negative embeddings: {len(user_emb)}")
        return np.concatenate([libri_emb, user_emb], axis=0)

    return libri_emb


# --- Training ---

def train(cfg):
    tc = cfg["training"]
    ac = cfg["audio"]
    output_dir = Path(tc["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    n_test = tc.get("n_test_samples", 4)
    np.random.seed(42)

    # Initialize embedding extractor
    print("Loading pretrained embedding models...")
    extractor = EmbeddingExtractor()

    # For 2s audio: 197 mel frames -> (197-76)//8 + 1 = 16 embedding frames
    n_frames = 16

    # Load positives
    print("\n=== Loading positive samples ===")
    all_positives = load_wavs(tc["positive_glob"], ac["sample_rate"], ac["window_sec"])
    print(f"Found {len(all_positives)} recordings")

    # Split: train / val / test at recording level
    indices = np.random.permutation(len(all_positives))
    n_val = 2
    test_indices = indices[-n_test:]
    val_indices = indices[-(n_test + n_val):-n_test]
    train_indices = indices[:-(n_test + n_val)]

    train_pos = [all_positives[i] for i in train_indices]
    val_pos = [all_positives[i] for i in val_indices]
    test_pos = [all_positives[i] for i in test_indices]

    print(f"Train: {len(train_pos)}, Val: {len(val_pos)}, Test: {len(test_pos)}")

    # Embed positives
    n_aug = tc["augmentations_per_sample"]
    print(f"\n=== Embedding train positives ({n_aug}x aug) ===")
    train_pos_emb = embed_samples(extractor, train_pos, cfg, n_aug=n_aug, n_frames=n_frames)
    print(f"Train positive embeddings: {len(train_pos_emb)}")

    print(f"\n=== Embedding val positives (50x aug) ===")
    val_pos_emb = embed_samples(extractor, val_pos, cfg, n_aug=50, n_frames=n_frames)

    print(f"\n=== Embedding test positives (50x aug) ===")
    test_pos_emb = embed_samples(extractor, test_pos, cfg, n_aug=50, n_frames=n_frames)

    # Negatives
    print("\n=== Preparing negatives ===")
    all_neg_emb = prepare_negatives(extractor, cfg, n_frames=n_frames)

    # Split negatives
    neg_perm = np.random.permutation(len(all_neg_emb))
    s1 = int(len(all_neg_emb) * 0.6)
    s2 = int(len(all_neg_emb) * 0.8)
    train_neg_emb = all_neg_emb[neg_perm[:s1]]
    val_neg_emb = all_neg_emb[neg_perm[s1:s2]]
    test_neg_emb = all_neg_emb[neg_perm[s2:]]

    # Balance: 2x negatives per positive
    for name, pos_n in [("train", len(train_pos_emb)), ("val", len(val_pos_emb)), ("test", len(test_pos_emb))]:
        max_neg = pos_n * 2
        if name == "train" and len(train_neg_emb) > max_neg:
            train_neg_emb = train_neg_emb[np.random.choice(len(train_neg_emb), max_neg, replace=False)]
        elif name == "val" and len(val_neg_emb) > max_neg:
            val_neg_emb = val_neg_emb[np.random.choice(len(val_neg_emb), max_neg, replace=False)]
        elif name == "test" and len(test_neg_emb) > max_neg:
            test_neg_emb = test_neg_emb[np.random.choice(len(test_neg_emb), max_neg, replace=False)]

    print(f"\nTrain: {len(train_pos_emb)} pos + {len(train_neg_emb)} neg")
    print(f"Val:   {len(val_pos_emb)} pos + {len(val_neg_emb)} neg")
    print(f"Test:  {len(test_pos_emb)} pos + {len(test_neg_emb)} neg")

    # Build tensors
    X_train = torch.from_numpy(np.concatenate([train_pos_emb, train_neg_emb]))
    y_train = torch.cat([torch.ones(len(train_pos_emb)), torch.zeros(len(train_neg_emb))])
    perm = torch.randperm(len(X_train))
    X_train, y_train = X_train[perm], y_train[perm]

    X_val = torch.from_numpy(np.concatenate([val_pos_emb, val_neg_emb]))
    y_val = torch.cat([torch.ones(len(val_pos_emb)), torch.zeros(len(val_neg_emb))])

    X_test = torch.from_numpy(np.concatenate([test_pos_emb, test_neg_emb]))
    y_test = torch.cat([torch.ones(len(test_pos_emb)), torch.zeros(len(test_neg_emb))])

    train_dl = DataLoader(TensorDataset(X_train, y_train), batch_size=tc["batch_size"], shuffle=True)
    val_dl = DataLoader(TensorDataset(X_val, y_val), batch_size=tc["batch_size"])
    test_dl = DataLoader(TensorDataset(X_test, y_test), batch_size=tc["batch_size"])

    # Model
    mc = cfg["model"]
    model = WakeWordClassifier(
        n_frames=n_frames,
        embedding_dim=96,
        hidden=mc.get("hidden", 64),
        dropout=mc.get("dropout", 0.3),
    )
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\nClassifier: {param_count:,} parameters")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=tc["learning_rate"], weight_decay=tc["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    pos_weight = len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1)
    print(f"Positive class weight: {pos_weight:.2f}")

    best_val_loss = float("inf")
    patience_counter = 0

    print("\n=== Training ===")
    for epoch in range(tc["epochs"]):
        model.train()
        train_loss = train_correct = train_total = 0

        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            weights = torch.where(yb == 1, pos_weight, 1.0).to(device)
            loss = nn.functional.binary_cross_entropy(pred, yb, weight=weights)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(xb)
            train_correct += ((pred > 0.5) == yb).sum().item()
            train_total += len(xb)

        train_loss /= train_total

        # Validate
        model.eval()
        val_loss = val_tp = val_fp = val_fn = val_total = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = nn.functional.binary_cross_entropy(pred, yb)
                val_loss += loss.item() * len(xb)
                predicted = pred > 0.5
                actual = yb.bool()
                val_tp += (predicted & actual).sum().item()
                val_fp += (predicted & ~actual).sum().item()
                val_fn += (~predicted & actual).sum().item()
                val_total += len(xb)

        val_loss /= val_total
        prec = val_tp / max(val_tp + val_fp, 1)
        rec = val_tp / max(val_tp + val_fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)

        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch+1:3d}/{tc['epochs']}  "
              f"loss: {train_loss:.4f}/{val_loss:.4f}  "
              f"P/R/F1: {prec:.3f}/{rec:.3f}/{f1:.3f}  lr: {lr:.1e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= tc["patience"]:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # Load best and evaluate
    model.load_state_dict(torch.load(output_dir / "best_model.pt", weights_only=True))
    model.eval()

    print("\n=== Test set evaluation ===")
    test_tp = test_fp = test_fn = test_tn = 0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for xb, yb in test_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            predicted = pred > 0.5
            actual = yb.bool()
            test_tp += (predicted & actual).sum().item()
            test_fp += (predicted & ~actual).sum().item()
            test_fn += (~predicted & actual).sum().item()
            test_tn += (~predicted & ~actual).sum().item()
            all_probs.extend(pred.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    test_prec = test_tp / max(test_tp + test_fp, 1)
    test_rec = test_tp / max(test_tp + test_fn, 1)
    test_f1 = 2 * test_prec * test_rec / max(test_prec + test_rec, 1e-8)

    print(f"Precision: {test_prec:.3f}  Recall: {test_rec:.3f}  F1: {test_f1:.3f}")
    print(f"TP={test_tp} FP={test_fp} FN={test_fn} TN={test_tn}")
    print(f"Positive avg conf: {all_probs[all_labels == 1].mean():.3f}")
    print(f"Negative avg conf: {all_probs[all_labels == 0].mean():.3f}")

    # Export
    model = model.to("cpu")
    export_classifier_onnx(model, str(output_dir / "classifier.onnx"), n_frames, 96)

    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)
