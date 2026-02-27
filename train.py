"""
Wake word training pipeline using Google's speech embedding model.

Embeds each audio file once and caches. Positive augmentation done on audio
before embedding, also cached. Subsequent runs load from cache instantly.

Usage: python train.py [--config config.yaml]
"""

import argparse
import glob
import hashlib
import tarfile
import time
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


class EmbeddingExtractor:
    def __init__(self, melspec_path="models/melspectrogram.onnx",
                 embedding_path="models/embedding_model.onnx"):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.melspec_sess = ort.InferenceSession(melspec_path, opts)
        self.embed_sess = ort.InferenceSession(embedding_path, opts)

    def extract_fixed(self, audio_float, sr=16000, n_frames=16):
        audio_int16 = (audio_float * 32767).clip(-32768, 32767).astype(np.int16)
        x = audio_int16.astype(np.float32)[None, :]
        mel = self.melspec_sess.run(None, {'input': x})[0]
        mel = np.squeeze(mel) / 10 + 2

        windows = []
        for i in range(0, mel.shape[0], 8):
            w = mel[i:i+76]
            if w.shape[0] == 76:
                windows.append(w)

        if not windows:
            return np.zeros((n_frames, 96), dtype=np.float32)

        batch = np.array(windows, dtype=np.float32)[:, :, :, None]
        emb = self.embed_sess.run(None, {'input_1': batch})[0].squeeze(axis=(1, 2))

        if emb.shape[0] >= n_frames:
            return emb[:n_frames]
        return np.vstack([emb, np.zeros((n_frames - emb.shape[0], 96), dtype=np.float32)])


def _files_hash(file_list):
    h = hashlib.md5()
    for f in sorted(file_list):
        h.update(f.encode())
        h.update(str(Path(f).stat().st_mtime_ns).encode())
    return h.hexdigest()[:12]


def embed_files(extractor, file_list, cache_path, label, sr=16000, window_sec=2.0,
                n_frames=16, n_aug=0):
    """Embed audio files with optional audio augmentation. Cached."""
    fhash = _files_hash(file_list)
    cache_key = f"{fhash}_aug{n_aug}"

    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        if str(cached.get("key", "")) == cache_key:
            embs = cached["embeddings"]
            print(f"  {label}: {len(embs)} embeddings (cached)")
            return embs

    window_samples = int(window_sec * sr)
    embeddings = []
    for i, f in enumerate(file_list):
        audio, _ = librosa.load(f, sr=sr)
        audio = pad_or_trim(audio, window_samples)
        # Original
        embeddings.append(extractor.extract_fixed(audio, sr, n_frames))
        # Audio augmentations
        for _ in range(n_aug):
            aug = augment_one(audio, sr)
            aug = pad_or_trim(aug, window_samples)
            embeddings.append(extractor.extract_fixed(aug, sr, n_frames))
        if (i + 1) % 50 == 0:
            print(f"  {label}: {i + 1}/{len(file_list)} files...")

    embeddings = np.array(embeddings)
    np.savez(cache_path, embeddings=embeddings, key=cache_key)
    print(f"  {label}: {len(embeddings)} embeddings (computed & cached)")
    return embeddings


def make_partial_negatives(extractor, pos_files, sr=16000, window_sec=2.0, n_frames=16, n_aug=10):
    """Split positive recordings into thirds — each partial is a hard negative.
    Trim leading/trailing silence first, then split the voiced portion."""
    window_samples = int(window_sec * sr)
    embeddings = []
    for f in pos_files:
        audio, _ = librosa.load(f, sr=sr)
        # Trim silence (threshold -30dB)
        trimmed, _ = librosa.effects.trim(audio, top_db=30)
        if len(trimmed) < sr * 0.3:  # skip if too short after trim
            continue
        half = len(trimmed) // 2
        # First half ("hey") and second half ("eliezer") as negatives
        for start, end in [(0, half), (half, len(trimmed))]:
            chunk = trimmed[start:end]
            # Zero-pad to full window
            padded = pad_or_trim(chunk, window_samples)
            embeddings.append(extractor.extract_fixed(padded, sr, n_frames))
            for _ in range(n_aug):
                aug = augment_one(padded, sr)
                aug = pad_or_trim(aug, window_samples)
                embeddings.append(extractor.extract_fixed(aug, sr, n_frames))
    return np.array(embeddings) if embeddings else None


# --- LibriSpeech ---

_last_pct = -1

def _progress_hook(block_num, block_size, total_size):
    global _last_pct
    downloaded = block_num * block_size
    pct = min(100, downloaded * 100 // total_size)
    if pct != _last_pct:
        _last_pct = pct
        print(f"\r  {downloaded/1e6:.0f}/{total_size/1e6:.0f} MB ({pct}%)", end="", flush=True)


def prepare_librispeech(extractor, cfg, n_frames=16):
    tc = cfg["training"]
    ac = cfg["audio"]
    data_dir = Path(tc["data_dir"])
    cache_file = data_dir / "negative_embeddings.npy"

    if cache_file.exists():
        embs = np.load(cache_file)
        print(f"  LibriSpeech: {len(embs)} embeddings (cached)")
        return embs

    data_dir.mkdir(parents=True, exist_ok=True)
    tar_path = data_dir / "dev-clean.tar.gz"
    if not tar_path.exists():
        print(f"  Downloading LibriSpeech dev-clean...")
        urllib.request.urlretrieve(tc["negative_data_url"], tar_path, _progress_hook)
        print()

    extract_dir = data_dir / "LibriSpeech"
    if not extract_dir.exists():
        print("  Extracting...")
        with tarfile.open(tar_path) as tar:
            tar.extractall(data_dir)

    flac_files = sorted(glob.glob(str(data_dir / "LibriSpeech" / "dev-clean" / "**" / "*.flac"), recursive=True))
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
            embeddings.append(extractor.extract_fixed(clip, sr, n_frames))
        if len(embeddings) % 500 == 0 and len(embeddings) > 0:
            print(f"  {len(embeddings)}/{max_samples} clips...")

    for _ in range(200):
        embeddings.append(extractor.extract_fixed(np.zeros(window_samples, dtype=np.float32), sr, n_frames))

    libri_emb = np.array(embeddings)
    np.save(cache_file, libri_emb)
    print(f"  LibriSpeech: {len(libri_emb)} embeddings (cached)")
    return libri_emb


# --- Training ---

def train(cfg):
    t0 = time.time()
    tc = cfg["training"]
    ac = cfg["audio"]
    output_dir = Path(tc["output_dir"])
    data_dir = Path(tc["data_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    n_test = tc.get("n_test_samples", 4)
    n_frames = 16
    sr = ac["sample_rate"]
    n_aug = tc["augmentations_per_sample"]
    np.random.seed(42)

    print("Loading embedding models...")
    extractor = EmbeddingExtractor()

    # --- Embed everything (cached after first run) ---
    print("\n=== Embedding audio ===")

    # Positives with audio augmentation
    pos_files = sorted(glob.glob(tc["positive_glob"]))
    pos_all = embed_files(extractor, pos_files, data_dir / "pos_embeddings.npz",
                          "Positives", sr, ac["window_sec"], n_frames, n_aug=n_aug)

    # Each file produced (1 + n_aug) embeddings, reshape to (n_files, 1+n_aug, n_frames, 96)
    emb_per_file = 1 + n_aug
    pos_by_file = pos_all.reshape(len(pos_files), emb_per_file, n_frames, 96)

    # YouTube negatives (raw, no augmentation)
    yt_glob = tc.get("youtube_negative_glob", "train_data/youtube/*_16k.wav")
    yt_files = sorted(glob.glob(yt_glob))
    yt_emb = None
    if yt_files:
        yt_emb = embed_files(extractor, yt_files, data_dir / "youtube_embeddings.npz",
                             "YouTube negs", sr, ac["window_sec"], n_frames)

    # Hand-recorded negatives (with some augmentation)
    neg_files = sorted(glob.glob(tc.get("negative_glob", "train_data/neg*_16k.wav")))
    user_emb = None
    if neg_files:
        user_emb = embed_files(extractor, neg_files, data_dir / "user_neg_embeddings.npz",
                               "Hand-recorded negs", sr, ac["window_sec"], n_frames, n_aug=30)

    # LibriSpeech
    libri_emb = prepare_librispeech(extractor, cfg, n_frames)

    t_embed = time.time() - t0
    print(f"\nEmbedding: {t_embed:.1f}s")

    # --- Split positives at recording level ---
    indices = np.random.permutation(len(pos_files))
    n_val = 2
    test_idx = indices[-n_test:]
    val_idx = indices[-(n_test + n_val):-n_test]
    train_idx = indices[:-(n_test + n_val)]

    # Split positive embeddings (all augmentations stay with their source recording)
    train_pos = pos_by_file[train_idx].reshape(-1, n_frames, 96)
    val_pos = pos_by_file[val_idx].reshape(-1, n_frames, 96)
    test_pos = pos_by_file[test_idx].reshape(-1, n_frames, 96)

    print(f"Positives: {len(train_idx)} train ({len(train_pos)}), "
          f"{len(val_idx)} val ({len(val_pos)}), {len(test_idx)} test ({len(test_pos)})")

    # --- Combine negatives ---
    neg_parts = [libri_emb]
    if yt_emb is not None:
        neg_parts.append(yt_emb)
    if user_emb is not None:
        neg_parts.append(user_emb)
    all_neg = np.concatenate(neg_parts)

    neg_perm = np.random.permutation(len(all_neg))
    s1 = int(len(all_neg) * 0.6)
    s2 = int(len(all_neg) * 0.8)
    train_neg = all_neg[neg_perm[:s1]]
    val_neg = all_neg[neg_perm[s1:s2]]
    test_neg = all_neg[neg_perm[s2:]]

    # Cap negatives to 2x positives
    for name, pos_n in [("train", len(train_pos)), ("val", len(val_pos)), ("test", len(test_pos))]:
        max_neg = pos_n * 2
        if name == "train" and len(train_neg) > max_neg:
            train_neg = train_neg[np.random.choice(len(train_neg), max_neg, replace=False)]
        elif name == "val" and len(val_neg) > max_neg:
            val_neg = val_neg[np.random.choice(len(val_neg), max_neg, replace=False)]
        elif name == "test" and len(test_neg) > max_neg:
            test_neg = test_neg[np.random.choice(len(test_neg), max_neg, replace=False)]

    print(f"Train: {len(train_pos)} pos + {len(train_neg)} neg")
    print(f"Val:   {len(val_pos)} pos + {len(val_neg)} neg")
    print(f"Test:  {len(test_pos)} pos + {len(test_neg)} neg")

    # --- Build tensors & train ---
    X_train = torch.from_numpy(np.concatenate([train_pos, train_neg]))
    y_train = torch.cat([torch.ones(len(train_pos)), torch.zeros(len(train_neg))])
    perm = torch.randperm(len(X_train))
    X_train, y_train = X_train[perm], y_train[perm]

    X_val = torch.from_numpy(np.concatenate([val_pos, val_neg]))
    y_val = torch.cat([torch.ones(len(val_pos)), torch.zeros(len(val_neg))])

    X_test = torch.from_numpy(np.concatenate([test_pos, test_neg]))
    y_test = torch.cat([torch.ones(len(test_pos)), torch.zeros(len(test_neg))])

    train_dl = DataLoader(TensorDataset(X_train, y_train), batch_size=tc["batch_size"], shuffle=True)
    val_dl = DataLoader(TensorDataset(X_val, y_val), batch_size=tc["batch_size"])
    test_dl = DataLoader(TensorDataset(X_test, y_test), batch_size=tc["batch_size"])

    mc = cfg["model"]
    model = WakeWordClassifier(
        n_frames=n_frames, embedding_dim=96,
        hidden=mc.get("hidden", 64), dropout=mc.get("dropout", 0.3),
    )
    print(f"\nClassifier: {sum(p.numel() for p in model.parameters()):,} params")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=tc["learning_rate"], weight_decay=tc["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    pos_weight = len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1)
    print(f"Pos weight: {pos_weight:.2f}")

    best_val_loss = float("inf")
    patience_counter = 0

    print("\n=== Training ===")
    for epoch in range(tc["epochs"]):
        model.train()
        train_loss = train_total = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            weights = torch.where(yb == 1, pos_weight, 1.0).to(device)
            loss = nn.functional.binary_cross_entropy(pred, yb, weight=weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
            train_total += len(xb)
        train_loss /= train_total

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

        print(f"  {epoch+1:3d}/{tc['epochs']}  "
              f"loss: {train_loss:.4f}/{val_loss:.4f}  "
              f"P/R/F1: {prec:.3f}/{rec:.3f}/{f1:.3f}  "
              f"lr: {optimizer.param_groups[0]['lr']:.1e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= tc["patience"]:
                print(f"  Early stop at epoch {epoch + 1}")
                break

    # --- Evaluate ---
    model.load_state_dict(torch.load(output_dir / "best_model.pt", weights_only=True))
    model.eval()
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

    print(f"\n=== Test: P={test_prec:.3f} R={test_rec:.3f} F1={test_f1:.3f} ===")
    print(f"TP={test_tp} FP={test_fp} FN={test_fn} TN={test_tn}")
    print(f"Pos avg: {all_probs[all_labels == 1].mean():.3f}  "
          f"Neg avg: {all_probs[all_labels == 0].mean():.3f}")

    model = model.to("cpu")
    export_classifier_onnx(model, str(output_dir / "classifier.onnx"), n_frames, 96)
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)

    print(f"\nTotal: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)
