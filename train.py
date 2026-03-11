"""
Wake word training pipeline.

Uses Google's pretrained speech embedding as frozen feature extractor,
trains a small attention classifier on top. All embeddings are cached —
subsequent runs with the same audio files complete in seconds.

Negative sources (in order of importance):
  1. Hand-recorded negatives (train_data/neg*_16k.wav) — user's voice, augmented
  2. Voice recording — long audio file auto-sliced into 2s WAV clips
  3. LibriSpeech dev-clean — generic English speech

Usage: python train.py [--config config.yaml]
"""

import argparse
import glob
import hashlib
import subprocess
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
from embedding import EmbeddingExtractor
from model import WakeWordClassifier, export_classifier_onnx


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


# --- Caching ---

def _files_hash(file_list):
    h = hashlib.md5()
    for f in sorted(file_list):
        h.update(f.encode())
        h.update(str(Path(f).stat().st_mtime_ns).encode())
    return h.hexdigest()[:12]


def vad_filter(vad, file_list, cache_path, sr=16000, threshold=0.5, chunk_size=512):
    """Filter file list to only those containing speech. Cached."""
    fhash = _files_hash(file_list)
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        if str(cached.get("key", "")) == fhash:
            kept = list(cached["files"])
            print(f"  VAD filter: {len(kept)}/{len(file_list)} have speech (cached)")
            return kept

    kept = []
    for f in file_list:
        vad.reset_states()
        audio, _ = librosa.load(f, sr=sr)
        has_speech = False
        for i in range(0, len(audio) - chunk_size + 1, chunk_size):
            prob = vad(torch.from_numpy(audio[i:i+chunk_size]), sr).item()
            if prob > threshold:
                has_speech = True
                break
        if has_speech:
            kept.append(f)

    np.savez(cache_path, files=kept, key=fhash)
    print(f"  VAD filter: {len(kept)}/{len(file_list)} have speech")
    return kept


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
        embeddings.append(extractor.extract_fixed(audio, sr, n_frames))
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


# --- Voice recording slicing ---

def slice_voice_recording(audio_path, clip_dir, sr=16000, window_sec=2.0):
    """Slice a long audio file into 2s WAV clips using ffmpeg segment.

    Returns sorted list of WAV paths. Clips regenerated only when source changes.
    """
    audio_path = Path(audio_path)
    clip_dir = Path(clip_dir)
    if not audio_path.exists():
        return []

    marker = clip_dir / ".source_mtime"
    current_mtime = str(audio_path.stat().st_mtime_ns)
    if marker.exists() and marker.read_text().strip() == current_mtime:
        clips = sorted(glob.glob(str(clip_dir / "voice*_16k.wav")))
        if clips:
            return clips

    clip_dir.mkdir(parents=True, exist_ok=True)
    for f in clip_dir.glob("voice*_16k.wav"):
        f.unlink()

    subprocess.run(
        ["ffmpeg", "-y", "-i", str(audio_path), "-ac", "1", "-ar", str(sr),
         "-f", "segment", "-segment_time", str(window_sec),
         str(clip_dir / "voice%03d_16k.wav")],
        capture_output=True, timeout=120,
    )
    marker.write_text(current_mtime)

    clips = sorted(glob.glob(str(clip_dir / "voice*_16k.wav")))
    print(f"  Voice recording: sliced into {len(clips)} clips")
    return clips


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

    n_frames = 16
    sr = ac["sample_rate"]
    n_aug = tc["augmentations_per_sample"]
    n_seeds = tc.get("n_seeds", 5)
    np.random.seed(42)

    # --- Load positives from cache ---
    print("\n=== Loading cached embeddings ===")

    pos_files = sorted(glob.glob(str(Path(tc["positive_dir"]) / "*.wav")))
    pos_cache = data_dir / "pos_embeddings.npz"
    if not pos_cache.exists():
        print("ERROR: no positive embeddings found. Run --embed pos first.")
        return
    pos_all = np.load(pos_cache, allow_pickle=True)["embeddings"]
    print(f"  Positives: {len(pos_all)} embeddings")
    emb_per_file = 1 + n_aug
    pos_by_file = pos_all.reshape(len(pos_files), emb_per_file, n_frames, 96)

    # TTS positives
    tts_all = None
    tts_cache = data_dir / "tts_pos_embeddings.npz"
    if tts_cache.exists():
        tts_all = np.load(tts_cache, allow_pickle=True)["embeddings"]
        print(f"  TTS positives: {len(tts_all)} embeddings")

    # --- Split positives: train + val only (external test_data/ for eval) ---
    n_val = tc.get("n_test_samples", 4)
    indices = np.random.permutation(len(pos_files))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_pos = pos_by_file[train_idx].reshape(-1, n_frames, 96)
    val_pos = pos_by_file[val_idx].reshape(-1, n_frames, 96)

    # Add TTS positives to training (all of them, no val split)
    if tts_all is not None:
        train_pos = np.concatenate([train_pos, tts_all])

    print(f"Positives: {len(train_idx)} real train ({pos_by_file[train_idx].reshape(-1, n_frames, 96).shape[0]})"
          f" + {len(tts_all) if tts_all is not None else 0} TTS"
          f" = {len(train_pos)}, val={len(val_pos)}")

    # --- Combine negatives from group caches ---
    neg_parts = []
    # LibriSpeech (legacy cache)
    libri_path = data_dir / "negative_embeddings.npy"
    if libri_path.exists():
        libri_emb = np.load(libri_path)
        print(f"  LibriSpeech: {len(libri_emb)} embeddings")
        neg_parts.append(libri_emb)
    # Group caches
    for g in NEG_GROUPS:
        cache = data_dir / f"neg_{g}_embeddings.npz"
        if cache.exists():
            embs = np.load(cache, allow_pickle=True)["embeddings"]
            print(f"  {g}: {len(embs)} embeddings")
            neg_parts.append(embs)
    if not neg_parts:
        print("ERROR: no negative embeddings found. Run --embed neg first.")
        return
    all_neg = np.concatenate(neg_parts)

    neg_perm = np.random.permutation(len(all_neg))
    split = int(len(all_neg) * 0.8)
    train_neg = all_neg[neg_perm[:split]]
    val_neg = all_neg[neg_perm[split:]]

    print(f"Train: {len(train_pos)} pos + {len(train_neg)} neg")
    print(f"Val:   {len(val_pos)} pos + {len(val_neg)} neg")

    # --- Build tensors ---
    X_train = torch.from_numpy(np.concatenate([train_pos, train_neg]))
    y_train = torch.cat([torch.ones(len(train_pos)), torch.zeros(len(train_neg))])

    X_val = torch.from_numpy(np.concatenate([val_pos, val_neg]))
    y_val = torch.cat([torch.ones(len(val_pos)), torch.zeros(len(val_neg))])

    pos_weight = len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1)

    mc = cfg["model"]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    best_seed_f1 = -1

    print(f"\n=== Training {n_seeds} seeds ===")
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        perm = torch.randperm(len(X_train))
        Xt, yt = X_train[perm], y_train[perm]

        train_dl = DataLoader(TensorDataset(Xt, yt), batch_size=tc["batch_size"], shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
        val_dl = DataLoader(TensorDataset(X_val, y_val), batch_size=tc["batch_size"])

        model = WakeWordClassifier(
            n_frames=n_frames, embedding_dim=96,
            hidden=mc.get("hidden", 64), dropout=mc.get("dropout", 0.3),
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=tc["learning_rate"], weight_decay=tc["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        best_val_loss = float("inf")
        patience_counter = 0
        best_f1 = 0

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
            best_f1 = max(best_f1, f1)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), output_dir / f"model_seed{seed}.pt")
            else:
                patience_counter += 1
                if patience_counter >= tc["patience"]:
                    break

        print(f"  Seed {seed}: train_loss={train_loss:.4f}  val_loss={best_val_loss:.4f}  best_f1={best_f1:.3f}  epochs={epoch+1}")

        if best_f1 > best_seed_f1:
            best_seed_f1 = best_f1
            best_seed = seed

    # --- Load best seed model ---
    print(f"\nBest seed: {best_seed} (F1={best_seed_f1:.3f})")
    model = WakeWordClassifier(
        n_frames=n_frames, embedding_dim=96,
        hidden=mc.get("hidden", 64), dropout=mc.get("dropout", 0.3),
    ).to(device)
    model.load_state_dict(torch.load(output_dir / f"model_seed{best_seed}.pt", weights_only=True))
    torch.save(model.state_dict(), output_dir / "best_model.pt")

    # Clean up per-seed checkpoints
    for seed in range(n_seeds):
        p = output_dir / f"model_seed{seed}.pt"
        if p.exists():
            p.unlink()

    # Backup previous classifier
    prev = output_dir / "classifier.onnx"
    if prev.exists():
        from datetime import datetime
        mtime = datetime.fromtimestamp(prev.stat().st_mtime)
        tag = mtime.strftime("%Y%m%d_%H%M%S")
        prev.rename(output_dir / f"classifier_{tag}.onnx")
        prev_data = output_dir / "classifier.onnx.data"
        if prev_data.exists():
            prev_data.rename(output_dir / f"classifier_{tag}.onnx.data")
        print(f"  Backed up previous classifier as classifier_{tag}.onnx")

    model = model.to("cpu")
    export_classifier_onnx(model, str(output_dir / "classifier.onnx"), n_frames, 96)
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)

    print(f"\nTotal: {time.time() - t0:.1f}s")


def embed_neg_group(cfg, group):
    """Embed one negative group. Groups: user, partial, podcast1, podcast2, tts_neg, voice."""
    tc = cfg["training"]
    ac = cfg["audio"]
    data_dir = Path(tc["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)
    sr = ac["sample_rate"]

    neg_files = sorted(glob.glob(str(Path(tc["negative_dir"]) / "*.wav")))
    group_files = []
    for f in neg_files:
        name = Path(f).name
        if group == "podcast1" and name.startswith("podcast1"):
            group_files.append(f)
        elif group == "podcast2" and name.startswith("podcast2"):
            group_files.append(f)
        elif group == "tts_neg" and name.startswith("tts_neg"):
            group_files.append(f)
        elif group == "voice" and name.startswith("voice"):
            group_files.append(f)
        elif group == "partial" and name.startswith("partial"):
            group_files.append(f)
        elif group == "user" and not any(name.startswith(p) for p in
                ["podcast1", "podcast2", "tts_neg", "voice", "partial"]):
            group_files.append(f)

    if not group_files:
        print(f"No files for group '{group}'")
        return

    print(f"Group '{group}': {len(group_files)} files")
    extractor = EmbeddingExtractor()

    # VAD filter (skip for tts_neg — all speech)
    if group != "tts_neg":
        vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad",
                                verbose=False, onnx=False, trust_repo=True)
        group_files = vad_filter(vad, group_files,
                                 data_dir / f"neg_{group}_vad.npz", sr)

    embed_files(extractor, group_files, data_dir / f"neg_{group}_embeddings.npz",
                group, sr, ac["window_sec"], 16, n_aug=0)


NEG_GROUPS = ["user", "partial", "podcast1", "podcast2", "tts_neg", "voice"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--embed", choices=["pos", "tts", "neg"] + NEG_GROUPS,
                        help="Embed one group only, then exit")
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.embed == "pos":
        tc = cfg["training"]
        ac = cfg["audio"]
        data_dir = Path(tc["data_dir"])
        data_dir.mkdir(parents=True, exist_ok=True)
        extractor = EmbeddingExtractor()
        pos_files = sorted(glob.glob(str(Path(tc["positive_dir"]) / "*.wav")))
        embed_files(extractor, pos_files, data_dir / "pos_embeddings.npz",
                    "Positives", ac["sample_rate"], ac["window_sec"], 16,
                    n_aug=tc["augmentations_per_sample"])
    elif args.embed == "tts":
        tc = cfg["training"]
        ac = cfg["audio"]
        data_dir = Path(tc["data_dir"])
        data_dir.mkdir(parents=True, exist_ok=True)
        extractor = EmbeddingExtractor()
        tts_files = sorted(glob.glob(str(Path(tc["tts_positive_dir"]) / "*.wav")))
        tts_n_aug = max(1, tc["augmentations_per_sample"] // 10)
        embed_files(extractor, tts_files, data_dir / "tts_pos_embeddings.npz",
                    "TTS positives", ac["sample_rate"], ac["window_sec"], 16,
                    n_aug=tts_n_aug)
    elif args.embed == "neg":
        for g in NEG_GROUPS:
            embed_neg_group(cfg, g)
    elif args.embed in NEG_GROUPS:
        embed_neg_group(cfg, args.embed)
    else:
        train(cfg)
