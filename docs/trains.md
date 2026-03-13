# Training Log

## Format
Each entry: date, arch, data sizes, duration, train results (loss, acc), eval results, notes.

## Runs

### 2025-03-10a — ConvAttn baseline (1s center-pad)
- **Embedding**: Google speech embedding (melspec→embed ONNX), 16×96
- **Classifier**: ConvAttn (conv1d + attention pool), 37K params, hidden=64, dropout=0.3
- **Data**: 62 real pos + 171 TTS pos, ~5035 neg (6 groups + LibriSpeech), aug=50
- **Duration**: unknown (not recorded)
- **Eval**: TP=27 FP=8 FN=2 F1=0.844
- **Notes**: First run with 1s clips center-padded to 2s. FPs on user-voice negs (neg7,11,12 counted 2x from split). FNs (pos7,pos9) at ~0.80, sound clear after normalization.

### 2025-03-10b — ConvAttn baseline (cached, 3 seeds)
- **Embedding**: Google speech embedding (melspec→embed ONNX), 16×96
- **Classifier**: ConvAttn (conv1d + attention pool), 37K params, hidden=64, dropout=0.3
- **Data**: 62 real pos (3264 w/aug) + 171 TTS pos (1026 w/aug) = 4086 pos, 10548 neg
- **Train**: seed1 best, val_loss=0.0004, val_f1=1.000, 25 epochs
- **Duration**: 82s (3 seeds, cached embeddings)
- **Eval**: TP=28 FP=2 FN=0 F1=0.966
- **Notes**: 0 FNs! pos7 now 0.998. 2 FPs both neg12 (0.953, same original split). neg7→0.728, neg11→0.580. Train-only (no embed step).

### 2025-03-10c — Context negatives (ask/about/call/is/mister eliezer)
- **Embedding**: Google speech embedding (melspec→embed ONNX), 16×96
- **Classifier**: ConvAttn, 37K params, hidden=64, dropout=0.3
- **Data**: 4086 pos, 10620 neg (+90 context negatives in tts_neg group, total 115)
- **Train**: seed2 best, val_loss=0.0001, val_f1=1.000, 32 epochs
- **Duration**: 88s (3 seeds)
- **Eval**: TP=26 FP=2 FN=2 F1=0.929
- **Notes**: neg12 dropped 0.953→0.738 (no longer FP), but neg11 rose 0.580→0.886 (new FP). 2 new FNs: pos10=0.733, pos33=0.588. Regression — context negs hurt recall without fixing FPs. Reverted.

### 2025-03-10d — Hidden 128 (bigger model)
- **Embedding**: Google speech embedding, 16×96
- **Classifier**: ConvAttn, ~130K params, hidden=128, dropout=0.3
- **Data**: 4086 pos, 10548 neg (same as baseline)
- **Train**: seed0 best, train_loss=0.0013, val_loss=0.0000, val_f1=1.000, 38 epochs
- **Duration**: 116s (3 seeds)
- **Eval**: TP=28 FP=2 FN=0 F1=0.966
- **Notes**: Same F1 as baseline but better margins. neg12: 0.953→0.893. pos10: 0.988→1.000. All positives ≥0.990. Still no overfitting.

### 2025-03-10e — Hidden 128, dropout 0.2, 8 val files
- **Embedding**: Google speech embedding, 16×96
- **Classifier**: ConvAttn, ~130K params, hidden=128, dropout=0.2
- **Data**: 3882 pos (8 val files), 10548 neg
- **Train**: seed0 best, train_loss=0.0007, val_loss=0.0006, val_f1=0.999, 34 epochs
- **Duration**: 95s (3 seeds)
- **Eval**: TP=28 FP=0 FN=0 F1=1.000
- **Notes**: Perfect eval! neg12=0.834 (just below 0.85). All positives ≥0.986. Bigger model + less dropout + more val files = better seed selection + sharper boundary.

### 2025-03-10f — +5 context negatives (1 per phrase)
- **Embedding**: Google speech embedding, 16×96
- **Classifier**: ConvAttn, ~130K params, hidden=128, dropout=0.2
- **Data**: 3882 pos, 10552 neg (+5 context negs: about/ask/call/is/mister eliezer, 1 voice each)
- **Train**: seed1 best, train_loss=0.0030, val_loss=0.0003, val_f1=1.000, 22 epochs
- **Duration**: 76s (3 seeds)
- **Eval**: TP=28 FP=0 FN=0 F1=1.000
- **Notes**: neg12: 0.834→0.217! neg11 also down. All positives ≥0.952. 5 context negs = sweet spot vs 90 that killed recall.

## Embedding Analysis
Most discriminative dims (pos vs neg, temporal variance):
- **dim 95**: strong downward sweep (37→3) in pos, flat (~20) in neg — most discriminative
- **dim 42**: bell curve peaking frames 3-4 (~67) in pos, flat (~44) in neg — "hey" energy
- **dim 77**: dips to -20 at frames 9-11 in pos, near 0 in neg — "eliezer" tail
- **dim 0**: V-shape dip to -23 at frames 6-7 in pos, flat in neg
- **dim 37, 12, 46, 73**: also show clear pos/neg temporal separation

Future: consider feeding only top discriminative dims to classifier, or using dim-specific attention weights.
