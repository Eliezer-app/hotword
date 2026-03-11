# E2E Wake Word Model — Findings & Plan

## Problem Statement
The original pipeline (Google pretrained embedding + trained classifier) has too many FPs and FNs in production. The embedding can't separate "hey eliezer" from other speech containing similar sounds. Goal: train an end-to-end model that learns phoneme patterns directly.

## Data

### Train positives (70 real + 168 TTS)
- 30 `wake*` + 2 `pos*` = 32 original hand-recordings (first session)
- 23 `hit*` + 15 `near*` = 38 from detector use (later sessions)
- 168 TTS in `tts_positive/` (28 Kokoro voices × 2 phrases × 3 speeds)
- TTS gets 10% augmentation multiplier to avoid TTS bias

### Train negatives (327 clips + 2703 LibriSpeech)
- 45 original hand-recorded negatives
- 83 `voice2_*` from `negative-audio2.mp4` (user's voice, no wake word)
- 20 `hit*` + 63 `near*` = 83 FPs collected from detector
- 116 `partial_*` (30% cuts of positives, silence-padded, VAD-filtered)
- 2703 LibriSpeech dev-clean (other speakers reading books, capped at 500 for fast iteration)
- 166 non-speech files moved to `negative_no_speech/` (excluded)

### Key imbalance
Only ~128 clips of user's voice as negatives vs 2703 LibriSpeech from strangers.
Model sees "other people = negative" and learns "user's voice = positive" as shortcut.

### Test data
- 34 positives, 97 negatives (all VAD-filtered)

### VAD rule
Never train or eval on samples that don't pass Silero VAD (matches inference).

## Architecture Experiments

### 1. CNN with freq-only convolutions (original)
- Conv2d(5,1) along frequency only → self-attention → attention pooling → classifier
- **Result**: Learns speaker identity, not phonemes. Each time frame processed independently — can't see spectro-temporal transitions.
- F1 ~0.72 with high FPs on user's voice saying other things

### 2. CNN with 2D convolutions (3,3)
- Conv2d(3,3) over both time and frequency → self-attention → attention pooling
- **Key finding: hop_length is critical**
  - 10ms hop (201 frames): F1=0.767, 3 FPs — too fine, overfits
  - 20ms hop (101 frames): F1=0.806
  - 30ms hop (67 frames): F1=0.892
  - **40ms hop (51 frames): F1=0.957** — best CNN result (TP=33, FP=2, FN=1)
  - 50ms hop (41 frames): F1=0.928 — too coarse
- **But**: live testing revealed model reacts to envelope/energy, not phonemes. 2D CNN still fundamentally wrong for speech.

### 3. MFCC + GRU (current)
- MFCC(13) + deltas + delta-deltas = 39 features → 2-layer bidirectional GRU → attention pooling → classifier
- CMVN (cepstral mean normalization) applied
- **Best recall**: TP=33, FP=11, FN=1 (R=0.971) with hidden=64, 20ms hop
- Hidden=32: TP=26 FP=7 FN=8 (fewer FPs but lower recall)
- **Problem**: still learning speaker identity — 11 FPs all high-confidence on user's voice

## Key Findings

### What works
- **VAD gating**: pre-filter all data through Silero VAD. Matches inference, removes easy negatives
- **RMS normalization**: global RMS to 0.1 before feature extraction. Per-frame normalization hurts
- **Bandwidth 60-7500Hz**: better than 80-4000Hz or full spectrum
- **Coarser time resolution**: 20-40ms hop generalizes better than 10ms
- **MFCCs + CMVN**: better phoneme features than raw mel
- **Partial negatives**: 30% cuts of positives as hard negatives (silence-padded, VAD-filtered)
- **Same-speaker negatives**: negative-audio2.mp4 clips help but don't solve FPs

### What doesn't work
- **2D CNN on spectrograms**: learns energy/envelope, not phonemes
- **Freq-only convolutions (5,1)**: can't see temporal transitions
- **Per-frame mel normalization**: destroys useful amplitude relationships between frames
- **Low dropout (0.15)**: extreme precision, terrible recall
- **Too much TTS**: biases toward synthetic voices, misses real voice
- **Too many partial negatives (50% cut)**: contain nearly full wake word, confuse model
- **Filled partials (half wake word + random speech)**: risk accidentally creating the wake word

### The core unsolved problem
With only ~1 speaker's positives, any model learns "Victor's voice" as a shortcut. TTS positives help marginally but don't solve it. The model needs to learn the phoneme sequence "hei-eli:-zər" independent of speaker, but all high-confidence FPs are the same speaker saying other things.

## Current Best Configs

### CNN (for reference, not phoneme-aware)
```
model_e2e.py: 2D Conv(3,3), 40ms hop, 60-7500Hz, global RMS, dropout=0.3
Best: TP=33 FP=2 FN=1, F1=0.957 (but fails live — envelope-based)
```

### GRU (phoneme-aware, but FP issue)
```
model_e2e.py: MFCC(13)+deltas, 2-layer bidir GRU(64), 20ms hop, CMVN, dropout=0.3
Best: TP=33 FP=11 FN=1, R=0.971 (great recall, too many FPs)
```

## Plan Forward

### Next: pretrained phoneme features (highest priority)
The core issue: training a binary classifier from scratch forces the model to simultaneously learn what phonemes are AND what "hey eliezer" is. With only 70 positive samples, it takes the speaker-identity shortcut instead.

**Solution**: use a pretrained model that already understands phonemes as a frozen feature extractor, then train a small classifier on top. Options:
- **wav2vec2-base** (~95M params, but we freeze it — only run once to extract features, like we did with Google embeddings). Already knows phonemes from pretraining on 960h of speech.
- **HuBERT-base** — similar, may have better phoneme representations.
- Fine-tuned variants exist specifically for phoneme recognition (e.g., `wav2vec2-lv-60-espeak-cv-ft`).

This is the same approach as the Google embedding pipeline, but with a model that actually encodes phoneme content rather than generic audio features. The classifier stays tiny — it just needs to match "hei-eli:-zər" in the phoneme-aware embeddings.

### Hebrew STT as verifier (validated)
Apple's on-device SFSpeechRecognizer with Hebrew (`he-IL`) reliably transcribes "hey eliezer" as **היי אליעזר**. "Eliezer" is a Hebrew name (אליעזר) so the Hebrew model has it natively in its vocabulary — unlike English STT which guesses "Elias", "Eliza", "loser", etc.

**Key findings**:
- Requires Hebrew dictation language installed in System Settings → Keyboard → Dictation
- CLI tool `hear -d -l he-IL -i clip.wav` — on-device, no network needed
- **Latency: ~350ms** for a 2s clip — fast enough for wake word verification
- On clean speech, consistently gets "היי אליעזר" (full match)
- On test data: 11/34 positives get full "היי אליעזר", 0/97 negatives do
- Match on substring "אליעז" to catch minor transcription variations
- Romanian STT returns nothing despite same pronunciation and name existing in Romania

**Two-stage architecture**:
1. Permissive detector (high recall, tolerate FPs) — GRU already has R=0.971
2. Hebrew STT verification on triggered clips — check for "אליעז" in transcript
3. Only fires wake word if both stages agree

### Other ideas
1. **Contrastive/triplet loss**: instead of binary classification, force the model to find what's DIFFERENT about "hey eliezer" vs other speech
2. **CTC or sequence-level loss**: explicitly teach phoneme recognition, not binary classification
3. **Hard negative mining**: deploy current model, collect FPs, add to training

## File Summary
- `model_e2e.py` — model architecture + feature extraction (compute_log_mel)
- `train_e2e.py` — training pipeline (supports --seeds, --epochs, --aug, --max-neg)
- `eval_e2e.py` — evaluation against test_data/
- `detect_e2e.py` — live detection with VAD gating
- `vad.py` — Silero VAD utility (has_speech, filter_speech_files)
- `Makefile` — train-e2e, eval-e2e, detect-e2e targets
- `train_data/negative_no_speech/` — VAD-rejected negatives (kept aside)
- `train_data/negative_partials/` — unused partials backup
- `train_data/tts_positive/` — 168 Kokoro TTS "hey eliezer" samples
