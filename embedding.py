"""
Shared embedding extraction pipeline.

Loads pretrained melspectrogram + embedding ONNX models.
Used by both detect.py (inference) and train.py (training).
"""

import numpy as np
import onnxruntime as ort


class EmbeddingExtractor:
    def __init__(self, melspec_path="models/melspectrogram.onnx",
                 embedding_path="models/embedding_model.onnx"):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.melspec_sess = ort.InferenceSession(melspec_path, opts)
        self.embed_sess = ort.InferenceSession(embedding_path, opts)

    def extract_fixed(self, audio_float, sr=16000, n_frames=16):
        """Extract fixed-size embeddings from audio. RMS-normalizes amplitude."""
        rms = np.sqrt(np.mean(audio_float ** 2))
        if rms > 0:
            audio_float = audio_float / rms * 0.1
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
