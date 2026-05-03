#!/usr/bin/env python3
"""Train standalone Transformer classifier head on existing preprocessor tensors (additive artifact)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.helpers import MODELS_DIR, ensure_directories  # noqa: E402
from utils.preprocessing import build_feature_frame, chronological_split, load_raw_dataset, transform_with_preprocessor  # noqa: E402

SEQ_LEN = 12


def _windows(X: np.ndarray, y: np.ndarray, seq: int):
    xs, ys = [], []
    for end in range(seq - 1, len(X)):
        xs.append(X[end - seq + 1 : end + 1])
        ys.append(int(y[end]))
    return np.stack(xs, dtype=np.float32), np.asarray(ys, dtype=np.int32)


def main() -> None:
    ensure_directories()
    risk_blob_path = MODELS_DIR / "risk_model.pkl"
    if not risk_blob_path.exists():
        raise SystemExit("risk_model.pkl missing; run python train_model.py first")
    risk_blob = joblib.load(risk_blob_path)
    pre = risk_blob["preprocessor"]

    raw = load_raw_dataset()
    feat_df = build_feature_frame(raw).sort_values("pub_ts").reset_index(drop=True)
    train_df, test_df, _cut = chronological_split(feat_df, test_fraction=0.2)

    X_tr = transform_with_preprocessor(pre, train_df).astype(np.float32)
    X_te = transform_with_preprocessor(pre, test_df).astype(np.float32)
    y_tr = train_df["risk_class"].to_numpy(dtype=np.int32)
    y_te = test_df["risk_class"].to_numpy(dtype=np.int32)

    Xw_tr, yw_tr = _windows(X_tr, y_tr, SEQ_LEN)
    Xw_te, yw_te = _windows(X_te, y_te, SEQ_LEN)

    try:
        from tensorflow import keras  # noqa: PLC0415
        from tensorflow.keras import layers  # noqa: PLC0415
    except ImportError as exc:
        raise SystemExit("TensorFlow required for transformer export") from exc

    inp = keras.Input(shape=(SEQ_LEN, Xw_tr.shape[2]))
    y = inp
    y = layers.Dense(96)(y)
    y = layers.MultiHeadAttention(num_heads=4, key_dim=32)(query=y, value=y)
    y = layers.GlobalAveragePooling1D()(y)
    y = layers.Dense(64, activation="relu")(y)
    out = layers.Dense(4, activation="softmax")(y)

    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(Xw_tr, yw_tr, validation_data=(Xw_te, yw_te), epochs=5, batch_size=512, verbose=1)

    prob = model.predict(Xw_te, verbose=0)
    acc = accuracy_score(yw_te, np.argmax(prob, axis=1))
    out_path = MODELS_DIR / "transformer_model.keras"
    model.save(out_path)

    metrics = MODELS_DIR / "transformer_head_metrics.json"
    metrics.write_text(json.dumps({"seq_len": SEQ_LEN, "val_accuracy_approx": float(acc)}, indent=2), encoding="utf-8")

    print(f"Saved {out_path} acc~={acc:.4f}")


if __name__ == "__main__":
    main()
