#!/usr/bin/env python3
"""
End-to-end training pipeline for AURA.
Fixed Import: build_feature_frame added to preprocessing imports.
"""

from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple
import joblib
import numpy as np
from sklearn.ensemble import (
    GradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.helpers import LOGS_DIR, MODELS_DIR, ensure_directories, setup_logging
from utils.preprocessing import (
    RISK_NAMES,
    build_feature_frame,       # Added missing import
    build_lstm_split_by_cut,
    chronological_split,
    extract_labels_and_meta,
    fit_transform_preprocessor,
    load_raw_dataset,
)

logger = setup_logging("aura.train", log_file=LOGS_DIR / "training.log")

SEQ_LEN = 12
LSTM_TRAIN_CAP = 24_000
LSTM_TEST_CAP = 6_000

def _evaluate_classifier(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }

def _evaluate_iforest_vs_high_risk(y_risk_test: np.ndarray, if_pred: np.ndarray) -> Dict[str, float]:
    y_alert = (y_risk_test >= 2).astype(int)
    y_hat = (if_pred == -1).astype(int)
    return {
        "if_precision_vs_high_risk": float(precision_score(y_alert, y_hat, zero_division=0)),
        "if_recall_vs_high_risk": float(recall_score(y_alert, y_hat, zero_division=0)),
        "if_f1_vs_high_risk": float(f1_score(y_alert, y_hat, zero_division=0)),
    }

def _subsample_seq(X: np.ndarray, y: np.ndarray, max_n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(X) <= max_n:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=max_n, replace=False)
    return X[idx], y[idx]

def _train_keras_lstm(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, n_classes: int) -> Tuple[Any, Dict[str, Any]]:
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
        
        seq_len, n_features = X_train.shape[1], X_train.shape[2]
        inputs = keras.Input(shape=(seq_len, n_features))
        x = layers.LSTM(96, return_sequences=True)(inputs)
        x = layers.Dropout(0.25)(x)
        x = layers.LSTM(48)(x)
        x = layers.Dropout(0.25)(x)
        outputs = layers.Dense(n_classes, activation="softmax")(x)
        model = keras.Model(inputs, outputs)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        
        early = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_accuracy")
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=512, callbacks=[early], verbose=1)
        
        y_prob = model.predict(X_test, verbose=0)
        y_hat = np.argmax(y_prob, axis=1)
        metrics = _evaluate_classifier(y_test, y_hat)
        hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        return model, {"lstm_test": metrics, "lstm_history": hist}
    except ImportError:
        logger.warning("TensorFlow not found. Skipping LSTM training.")
        return None, {"lstm_test": "Skipped", "lstm_history": {}}

def train_pipeline() -> Dict[str, Any]:
    ensure_directories()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset...")
    raw = load_raw_dataset()
    feature_df = build_feature_frame(raw)
    train_df, test_df, cut_index = chronological_split(feature_df, test_fraction=0.2)

    y_train_risk, y_train_cvss, _ = extract_labels_and_meta(train_df)
    y_test_risk, y_test_cvss, _ = extract_labels_and_meta(test_df)

    X_train, X_test, preprocessor = fit_transform_preprocessor(train_df, test_df)
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)

    logger.info("Training Isolation Forest (Anomaly Detection)")
    isolation = IsolationForest(n_estimators=220, contamination=0.06, random_state=42, n_jobs=-1)
    isolation.fit(X_train)
    
    logger.info("Training Random Forest (Risk Classifier)")
    clf = RandomForestClassifier(n_estimators=140, max_depth=28, class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train_risk)
    y_hat_risk = clf.predict(X_test)
    clf_metrics = _evaluate_classifier(y_test_risk, y_hat_risk)

    logger.info("Training Gradient Boosting (CVSS Regressor)")
    reg = GradientBoostingRegressor(random_state=42, max_depth=5, learning_rate=0.08, n_estimators=200)
    reg.fit(X_train, y_train_cvss)
    y_hat_cvss = np.clip(reg.predict(X_test), 0.0, 10.0)
    rmse = float(np.sqrt(mean_squared_error(y_test_cvss, y_hat_cvss)))

    # --- LSTM Section ---
    logger.info("Preparing LSTM sequence tensors")
    X_full = np.vstack([X_train, X_test]).astype(np.float32)
    y_full_risk = np.concatenate([y_train_risk, y_test_risk]).astype(np.int32)
    X_tr_seq, y_tr_seq, X_te_seq, y_te_seq = build_lstm_split_by_cut(X_full, y_full_risk, seq_len=SEQ_LEN, cut_index=cut_index)
    X_tr_seq, y_tr_seq = _subsample_seq(X_tr_seq, y_tr_seq, LSTM_TRAIN_CAP, seed=11)
    X_te_seq, y_te_seq = _subsample_seq(X_te_seq, y_te_seq, LSTM_TEST_CAP, seed=13)

    lstm_model, lstm_meta = _train_keras_lstm(X_tr_seq, y_tr_seq, X_te_seq, y_te_seq, n_classes=len(RISK_NAMES))
    if lstm_model:
        lstm_path = MODELS_DIR / "lstm_model.h5"
        lstm_model.save(lstm_path)
        logger.info("Saved LSTM Keras model.")

    # --- Saving Blobs ---
    risk_blob = {
        "classifier": clf,
        "isolation_forest": isolation,
        "preprocessor": preprocessor,
        "lstm_seq_len": SEQ_LEN,
        "metrics": {"classifier_test": clf_metrics, **lstm_meta},
    }
    joblib.dump(risk_blob, MODELS_DIR / "risk_model.pkl")

    cvss_blob = {
        "regressor": reg,
        "preprocessor": preprocessor,
        "metrics": {"cvss_rmse_test": rmse},
    }
    joblib.dump(cvss_blob, MODELS_DIR / "cvss_model.pkl")

    summary = {"classifier_accuracy": clf_metrics["accuracy"], "cvss_rmse": rmse}
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    full_metrics = {
        "classifier_test": clf_metrics,
        "cvss_rmse_test": rmse,
        "lstm": lstm_meta,
    }
    (LOGS_DIR / "last_training_metrics.json").write_text(json.dumps(full_metrics, indent=2, default=str), encoding="utf-8")

    print("\n--- Training Complete ---")
    print(json.dumps(summary, indent=2))
    return summary

if __name__ == "__main__":
    train_pipeline()