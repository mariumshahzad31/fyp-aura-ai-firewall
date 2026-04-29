"""
Data loading, cleaning, feature engineering, and sequence construction for AURA.
All data is sourced exclusively from data/Dataset-Attacks-Firewall.csv.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils.helpers import DEFAULT_DATA_PATH, setup_logging

logger = setup_logging("aura.preprocessing")

RISK_NAMES: List[str] = ["Safe", "Suspicious", "Malicious", "Critical"]

FIREWALL_REQUIRED_COLUMNS: Set[str] = {
    "Data",
    "mod_date",
    "pub_date",
    "cvss",
    "Firewall Traffics",
    "cwe_code",
    "cwe_name",
    "summary",
    "access_authentication",
    "access_complexity",
    "access_vector",
    "impact_availability",
    "impact_confidentiality",
    "impact_integrity",
}

CAT_COLS = [
    "access_authentication",
    "access_complexity",
    "access_vector",
    "impact_availability",
    "impact_confidentiality",
    "impact_integrity",
]

GENERIC_MAX_ONEHOT_CATEGORIES = 40
GENERIC_RARE_CATEGORY_THRESHOLD = 0.04
GENERIC_ID_UNIQUENESS_THRESHOLD = 0.95

NUM_COLS = [
    "cwe_code",
    "fw_o1",
    "fw_o2",
    "fw_o3",
    "fw_o4",
    "summary_len",
    "pub_ts",
    "mod_ts",
]


def cvss_to_risk_class(cvss: float) -> int:
    """Map CVSS base score to ordered risk class (training label)."""
    if cvss < 4.0:
        return 0
    if cvss < 6.0:
        return 1
    if cvss < 9.0:
        return 2
    return 3


def parse_firewall_octets(value: Any) -> Tuple[float, float, float, float]:
    """Parse 'Firewall Traffics' field into up to four numeric octet-like parts."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0, 0.0, 0.0, 0.0
    text = str(value).strip()
    if not text:
        return 0.0, 0.0, 0.0, 0.0
    parts = [p for p in text.replace("..", ".").split(".") if p != ""]
    nums: List[float] = []
    for p in parts[:4]:
        try:
            nums.append(float(p))
        except ValueError:
            nums.append(0.0)
    while len(nums) < 4:
        nums.append(0.0)
    return tuple(nums[:4])  # type: ignore[return-value]


@lru_cache(maxsize=4)
def _cached_load_raw_dataset(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    logger.info("Loaded dataset rows=%s cols=%s from %s", len(df), len(df.columns), path)
    return df


def load_raw_dataset(csv_path: Optional[str] = None) -> pd.DataFrame:
    """Load the firewall dataset CSV with caching for repeated access."""
    path = DEFAULT_DATA_PATH if csv_path is None else Path(csv_path)
    expected = FIREWALL_REQUIRED_COLUMNS
    df = _cached_load_raw_dataset(str(path))
    if not expected.issubset(set(df.columns)):
        missing = expected - set(df.columns)
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")
    return df.copy()


def _is_firewall_dataset(df: pd.DataFrame) -> bool:
    return FIREWALL_REQUIRED_COLUMNS.issubset(set(df.columns))


def _infer_generic_column_types(df: pd.DataFrame) -> Dict[str, Any]:
    detected = {
        "numeric": [],
        "datetime": [],
        "categorical": [],
        "id": [],
        "timestamp": [],
        "entity": [],
    }
    n = len(df)
    for col in df.columns:
        series = df[col]
        if series.dtype.kind in "biufc":
            detected["numeric"].append(col)
        else:
            numeric = pd.to_numeric(series, errors="coerce")
            numeric_ratio = numeric.notna().sum() / max(1, n)
            dt = pd.to_datetime(series, errors="coerce")
            datetime_ratio = dt.notna().sum() / max(1, n)
            unique_ratio = series.nunique(dropna=True) / max(1, n)
            if datetime_ratio >= 0.75:
                detected["datetime"].append(col)
                detected["timestamp"].append(col)
            elif numeric_ratio >= 0.75:
                detected["numeric"].append(col)
            else:
                detected["categorical"].append(col)
            if unique_ratio >= GENERIC_ID_UNIQUENESS_THRESHOLD:
                detected["id"].append(col)
            if 0.05 < unique_ratio < 0.95 and col not in detected["numeric"] and col not in detected["datetime"]:
                detected["entity"].append(col)
    return detected


def _safe_transform_datetime(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
    if dt.isna().all():
        return pd.Series([pd.NaT] * len(series), index=series.index)
    median = dt.median()
    return dt.fillna(median)


def _cast_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _build_generic_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    schema = _infer_generic_column_types(out)

    for col in schema["datetime"]:
        out[col] = _safe_transform_datetime(out[col])
    for col in schema["numeric"]:
        out[col] = _cast_numeric(out[col])

    # Basic safe fill strategy
    for col in out.columns:
        if col in schema["numeric"]:
            out[col] = out[col].fillna(out[col].median() if not out[col].isna().all() else 0.0).astype(float)
        elif col in schema["datetime"]:
            out[col] = out[col].fillna(out[col].median())
        else:
            out[col] = out[col].fillna("MISSING").astype(str)

    # Normalize timestamp to epoch seconds
    for col in schema["datetime"]:
        out[f"{col}_epoch"] = out[col].astype("int64") // 10**9
        out[f"{col}_hour"] = out[col].dt.hour.fillna(0).astype(int)
        out[f"{col}_weekday"] = out[col].dt.weekday.fillna(0).astype(int)
        out[f"{col}_day"] = out[col].dt.day.fillna(0).astype(int)

    # Frequency and rarity features for categorical columns
    for col in schema["categorical"]:
        value_counts = out[col].value_counts(dropna=False)
        freq = out[col].map(value_counts).fillna(0).astype(float)
        out[f"{col}_freq"] = freq
        out[f"{col}_rare"] = (freq < max(1.0, len(out) * GENERIC_RARE_CATEGORY_THRESHOLD)).astype(int)

    # Entity-based temporal features if a timestamp exists
    primary_ts = schema["timestamp"][0] if schema["timestamp"] else None
    if primary_ts and schema["entity"]:
        out = out.sort_values(primary_ts).reset_index(drop=True)
        for entity_col in schema["entity"]:
            groups = out.groupby(entity_col)[primary_ts]
            delta = groups.diff().dt.total_seconds().fillna(0.0)
            out[f"{entity_col}_event_interval"] = delta.fillna(0.0).astype(float)
            out[f"{entity_col}_frequency"] = groups.transform("count").astype(float)

    # Numeric trend features and score proxies
    for col in schema["numeric"]:
        median = out[col].median() if not out[col].isna().all() else 0.0
        std = out[col].std() if not out[col].isna().all() else 0.0
        out[f"{col}_zscore"] = ((out[col] - median) / (std if std else 1.0)).fillna(0.0).astype(float)
        out[f"{col}_log"] = np.log1p(out[col].abs()).astype(float)

    if schema["timestamp"]:
        ts = out[schema["timestamp"][0]].astype("int64") // 10**9
        out["age_seconds"] = (ts - ts.min()).astype(float)
        out["trend_delta"] = ts.diff().fillna(0).astype(float)

    return out


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Clean rows and engineer tabular features used by all models."""
    if _is_firewall_dataset(df):
        out = df.copy()
        out["summary"] = out["summary"].fillna("").astype(str)
        out["summary_len"] = out["summary"].str.len().clip(0, 20000)

        octets = out["Firewall Traffics"].apply(parse_firewall_octets)
        out["fw_o1"] = octets.apply(lambda t: t[0])
        out["fw_o2"] = octets.apply(lambda t: t[1])
        out["fw_o3"] = octets.apply(lambda t: t[2])
        out["fw_o4"] = octets.apply(lambda t: t[3])

        for col in CAT_COLS:
            out[col] = out[col].fillna("MISSING").astype(str)

        out["pub_ts"] = pd.to_datetime(out["pub_date"], dayfirst=True, errors="coerce")
        out["mod_ts"] = pd.to_datetime(out["mod_date"], dayfirst=True, errors="coerce")
        median_pub = out["pub_ts"].median()
        median_mod = out["mod_ts"].median()
        out["pub_ts"] = out["pub_ts"].fillna(median_pub)
        out["mod_ts"] = out["mod_ts"].fillna(median_mod)
        out["pub_ts"] = out["pub_ts"].astype("int64") // 10**9
        out["mod_ts"] = out["mod_ts"].astype("int64") // 10**9

        out["cwe_code"] = pd.to_numeric(out["cwe_code"], errors="coerce").fillna(0.0)
        out["cvss"] = pd.to_numeric(out["cvss"], errors="coerce")
        if out["cvss"].isna().any():
            raise ValueError("CVSS column contains non-numeric values after coercion")

        out["risk_class"] = out["cvss"].apply(cvss_to_risk_class)
        return out
    return _build_generic_feature_frame(df)


def make_preprocessor() -> ColumnTransformer:
    """Build sklearn preprocessor for numeric and categorical columns."""
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=50),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUM_COLS),
            ("cat", categorical_pipe, CAT_COLS),
        ],
        remainder="drop",
    )


def extract_labels_and_meta(feature_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Extract targets and metadata aligned with feature rows."""
    meta = feature_df[
        ["Data", "cvss", "risk_class", "Firewall Traffics", "cwe_name", "summary"]
    ].copy()
    y_risk = feature_df["risk_class"].to_numpy(dtype=np.int32)
    y_cvss = feature_df["cvss"].to_numpy(dtype=np.float64)
    return y_risk, y_cvss, meta


def fit_transform_preprocessor(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, ColumnTransformer]:
    """Fit preprocessor on train only; transform train and test (no label leakage)."""
    preprocessor = make_preprocessor()
    X_train = preprocessor.fit_transform(train_df[NUM_COLS + CAT_COLS])
    X_test = preprocessor.transform(test_df[NUM_COLS + CAT_COLS])
    logger.info("X_train shape=%s X_test shape=%s", X_train.shape, X_test.shape)
    return X_train, X_test, preprocessor


def transform_with_preprocessor(
    preprocessor: ColumnTransformer, feature_df: pd.DataFrame
) -> np.ndarray:
    """Apply a fitted preprocessor to new rows."""
    return preprocessor.transform(feature_df[NUM_COLS + CAT_COLS])


def dataframe_train_test_split(
    feature_df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified row split on risk_class at the DataFrame level."""
    y = feature_df["risk_class"].to_numpy()
    train_idx, test_idx = train_test_split(
        np.arange(len(feature_df)),
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    train_df = feature_df.iloc[train_idx].reset_index(drop=True)
    test_df = feature_df.iloc[test_idx].reset_index(drop=True)
    return train_df, test_df


def chronological_split(
    feature_df: pd.DataFrame,
    test_fraction: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Order rows by publication time, then split into early train and late test.
    Returns train_df, test_df, and cut_index (first row index belonging to test in the ordered frame).
    """
    ordered = feature_df.sort_values("pub_ts").reset_index(drop=True)
    n = len(ordered)
    if n < 2:
        raise ValueError("Dataset too small for chronological split")
    cut = int(n * (1.0 - test_fraction))
    cut = max(min(cut, n - 1), 1)

    train_df = ordered.iloc[:cut].copy()
    test_df = ordered.iloc[cut:].copy()
    logger.info("Chronological split train_rows=%s test_rows=%s cut_index=%s", len(train_df), len(test_df), cut)
    return train_df, test_df, cut


def build_lstm_split_by_cut(
    X: np.ndarray,
    y_risk: np.ndarray,
    seq_len: int,
    cut_index: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sliding windows over ordered feature rows; train sequences end before cut_index,
    test sequences end at or after cut_index (matches chronological firewall timeline).
    """
    if seq_len < 2:
        raise ValueError("seq_len must be at least 2")
    n = X.shape[0]
    if n < seq_len:
        raise ValueError("Not enough rows to build LSTM sequences")
    X_train_list: List[np.ndarray] = []
    y_train_list: List[int] = []
    X_test_list: List[np.ndarray] = []
    y_test_list: List[int] = []
    for end in range(seq_len - 1, n):
        start = end - seq_len + 1
        window = X[start : end + 1].astype(np.float32)
        label = int(y_risk[end])
        if end < cut_index:
            X_train_list.append(window)
            y_train_list.append(label)
        else:
            X_test_list.append(window)
            y_test_list.append(label)
    if not X_train_list or not X_test_list:
        raise ValueError("LSTM split produced empty train or test sequence set; adjust test_fraction or seq_len")
    X_tr = np.stack(X_train_list, axis=0)
    y_tr = np.array(y_train_list, dtype=np.int32)
    X_te = np.stack(X_test_list, axis=0)
    y_te = np.array(y_test_list, dtype=np.int32)
    logger.info("LSTM train_seq=%s test_seq=%s", X_tr.shape, X_te.shape)
    return X_tr, y_tr, X_te, y_te
