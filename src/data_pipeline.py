import numpy as np
import pandas as pd

from src.config import RAW, PROCESSED, COLUMNS, SETTINGS, SENSORS, FEATURE_SENSORS, RUL_CAP

SYNTHETIC_MARKER = RAW / "SYNTHETIC_DATA_NOTICE.txt"


def read_file(path):
    df = pd.read_csv(path, sep=r"\s+", header=None).dropna(axis=1, how="all")
    if df.shape[1] != len(COLUMNS):
        raise ValueError(f"{path.name}: expected 26 columns, found {df.shape[1]}. Check the file.")
    df.columns = COLUMNS
    return df


def load_data():
    paths = [RAW / "train_FD001.txt", RAW / "test_FD001.txt", RAW / "RUL_FD001.txt"]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing dataset files: {', '.join(missing)}. Run `python generate_demo_data.py` "
            "for synthetic smoke-test data, or add the genuine FD001 files to data/raw/."
        )
    train, test = read_file(paths[0]), read_file(paths[1])
    rul_frame = pd.read_csv(paths[2], sep=r"\s+", header=None)
    if rul_frame.shape[1] != 1:
        raise ValueError("RUL_FD001.txt should contain one RUL value per row.")
    test_rul = pd.to_numeric(rul_frame.iloc[:, 0], errors="coerce")
    if test_rul.isna().any() or (test_rul < 0).any():
        raise ValueError("Test RUL labels must be numeric and non-negative.")
    if len(test_rul) != test["engine_id"].nunique():
        raise ValueError("RUL label count must match the number of test engines.")
    return train, test, test_rul.astype(float)


def validate(df, label, imputation_values=None):
    if df.empty:
        raise ValueError(f"{label} data is empty.")
    out = df.copy()
    # Convert all expected columns to numeric; malformed strings become missing values.
    for col in COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)

    if out["engine_id"].isna().any() or out["cycle"].isna().any():
        raise ValueError(f"{label} data contains missing/invalid engine IDs or cycle numbers.")
    if (out["engine_id"] < 1).any() or (out["cycle"] < 1).any():
        raise ValueError(f"{label} data contains engine IDs or cycles below 1.")
    if not np.allclose(out["engine_id"], out["engine_id"].round()):
        raise ValueError(f"{label} engine IDs must be integers.")
    if not np.allclose(out["cycle"], out["cycle"].round()):
        raise ValueError(f"{label} cycle numbers must be integers.")

    sensor_and_setting_cols = SETTINGS + SENSORS
    all_missing = [c for c in sensor_and_setting_cols if out[c].isna().all()]
    if all_missing:
        raise ValueError(f"{label} columns contain no valid readings: {all_missing}")
    missing_counts = out[sensor_and_setting_cols].isna().sum().to_dict()

    # Fit imputation values on training data only, then reuse them for validation/test data.
    if imputation_values is None:
        imputation_values = {
            col: out[col].median()
            for col in sensor_and_setting_cols
        }
    for col in sensor_and_setting_cols:
        if out[col].isna().any():
            out[col] = out[col].fillna(imputation_values[col])

    out = out.sort_values(["engine_id", "cycle"]).reset_index(drop=True)
    if out.duplicated(["engine_id", "cycle"]).any():
        duplicates = int(out.duplicated(["engine_id", "cycle"]).sum())
        raise ValueError(f"{label} data has {duplicates} duplicate engine/cycle rows.")
    out.attrs["missing_value_counts_before_imputation"] = missing_counts
    out.attrs["imputation_values"] = imputation_values
    return out


def add_features(df):
    """Denoise sensor readings with a centered 3-cycle median, then engineer per-engine features.

    Raw columns remain unchanged; *_denoised columns are used for rolling features/differences.
    Centered filtering is for retrospective analysis, not a causal live-stream filter.
    """
    out = df.sort_values(["engine_id", "cycle"]).copy()
    cols = []
    for sensor in FEATURE_SENSORS:
        denoised = f"{sensor}_denoised"
        out[denoised] = out.groupby("engine_id")[sensor].transform(
            lambda s: s.rolling(window=3, min_periods=1, center=True).median()
        )
        out[f"{sensor}_mean5"] = out.groupby("engine_id")[denoised].transform(
            lambda s: s.rolling(window=5, min_periods=1).mean()
        )
        out[f"{sensor}_diff"] = out.groupby("engine_id")[denoised].diff().fillna(0.0)
        cols.extend([f"{sensor}_mean5", f"{sensor}_diff"])
    cols.extend(SETTINGS)
    return out, cols


def prepare():
    train, test, test_rul = load_data()
    train = validate(train, "Training")
    train_imputation_values = train.attrs["imputation_values"]
    test = validate(test, "Test", imputation_values=train_imputation_values)

    # C-MAPSS RUL labels for training are derived from the last observed cycle per engine.
    train["rul"] = (
        train.groupby("engine_id")["cycle"].transform("max") - train["cycle"]
    ).clip(upper=RUL_CAP)

    train_features, feature_cols = add_features(train)
    test_features, _ = add_features(test)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    train_features.to_csv(PROCESSED / "train_features.csv", index=False)
    test_features.to_csv(PROCESSED / "test_features.csv", index=False)

    data_source = "synthetic_demo" if SYNTHETIC_MARKER.exists() else "cmapss_fd001"
    quality = {
        "data_source": data_source,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_engines": int(train["engine_id"].nunique()),
        "test_engines": int(test["engine_id"].nunique()),
        "train_missing_values_imputed": int(sum(train.attrs.get("missing_value_counts_before_imputation", {}).values())),
        "test_missing_values_imputed": int(sum(test.attrs.get("missing_value_counts_before_imputation", {}).values())),
        "duplicate_engine_cycle_rows": 0,
        "feature_count": len(feature_cols),
    }
    return train_features, test_features, test_rul, feature_cols, quality
