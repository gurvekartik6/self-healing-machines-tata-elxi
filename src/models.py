import json
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit

from src.config import MODELS, REPORTS, RUL_CAP
from src.data_pipeline import prepare


def _regression_metrics(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
    }


def run_models():
    train, test, test_rul, features, quality = prepare()
    X = train[features].astype(float)
    y = train["rul"].astype(float)
    groups = train["engine_id"]

    # Split by engine, not by individual rows, to avoid putting cycles from one engine
    # into both the fit and validation sets.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    fit_idx, val_idx = next(splitter.split(X, y, groups=groups))
    X_fit, y_fit = X.iloc[fit_idx], y.iloc[fit_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    baseline = DummyRegressor(strategy="median").fit(X_fit, y_fit)
    baseline_pred = np.clip(baseline.predict(X_val), 0, RUL_CAP)
    baseline_val = _regression_metrics(y_val, baseline_pred)

    validation_model = RandomForestRegressor(
        n_estimators=140, min_samples_leaf=2, random_state=42, n_jobs=-1
    ).fit(X_fit, y_fit)
    val_pred = np.clip(validation_model.predict(X_val), 0, RUL_CAP)
    validation = _regression_metrics(y_val, val_pred)

    # Final model is retrained on all available training engines.
    model = RandomForestRegressor(
        n_estimators=140, min_samples_leaf=2, random_state=42, n_jobs=-1
    ).fit(X, y)
    anomaly_model = IsolationForest(
        n_estimators=120, contamination=0.04, random_state=42
    ).fit(X)

    latest = (
        test.sort_values(["engine_id", "cycle"])
        .groupby("engine_id", as_index=False)
        .tail(1)
        .sort_values("engine_id")
        .reset_index(drop=True)
    )
    X_latest = latest[features].astype(float)
    pred_rul = np.clip(model.predict(X_latest), 0, RUL_CAP)
    anomaly_labels = anomaly_model.predict(X_latest)

    predictions = latest[["engine_id", "cycle"]].copy()
    predictions["predicted_rul"] = np.round(pred_rul, 1)
    predictions["anomaly"] = anomaly_labels == -1
    predictions["anomaly_score"] = np.round(anomaly_model.decision_function(X_latest), 4)
    predictions["true_rul"] = np.asarray(test_rul, dtype=float)
    for feature in features:
        predictions[feature] = X_latest[feature].to_numpy()
    test_metrics = _regression_metrics(predictions["true_rul"], predictions["predicted_rul"])

    MODELS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS / "candidate_rul_model.joblib")
    joblib.dump(anomaly_model, MODELS / "anomaly_model.joblib")
    joblib.dump(features, MODELS / "feature_columns.joblib")
    predictions.to_csv(REPORTS / "predictions.csv", index=False)

    metrics = {
        "data_source": quality["data_source"],
        "data_quality": quality,
        "baseline_validation": {k: round(v, 4) for k, v in baseline_val.items()},
        "random_forest_validation": {k: round(v, 4) for k, v in validation.items()},
        "test": {k: round(v, 4) for k, v in test_metrics.items()},
        "test_engine_count": int(len(predictions)),
        "feature_count": int(len(features)),
        "rul_cap": RUL_CAP,
        "note": (
            "Synthetic results are only a software smoke test, not benchmark evidence."
            if quality["data_source"] == "synthetic_demo"
            else "Test metrics use the provided FD001 RUL labels; report them with the dataset and validation method."
        ),
    }
    # Anomaly detection is unsupervised: report flag rates and overlap with low predicted RUL,
    # but do not claim anomalies are verified failures without failure labels.
    anomaly_count = int(predictions["anomaly"].sum())
    low_rul = predictions["predicted_rul"] <= 30
    metrics["anomaly_evaluation"] = {
        "flagged_engines": anomaly_count,
        "flag_rate": round(float(predictions["anomaly"].mean()), 4),
        "flagged_and_rul_le_30": int((predictions["anomaly"] & low_rul).sum()),
        "low_rul_engines": int(low_rul.sum()),
        "interpretation": "Unsupervised outlier flags are not ground-truth failure labels; inspect sensor traces and maintenance evidence."
    }
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    from src.lifecycle import record_candidate, monitor_predictions
    lifecycle = record_candidate(metrics)
    metrics["model_lifecycle"] = lifecycle
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    monitor_predictions(predictions)
    try:
        from src.lifecycle import write_sensor_plot
        write_sensor_plot()
    except Exception as exc:
        warnings.warn(f"Sensor plot skipped: {exc}")

    # MLflow is optional for local use; persist model artifacts and metrics if available.
    try:
        import mlflow
        mlflow.set_tracking_uri((Path(__file__).resolve().parents[1] / "mlruns").as_uri())
        mlflow.set_experiment("teliport-self-healing-machines")
        with mlflow.start_run():
            mlflow.log_params({
                "model": "RandomForestRegressor",
                "n_estimators": 140,
                "features": len(features),
                "data_source": quality["data_source"],
                "rul_cap": RUL_CAP,
            })
            mlflow.log_metrics({
                "baseline_validation_mae": baseline_val["mae"],
                "baseline_validation_rmse": baseline_val["rmse"],
                "random_forest_validation_mae": validation["mae"],
                "random_forest_validation_rmse": validation["rmse"],
                "test_mae": test_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
            })
            mlflow.log_artifact(str(REPORTS / "metrics.json"))
            mlflow.log_artifact(str(MODELS / "rul_model.joblib"))
            mlflow.log_artifact(str(MODELS / "anomaly_model.joblib"))
            mlflow.log_artifact(str(MODELS / "feature_columns.joblib"))
    except Exception as exc:
        warnings.warn(f"MLflow tracking skipped; JSON metrics were saved. Reason: {exc}")

    return predictions, metrics
