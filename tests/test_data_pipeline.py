import pandas as pd
import pytest
from src.config import COLUMNS
from src.data_pipeline import validate, add_features

def sample():
    row1 = [1, 1, 0, 0, 100] + [10.0] * 21
    row2 = [1, 2, 0, 0, 100] + [12.0] * 21
    return pd.DataFrame([row1, row2], columns=COLUMNS)

def test_validate_sorts_and_preserves_rows():
    df = sample().iloc[::-1].reset_index(drop=True)
    result = validate(df, "test")
    assert result.cycle.tolist() == [1, 2]

def test_validate_rejects_duplicate_engine_cycle():
    df = pd.concat([sample(), sample().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate(df, "test")

def test_features_do_not_mix_engines():
    df = sample()
    other = sample().copy()
    other["engine_id"] = 2
    other["sensor_2"] = [1000.0, 1002.0]
    combined = pd.concat([df, other], ignore_index=True)
    features, _ = add_features(combined)
    assert features.loc[features.engine_id == 2, "sensor_2_mean5"].iloc[0] == 1001.0
