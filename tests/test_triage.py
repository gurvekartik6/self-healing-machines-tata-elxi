import pandas as pd
from src.triage import decision

def test_urgent():
    assert decision(pd.Series({"predicted_rul": 5, "anomaly": False}))[0] == "URGENT_REVIEW"

def test_plan():
    assert decision(pd.Series({"predicted_rul": 20, "anomaly": False}))[0] == "PLAN_MAINTENANCE"

def test_inspect():
    assert decision(pd.Series({"predicted_rul": 80, "anomaly": True}))[0] == "INSPECT"

def test_monitor():
    assert decision(pd.Series({"predicted_rul": 80, "anomaly": False}))[0] == "MONITOR"
