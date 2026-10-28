import json
from src.models import run_models
from src.triage import create_orders
from src.config import REPORTS

def run_pipeline():
    print("TELIPORT Self-Healing Machines")
    print("1/3 Cleaning data, engineering features, training and evaluating models...")
    predictions, metrics = run_models()
    print("2/3 Applying triage rules, logging decisions and creating actionable work orders...")
    orders, decisions = create_orders(predictions)
    print("3/3 Saving reports...")
    print(json.dumps(metrics, indent=2))
    print(f"Predictions: {REPORTS / 'predictions.csv'}")
    print(f"Triage decisions: {REPORTS / 'triage_decisions.csv'}")
    print(f"New work orders created: {len(orders)}")
    print(f"Engine decisions recorded: {len(decisions)}")
    print("Pipeline completed.")
    if metrics.get("data_source") == "synthetic_demo":
        print("WARNING: synthetic results are only a software smoke test, not NASA benchmark evidence.")
    return predictions, metrics, orders, decisions

if __name__ == "__main__":
    run_pipeline()
