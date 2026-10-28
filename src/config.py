from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
DB = ROOT / "work_orders.db"
for p in (RAW, PROCESSED, MODELS, REPORTS):
    p.mkdir(parents=True, exist_ok=True)
SETTINGS = [f"setting_{i}" for i in range(1, 4)]
SENSORS = [f"sensor_{i}" for i in range(1, 22)]
COLUMNS = ["engine_id", "cycle", *SETTINGS, *SENSORS]
FEATURE_SENSORS = ["sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_11", "sensor_12", "sensor_15", "sensor_17", "sensor_20", "sensor_21"]
RUL_CAP = 125
