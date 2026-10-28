import sqlite3
from datetime import datetime, timezone
import pandas as pd
from src.config import DB, REPORTS


def decision(row):
    rul, anomaly = float(row["predicted_rul"]), bool(row["anomaly"])
    if rul <= 10:
        return "URGENT_REVIEW", "Estimated RUL is 10 cycles or less; request human assessment."
    if rul <= 30:
        return "PLAN_MAINTENANCE", "Estimated RUL is 30 cycles or less; plan maintenance."
    if anomaly:
        return "INSPECT", "Unusual sensor pattern flagged; inspect despite higher estimated RUL."
    return "MONITOR", "No urgent action indicated by current rules."


def init_db():
    with sqlite3.connect(DB) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS work_orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_key TEXT NOT NULL,
            engine_id INTEGER NOT NULL,
            cycle INTEGER NOT NULL,
            predicted_rul REAL NOT NULL,
            anomaly INTEGER NOT NULL,
            priority TEXT NOT NULL,
            reason TEXT NOT NULL,
            status TEXT NOT NULL,
            assigned_to TEXT,
            acknowledged_at TEXT,
            escalation_level INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT,
            created_at TEXT NOT NULL
        )""")
        con.execute("""CREATE TABLE IF NOT EXISTS triage_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_key TEXT NOT NULL,
            engine_id INTEGER NOT NULL,
            cycle INTEGER NOT NULL,
            predicted_rul REAL NOT NULL,
            anomaly INTEGER NOT NULL,
            decision TEXT NOT NULL,
            reason TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(run_key, engine_id)
        )""")
        # Safe schema upgrades for databases created by earlier versions.
        existing = {r[1] for r in con.execute("PRAGMA table_info(work_orders)").fetchall()}
        for name, definition in [("assigned_to", "TEXT"), ("acknowledged_at", "TEXT"), ("escalation_level", "INTEGER NOT NULL DEFAULT 0"), ("updated_at", "TEXT")]:
            if name not in existing:
                con.execute(f"ALTER TABLE work_orders ADD COLUMN {name} {definition}")
        con.execute("""CREATE TABLE IF NOT EXISTS work_order_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT, work_order_id INTEGER NOT NULL,
            action TEXT NOT NULL, actor TEXT NOT NULL, details TEXT NOT NULL, created_at TEXT NOT NULL
        )""")
        # Prevent duplicate active work orders for the same engine across repeated runs.
        con.execute("""CREATE UNIQUE INDEX IF NOT EXISTS ux_active_work_order_engine
            ON work_orders(engine_id)
            WHERE status IN ('OPEN', 'ASSIGNED', 'IN_PROGRESS')""")


def create_orders(predictions):
    init_db()
    run_key = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    decision_rows = []
    actionable_rows = []

    for _, r in predictions.iterrows():
        priority, reason = decision(r)
        engine_id, cycle = int(r.engine_id), int(r.cycle)
        decision_rows.append((
            run_key, engine_id, cycle, float(r.predicted_rul), int(bool(r.anomaly)),
            priority, reason, now
        ))
        # Monitoring decisions belong in the decision log, not the work-order queue.
        if priority != "MONITOR":
            actionable_rows.append((
                run_key, engine_id, cycle, float(r.predicted_rul), int(bool(r.anomaly)),
                priority, reason, "OPEN", now
            ))

    with sqlite3.connect(DB) as con:
        con.executemany("""INSERT OR IGNORE INTO triage_decisions
            (run_key,engine_id,cycle,predicted_rul,anomaly,decision,reason,created_at)
            VALUES (?,?,?,?,?,?,?,?)""", decision_rows)

        created = []
        for row in actionable_rows:
            try:
                cur = con.execute("""INSERT INTO work_orders
                    (run_key,engine_id,cycle,predicted_rul,anomaly,priority,reason,status,created_at)
                    VALUES (?,?,?,?,?,?,?,?,?)""", row)
                created.append(cur.lastrowid)
            except sqlite3.IntegrityError:
                # An active order already exists for this engine; do not create a duplicate.
                pass
        if created:
            placeholders = ",".join("?" for _ in created)
            orders = pd.read_sql_query(
                f"SELECT * FROM work_orders WHERE id IN ({placeholders}) ORDER BY predicted_rul",
                con, params=created
            )
        else:
            orders = pd.DataFrame(columns=[
                "id", "run_key", "engine_id", "cycle", "predicted_rul", "anomaly",
                "priority", "reason", "status", "created_at"
            ])

        decisions = pd.read_sql_query(
            "SELECT * FROM triage_decisions WHERE run_key=? ORDER BY predicted_rul",
            con, params=(run_key,)
        )

    REPORTS.mkdir(parents=True, exist_ok=True)
    orders.to_csv(REPORTS / "work_orders.csv", index=False)
    decisions.to_csv(REPORTS / "triage_decisions.csv", index=False)
    return orders, decisions


def read_orders(limit=500):
    init_db()
    with sqlite3.connect(DB) as con:
        return pd.read_sql_query(
            "SELECT * FROM work_orders ORDER BY id DESC LIMIT ?",
            con, params=(int(limit),)
        )


def _audit(con, order_id, action, actor, details):
    con.execute("INSERT INTO work_order_audit(work_order_id,action,actor,details,created_at) VALUES (?,?,?,?,?)",
                (int(order_id), action, actor, details, datetime.now(timezone.utc).isoformat(timespec="seconds")))

def update_status(order_id, status, actor="operator"):
    allowed = {"OPEN", "ASSIGNED", "IN_PROGRESS", "RESOLVED", "CANCELLED"}
    if status not in allowed:
        raise ValueError(f"Invalid status: {status}")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with sqlite3.connect(DB) as con:
        cur = con.execute("UPDATE work_orders SET status=?, updated_at=? WHERE id=?", (status, now, int(order_id)))
        if cur.rowcount:
            _audit(con, order_id, "STATUS_CHANGED", actor, status)

def assign_order(order_id, assigned_to, actor="operator"):
    assigned_to = str(assigned_to).strip()
    if not assigned_to: raise ValueError("Assignee cannot be empty")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with sqlite3.connect(DB) as con:
        cur = con.execute("UPDATE work_orders SET assigned_to=?, status='ASSIGNED', updated_at=? WHERE id=?", (assigned_to, now, int(order_id)))
        if cur.rowcount: _audit(con, order_id, "ASSIGNED", actor, assigned_to)

def acknowledge_order(order_id, actor="operator"):
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with sqlite3.connect(DB) as con:
        cur = con.execute("UPDATE work_orders SET acknowledged_at=?, updated_at=? WHERE id=?", (now, now, int(order_id)))
        if cur.rowcount: _audit(con, order_id, "ACKNOWLEDGED", actor, "Work order acknowledged")

def escalate_order(order_id, actor="operator"):
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with sqlite3.connect(DB) as con:
        cur = con.execute("UPDATE work_orders SET escalation_level=escalation_level+1, updated_at=? WHERE id=?", (now, int(order_id)))
        if cur.rowcount: _audit(con, order_id, "ESCALATED", actor, "Escalation level incremented")

def read_audit(order_id=None):
    init_db()
    with sqlite3.connect(DB) as con:
        if order_id is None: return pd.read_sql_query("SELECT * FROM work_order_audit ORDER BY id DESC LIMIT 1000", con)
        return pd.read_sql_query("SELECT * FROM work_order_audit WHERE work_order_id=? ORDER BY id DESC", con, params=(int(order_id),))
