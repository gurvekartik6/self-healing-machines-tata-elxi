"""Persistent investigation history and technician feedback."""
from __future__ import annotations
import json, sqlite3
from datetime import datetime, timezone
from pathlib import Path
from src.config import REPORTS
DB=REPORTS/'investigations.db'
def _connect():
    DB.parent.mkdir(parents=True,exist_ok=True); c=sqlite3.connect(DB); c.row_factory=sqlite3.Row
    c.execute('''CREATE TABLE IF NOT EXISTS investigations(id INTEGER PRIMARY KEY AUTOINCREMENT,created_at TEXT NOT NULL,engine_id INTEGER NOT NULL,report_path TEXT NOT NULL,summary TEXT NOT NULL,model TEXT,review_status TEXT NOT NULL DEFAULT 'PENDING',reviewer TEXT,feedback TEXT,helpful INTEGER,correctness TEXT)'''); c.commit(); return c
def save_investigation(engine_id:int, report_path:str, report:dict):
    c=_connect(); c.execute('INSERT INTO investigations(created_at,engine_id,report_path,summary,model) VALUES(?,?,?,?,?)',(datetime.now(timezone.utc).isoformat(),int(engine_id),report_path,str(report.get('analysis',{}).get('summary','')),report.get('analysis',{}).get('model'))); c.commit(); c.close()
def list_investigations(limit:int=100):
    c=_connect(); rows=[dict(x) for x in c.execute('SELECT * FROM investigations ORDER BY id DESC LIMIT ?',(max(1,min(limit,1000)),))]; c.close(); return rows
def submit_feedback(investigation_id:int, reviewer:str, helpful:bool, correctness:str, feedback:str, review_status:str='REVIEWED'):
    if not reviewer.strip(): raise ValueError('Reviewer name is required')
    if review_status not in {'REVIEWED','NEEDS_CORRECTION','REJECTED'}: raise ValueError('Invalid review status')
    c=_connect(); cur=c.execute('UPDATE investigations SET review_status=?,reviewer=?,feedback=?,helpful=?,correctness=? WHERE id=?',(review_status,reviewer,feedback,int(helpful),correctness,int(investigation_id))); c.commit(); c.close()
    if not cur.rowcount: raise ValueError('Investigation not found')
