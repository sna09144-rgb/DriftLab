import json, sqlite3
from io import StringIO
from pathlib import Path
import pandas as pd
from .config import FORMAT_VERSION

def default_db_path():
    return Path(__file__).resolve().parents[1] / "results" / "driftlab.sqlite3"

def _connect(path=None):
    p = Path(path or default_db_path())
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p)
    con.execute("""CREATE TABLE IF NOT EXISTS experiments (
        experiment_id TEXT PRIMARY KEY, created_at TEXT NOT NULL,
        config_json TEXT NOT NULL, detail_json TEXT NOT NULL,
        aggregate_json TEXT NOT NULL, format_version INTEGER NOT NULL)""")
    con.commit()
    return con

def save_result(result, path=None):
    con = _connect(path)
    try:
        con.execute("INSERT OR REPLACE INTO experiments VALUES (?, ?, ?, ?, ?, ?)", (
            result.experiment_id, result.created_at, json.dumps(result.config, ensure_ascii=False),
            result.detail.to_json(orient="records"), result.aggregate.to_json(orient="records"), FORMAT_VERSION))
        con.commit()
    finally: con.close()
    return str(Path(path or default_db_path()))

def list_experiments(path=None):
    con = _connect(path)
    try:
        rows = con.execute("SELECT experiment_id, created_at, config_json, format_version FROM experiments ORDER BY created_at DESC").fetchall()
    finally: con.close()
    out=[]
    for eid, created, cfg, version in rows:
        try: config=json.loads(cfg)
        except json.JSONDecodeError: config={}
        out.append({"experiment_id":eid,"created_at":created,"config":config,"format_version":version})
    return out

def load_result(experiment_id, path=None):
    con = _connect(path)
    try: row=con.execute("SELECT created_at, config_json, detail_json, aggregate_json FROM experiments WHERE experiment_id=?", (experiment_id,)).fetchone()
    finally: con.close()
    if row is None: raise KeyError(f"Эксперимент {experiment_id} не найден")
    created, cfg, detail, aggregate = row
    return {"experiment_id":experiment_id,"created_at":created,"config":json.loads(cfg),
            "detail":pd.read_json(StringIO(detail), orient="records"),"aggregate":pd.read_json(StringIO(aggregate), orient="records")}

def clear_history(path=None):
    # Explicitly opt-in only; the UI does not call this automatically.
    con=_connect(path)
    try: con.execute("DELETE FROM experiments"); con.commit()
    finally: con.close()
