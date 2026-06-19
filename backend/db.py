import json
import sqlite3
from pathlib import Path
from typing import Optional

ROOT    = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "annotation_tool.db"

DEFAULTS = {
    "classes_dir":        str(ROOT / "annotatted_classes"),
    "output_base_dir":    str(ROOT / "annotation_jobs"),
    "default_labeler":    "yolo",
    "default_yolo_model": "",
    "roboflow_api_key":   "",
    "roboflow_workspace": "",
}


def _conn():
    c = sqlite3.connect(str(DB_PATH))
    c.row_factory = sqlite3.Row
    return c


def init_db():
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id             TEXT PRIMARY KEY,
                name           TEXT NOT NULL,
                status         TEXT NOT NULL DEFAULT 'running',
                output_dir     TEXT NOT NULL,
                logs           TEXT NOT NULL DEFAULT '[]',
                saved_frames   TEXT NOT NULL DEFAULT '[]',
                skipped_frames TEXT NOT NULL DEFAULT '[]',
                config         TEXT NOT NULL DEFAULT '{}',
                created_at     TEXT NOT NULL
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        for k, v in DEFAULTS.items():
            c.execute(
                "INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)", (k, v)
            )
        c.commit()


def get_settings() -> dict:
    with _conn() as c:
        rows = c.execute("SELECT key, value FROM settings").fetchall()
    return {r["key"]: r["value"] for r in rows}


def update_settings(updates: dict):
    with _conn() as c:
        for k, v in updates.items():
            c.execute(
                "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (k, str(v))
            )
        c.commit()


def create_job(job_id: str, name: str, output_dir: str, config: dict, created_at: str):
    with _conn() as c:
        c.execute(
            "INSERT INTO jobs (id, name, status, output_dir, config, created_at) "
            "VALUES (?, ?, 'running', ?, ?, ?)",
            (job_id, name, output_dir, json.dumps(config), created_at),
        )
        c.commit()


def get_job(job_id: str) -> Optional[dict]:
    with _conn() as c:
        row = c.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["logs"]           = json.loads(d["logs"])
    d["saved_frames"]   = json.loads(d["saved_frames"])
    d["skipped_frames"] = json.loads(d["skipped_frames"])
    d["config"]         = json.loads(d["config"])
    return d


def list_jobs() -> list:
    with _conn() as c:
        rows = c.execute(
            "SELECT id, name, status, saved_frames, skipped_frames, created_at "
            "FROM jobs ORDER BY created_at DESC"
        ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["saved"]   = len(json.loads(d.pop("saved_frames")))
        d["skipped"] = len(json.loads(d.pop("skipped_frames")))
        result.append(d)
    return result


def append_log(job_id: str, lines: list):
    with _conn() as c:
        row = c.execute("SELECT logs FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not row:
            return
        logs = json.loads(row["logs"])
        logs.extend(lines)
        c.execute("UPDATE jobs SET logs = ? WHERE id = ?", (json.dumps(logs), job_id))
        c.commit()


def set_status(job_id: str, status: str):
    with _conn() as c:
        c.execute("UPDATE jobs SET status = ? WHERE id = ?", (status, job_id))
        c.commit()


def mark_saved(job_id: str, filename: str):
    with _conn() as c:
        row = c.execute(
            "SELECT saved_frames, skipped_frames FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
        saved   = json.loads(row["saved_frames"])
        skipped = json.loads(row["skipped_frames"])
        if filename not in saved:
            saved.append(filename)
        skipped = [s for s in skipped if s != filename]
        c.execute(
            "UPDATE jobs SET saved_frames = ?, skipped_frames = ? WHERE id = ?",
            (json.dumps(saved), json.dumps(skipped), job_id),
        )
        c.commit()


def mark_skipped(job_id: str, filename: str):
    with _conn() as c:
        row = c.execute(
            "SELECT saved_frames, skipped_frames FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
        saved   = json.loads(row["saved_frames"])
        skipped = json.loads(row["skipped_frames"])
        if filename not in skipped:
            skipped.append(filename)
        saved = [s for s in saved if s != filename]
        c.execute(
            "UPDATE jobs SET saved_frames = ?, skipped_frames = ? WHERE id = ?",
            (json.dumps(saved), json.dumps(skipped), job_id),
        )
        c.commit()


def delete_job(job_id: str):
    with _conn() as c:
        c.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        c.commit()
