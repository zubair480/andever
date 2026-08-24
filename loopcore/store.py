"""Persistence for hypotheses, scores and the preference dataset.

Everything the loop produces lands in one SQLite file so runs are comparable
across sessions, and every scored hypothesis also gets written out as
preference-pair training data (DPO format) and supervised examples for the
top scorers. That is the artefact that closes the loop back onto the model:
the harness result becomes training signal.
"""

from . import compat  # noqa: F401

import json
import os
import sqlite3
import threading
import time

# Container hosts run the image as an unprivileged uid that does not own the
# directory the code was copied into, so a fixed path next to the package is not
# always writable. LONGEVITY_LOOP_RUNTIME moves the database and the exported
# dataset somewhere that is; locally it is unset and nothing changes.
RUNTIME = os.environ.get("LONGEVITY_LOOP_RUNTIME") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runtime")
DB_PATH = os.path.join(RUNTIME, "loop.db")
DPO_PATH = os.path.join(RUNTIME, "dpo_pairs.jsonl")
SFT_PATH = os.path.join(RUNTIME, "sft_dataset.jsonl")

_LOCK = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    created_at  REAL,
    backend     TEXT,
    iterations  INTEGER,
    profile     TEXT,
    baseline    TEXT,
    reference   TEXT,
    status      TEXT
);
CREATE TABLE IF NOT EXISTS hypotheses (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT,
    iteration   INTEGER,
    mode        TEXT,
    title       TEXT,
    mechanism   TEXT,
    rationale   TEXT,
    targets     TEXT,
    protocol    TEXT,
    reward      REAL,
    score       REAL,
    years       REAL,
    gap         REAL,
    detail      TEXT,
    created_at  REAL
);
CREATE INDEX IF NOT EXISTS hyp_run ON hypotheses(run_id, reward DESC);
CREATE TABLE IF NOT EXISTS insights (
    run_id      TEXT,
    iteration   INTEGER,
    text        TEXT
);
"""


def connect():
    os.makedirs(RUNTIME, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    with conn:
        conn.executescript(SCHEMA)
    return conn


_CONN = None


def db():
    global _CONN
    if _CONN is None:
        _CONN = connect()
    return _CONN


def start_run(run_id, backend, iterations, profile, baseline, reference):
    with _LOCK, db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?)",
            (run_id, time.time(), backend, iterations, json.dumps(profile),
             json.dumps(baseline), json.dumps(reference), "running"),
        )


def finish_run(run_id, status="complete"):
    with _LOCK, db() as conn:
        conn.execute("UPDATE runs SET status=? WHERE run_id=?", (status, run_id))


def record(run_id, iteration, mode, hypothesis, scored, display_score):
    row = (
        run_id, iteration, mode, hypothesis["title"], hypothesis["mechanism_class"],
        hypothesis["rationale"], json.dumps(hypothesis["targets"]),
        json.dumps(hypothesis.get("protocol", [])),
        float(scored["reward"]), float(display_score),
        float(scored.get("years_reversed", 0.0)),
        float(scored.get("generalisation_gap", 0.0)),
        json.dumps(dict(gains=scored.get("gains", {}),
                        penalties=scored.get("penalties", {}),
                        benefit=scored.get("benefit"),
                        held_out=scored.get("held_out", []),
                        applied=scored.get("applied", {}),
                        primary_endpoint=hypothesis.get("primary_endpoint"),
                        falsifier=hypothesis.get("falsifier"),
                        protocol_quality=hypothesis.get("protocol_quality", {}))),
        time.time(),
    )
    with _LOCK, db() as conn:
        conn.execute(
            "INSERT INTO hypotheses (run_id, iteration, mode, title, mechanism,"
            " rationale, targets, protocol, reward, score, years, gap, detail,"
            " created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", row)


def add_insight(run_id, iteration, text):
    with _LOCK, db() as conn:
        conn.execute("INSERT INTO insights VALUES (?,?,?)", (run_id, iteration, text))


def leaderboard(run_id, limit=50):
    with _LOCK:
        rows = db().execute(
            "SELECT * FROM hypotheses WHERE run_id=? ORDER BY reward DESC LIMIT ?",
            (run_id, limit)).fetchall()
    return [_row(r) for r in rows]


def timeline(run_id):
    with _LOCK:
        rows = db().execute(
            "SELECT * FROM hypotheses WHERE run_id=? ORDER BY iteration ASC",
            (run_id,)).fetchall()
    return [_row(r) for r in rows]


def run_meta(run_id):
    with _LOCK:
        row = db().execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
    if not row:
        return None
    return dict(run_id=row["run_id"], created_at=row["created_at"],
                backend=row["backend"], iterations=row["iterations"],
                profile=json.loads(row["profile"]),
                baseline=json.loads(row["baseline"]),
                reference=json.loads(row["reference"]), status=row["status"])


def recent_runs(limit=20):
    with _LOCK:
        rows = db().execute(
            "SELECT run_id, created_at, backend, iterations, status FROM runs"
            " ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def _row(r):
    detail = json.loads(r["detail"])
    return dict(
        iteration=r["iteration"], mode=r["mode"], title=r["title"],
        mechanism=r["mechanism"], rationale=r["rationale"],
        targets=json.loads(r["targets"]), protocol=json.loads(r["protocol"]),
        reward=r["reward"], score=r["score"], years_reversed=r["years"],
        generalisation_gap=r["gap"], **detail)


# ---------------------------------------------------------------------------
# Preference dataset export
# ---------------------------------------------------------------------------

def export_preferences(run_id, prompt_context):
    """Write DPO pairs and SFT examples from one run.

    Every hypothesis is paired against every hypothesis that scored materially
    worse on the same prompt context, which is exactly the (prompt, chosen,
    rejected) triple that DPO consumes. The top scorers additionally become
    supervised examples.
    """
    rows = leaderboard(run_id, limit=200)
    if len(rows) < 2:
        return dict(pairs=0, sft=0, dpo_path=DPO_PATH, sft_path=SFT_PATH)

    os.makedirs(RUNTIME, exist_ok=True)
    margin = max(0.05, 0.12 * (rows[0]["reward"] - rows[-1]["reward"]))

    pairs = 0
    with open(DPO_PATH, "a", encoding="utf-8") as fh:
        for i, better in enumerate(rows):
            for worse in rows[i + 1:]:
                if better["reward"] - worse["reward"] < margin:
                    continue
                fh.write(json.dumps(dict(
                    run_id=run_id,
                    prompt=prompt_context,
                    chosen=_as_completion(better),
                    rejected=_as_completion(worse),
                    margin=round(better["reward"] - worse["reward"], 4),
                    chosen_reward=round(better["reward"], 4),
                    rejected_reward=round(worse["reward"], 4),
                ), ensure_ascii=False) + "\n")
                pairs += 1

    keep = [r for r in rows[:5] if r["reward"] > 0]
    with open(SFT_PATH, "a", encoding="utf-8") as fh:
        for r in keep:
            fh.write(json.dumps(dict(
                run_id=run_id,
                messages=[
                    {"role": "user", "content": prompt_context},
                    {"role": "assistant", "content": _as_completion(r)},
                ],
                reward=round(r["reward"], 4),
            ), ensure_ascii=False) + "\n")

    return dict(pairs=pairs, sft=len(keep), dpo_path=DPO_PATH, sft_path=SFT_PATH)


def _as_completion(row):
    return json.dumps(dict(
        title=row["title"],
        mechanism_class=row["mechanism"],
        rationale=row["rationale"],
        targets=row["targets"],
        primary_endpoint=row.get("primary_endpoint"),
        falsifier=row.get("falsifier"),
    ), ensure_ascii=False, indent=2)
