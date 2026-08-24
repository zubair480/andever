"""In-memory stand-in for ``loopcore.store``.

Used by the hosted build. It has the same call surface as the SQLite store but
keeps everything in the request that produced it and writes nothing anywhere,
because the hosted endpoint takes health data from strangers and should not be
holding any of it once the response is sent.

The preference pairs still get built; they are returned inline for the caller to
download rather than appended to a file on the server.
"""

from . import compat  # noqa: F401

import json


class MemoryStore:
    def __init__(self):
        self.rows = []
        self.insights = []
        self.meta = None
        self.exported = None

    # -- same surface as loopcore.store ----------------------------------
    def start_run(self, run_id, backend, iterations, profile, baseline, reference):
        self.meta = dict(run_id=run_id, backend=backend, iterations=iterations,
                         reference=reference)

    def finish_run(self, run_id, status="complete"):
        if self.meta:
            self.meta["status"] = status

    def record(self, run_id, iteration, mode, hypothesis, scored, display_score):
        self.rows.append(dict(
            iteration=iteration, mode=mode, title=hypothesis["title"],
            mechanism=hypothesis["mechanism_class"],
            rationale=hypothesis["rationale"], targets=hypothesis["targets"],
            protocol=hypothesis.get("protocol", []),
            reward=float(scored["reward"]), score=float(display_score),
            years_reversed=float(scored.get("years_reversed", 0.0)),
            generalisation_gap=float(scored.get("generalisation_gap", 0.0)),
            gains=scored.get("gains", {}), penalties=scored.get("penalties", {}),
            primary_endpoint=hypothesis.get("primary_endpoint"),
            falsifier=hypothesis.get("falsifier"),
        ))

    def add_insight(self, run_id, iteration, text):
        self.insights.append(dict(iteration=iteration, text=text))

    def leaderboard(self, run_id, limit=50):
        return sorted(self.rows, key=lambda r: -r["reward"])[:limit]

    def export_preferences(self, run_id, prompt_context):
        """Build the DPO pairs and hand them back instead of writing a file."""
        rows = self.leaderboard(run_id, limit=200)
        if len(rows) < 2:
            self.exported = dict(pairs=0, sft=0, dpo=[], sft_rows=[])
            return self.exported

        margin = max(0.05, 0.12 * (rows[0]["reward"] - rows[-1]["reward"]))
        dpo = []
        for i, better in enumerate(rows):
            for worse in rows[i + 1:]:
                if better["reward"] - worse["reward"] < margin:
                    continue
                dpo.append(dict(
                    prompt=prompt_context,
                    chosen=_completion(better),
                    rejected=_completion(worse),
                    margin=round(better["reward"] - worse["reward"], 4),
                    chosen_reward=round(better["reward"], 4),
                    rejected_reward=round(worse["reward"], 4),
                ))

        keep = [r for r in rows[:5] if r["reward"] > 0]
        sft_rows = [dict(messages=[{"role": "user", "content": prompt_context},
                                   {"role": "assistant", "content": _completion(r)}],
                         reward=round(r["reward"], 4)) for r in keep]

        self.exported = dict(pairs=len(dpo), sft=len(sft_rows),
                             dpo=dpo, sft_rows=sft_rows, inline=True)
        return self.exported


def _completion(row):
    return json.dumps(dict(
        title=row["title"], mechanism_class=row["mechanism"],
        rationale=row["rationale"], targets=row["targets"],
        primary_endpoint=row.get("primary_endpoint"),
        falsifier=row.get("falsifier"),
    ), ensure_ascii=False, indent=2)
