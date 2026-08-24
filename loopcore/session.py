"""Shared state between the browser and a connected agent.

The MCP server and the HTTP server run in one process, so a coding agent that
pushes a subject profile or starts a run shows up live in the open browser tab
rather than in a separate silo.
"""

from . import compat  # noqa: F401

import threading
import time

_LOCK = threading.Lock()
_STATE = {
    "client": None,          # name the connected agent reported
    "client_at": None,
    "calls": 0,
    "profile": None,         # profile pushed by the agent, not yet consumed
    "profile_at": None,
    "profile_source": None,
    "active_run_id": None,
    "last_run_id": None,
    "last_summary": None,
    "log": [],               # short activity trail shown in the connect panel
}


def _note(message):
    _STATE["log"].append(dict(at=time.time(), message=message))
    del _STATE["log"][:-12]


def note_client(name, transport="streamable-http"):
    with _LOCK:
        first = _STATE["client"] != name
        _STATE["client"] = name
        _STATE["client_at"] = time.time()
        _STATE["calls"] += 1
        if first:
            _note(f"{name} connected over {transport}")


def set_profile(profile, source="agent"):
    with _LOCK:
        _STATE["profile"] = dict(profile)
        _STATE["profile_at"] = time.time()
        _STATE["profile_source"] = source
        fields = len([v for v in profile.values() if v not in (None, "")])
        _note(f"{source} pushed a subject profile with {fields} fields")
    return dict(profile)


def take_profile():
    """Read the pending profile without clearing it, for the browser to mirror."""
    with _LOCK:
        return dict(_STATE["profile"]) if _STATE["profile"] else None


def set_active_run(run_id):
    with _LOCK:
        _STATE["active_run_id"] = run_id
        if run_id:
            _note(f"run {run_id} started")


def finish_run(run_id, summary):
    with _LOCK:
        _STATE["active_run_id"] = None
        _STATE["last_run_id"] = run_id
        _STATE["last_summary"] = summary
        _note(f"run {run_id} finished")


def last_summary():
    with _LOCK:
        return _STATE["last_summary"]


def snapshot():
    with _LOCK:
        return dict(
            client=_STATE["client"],
            client_at=_STATE["client_at"],
            connected=bool(_STATE["client"]),
            calls=_STATE["calls"],
            profile=dict(_STATE["profile"]) if _STATE["profile"] else None,
            profile_at=_STATE["profile_at"],
            profile_source=_STATE["profile_source"],
            active_run_id=_STATE["active_run_id"],
            last_run_id=_STATE["last_run_id"],
            log=list(_STATE["log"]),
        )
