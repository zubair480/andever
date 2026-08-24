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


# ---------------------------------------------------------------------------
# Mirrors
# ---------------------------------------------------------------------------
# A shared instance cannot use the single module-wide profile slot to show an
# agent's work in the browser: one process serves everyone, so a second visitor
# would see the first one's data. A mirror is that slot made per-session. The
# browser generates a code, shows it, and an agent that passes the same code
# gets its run streamed to that one page. No code means no mirroring, which is
# the safe default for anyone who does not opt in.

_MIRRORS = {}
_MIRROR_TTL = 45 * 60          # a browser tab left open all day is not a claim
_MIRROR_LIMIT = 64             # bounded so a stranger cannot grow this forever


def _sweep_mirrors(now):
    stale = [code for code, m in _MIRRORS.items() if now - m["seen"] > _MIRROR_TTL]
    for code in stale:
        _MIRRORS.pop(code, None)
    while len(_MIRRORS) > _MIRROR_LIMIT:
        oldest = min(_MIRRORS, key=lambda c: _MIRRORS[c]["seen"])
        _MIRRORS.pop(oldest, None)


def open_mirror(code):
    """Register a browser session so an agent can stream into it."""
    now = time.time()
    with _LOCK:
        _sweep_mirrors(now)
        entry = _MIRRORS.get(code)
        if entry is None:
            entry = _MIRRORS[code] = dict(events=[], created=now, seen=now,
                                          calls=0)
        entry["seen"] = now
    return code


def mirror_exists(code):
    with _LOCK:
        return code in _MIRRORS


def mirror_event(code, event):
    """Append one loop event to a mirror. False when nobody is listening."""
    if not code:
        return False
    now = time.time()
    with _LOCK:
        entry = _MIRRORS.get(code)
        if entry is None:
            return False
        entry["events"].append(event)
        entry["seen"] = now
        # A run is a few hundred events; keep the tail so a tab that reconnects
        # still sees the end of it without letting this grow without bound.
        del entry["events"][:-400]
        return True


def mirror_note_call(code, tool):
    with _LOCK:
        entry = _MIRRORS.get(code)
        if entry is None:
            return False
        entry["calls"] += 1
        entry["seen"] = time.time()
        entry["last_tool"] = tool
        return True


def mirror_poll(code, since=0):
    """Events after ``since`` for one mirror, plus how many there are now."""
    now = time.time()
    with _LOCK:
        entry = _MIRRORS.get(code)
        if entry is None:
            return dict(known=False, events=[], cursor=0, calls=0)
        entry["seen"] = now
        events = entry["events"][int(since):]
        return dict(known=True, events=list(events),
                    cursor=len(entry["events"]), calls=entry["calls"],
                    last_tool=entry.get("last_tool"))
