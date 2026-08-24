"""Standard-library HTTP server with server-sent events.

Deliberately no web framework: the eval harness is CPU-bound and blocking, and
a threading HTTP server with one queue per run is both simpler and more robust
here than an async stack. Endpoints:

    GET  /                     the interface
    GET  /api/meta             backends, axes, panel, reference cohort
    POST /api/run              start a loop, returns a run id
    GET  /api/stream/<run_id>  server-sent events for that run
    GET  /api/run/<run_id>     leaderboard and timeline for a finished run
    GET  /api/runs             recent runs
    GET  /api/dataset/<kind>   download the exported dpo or sft dataset
"""

from . import compat  # noqa: F401

import json
import mimetypes
import os
import queue
import threading
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from . import agent as agents
from . import evidence, looprunner, panels, session, store

WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "web")

_RUNS = {}
_RUNS_LOCK = threading.Lock()
_SENTINEL = object()

# Client hang-ups look different on each platform; treat them all as normal.
_DISCONNECTED = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError,
                 TimeoutError)

# Filled in by serve(), read by the MCP tools when they tell an agent where the
# interface is.
BOUND_HOST = "127.0.0.1"
BOUND_PORT = 8770
MCP_PORT = 8771
MCP_READY = False

# BOUND_HOST/BOUND_PORT describe the socket, which behind a proxy is not the
# address anyone can type. A host that knows its public address sets this so the
# MCP tools point an agent at somewhere reachable.
PUBLIC_URL = os.environ.get("LONGEVITY_LOOP_PUBLIC_URL") or ""


def public_url():
    """Where a person's browser can actually reach the interface."""
    return PUBLIC_URL or f"http://{BOUND_HOST}:{BOUND_PORT}"


def meta():
    """Everything the interface needs to draw itself before any run starts."""
    backends = agents.available_backends()
    return dict(
        backends=backends,
        default_backend=backends[0],
        claude_available="claude" in backends,
        model=agents.MODEL,
        axes={k: {kk: vv for kk, vv in v.items() if kk != "files"}
              for k, v in panels.AXES.items()},
        panel={k: dict(v) for k, v in panels.EVAL_PANEL.items()},
        holdout=panels.GENERALISATION_HOLDOUT,
        interventions=[dict(name=i["name"], category=i["category"],
                            grade=i["grade"], risk=i["risk"])
                       for i in evidence.INTERVENTIONS],
        reference=looprunner.reference_info(),
    )


def mcp_setup():
    """Copy-paste connection instructions for the two agent clients."""
    url = f"http://{BOUND_HOST}:{MCP_PORT}/mcp"
    stdio_cmd = f"python -m loopcore.mcp_app"
    cwd = os.path.dirname(WEB_DIR)
    return dict(
        ready=MCP_READY,
        url=url,
        clients=[
            dict(
                id="claude-code",
                name="Claude Code",
                transport="HTTP",
                steps=["Run this in any terminal, then ask Claude Code to set "
                       "your profile and run the loop."],
                command=f"claude mcp add --transport http longevity {url}",
                verify="claude mcp list",
            ),
            dict(
                id="codex",
                name="Codex",
                transport="stdio",
                steps=[f"Add this to ~/.codex/config.toml. The command must run "
                       f"from {cwd}."],
                command=(f'[mcp_servers.longevity]\n'
                         f'command = "python"\n'
                         f'args = ["-m", "loopcore.mcp_app"]\n'
                         f'cwd = "{cwd.replace(chr(92), "/")}"'),
                verify="codex mcp list",
            ),
            dict(
                id="other",
                name="Any other MCP client",
                transport="HTTP or stdio",
                steps=["Streamable HTTP endpoint, or stdio with the command "
                       "below run from the project directory."],
                command=f"{url}\n\n{stdio_cmd}",
                verify="",
            ),
        ],
        tools=["describe_inputs", "set_subject_profile", "run_longevity_loop",
               "get_last_report", "describe_harness"],
    )


class Run:
    def __init__(self, run_id):
        self.run_id = run_id
        self.queue = queue.Queue()
        self.events = []
        self.done = False
        # Lets an MCP tool call block on the run it just started.
        self.done_event = threading.Event()

    def emit(self, event):
        self.events.append(event)
        self.queue.put(event)

    def finish(self):
        self.done = True
        self.queue.put(_SENTINEL)
        self.done_event.set()


def get_run(run_id):
    with _RUNS_LOCK:
        return _RUNS.get(run_id)


def start_run(payload):
    run_id = uuid.uuid4().hex[:12]
    run = Run(run_id)
    with _RUNS_LOCK:
        _RUNS[run_id] = run

    profile = payload.get("profile", {})
    iterations = int(payload.get("iterations", 10))
    iterations = max(2, min(iterations, 40))
    backend = payload.get("backend", "auto")
    seed = int(payload.get("seed", 0) or 0)

    def worker():
        try:
            report = looprunner.run(profile, iterations=iterations,
                                    backend=backend, emit=run.emit, seed=seed,
                                    run_id=run_id)
            session.finish_run(run_id, report)
        except Exception as exc:
            run.emit(dict(type="error", message=f"{type(exc).__name__}: {exc}",
                          traceback=traceback.format_exc()[-2000:]))
            store.finish_run(run_id, status="failed")
            session.set_active_run(None)
        finally:
            run.finish()

    threading.Thread(target=worker, daemon=True, name=f"loop-{run_id}").start()
    return run_id


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "LongevityLoop/0.1"

    def log_message(self, fmt, *args):  # keep the console readable
        pass

    # -- helpers ---------------------------------------------------------
    def _send(self, status, body, content_type="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, default=_jsonable).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except _DISCONNECTED:
            # The browser polls and navigates away constantly; a client that
            # hangs up mid-response is normal, not an error worth a traceback.
            self.close_connection = True

    def _body(self):
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return {}

    # -- routes ----------------------------------------------------------
    def do_GET(self):
        path = urlparse(self.path).path
        try:
            if path in ("/", "/index.html"):
                return self._static("index.html")
            if path.startswith("/static/"):
                return self._static(path[len("/static/"):])
            if path == "/api/meta":
                return self._send(200, meta())
            if path == "/api/connection":
                return self._send(200, session.snapshot())
            if path == "/api/mcp-setup":
                return self._send(200, mcp_setup())
            if path == "/api/runs":
                return self._send(200, dict(runs=store.recent_runs()))
            if path.startswith("/api/stream/"):
                return self._stream(path.rsplit("/", 1)[-1])
            if path.startswith("/api/run/"):
                run_id = path.rsplit("/", 1)[-1]
                meta = store.run_meta(run_id)
                if not meta:
                    return self._send(404, dict(error="unknown run"))
                return self._send(200, dict(
                    meta=meta, leaderboard=store.leaderboard(run_id),
                    timeline=store.timeline(run_id)))
            if path.startswith("/api/dataset/"):
                return self._dataset(path.rsplit("/", 1)[-1])
            return self._send(404, dict(error="not found"))
        except _DISCONNECTED:
            self.close_connection = True
        except Exception as exc:
            self._send(500, dict(error=f"{type(exc).__name__}: {exc}"))

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            if path == "/api/run":
                payload = self._body()
                if not payload.get("profile", {}).get("age"):
                    return self._send(400, dict(error="profile.age is required"))
                run_id = start_run(payload)
                session.set_active_run(run_id)
                return self._send(200, dict(run_id=run_id))
            if path == "/api/profile":
                payload = self._body()
                if not payload.get("age"):
                    return self._send(400, dict(error="age is required"))
                return self._send(200, dict(
                    profile=session.set_profile(payload, source="browser")))
            return self._send(404, dict(error="not found"))
        except _DISCONNECTED:
            self.close_connection = True
        except Exception as exc:
            self._send(500, dict(error=f"{type(exc).__name__}: {exc}"))

    # -- implementations --------------------------------------------------
    def _stream(self, run_id):
        with _RUNS_LOCK:
            run = _RUNS.get(run_id)
        if run is None:
            return self._send(404, dict(error="unknown run"))

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        sent = 0
        try:
            # Replay anything the run already produced before this connection.
            while True:
                while sent < len(run.events):
                    self._event(run.events[sent])
                    sent += 1
                if run.done and sent >= len(run.events):
                    break
                try:
                    item = run.queue.get(timeout=15)
                except queue.Empty:
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
                    continue
                if item is _SENTINEL:
                    while sent < len(run.events):
                        self._event(run.events[sent])
                        sent += 1
                    break
            self.wfile.write(b"event: end\ndata: {}\n\n")
            self.wfile.flush()
        except (*_DISCONNECTED, OSError):
            self.close_connection = True

    def _event(self, event):
        payload = json.dumps(event, default=_jsonable)
        self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
        self.wfile.flush()

    def _dataset(self, kind):
        path = store.DPO_PATH if kind == "dpo" else store.SFT_PATH
        name = os.path.basename(path)
        if not os.path.exists(path):
            return self._send(404, dict(error="dataset not generated yet"))
        with open(path, "rb") as fh:
            body = fh.read()
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Content-Disposition", f'attachment; filename="{name}"')
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _static(self, name):
        safe = os.path.normpath(name).replace("\\", "/").lstrip("/")
        if safe.startswith(".."):
            return self._send(403, dict(error="forbidden"))
        path = os.path.join(WEB_DIR, safe)
        if not os.path.isfile(path):
            return self._send(404, dict(error="not found"))
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript",):
            ctype += "; charset=utf-8"
        with open(path, "rb") as fh:
            self._send(200, fh.read(), ctype)


def _jsonable(value):
    try:
        import numpy as np
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    return str(value)


def serve(host="127.0.0.1", port=8770, warm=True, mcp_port=None):
    global BOUND_HOST, BOUND_PORT, MCP_PORT, MCP_READY
    BOUND_HOST, BOUND_PORT = host, port
    MCP_PORT = mcp_port or (port + 1)

    if warm:
        print("Warming the bio-eval harness (first run downloads the GEO cohort)")
        looprunner.bootstrap(lambda m: print("  ", m))

    try:
        from . import mcp_app
        mcp_app.serve_http_in_thread(host, MCP_PORT)
        MCP_READY = True
    except Exception as exc:
        print(f"  MCP endpoint disabled: {type(exc).__name__}: {exc}")

    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    print(f"\n  Longevity AI Loop   http://{host}:{port}")
    if MCP_READY:
        print(f"  MCP endpoint        http://{host}:{MCP_PORT}/mcp")
        print(f"  Connect an agent    claude mcp add --transport http "
              f"longevity http://{host}:{MCP_PORT}/mcp")
    print()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
