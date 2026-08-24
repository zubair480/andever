"""Entry point for hosts that expose exactly one port.

    python serve_single_port.py --host 0.0.0.0 --port 7860

``run.py`` binds two sockets: the interface on PORT and the MCP endpoint on
PORT+1. Container hosts route a single port into the container -- Hugging Face
Spaces routes whatever ``app_port`` says and nothing else -- so on those the
MCP endpoint is simply unreachable. This serves both from one port by adding
the interface routes to FastMCP's own streamable-HTTP app, which is the shape
``api/index.py`` already uses on Vercel.

The difference from the Vercel build is that this one stays stateful. Runs live
in this process, so an agent calling ``set_subject_profile`` still fills the
form in the open browser tab and ``run_longevity_loop`` still streams into it,
exactly as it does locally. ``run.py`` is untouched and remains the way to run
this on your own machine.

One caveat worth knowing before pointing the public at it: the stateful session
is process-wide, so two people using the same instance at the same time share
one pending profile. That is fine for a demo and wrong for a service.
"""

import argparse
import json
import os
import queue
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# loopcore.compat has to win the race for the OpenMP runtime, so import the
# package before anything that might pull numpy in behind our back.
from loopcore import looprunner, mcp_app, server, session, store  # noqa: E402

from starlette.responses import (FileResponse, PlainTextResponse,  # noqa: E402
                                 Response, StreamingResponse)
from starlette.routing import Route                                # noqa: E402

WEB_DIR = os.path.join(_ROOT, "web")


def _json(payload, status=200):
    """JSON with the same numpy-tolerant encoding the stdlib server uses."""
    return Response(json.dumps(payload, default=server._jsonable),
                    status_code=status, media_type="application/json",
                    headers={"Cache-Control": "no-store"})


def _public_base(request):
    """The origin a client outside the proxy should use to reach us."""
    host = request.headers.get("x-forwarded-host") or request.headers.get("host", "")
    proto = request.headers.get("x-forwarded-proto") or request.url.scheme or "https"
    return f"{proto}://{host}" if host else ""


# ---------------------------------------------------------------------------
# Static
# ---------------------------------------------------------------------------

async def index(request):
    return FileResponse(os.path.join(WEB_DIR, "index.html"),
                        media_type="text/html; charset=utf-8")


async def static(request):
    name = os.path.normpath(request.path_params["path"]).replace("\\", "/")
    if name.startswith("..") or name.startswith("/"):
        return PlainTextResponse("forbidden", status_code=403)
    path = os.path.join(WEB_DIR, name)
    if not os.path.isfile(path):
        return PlainTextResponse("not found", status_code=404)
    kind = ("text/css; charset=utf-8" if name.endswith(".css")
            else "application/javascript; charset=utf-8" if name.endswith(".js")
            else None)
    return FileResponse(path, media_type=kind)


# ---------------------------------------------------------------------------
# API, one route per branch of loopcore.server.Handler
# ---------------------------------------------------------------------------

async def meta(request):
    return _json(server.meta())


async def connection(request):
    return _json(session.snapshot())


async def mcp_setup(request):
    """Connection instructions, pointing at /mcp on this same origin.

    The two-port version in ``loopcore.server`` offers Codex a stdio command
    run from a local checkout, which makes no sense against a remote instance.
    Everything here is the one HTTP endpoint.
    """
    url = f"{_public_base(request)}/mcp"
    return _json(dict(
        ready=True,
        url=url,
        clients=[
            dict(id="claude-code", name="Claude Code", transport="HTTP",
                 steps=["Run this in any terminal, then ask Claude Code to set "
                        "your profile and run the loop."],
                 command=f"claude mcp add --transport http longevity {url}",
                 verify="claude mcp list"),
            dict(id="codex", name="Codex", transport="HTTP",
                 steps=["Add this to ~/.codex/config.toml."],
                 command=f'[mcp_servers.longevity]\nurl = "{url}"',
                 verify="codex mcp list"),
            dict(id="other", name="Any other MCP client", transport="HTTP",
                 steps=["Streamable HTTP endpoint, served from the same origin "
                        "as this page."],
                 command=url, verify=""),
        ],
        tools=["describe_inputs", "set_subject_profile", "run_longevity_loop",
               "get_last_report", "describe_harness"],
    ))


async def runs(request):
    return _json(dict(runs=store.recent_runs()))


async def run_detail(request):
    run_id = request.path_params["run_id"]
    row = store.run_meta(run_id)
    if not row:
        return _json(dict(error="unknown run"), 404)
    return _json(dict(meta=row, leaderboard=store.leaderboard(run_id),
                      timeline=store.timeline(run_id)))


async def start_run(request):
    try:
        payload = await request.json()
    except Exception:
        return _json(dict(error="invalid JSON"), 400)
    if not (payload.get("profile") or {}).get("age"):
        return _json(dict(error="profile.age is required"), 400)
    run_id = server.start_run(payload)
    session.set_active_run(run_id)
    return _json(dict(run_id=run_id))


async def set_profile(request):
    try:
        payload = await request.json()
    except Exception:
        return _json(dict(error="invalid JSON"), 400)
    if not payload.get("age"):
        return _json(dict(error="age is required"), 400)
    return _json(dict(profile=session.set_profile(payload, source="browser")))


async def stream(request):
    run = server.get_run(request.path_params["run_id"])
    if run is None:
        return _json(dict(error="unknown run"), 404)

    def events():
        # A sync generator, deliberately: Starlette drains it on a worker
        # thread, so blocking on the run's queue here never stalls the event
        # loop. run.events is the source of truth and the queue only wakes us,
        # which is what lets a reconnecting browser replay from the start.
        sent = 0
        while True:
            while sent < len(run.events):
                yield ("data: " + json.dumps(run.events[sent],
                                             default=server._jsonable) + "\n\n")
                sent += 1
            if run.done and sent >= len(run.events):
                break
            try:
                run.queue.get(timeout=15)
            except queue.Empty:
                yield ": keep-alive\n\n"
        yield "event: end\ndata: {}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-store",
                                      "X-Accel-Buffering": "no"})


async def dataset(request):
    kind = request.path_params["kind"]
    path = store.DPO_PATH if kind == "dpo" else store.SFT_PATH
    if not os.path.exists(path):
        return _json(dict(error="dataset not generated yet"), 404)
    return FileResponse(path, media_type="application/x-ndjson",
                        filename=os.path.basename(path))


# ---------------------------------------------------------------------------
# App: the MCP streamable-HTTP app, with our routes added to it.
# ---------------------------------------------------------------------------
# Extending its router rather than mounting it as a sub-app keeps FastMCP's own
# lifespan, which its session manager needs in order to start at all.

app = mcp_app.mcp.streamable_http_app()
app.router.routes.extend([
    Route("/", index),
    Route("/index.html", index),
    Route("/api/meta", meta),
    Route("/api/connection", connection),
    Route("/api/mcp-setup", mcp_setup),
    Route("/api/runs", runs),
    Route("/api/run", start_run, methods=["POST"]),
    Route("/api/profile", set_profile, methods=["POST"]),
    Route("/api/stream/{run_id}", stream),
    Route("/api/run/{run_id}", run_detail),
    Route("/api/dataset/{kind}", dataset),
    Route("/static/{path:path}", static),
])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("PORT", 7860)))
    parser.add_argument("--no-warm", action="store_true",
                        help="bind the port before loading the panel")
    args = parser.parse_args(argv)

    server.BOUND_HOST, server.BOUND_PORT = args.host, args.port
    server.MCP_READY = True

    # Spaces publishes the address it put in front of us. Without it the MCP
    # tools would hand an agent the container-internal 0.0.0.0:7860.
    space_host = os.environ.get("SPACE_HOST")
    if space_host and not server.PUBLIC_URL:
        server.PUBLIC_URL = f"https://{space_host}"

    if not args.no_warm:
        # The reference epigenome and the clock coefficients both come off
        # local disk, so this is seconds rather than the GEO download the
        # first local start does. Failing it should not stop the port opening.
        try:
            looprunner.bootstrap(lambda m: print("  ", m))
        except Exception as exc:
            print(f"  warm-up skipped: {type(exc).__name__}: {exc}")

    import uvicorn

    print(f"\n  Longevity AI Loop   {server.public_url()}")
    print(f"  MCP endpoint        {server.public_url()}/mcp\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
