# Deploying to Render (free tier)

Render's free web service runs the interface and the MCP endpoint from one
port, which is the same shape the Hugging Face build uses. The difference is
that Render has no image build, so `render.yaml` and `requirements.txt`
are the whole configuration.

Two ways in. The Blueprint path reads `render.yaml` and creates the service
for you. The manual path clicks the same settings in by hand. They produce the
same service, so pick whichever you find easier to trust.

Read [What free actually gives you](#what-free-actually-gives-you) before you
point anyone at the URL. Some of it is worse than it sounds.

## Before you start

You need a GitHub (or GitLab, or Bitbucket) repo containing this project, with
`render.yaml` and `requirements.txt` at its root. Render deploys from a
git remote, not from your laptop, so a local-only folder will not work.

```bash
git init
git add .
git commit -m "longevity loop"
git remote add origin git@github.com:YOUR_NAME/longevity-loop.git
git push -u origin main
```

A free Render account needs an email address and nothing else. No credit card.

**Check one thing first.** `requirements.txt` lists numpy and pandas and
not biolearn, because the eval path is meant to be the pure-numpy one in
`loopcore/fastpanel.py`. If that file is not in your checkout yet, the import
chain still runs through `loopcore/panels.py`, which does `from biolearn.util
import get_data_file` at module load, and the service will fail to boot with
`ModuleNotFoundError: No module named 'biolearn'`. The bottom of
`requirements.txt` has a commented stopgap for that case. Uncomment it,
or land fastpanel first.

## Path A: the Blueprint

This is the shorter one and it is reproducible, because the settings live in a
file you can diff rather than in a form you filled in six months ago.

1. Sign in at <https://dashboard.render.com>.
2. In the top menu click **New**, then **Blueprint**.
3. Find your repo in the list and click **Connect**. First time through, Render
   sends you to GitHub to install the Render app. Grant it access to that one
   repo rather than to everything.
4. Give the Blueprint a name and pick the branch. Leave **Blueprint Path**
   empty if `render.yaml` is at the repo root, which is where it is.
5. Render shows you what it is about to create: one web service called
   `longevity-loop`, on the free plan. Check that it says free.
6. Click **Deploy Blueprint**.

The first build takes a few minutes, most of it pip installing numpy and
pandas. When it finishes the service page shows a URL of the form
`https://longevity-loop.onrender.com`. That is the interface. The MCP endpoint
is the same URL with `/mcp` on the end.

After this, every push to the linked branch that touches `render.yaml`
re-syncs the Blueprint, and every push at all redeploys the service, because
`autoDeployTrigger: commit` is set. To stop the Blueprint auto-syncing, set
**Auto Sync** to **No** on its Settings page.

## Path B: New Web Service, by hand

Use this if you want to see every setting, or if you would rather not give
Render a file that can create resources.

1. Sign in at <https://dashboard.render.com>.
2. Click **New**, then **Web Service**.
3. Connect your repo, same GitHub step as above.
4. Fill the form in:

   | Field | Value |
   |---|---|
   | Name | `longevity-loop` |
   | Language / Runtime | `Python 3` |
   | Branch | your default branch |
   | Region | Oregon, or whichever is closest to you |
   | Root Directory | leave empty |
   | Build Command | `pip install -r requirements.txt` |
   | Start Command | `uvicorn serve_single_port:app --host 0.0.0.0 --port $PORT --workers 1 --timeout-keep-alive 75` |
   | Instance Type | **Free** |

5. Open **Advanced** and set **Health Check Path** to `/api/connection`.
6. Still under **Advanced**, add the environment variables. These are the same
   ones `render.yaml` sets, and the reasons are in the comments there.

   | Key | Value |
   |---|---|
   | `PYTHON_VERSION` | `3.12.11` |
   | `LONGEVITY_LOOP_RUNTIME` | `/tmp/longevity-runtime` |
   | `XDG_CACHE_HOME` | `/tmp/cache` |
   | `MPLCONFIGDIR` | `/tmp/matplotlib` |
   | `OMP_NUM_THREADS` | `1` |
   | `PYTHONUNBUFFERED` | `1` |
   | `PIP_DISABLE_PIP_VERSION_CHECK` | `1` |
   | `PIP_NO_CACHE_DIR` | `1` |

7. Click **Create Web Service**.

Two of those are worth not skipping. `--workers 1` is a correctness
requirement, not a performance one: the session state, the run registry and
the SQLite handle all live in module globals in a single process, so a second
worker would serve a browser tab that cannot see the run an MCP client just
started. And `PYTHON_VERSION` matters because Render's default moved to 3.14 in
February 2026, which is new enough that a dependency without a wheel would try
to compile inside 512 MB and fail.

## Connect an agent to it

The interface has a **Connect agent** panel that prints the exact command with
your real URL in it. It works out the origin from the `X-Forwarded-Host` header
Render's proxy sends, so it is correct without you configuring anything.

```bash
claude mcp add --transport http longevity https://YOUR-SERVICE.onrender.com/mcp
```

One optional extra. `loopcore`'s MCP tools also mention the interface URL in
some of their replies, and they read it from `LONGEVITY_LOOP_PUBLIC_URL`, which
`render.yaml` cannot set because the URL does not exist until the service does.
If you want that text to name the public URL, add it by hand after the first
deploy: service page, **Environment**, **Add Environment Variable**, key
`LONGEVITY_LOOP_PUBLIC_URL`, value `https://YOUR-SERVICE.onrender.com`.

Same place is where `ANTHROPIC_API_KEY` goes if you want the Claude-backed
hypothesis agent instead of the built-in optimiser. Do not put it in
`render.yaml`, which lives in the repo.

## Where the logs are

Three different things get called logs and only one of them is the one you
want.

- **Build failed, service never started.** Service page, **Logs** tab, or the
  build output linked from the failed entry in **Events**. Missing dependency,
  no wheel, pip out of memory: all here.
- **Build succeeded, service will not stay up.** Same **Logs** tab, but you are
  looking for the Python traceback after `==> Running 'uvicorn ...'`. A health
  check that never passes shows up as Render repeatedly restarting the
  instance. `PYTHONUNBUFFERED=1` is set precisely so this arrives while it is
  happening rather than in a burst afterwards.
- **Service is up and a request misbehaved.** Same **Logs** tab, filtered.
  uvicorn's access lines are here.
- **What happened and when.** The **Events** tab is the timeline: deploys,
  restarts, spin-downs, spin-ups. This is where you confirm the instance went
  to sleep rather than crashed.

To redeploy without pushing: **Events** tab, **Manual Deploy** dropdown, then
**Deploy latest commit**. The same dropdown has **Clear build cache & deploy**,
which you want if you changed `requirements.txt` and pip seems to be
ignoring it, and **Restart service**.

## Streaming: does it survive Render's proxy?

Short answer: yes for this app, with one client-side bug worth fixing and one
scenario that will bite you. The long answer matters because both the
interface and the MCP endpoint depend on streaming, so if it broke, nothing
would work.

### What actually streams

The premise that this app streams out of `POST /api/run` is true of the Vercel
build and not of this one. `web/app.js` branches on `state.meta.hosted`, which
only `api/index.py` sets. On Render you get the other path:

| Request | Shape | Duration |
|---|---|---|
| `POST /api/run` | plain JSON, returns a `run_id` | milliseconds |
| `GET /api/stream/{run_id}` | `text/event-stream`, `EventSource` | the whole run |
| `POST /mcp` | `text/event-stream`, one JSON-RPC frame | milliseconds |
| `GET /mcp` | `text/event-stream`, server to client channel | open-ended |

The run itself is not on the connection. `loopcore/server.start_run` puts it on
a daemon thread and the SSE route only reads `run.events`, replaying from index
zero each time it is opened. That decoupling is the single most useful fact
here: a dropped stream loses the live view, not the run.

Note that MCP's streamable HTTP returns `text/event-stream` even for a
one-shot `initialize`. This is not optional plumbing you could route around.

### Findings

**Render sits behind Cloudflare, on every request.** Render states that "All
inbound traffic to Render web services passes through Cloudflare's global
network" ([how Render handles DDoS
attacks](https://render.com/articles/how-render-handles-ddos-attacks)). So
there are two proxy layers between uvicorn and the browser, not one.

**Compression should not touch SSE.** `text/event-stream` is not in
Cloudflare's default compressible content-type list ([Cloudflare
compression](https://developers.cloudflare.com/speed/optimization/content/compression/)),
and the same page documents `cache-control: no-transform` as the way to stop
Cloudflare altering compression. The MCP endpoint already sends exactly that
header, which was confirmed by reading the response locally.

**Buffering is not documented either way.** Render publishes nothing about
response buffering, no setting to control it, and no statement about honouring
`X-Accel-Buffering: no`. That header is sent by both `serve_single_port.stream`
and the MCP layer, and it is free to send, but nobody at Render has said in
writing that it is honoured. What is on the record is circumstantial and
consistent: Render's own [Go and Gin
quickstart](https://render.com/docs/deploy-go-gin) deploys an SSE app, and a
Render article recommends SSE, saying it is "often the simplest and most
reliable solution" for read-only streaming to a UI ([real-time AI
chat](https://render.com/articles/real-time-ai-chat-websockets-infrastructure)).
Render would not build a quickstart on a transport its own proxy broke.

Worth knowing: most of the folklore on this subject lives in threads on
`community.render.com`, which Render [sunset on 24 March
2026](https://render.com/docs/community). Those URLs no longer resolve, so
anyone quoting a Render staff answer about SSE buffering today is quoting
something you cannot go and check.

**The request timeout is 100 minutes.** Render states in two comparison pages
and one article that HTTP responses may take up to 100 minutes ([vs
Vercel](https://render.com/docs/render-vs-vercel-comparison), [vs
Heroku](https://render.com/docs/render-vs-heroku-comparison)). A ten iteration
run does not come close, even at 0.1 vCPU. Forty iterations on a cold, throttled
instance is the only case worth thinking about, and 100 minutes is still a lot
of headroom.

The 100 second and 5 minute figures that circulate are not Render numbers. The
100 second one is Cloudflare's 524, misremembered: Cloudflare documents
[125 seconds](https://developers.cloudflare.com/support/troubleshooting/http-status-codes/cloudflare-5xx-errors/error-524/),
and 524 fires when the origin has not responded at all, so it does not apply
once bytes are flowing. The 5 minute one traces to a dead forum thread.

**There is no documented idle timeout, so assume there is an undocumented
one.** Render publishes no read timeout for a connection with no bytes moving.
For WebSockets it says outright that "your web service and its connected
clients should periodically send each other keepalive messages"
([WebSockets](https://render.com/docs/websocket)), which is the same advice
applied to a different transport. This app already does it: the SSE generator
in `serve_single_port.py` emits `: keep-alive\n\n` after 15 seconds of silence,
so the stream is never quiet for longer than that. Do not remove that.

**Free instances do not restrict streaming.** The [free
tier](https://render.com/docs/free) unsupported list covers scaling past one
instance, persistent disks, edge caching, one-off jobs and shell access. SSE,
WebSockets and request duration are not on it.

### The one that will bite you

Render spins a free service down after "15 minutes without receiving any
inbound traffic. This includes both HTTP requests and WebSocket messages from
existing connections" ([free tier](https://render.com/docs/free)). Render
[extended that definition to inbound WebSocket messages in February
2026](https://render.com/changelog/free-web-services-now-remain-active-while-receiving-websocket-messages),
and said nothing about SSE or long-lived HTTP responses.

That gap matters, because an SSE stream is one inbound request followed by
purely outbound bytes. The initial GET resets the clock and then nothing does.
So a quiet SSE stream almost certainly does not count as activity, and a long
run watched over SSE alone could be spun down mid-run.

For the browser this is already solved, by accident rather than design.
`web/app.js` line 84 runs `setInterval(pollConnection, 2500)`, so an open tab
sends an inbound `GET /api/connection` every 2.5 seconds for as long as it is
open. That is inbound HTTP traffic on the documented list, and it resets the
idle clock 24 times a minute. As long as somebody has the page open, spin-down
during a run cannot happen.

The exposed case is an agent driving the loop over `/mcp` with no browser tab
open. The stateful `run_longevity_loop` blocks until the run finishes, which
the docstring puts at ten to thirty seconds locally and which will be several
minutes on 0.1 vCPU. That is inside 15 minutes, so it is a risk rather than a
certainty, and it gets worse the more iterations are asked for.

### What would need to change

Nothing is required to deploy. These are ordered by how much they buy you.

1. **`web/app.js` line 427 defeats `EventSource` reconnection.**

   ```js
   source.onerror = () => { source.close(); state.source = null; resetButton(); };
   ```

   `EventSource` reconnects on its own after a dropped connection, and this
   handler closes it before it can, then resets the run button as though the
   run had ended. It has not: the daemon thread is still going and
   `GET /api/stream/{run_id}` replays every event from the start on reconnect,
   which is exactly what the code comment says it is for. The fix is to leave
   the source open on error and only close it on the `end` event, or to
   reconnect explicitly after a short delay. Until then, any drop from either
   proxy layer looks to the user like a failed run.

   This is worth doing whether or not Render ever drops a stream, because it
   also covers a laptop lid closing and a phone changing network.

2. **A finished run is recoverable and the UI does not try.** `GET
   /api/run/{run_id}` returns the meta, leaderboard and timeline for a run
   whether or not you watched it happen. On a reconnect failure the page could
   fetch that instead of showing "Run failed".

3. **Cap iterations lower than 40 on Render.** Not for the 100 minute limit,
   which is not close, but because every extra iteration on 0.1 vCPU widens the
   window in which a browserless MCP run can meet the 15 minute idle timer.

4. **If you ever see truncated or bursty streams in practice**, the thing to
   test is whether the two proxy layers coalesce chunks, since Render has not
   documented it. The mitigation is padding: a comment line of a couple of
   kilobytes at the top of the stream forces any buffer to flush early. Do not
   add this pre-emptively. It is ugly and there is no evidence it is needed.

## What free actually gives you

Render's own documentation opens the free tier page with "Free instances have
important limitations, described below. Do not use them for production
applications." Take that at face value. This is a good way to show someone the
loop and a bad way to run a service.

**It sleeps.** 15 minutes without inbound traffic and Render spins the instance
down. The next request wakes it, and Render says that "process takes about one
minute". So the first person to open the link after a quiet afternoon stares at
a blank page for a minute. There is no warm-up trick that fits inside the free
tier: pinging it to stay awake just burns the hour budget below.

**750 instance hours a month, per workspace.** A month is about 730 hours, so
one service that never sleeps would use the entire allowance and one that does
sleep will not come close. Two free services running full time will exceed it.
The spin-down is what makes the budget work, which is the argument against
defeating it with a keep-alive pinger.

**The filesystem is ephemeral and there is no disk to attach.** Render is
explicit that "Paid services can preserve local filesystem changes by attaching
a persistent disk, but Free web services cannot." This is not a small
footnote for this app. `LONGEVITY_LOOP_RUNTIME` points at `/tmp`, and
`loopcore/store.py` puts the SQLite database, `dpo_pairs.jsonl` and
`sft_dataset.jsonl` there. All three are gone on every spin-down, every
redeploy and every restart. Concretely:

- Run history is empty every time the instance wakes up.
- A `run_id` from before a sleep returns 404.
- `/api/dataset/dpo` and `/api/dataset/sft` only have the runs from this
  waking. Download anything you want to keep, in the session that produced it.

**One instance, and one shared session.** Free cannot scale past a single
instance, which suits this app because it needs exactly one anyway. But the
caveat `serve_single_port.py` already documents applies with more force in
public: the pending profile is process-wide, so two people using the same
instance at the same time share one. Fine for a demo, wrong for a service.

**No shell access and no one-off jobs.** You cannot SSH in or open a shell from
the dashboard on a free instance, so the logs are the only debugging tool you
have. This is why `/api/connection` is the health check and why
`PYTHONUNBUFFERED` is set.

**0.1 vCPU and 512 MB.** A tenth of a core is the reason `OMP_NUM_THREADS` is
pinned to 1 and the reason `requirements.txt` has no torch in it. Runs
will be noticeably slower than on your laptop. Nothing breaks; it just takes
longer.

## If it will not start

| Symptom in the logs | Cause |
|---|---|
| `ModuleNotFoundError: No module named 'biolearn'` | `loopcore/fastpanel.py` is not in place. See the stopgap at the bottom of `requirements.txt`. |
| Build killed with no error, or `Killed` during pip | Out of memory. Something in the requirements is too big to install in 512 MB. torch is the usual culprit. |
| `No open ports detected` | The start command is not reading `$PORT`, or bound to `127.0.0.1` instead of `0.0.0.0`. |
| Health check never passes, instance restarts in a loop | The app raised during import. Look above the restart for the traceback. |
| Interface loads, `/mcp` 404s | Something is serving only the static routes. `serve_single_port:app` is the MCP app with routes added to it, so this means the start command is pointing somewhere else. |
| Agent gets a session error on `/mcp` after a quiet period | The instance spun down and the MCP session went with it. Reconnect. |

## Verified locally

The start command in `render.yaml` was run against this checkout with
`PORT=7871` before any of this was written:

```bash
PORT=7871 LONGEVITY_LOOP_RUNTIME=/tmp/longevity-runtime OMP_NUM_THREADS=1 \
  uvicorn serve_single_port:app --host 0.0.0.0 --port $PORT \
  --workers 1 --timeout-keep-alive 75
```

`GET /` returned 200 and 9 KB of HTML. `GET /api/meta` returned 200 and 8.5 KB
of JSON. `GET /api/connection`, the health check path, returned 200 in 2.4 ms.
`POST /mcp` returned a JSON-RPC `initialize` result, as `text/event-stream`
with `cache-control: no-cache, no-transform` and `x-accel-buffering: no`. A two
iteration run streamed out of `GET /api/stream/{run_id}` as chunked
`text/event-stream`. `/api/mcp-setup` given an `X-Forwarded-Host` of
`longevity-loop.onrender.com` returned `https://longevity-loop.onrender.com/mcp`,
which is the mechanism the Connect agent panel relies on behind Render's proxy.
