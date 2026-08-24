# Longevity AI Loop

A self-improving agentic loop for longevity research, graded against real
[biolearn](https://bio-learn.github.io/) bio-evals.

```
user parameters
     -> agent generates a longevity hypothesis
     -> the hypothesis is applied to the subject's methylome
     -> a 12-model biolearn panel scores the result
     -> hypothesis + score are stored
     -> the agent reads the high scorers and the credit they earned
     -> next hypothesis
```

The report opens with two numbers:

1. **When, on the current trajectory** — median age at death for a cohort with
   this biomarker profile.
2. **When, if you act on aging** — the same figure if the loop's best-scoring
   protocol is followed and held.

Then the evidence-graded protocol behind number two, the full panel, and a
preference dataset in DPO format, which is the artefact that feeds harness
results back into model training.

## Quick start

```bash
pip install biolearn
```

```bash
python run.py
```

Opens `http://127.0.0.1:8770` and an MCP endpoint on `http://127.0.0.1:8771/mcp`.
The first start downloads and caches one GEO series (~2 minutes); every start
after that is instant.

### Connect an agent instead of filling in a form

The app runs its own MCP server, so Claude Code or Codex can read your health
data from wherever it lives, push the profile into the open browser tab and run
the loop, with results streaming back to the page.

```bash
claude mcp add --transport http longevity http://127.0.0.1:8771/mcp
```

For Codex, add to `~/.codex/config.toml`:

```toml
[mcp_servers.longevity]
command = "python"
args = ["-m", "loopcore.mcp_app"]
cwd = "/path/to/longevity_loop"
```

Then ask it: *read my health data and run the longevity loop on it*. The
**Connect agent** button in the app shows the same instructions with a live
connection indicator. Tools exposed: `describe_inputs`, `set_subject_profile`,
`run_longevity_loop`, `get_last_report`, `describe_harness`.

The form only needs an age. Everything else is optional and sharpens the
estimate.

Headless:

```bash
python run.py --headless --iterations 12
```

## Deploying a public endpoint (Vercel, free tier)

The repo carries a second build that runs the whole thing as one serverless
function, so other people can point their own MCP client at a URL.

```bash
cd longevity_loop
npx vercel        # first run links the project
npx vercel --prod
```

Then anyone can connect with a single command, no key and no account:

```bash
claude mcp add --transport http longevity https://<your-app>.vercel.app/mcp
```

The hosted build differs from the local one in three ways, all forced by
serverless:

| | local | hosted |
|---|---|---|
| dependencies | everything real (~780 MB) | torch, cvxpy, scipy, sklearn stubbed (~135 MB, under Vercel's 250 MB cap) |
| MCP tools | `set_subject_profile` then `run_longevity_loop`, sharing state with the open browser tab | one single-shot `run_longevity_loop(age, sex, ...)` |
| storage | SQLite, datasets written to `runtime/` | nothing at all; the DPO dataset comes back in the response |
| loop length | unlimited | capped at 14 iterations to stay inside the 60 s function limit |

`loopcore/slim.py` registers the stand-ins before biolearn is imported. The two
functions biolearn genuinely needs from those packages are reimplemented in
numpy: `scipy.stats.rankdata` (DunedinPACE quantile-normalises against its gold
standard) and `LinearRegression` (GrimAge regresses DNAmGrimAge on age to get
AgeAccelGrim). Everything else raises rather than returning a wrong number.

That substitution is checked, not assumed:

```bash
python -m tests.parity full > full.json
LONGEVITY_LOOP_SLIM=1 python -m tests.parity slim > slim.json
python -m tests.parity compare full.json slim.json
```

All twelve panel models come back bit-identical. There is no GEO download on
the server either: the fitted reference epigenome ships in
`loopcore/precomputed/` (857 KB), which also means a fresh clone runs offline.

Run the hosted build locally before deploying:

```bash
LONGEVITY_LOOP_SLIM=1 LONGEVITY_LOOP_HOSTED=1 python -m uvicorn api.index:app --port 8790
```

## What is real and what is simulated

This distinction matters, so it is stated in the interface too.

**Real, straight out of biolearn:**

| Piece | Source |
|---|---|
| 12-model eval panel | `biolearn.model_gallery` — Horvath, Hannum, PhenoAge, GrimAge V2, DunedinPACE, StocP, StocZ, YingDamAge, YingAdaptAge, DNAmTL, EpiTOC2, Zhang-10 |
| CpG sets behind every mechanism axis | biolearn coefficient tables (EpiTOC, Reinius deconvolution reference, McCartney smoking / BMI / lipid signatures, DNAmTL, Westerman CVD, the GrimAge sub-models) |
| Reference epigenome | Per-CpG regression on GEO series **GSE41169** (95 whole-blood 450k samples, ages 18-65) loaded through `biolearn.data_library` |
| Panel calibration | Each model's drift per year of age, regressed across real cohort samples |
| Clinical PhenoAge | `biolearn.hematology.phenotypic_age` (Levine 2018) from the submitted blood panel |
| Mortality hazard ratios | Published all-cause HRs for GrimAge acceleration (~1.09/yr), DunedinPACE (~1.26/SD) and PhenoAge acceleration (~1.045/yr), applied below their point estimates because the three are correlated |

**Simulated:** how an intervention moves methylation. No public dataset maps
arbitrary longevity protocols onto per-CpG deltas, so `intervention.py` models
it. Scores rank hypotheses; they are not predicted clinical effects.

## Why the harness is hard to game

An eval an agent can trivially max out teaches it nothing. Five constraints:

1. **Headroom.** Every CpG has a young-adult target from the cohort regression.
   An intervention closes part of the distance and can never overshoot, so the
   reward saturates instead of running away.
2. **User-specific headroom.** Lifestyle burden is written into the baseline
   methylome along real trait-signature directions. A never-smoker has no
   headroom on the smoking axis, so proposing cessation earns nothing.
3. **Five clock families.** Improvements are converted to a common unit — years
   of biological aging undone — by dividing each model's change by its own
   measured drift per year of age. Pushing one clock alone scores badly, and
   disagreement between families is charged as an incoherence penalty.
4. **Held-out clocks.** GrimAge V2, DunedinPACE and StocZ drop out of the reward
   on alternating iterations. The difference is reported as a generalisation gap.
5. **Costs.** Off-target methylation change, protocol burden, more than four
   axes, and total methylation change beyond a plausible one-year budget are all
   charged against the benefit.

The harness is a deterministic function of the intervention vector, so runs
reproduce and two hypotheses that differ only in prose score identically.

## The agent

Two interchangeable backends:

- **Claude** (`ANTHROPIC_API_KEY` set) — the Messages API with structured
  output, given the profile, the baseline readout, the axis catalogue with this
  subject's headroom, every scored hypothesis and an accumulating insight log.
- **Built-in optimiser** (no key needed, the default here) — probes single axes
  to measure their marginal value for this subject, ridge-regresses reward on
  the intensity vectors seen so far to assign credit, then evolves combinations
  with explore/exploit alternation. It is not a language model, and it makes the
  loop demonstrably self-improving with no network at all.

Both emit the same hypothesis shape and both attach a concrete protocol drawn
from a curated, evidence-graded intervention catalogue.

## Layout

```
run.py                 entry point
loopcore/
  compat.py            torch-first import shim (Windows OpenMP crash)
  panels.py            the biolearn assets: eval panel, mechanism axes, CpG universe
  reference.py         fits and caches the reference epigenome from the GEO cohort
  bioevals.py          runs the panel, calibrates it, scores a hypothesis
  intervention.py      the in-silico response model
  evidence.py          curated intervention catalogue with evidence grades
  agent.py             Claude-backed agent and the built-in optimiser
  looprunner.py        the loop
  report.py            the longevity report
  session.py           shared state between the browser and a connected agent
  mcp_app.py           MCP server, local (stateful) and hosted (single-shot) modes
  mortality.py         Gompertz-Makeham lifespan projection
  slim.py              numpy stand-ins for torch/cvxpy/scipy/sklearn
  memstore.py          in-memory store, so the hosted build writes nothing
  precomputed/         fitted reference epigenome, shipped so there is no download
api/index.py           Vercel entrypoint: one ASGI app, UI + API + MCP
tests/parity.py        proves the slim build matches the full one
  store.py             SQLite + DPO/SFT export
  server.py            stdlib HTTP + server-sent events
web/                   the interface
runtime/               caches, database and exported datasets
```

## Notes

- `pip install biolearn` pulls `ecos`, which has no Python 3.13 wheel and needs
  a C++ toolchain. biolearn never imports it, so
  `pip install --no-deps biolearn` plus `appdirs cvxpy matplotlib openpyxl pyyaml
  requests scikit-learn scipy xlrd` works.
- biolearn imports torch after numpy and scipy, which crashes the interpreter on
  Windows. `loopcore/compat.py` imports torch first; import it before biolearn
  anywhere else you use this code.
- `python run.py --rebuild` refits the reference epigenome.

## About the two lifespan numbers

They are cohort projections, not predictions about one person. A Gompertz-Makeham
baseline hazard is calibrated so someone with average biomarkers gets the median
age at death a modern high-income life table gives them, then a proportional-hazards
multiplier is applied from the panel. Epigenetic clocks are validated for ranking
mortality risk across groups; two people with identical panels routinely die twenty
years apart, which is why the report always prints the interquartile band next to
the median. The treated figure additionally assumes the modelled biomarker change
is achieved and then held.

## This is not medical advice

Nothing here is a diagnosis or a treatment plan. Several catalogue entries are
prescription-only and several are preclinical. The report labels both.
