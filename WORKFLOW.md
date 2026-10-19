# Working agreement for this project

How the physicist (Liangyu) and the assistant work on the Pandora optimization study. Standing rules; read with
`CLAUDE.md` (technical notes) and `pandora_opt/PROGRESS.md` (state, decisions, results).

## 1. Mode of work

- **Autonomous after discussion.** Directions are agreed in conversation; afterwards the assistant runs the study
  on its own: it decides methodological and physics details, executes, and records each decision with its rationale
  and a "to review" mark in `PROGRESS.md`. The physicist can overturn any such decision later.
- **Ask first only for:** a single computation above ~10 node-hours; a change of scope or main line; deleting samples
  or other non-regenerable data; any outward action (git push, contacting people, spending outside this account).
- **Do not block on confirmations.** When a reasonable default exists, take it and log it. Never end a turn with an
  open question if work can proceed under an assumption.
- **Single source of truth:** `pandora_opt/PROGRESS.md`, appended after every result (what was done, the number that
  matters, the decision taken, newest at the bottom). Another session can resume from it; a "Next actions" entry is
  written before any planned restart. Slides are updated whenever a result is stable.
- **Communication:** replies in Chinese, short, numbers in tables; code, comments, docs and slides in English.
  Report outcomes exactly (failed runs, skipped steps, unconfirmed numbers are said so).

## 2. Code

- Professional, clear, as short as possible. One shared engine (`pandora_opt/runner.py`) for every evaluation;
  designs (`param_space.py`), screening (`screen.py`), optimizers (`optimize.py`), physics evaluator (`evaluate.py`)
  and analyses are separate, small modules. Change existing code only when necessary; merge rather than duplicate.
- Efficient by construction: vectorised (numpy/awkward), one process per configuration over many input files to
  amortise the 44 s Pandora start-up, all cores of a node used, truth tables cached, PFO files deleted after
  evaluation, every long run resumable from its metrics file.
- Scripts live in files and are run from files. No code through stdin or `python -c` (it triggers permission
  prompts and leaves no record). Temporary helpers go to `data/analysis/pandora_opt/tmp/`.
- Every number that reaches `PROGRESS.md` or a slide is produced by a script that can be rerun; paired comparisons
  on the same events with bootstrap errors; confirmation on independent events before anything is called an
  improvement.

## 3. Compute on NERSC

- Interactive allocations only, never batch submission: `pflow_alloc_cpu 4` / `pflow_alloc_gpu 4` (4 h is the
  maximum; the interactive QOS allows two concurrent allocations per user, so hold two CPU nodes, or one CPU and one
  GPU node, and re-request as soon as one expires). Run work with `pflow_on <jobid>` / the runner's `--jobid a,b`.
- Long runs and multi-hour chains are launched detached (`setsid nohup script.sh &`, logs to files; session background
  tasks are capped at one hour) so that a session restart does not kill
  them; the session waits with background `until` loops, not polling. Keep the login node light: model fitting
  limited to 8 threads; anything heavier goes to the node.
- Allocations expire silently: design every run to resume (runner metrics cache, `.done` markers).
- Kill processes by PID, never `pkill -f <pattern>` from the assistant shell.

## 4. Evaluation protocol (fixed)

- Objective: calibration-invariant visible-energy sigma/mu (`sigma_mu_E_cal`) with constraints relative to the CLD
  default on the same events (efficiency, fake rate, per-category response band 5%, event bias 2%); raw sigma/mu
  reported alongside (`evaluate.py::objective`).
- Event tiers: screening/tuning seeds 1-20 (2000 events) and 121-500; confirmation seeds 21-120 (10k); seeds
  501-800 reserved for model selection; final evaluation seeds 801-1000 (20k), never touched before the paper.
- Equal budgets: one evaluation = Pandora on the fixed 2000 tuning events; LLM tokens and failed runs are counted
  and reported separately.

## 5. Slides

- Built from the physicist's SLAC template by `slides/build_deck.py` (python-pptx), figures in `slides/figs/`,
  output `slides/Agentic_opt_CLD_Pandora.pptx`. Accurate, concise English; every number verified in the session
  and traceable to `PROGRESS.md`; outdated slides are removed, not kept.
- After each build: validate (`scripts/office/validate.py` of the pptx skill) and render with the LibreOffice shifter
  image (`linuxserver/libreoffice:latest`, `--module=none`), then inspect every changed slide for overflow and
  overlap before copying the deck into `slides/`.
- Update cadence: whenever a result is confirmed on the 10k tier or a design decision changes; the status table and
  the step chevrons are kept current.

