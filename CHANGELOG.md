# Changelog

All notable changes to ForgeDistill are recorded here.

## Phase 0 — Baseline measurement (unmodified tree, 2026-09-04)

Commands run from a checkout with `pip install -e ".[dev]"` (existing `.venv`, pytest 9.1.1). Phase 0 made no code changes; this file is the record.

| Measurement | Command | Result |
|---|---|---|
| **N0** | `pytest -q \| tail -1` | `166 passed in 3.10s` (exit 0) |
| **PLANS / SKILLS** | `python -c "from agentic_plans import PLANS, SKILLS; print(len(PLANS), len(SKILLS))"` | `47 15` |
| **`plan_id` shape** | `python -c "from agentic_plans import build_chain; import random; print(build_chain(random.Random(0))['plan_id'])"` | `cal-fanout-two` — **no `plan_` prefix** |
| **`requirements-dev.txt`** | `cat` / `wc -c` | **10 bytes**, contents exactly `pytest>=8` plus newline |
| **Phase 0 gate** | `python -m build && twine check dist/*` | exit 0 (wheel and sdist PASSED) |

This filename (`CHANGELOG.md`) matches no `.gitignore` pattern.

### OQ2 — which `rejects_{prov}` file gets a cross-teacher answer-route failure

`grep -n "_append_reject" src/distill_tools.py` hits three sites:

- **454** — definition: writes `rejects_{prov}.jsonl` under `self.out_dir`.
- **1020** — call site 1 (validator reject after a kept-looking `ok` trace): **always the thought-teacher worker `prov`**.
- **1061** — call site 2 (`res["ok"]` is false: HTTP / FORMAT / GROUNDING / VERIFY): **defaults to worker `prov`, then overrides from `res["teacher"]` if that string contains `/`** via `attempted.split("/", 1)`.

The only producer of `res["teacher"]` on the fail path is a split thoughts/answer FORMAT failure (`fail["teacher"] = f"{ans_prov}/{ans_model}"`). A cross-teacher **answer-route parse failure** is therefore written to `rejects_{ans_prov}.jsonl` (the attempted answer teacher's provider). GROUNDING / VERIFY / assemble failures after a successful parse do not set `fail["teacher"]`, so those land under the thought-teacher worker `prov`.

## [0.4.0] - 2026-09-28 - Phase 1 production-readiness

Amendments **A1** (CWD-relative paths) and the G1–G5 realizations below. Features F1/F2 and the 0.5.0 version bump are not in this phase.

### G1 — CWD-relative paths (A1)

`distill_tools.ROSTER_PATH` / `OUT_DIR` are `Path("configs/roster.yaml")` and `Path("data/raw")`. `_resolve_out_dir` is `Path(path).expanduser().resolve()` (process CWD). `_load_external` reads `data/seeds/external.jsonl` relative to CWD. `ROOT` is deleted. `forge-status --raw-dir` (default `data/raw`) and `archive_data` defaults `data/raw` / `data/archive` are CWD-relative.

### G2 — wheel-smoke CI job

`.github/workflows/ci.yml` job `wheel-smoke` (Python 3.11/3.12, `needs: test`) builds a wheel, installs it into a fresh venv, `cd`s to `$RUNNER_TEMP`, and exercises CLIs. F1/F2 lines (`distill --preflight`, `forge-lineage --check`, `import preflight`) are not in this phase.

### G3 — operator loop scripts untracked

`prod_loop.sh` and `loop_watcher.sh` are `git rm --cached` (`.gitignore` lines kept). `tests/test_repo_hygiene.py` asserts they stay untracked and that tracked files contain no operator-home absolute paths (skipped when `git` is missing or the tree is not a checkout).

### G4 — internal route aliases removed from tests

`tests/test_dataset_publish.py` uses `example-provider/example-model-thinking` and `local-router/local-model-a`. `test_no_internal_route_aliases` greps `tests/`, `src/`, `configs/`, `README.md` for `lexgf|lexzm|nvdacf|kimcf`.

### G5 — `requirements-dev.txt` mirrors the dev extra

Pins `pytest>=8`, `build>=1.2`, `twine>=5` (comment style from `requirements.txt`). `tests/test_packaging.py::test_requirements_dev_mirrors_pyproject` asserts set equality ignoring comments.

### Phase 1 gate

- `pytest -q` → `174 passed` (exit 0; N0 + 8).
- `python -m build && twine check dist/*` → exit 0.
- `grep -n "parent.parent" src/distill_tools.py src/status.py src/archive_data.py` → empty.
