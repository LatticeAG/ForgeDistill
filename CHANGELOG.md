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

## [0.5.1] - 2026-09-28 - prose format contract (reminder + label fallback)

Measured on 60 live teacher replies (a stealth-preview teacher): 24 were
rejected purely for missing `<thought>` tags while every one of the 60 passed
grounding. The prompt already showed the tag template - some teachers read it
as illustrative and emit bare labelled lines instead. That is a parsing gap,
not a model defect, so the accepted input is widened while the output contract
stays exactly as it was.

- `build_prose_prompt` now ends with an explicit literal-syntax reminder.
- `parse_teacher_output` accepts a second input axis: labelled prose
  (`THOUGHTS:` / `FINAL_ANSWER:`, no tags). Deliberately strict - it needs at
  least n thought units of real length, and the grounding gate still validates
  the final answer afterwards.
- `assemble_trace` re-emits canonical `<thought>` tags for both axes, so the
  exported trace is identical whichever axis the teacher used. The axis is
  recorded on the trace as `prose_format` (`"tags"` | `"labels"`) for eval.
- `parse_final` now stops at a following labelled section, so a trailing
  `REASONING:` block (teacher scratch) cannot leak into the student's final turn.
- 7 new tests, 211 total.

## [0.5.0] - 2026-09-28 - external chain lane (corpus -> chain records)

The agentic task space is finite (47 plan templates, ~1.6k unique chains).
The declared escape hatch - `harvest_prompts.py` writing
`data/seeds/external.jsonl` - was dead end-to-end: its extractor understood
none of the declared source shapes and `self.external_prompts` was loaded
but never read. This release wires the inverse direction: the corpus
supplies the tool calls, the teacher still writes prose only.

- `src/external_chains.py` (operator script, deliberately unpackaged like
  `harvest_prompts.py`): fetches rows from the HF datasets-server API with
  stdlib urllib and emits one chain record per call segment.
  - `Team-ACE/ToolACE` (`default`/`train`, Apache-2.0): ShareGPT turns;
    `Name(args)` call blocks paired with recorded `{"name", "results"}`
    tool turns. Segments drop (never guess) on unmatched results, non-JSON
    payloads, base64/data:image payloads, prompt length outside 8..400,
    or more than 6 steps.
  - `lockon/xlam-function-calling-60k` (`dataset`/`train`, CC-BY-4.0):
    ships calls without results, so results are synthesized
    deterministically from `sha256(tool|args_json|prop)` over each
    declared schema property; declared args echo through. Honest caveat:
    xLAM `result` payloads are synthetic placeholders - they carry real
    call structure, not real API responses. `result_source` records
    `recorded` vs `synthesized` on every record.
  - CLI: `--source toolace|xlam|both`, `--max-rows`, `--out`
    (default `data/seeds/external_chains.jsonl`, CWD-relative), `--append`,
    `--check PATH` (per-source counts + duplicate record_ids).
- `src/agentic_plans.py`: `validate_external_chain` (generic structural
  rules only - the internal mock-surface laws do not apply to corpus tool
  names), `external_tier` (1 easy / 2 medium / 3-4 hard / 5+ expert),
  `external_skills` (only tags already in SKILLS), `build_external_chain`
  (`plan_id=ext-<source>-<record_id>`, `seed_class="agentic"`, raises
  ValueError on invalid records).
- `src/distill_tools.py`: `Distiller(external_chains_path=, external_frac=)`
  loads the JSONL pool once; explicit missing path is a hard error, missing
  default path logs one stderr note. In the worker loop a non-empty pool
  and `rng.random() < external_frac` draws the next unused record under the
  existing lock, applies the same trajectory_hash + shard_keeps +
  used_trajs claim, and falls through to the plan path on exhaustion,
  duplicate, or shard reject (logged once when the pool empties).
  `--external-chains` / `--external-frac` CLI flags; plumbed through
  `run_mp` payloads like `dpo_rate`. `generate_agentic_trace` dispatches
  to `validate_external_chain` when `traj["source"] == "external"`.
- `src/eval_card.py`: `gate_chain(steps, trace)` and
  `gate_dependency_fidelity` dispatch on the `ext-` plan_id prefix -
  corpus chains cannot re-execute against the mock executor, so fidelity
  is the messages<->chain match. `skills.n_templates_used` now counts
  internal plan ids only; `n_external_templates_used` reports the corpus
  side separately.
- `src/dpo_pairs.py`: `_four_gates` passes the trace to `gate_chain`.
- `src/harvest_prompts.py`: extractor learned the ShareGPT `from`/`value`
  shape (hermes, ToolACE) and the glaive plain-text `USER:`/`ASSISTANT:`
  chat blob; SOURCES gained ToolACE and xLAM; `OUT` is CWD-relative
  `data/seeds/external.jsonl` (ROOT deleted, matching the 0.4.0 tree
  convention). The prompt lane stays a prompt source and is still NOT
  wired into generation: an arbitrary external prompt over an unrelated
  deterministic chain would break the grounding gate.
- Tests: 204 (0.4.2 + 27), all offline.

## [0.4.2] - 2026-09-28 - version assertion is no longer a literal

`tests/test_packaging.py::test_pyproject_metadata` hardcoded `== "0.4.0"`, so the 0.4.1 bump failed the suite on a pure metadata change.

- The test now asserts semver shape plus a matching `## [version]` CHANGELOG heading, which is the invariant that actually matters.
- Tests: 177, green on 0.4.2.

## [0.4.1] - 2026-09-28 - public-bundle provenance scrub

Published `eval_card.json` shipped the raw provenance: `input_paths` named the internal trace shards and `teachers.routes` / `tokens.by_route` named the internal routes and upstream model ids.

- `dataset_publish.scrub_eval_card` now anonymises the card in the public bundle: route keys become `teacher-NN`, `input_paths` become `data/raw/traces_shard_N.jsonl`, every count is preserved, and a `provenance_note` records the substitution.
- `dataset_publish.assert_clean_card` fails the publish if any route identity survives scrubbing.
- `test_no_internal_route_aliases` now scans every tracked text file instead of only `tests/`, `src/`, `configs/`, `README.md` (a CHANGELOG edit had carried the aliases past the old scope).
- Tests: 177 (0.4.0 + 3).

## [0.4.0] - 2026-09-28 - Phase 1 production-readiness

Amendments **A1** (CWD-relative paths) and the G1–G5 realizations below. Features F1/F2 and the 0.5.0 version bump are not in this phase.

### G1 — CWD-relative paths (A1)

`distill_tools.ROSTER_PATH` / `OUT_DIR` are `Path("configs/roster.yaml")` and `Path("data/raw")`. `_resolve_out_dir` is `Path(path).expanduser().resolve()` (process CWD). `_load_external` reads `data/seeds/external.jsonl` relative to CWD. `ROOT` is deleted. `forge-status --raw-dir` (default `data/raw`) and `archive_data` defaults `data/raw` / `data/archive` are CWD-relative.

### G2 — wheel-smoke CI job

`.github/workflows/ci.yml` job `wheel-smoke` (Python 3.11/3.12, `needs: test`) builds a wheel, installs it into a fresh venv, `cd`s to `$RUNNER_TEMP`, and exercises CLIs. F1/F2 lines (`distill --preflight`, `forge-lineage --check`, `import preflight`) are not in this phase.

### G3 — operator loop scripts untracked

`prod_loop.sh` and `loop_watcher.sh` are `git rm --cached` (`.gitignore` lines kept). `tests/test_repo_hygiene.py` asserts they stay untracked and that tracked files contain no operator-home absolute paths (skipped when `git` is missing or the tree is not a checkout).

### G4 — internal route aliases removed from tests

`tests/test_dataset_publish.py` uses `example-provider/example-model-thinking` and `local-router/local-model-a`. `test_no_internal_route_aliases` scans every tracked text file (built from fragments so the test does not itself carry the names) for the internal route-alias pattern. The published `eval_card.json` is anonymised by `dataset_publish.scrub_eval_card`: route keys become `teacher-NN`, shard paths become `data/raw/traces_shard_N.jsonl`, counts are preserved, and `assert_clean_card` fails the publish if any identity survives.

### G5 — `requirements-dev.txt` mirrors the dev extra

Pins `pytest>=8`, `build>=1.2`, `twine>=5` (comment style from `requirements.txt`). `tests/test_packaging.py::test_requirements_dev_mirrors_pyproject` asserts set equality ignoring comments.

### Phase 1 gate

- `pytest -q` → `174 passed` (exit 0; N0 + 8).
- `python -m build && twine check dist/*` → exit 0.
- `grep -n "parent.parent" src/distill_tools.py src/status.py src/archive_data.py` → empty.
