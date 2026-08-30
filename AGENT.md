# Agent / contributor conventions

Python >= 3.11. Install from a clone:

```
pip install -e ".[dev]"
```

Tests: `pytest -q`. Do not add `sys.path` hacks in new tests;
`tests/conftest.py` already inserts `src/`.

## Do not

- Commit `configs/roster.yaml`, `.env`, or `data/`.
- `rm` `data/raw`. Call `archive_data` (or `distill --wipe`, which
  archives first).
- Unfreeze `--legacy-v1` (it exits 2).
- Add FIXME / TODO / HACK / XXX in `src/`.
- Add `src/__init__.py`. Public imports stay flat py-modules
  (`import distill_tools`, `import forge_assets`), not a nested
  `latticeag` package.
- Expand the plan library: assert `len(PLANS)==47` unless a later
  spec explicitly expands it.

## Console scripts

After install: `distill`, `eval_card`, `export_sft`, `dpo_pairs`,
`eval_live`, `archive_data`, `dataset_publish`, `forge-status`.
`harvest_prompts` and `dedup_filter` stay unpackaged legacy.
