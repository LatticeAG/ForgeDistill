# Security policy

Report vulnerabilities privately via GitHub Security Advisories on
[LatticeAG/ForgeDistill](https://github.com/LatticeAG/ForgeDistill).
Do not file public issues for unreleased security bugs.

If Advisories are unavailable, contact the org owners through GitHub.
There is no security@ mailbox for this project.

## API keys

- Never file API keys in issues, pull requests, or traces attached to
  public tickets.
- Keys are env-only. Roster YAML uses `key_env` (the name of an
  environment variable), never the secret itself.
- `configs/roster.yaml` is gitignored. `configs/roster.example.yaml`
  must stay key-free.

## Distillation output

- `data/` is gitignored. Do not attach traces that include teacher
  identities to public issues.
- Run `dataset_publish` first to scrub internal fields (`teacher`,
  `chain_steps`, `vars`, route identifiers) before sharing a bundle.
