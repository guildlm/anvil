# Contributing to GuildLM Anvil

Thanks for your interest in improving Anvil, GuildLM's training infrastructure.

## Development setup

Anvil is split into a **light core** (configs, data formatting, orchestration)
and a **heavy training extra** (torch, transformers, peft, trl, ...). You only
need the light core to develop and test most of the codebase.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"        # pyyaml, typer, pytest, ruff
```

To actually run training locally on a GPU box, also install the training extra:

```bash
pip install -e ".[train]"
```

## Ground rules

- **No heavy imports at module top level.** torch/transformers/peft/trl/datasets
  must be imported lazily inside functions, guarded with an informative
  `ImportError`. This keeps the package importable and unit-testable on CPU-only
  CI.
- **Pure-python core stays pure.** `config.py`, the formatting helpers in
  `data.py`, and the `plan_*`/`build_*`/`validate_*` functions must not depend on
  the ML stack and must be covered by tests that pass with only `pyyaml`+`pytest`.
- Type hints and docstrings on all public functions.
- Logging via the module `logger`, never bare `print`.

## Before opening a PR

```bash
ruff check src tests
pytest -q
```

Both must be green. CI runs the same checks across Python 3.10–3.12 **without**
installing torch, so make sure your change does not require it for import.

## Commit messages

Keep them clear and imperative. Reference the area you touched
(`config:`, `data:`, `train:`, `merge:`, `quantize:`).
