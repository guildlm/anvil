"""Publish trained artifacts to the HuggingFace Hub.

This is the **host** stage of the zero-cost GuildLM path::

    Kaggle (free GPU training) -> HuggingFace Hub (free hosting) -> Ollama (local serving)

After ``anvil-train`` produces a LoRA adapter (or ``anvil-merge`` produces a
merged model), ``anvil-push`` uploads it to a Hub repo and writes a minimal
model card describing it as a GuildLM Code Guild specialist.

Design
------
* The **pure-python** helpers (:func:`validate_push_args`,
  :func:`build_model_card`) have *no* network and *no* ``huggingface_hub``
  dependency, so they are fully unit-testable on a bare CI runner.
* ``huggingface_hub`` is imported **lazily** inside the upload functions (and
  guarded with a friendly install hint), so importing this module — and running
  ``anvil-push --help`` — never requires the Hub SDK to be installed.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "HubError",
    "DEFAULT_LICENSE",
    "validate_push_args",
    "build_model_card",
    "push_adapter",
    "push_merged",
    "main",
]

DEFAULT_LICENSE = "apache-2.0"

# owner/name where each segment is HF-legal: alphanumerics plus -, _ and .
_REPO_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class HubError(ValueError):
    """Raised for invalid push arguments or a missing Hub dependency."""


# --------------------------------------------------------------------------- #
# Pure-python helpers (no network / no huggingface_hub) -- unit tested
# --------------------------------------------------------------------------- #
def validate_push_args(
    repo_id: str,
    adapter_dir: str | None = None,
    *,
    require_dir: bool = True,
) -> str:
    """Validate a ``repo_id`` and (optionally) that ``adapter_dir`` exists.

    Args:
        repo_id: Target Hub repo, must be of the form ``owner/name``.
        adapter_dir: Local directory holding the artifact to upload.
        require_dir: When ``True`` (default) ``adapter_dir`` must be an existing
            directory; set ``False`` to validate just the ``repo_id``.

    Returns:
        The validated ``repo_id`` (unchanged).

    Raises:
        HubError: if the repo id is malformed or the directory is missing.
    """
    if not repo_id or not isinstance(repo_id, str):
        raise HubError("repo_id must be a non-empty string of the form 'owner/name'")

    parts = repo_id.split("/")
    if len(parts) != 2 or not all(parts):
        raise HubError(
            f"repo_id {repo_id!r} must be of the form 'owner/name' "
            "(exactly one '/', non-empty segments)"
        )
    for segment in parts:
        if not _REPO_SEGMENT.match(segment):
            raise HubError(
                f"Invalid segment {segment!r} in repo_id {repo_id!r}; allowed "
                "characters are letters, digits, '-', '_' and '.'"
            )

    if require_dir:
        if not adapter_dir:
            raise HubError("adapter_dir is required")
        path = Path(adapter_dir)
        if not path.is_dir():
            raise HubError(f"adapter_dir does not exist or is not a directory: {path}")

    return repo_id


def build_model_card(
    repo_id: str,
    base_model: str | None = None,
    *,
    merged: bool = False,
    license: str = DEFAULT_LICENSE,
    guild: str = "Code Guild (Go)",
    tool: str = "anvil",
) -> str:
    """Build a minimal Markdown model card (README.md) for a pushed artifact.

    The returned string is pure text (YAML front-matter + body) and embeds the
    base model, license and guild, so it is trivially unit-testable. ``merged``
    toggles the wording/library between a standalone LoRA *adapter* and a
    *merged* full model.

    Returns:
        The full README.md contents as a string.
    """
    kind = "merged model" if merged else "LoRA adapter"
    library = "transformers" if merged else "peft"
    base = base_model or "the configured base model"

    front_matter_lines = [
        "---",
        f"license: {license}",
        f"library_name: {library}",
        "tags:",
        "- guildlm",
        "- anvil",
        "- code-guild",
        "- go",
        "- qlora" if not merged else "- merged",
    ]
    if base_model:
        front_matter_lines.append(f"base_model: {base_model}")
    front_matter_lines.append("---")
    front_matter = "\n".join(front_matter_lines)

    merge_hint = (
        ""
        if merged
        else (
            "\n## Use with the base model\n\n"
            "This is a LoRA adapter — load it on top of the base model with PEFT, "
            "or merge it for serving:\n\n"
            "```bash\n"
            f"anvil-merge --base-model {base} \\\n"
            "    --adapter ./adapter --output-dir ./merged --dtype bfloat16\n"
            "```\n"
        )
    )

    body = f"""
# {repo_id}

A **GuildLM {guild}** Go specialist {kind}, trained with
[`{tool}`](https://github.com/guildlm/anvil), the GuildLM training forge.

| | |
| --- | --- |
| **Guild** | {guild} |
| **Artifact** | {kind} |
| **Base model** | `{base}` |
| **Training tool** | [`guildlm-{tool}`](https://github.com/guildlm/anvil) (QLoRA SFT) |
| **License** | {license} |

## What it is

This {kind} specializes {base!r} for Go code tasks (review, generation,
explanation, testing) as part of the [GuildLM](https://github.com/guildlm)
Code Guild. It was produced by Anvil's config-driven QLoRA supervised
fine-tuning pipeline.
{merge_hint}
## Training

- **Method:** QLoRA (4-bit NF4 base + LoRA adapters), supervised fine-tuning.
- **Tool:** `guildlm-{tool}` — see [TRAINING.md](https://github.com/guildlm/anvil/blob/main/TRAINING.md)
  for the end-to-end free recipe (Kaggle GPU -> HuggingFace Hub -> Ollama).

## Limitations

Quality is bounded by the training dataset. If trained on the offline-synthetic
smoke-test sample, this model only learns a placeholder format — use a real
teacher-generated dataset (forge online mode) for a shippable specialist.
""".rstrip()

    return f"{front_matter}\n{body}\n"


def _write_model_card(target_dir: Path, card: str) -> Path:
    """Write *card* to ``target_dir/README.md`` and return the path."""
    readme = target_dir / "README.md"
    readme.write_text(card, encoding="utf-8")
    return readme


# --------------------------------------------------------------------------- #
# Heavy path (lazy huggingface_hub import)
# --------------------------------------------------------------------------- #
def _require_hub():  # pragma: no cover - needs huggingface_hub installed
    """Import ``huggingface_hub`` with a friendly error if it is missing."""
    try:
        from huggingface_hub import HfApi, create_repo, upload_folder
    except ImportError as exc:
        raise HubError(
            "Pushing to the Hub requires 'huggingface_hub'. Install it with "
            "`pip install guildlm-anvil[train]` (or `pip install huggingface_hub`)."
        ) from exc
    return HfApi, create_repo, upload_folder


def _push_folder(
    repo_id: str,
    folder: str,
    *,
    token: str | None,
    private: bool,
    base_model: str | None,
    merged: bool,
    commit_message: str,
) -> str:  # pragma: no cover - network/SDK side effects
    """Shared upload path: validate, create repo, write card, upload folder."""
    validate_push_args(repo_id, folder, require_dir=True)
    _HfApi, create_repo, upload_folder = _require_hub()

    target = Path(folder)
    card = build_model_card(repo_id, base_model=base_model, merged=merged)
    _write_model_card(target, card)

    logger.info("Ensuring repo %s exists (private=%s)", repo_id, private)
    create_repo(repo_id, token=token, private=private, exist_ok=True, repo_type="model")

    logger.info("Uploading %s -> %s", target, repo_id)
    upload_folder(
        repo_id=repo_id,
        folder_path=str(target),
        token=token,
        commit_message=commit_message,
    )
    url = f"https://huggingface.co/{repo_id}"
    logger.info("Pushed to %s", url)
    return url


def push_adapter(
    repo_id: str,
    adapter_dir: str,
    token: str | None = None,
    private: bool = False,
    base_model: str | None = None,
) -> str:  # pragma: no cover - network/SDK side effects
    """Push a LoRA *adapter_dir* to ``repo_id`` and return the repo URL."""
    return _push_folder(
        repo_id,
        adapter_dir,
        token=token,
        private=private,
        base_model=base_model,
        merged=False,
        commit_message="Upload GuildLM Go specialist LoRA adapter (anvil)",
    )


def push_merged(
    repo_id: str,
    model_dir: str,
    token: str | None = None,
    private: bool = False,
    base_model: str | None = None,
) -> str:  # pragma: no cover - network/SDK side effects
    """Push a merged full model directory to ``repo_id`` and return the URL."""
    return _push_folder(
        repo_id,
        model_dir,
        token=token,
        private=private,
        base_model=base_model,
        merged=True,
        commit_message="Upload GuildLM Go specialist merged model (anvil)",
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anvil-push",
        description="GuildLM Anvil — push a trained adapter/model to the HuggingFace Hub",
    )
    parser.add_argument("--repo-id", required=True, help="Target Hub repo, e.g. user/go-reviewer-lora")
    parser.add_argument("--adapter", required=True, help="Local adapter (or merged model) directory")
    parser.add_argument("--base-model", default=None, help="Base model id (for the model card)")
    parser.add_argument("--token", default=None, help="HF token (else uses HF_TOKEN / cached login)")
    parser.add_argument("--private", action="store_true", help="Create the repo as private")
    parser.add_argument("--merged", action="store_true", help="Treat --adapter as a merged full model")
    return parser


def main(argv: list | None = None) -> None:
    """CLI entry point for ``anvil-push``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    args = _build_parser().parse_args(argv)
    # Validate cheaply before importing the Hub SDK so bad input fails fast.
    validate_push_args(args.repo_id, args.adapter, require_dir=True)
    import os

    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if args.merged:
        url = push_merged(args.repo_id, args.adapter, token=token, private=args.private, base_model=args.base_model)
    else:
        url = push_adapter(args.repo_id, args.adapter, token=token, private=args.private, base_model=args.base_model)
    logger.info("Done: %s", url)


if __name__ == "__main__":
    main()
