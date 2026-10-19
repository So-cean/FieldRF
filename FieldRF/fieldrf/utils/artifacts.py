"""Content-based RF checkpoint provenance, independent of filesystem layout."""

import hashlib
from pathlib import Path
import re


def resolve_artifact(path, root):
    value = Path(path).expanduser()
    return value if value.is_absolute() else Path(root) / value


def checkpoint_sha256(path):
    """Stream the actual bytes; shared-filesystem timestamps cannot prove identity."""
    path = Path(path).resolve(strict=True)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Checkpoint is not a nonempty file: {path}")
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_rf_base(state, base_checkpoint, *, root):
    """Require the exact RF base bytes; historical paths are provenance only.

    Legacy checkpoints can be checked while their recorded RF base is still
    accessible. Migrate them then to retain verification after moving files.
    """
    actual = resolve_artifact(base_checkpoint, root)
    actual_digest = checkpoint_sha256(actual)
    expected = state.get("base_checkpoint_sha256")
    if expected is not None:
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("Invalid base_checkpoint_sha256 in endpoint checkpoint")
    else:
        recorded = state.get("base_checkpoint")
        if not recorded:
            raise ValueError("Endpoint checkpoint has no RF base provenance")
        original = resolve_artifact(recorded, root)
        if not original.is_file():
            raise ValueError(
                "Legacy endpoint checkpoint has no RF base checksum and its "
                f"recorded base is unavailable: {original}. Restore access to "
                "the original base and run fieldrf-migrate-checkpoint before "
                "relocating it; the filename alone cannot verify identity."
            )
        expected = actual_digest if original.resolve() == actual.resolve() else checkpoint_sha256(original)
    if expected != actual_digest:
        raise ValueError(
            "Endpoint aligner was trained for a different RF base checkpoint: "
            f"SHA-256 {expected} != {actual_digest} ({actual})"
        )
    return actual_digest
