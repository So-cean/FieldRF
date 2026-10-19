"""Create a portable copy of a legacy endpoint checkpoint without changing weights."""

import argparse
from pathlib import Path

from fieldrf.utils.artifacts import validate_rf_base


def migrate_checkpoint(checkpoint, base_checkpoint, output, *, legacy_root):
    import torch
    from fieldrf.training.checkpoints import atomic_save

    checkpoint = Path(checkpoint)
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError(f"Output already exists; choose a new file: {output}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "endpoint_aligner" not in state or "base_checkpoint" not in state or "bridge" in state:
        raise ValueError("Migration requires an unpaired endpoint-alignment checkpoint")
    state["base_checkpoint_sha256"] = validate_rf_base(
        state, Path(base_checkpoint).resolve(), root=legacy_root
    )
    atomic_save(state, output)
    return state["base_checkpoint_sha256"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--output", required=True, help="New checkpoint; originals are preserved")
    parser.add_argument(
        "--legacy-root", default=str(Path(__file__).resolve().parents[2]),
        help="Old FieldRF working directory used to resolve the recorded RF base path",
    )
    args = parser.parse_args()
    digest = migrate_checkpoint(
        args.checkpoint, args.base_checkpoint,
        args.output, legacy_root=args.legacy_root,
    )
    print(f"Saved {args.output}; verified RF base SHA-256: {digest}")


if __name__ == "__main__":
    main()
