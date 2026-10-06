"""Command-line entry point for vae training."""

import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True,
                   help="Path to VAE finetune YAML config")
    p.add_argument("--max_volumes", type=int, default=0,
                   help="Max volumes per (modality, field) for smoke test (0 = all)")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override epochs from config (for smoke test)")
    initialization = p.add_mutually_exclusive_group()
    initialization.add_argument("--resume_ckpt", default=None, help="Restore a complete epoch-boundary training checkpoint")
    initialization.add_argument("--init_ckpt", default=None, help="Initialize weights without resuming training")
    p.add_argument("--dist", action="store_true")
    p.add_argument("--device", choices=["cpu", "cuda"], default=None)
    args = p.parse_args()

    from fieldrf.training.vae import run
    from fieldrf.training.distributed import teardown_distributed
    try:
        run(args)
    finally:
        teardown_distributed()
