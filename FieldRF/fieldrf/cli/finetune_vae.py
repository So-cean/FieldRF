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
    p.add_argument("--resume_ckpt", default=None)
    p.add_argument("--dist", action="store_true")
    args = p.parse_args()

    from fieldrf.training.vae import run
    run(args)
