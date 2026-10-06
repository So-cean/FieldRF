"""Command-line entry point for pretrain training."""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pretrain.yaml")
    parser.add_argument("--pretrain_ckpt", default=None)
    parser.add_argument("--resume", default=None,
                        help="Resume from checkpoint (e.g. runs/pretrain/fieldrf_epoch50.pt)")
    parser.add_argument("--dist", action="store_true")
    parser.add_argument("--n_epochs", type=int, default=None,
                        help="Override config train.n_epochs (for smoke testing)")
    parser.add_argument("--max_volumes", type=int, default=0,
                        help="Max volumes per field (0=all, for smoke testing)")
    args = parser.parse_args()

    from fieldrf.training.pretrain import run
    run(args)
