"""Compatibility entry point; implementation lives in fieldrf.cli.finetune_vae."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fieldrf.cli.finetune_vae import main

if __name__ == "__main__":
    main()
