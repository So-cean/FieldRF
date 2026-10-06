"""Compatibility entry point; implementation lives in fieldrf.visualization.pretrain."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fieldrf.visualization.pretrain import run_vis
