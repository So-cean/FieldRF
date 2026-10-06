"""Compatibility entry point; implementation lives in fieldrf.visualization.paired."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fieldrf.visualization.paired import run_validation
