"""Retry one mesh with logs and resource measurements, without training."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_pipeline.mesh_retry import main

if __name__ == "__main__":
    raise SystemExit(main())
