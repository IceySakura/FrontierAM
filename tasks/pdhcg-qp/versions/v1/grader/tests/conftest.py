from pathlib import Path
import sys

GRADER = Path(__file__).resolve().parents[1]
ROOT = GRADER.parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(GRADER))
