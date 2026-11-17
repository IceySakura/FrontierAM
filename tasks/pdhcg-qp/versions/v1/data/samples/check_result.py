"""Check one candidate NPZ with the same schema and KKT equations as the verifier."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "io"))
from qpbench import check_solution, load_output, load_problem

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
problem = load_problem(args.input)
result = load_output(args.output, expected_n=problem.n, expected_m=problem.m)
report = check_solution(problem, result)
print(json.dumps(report, allow_nan=False))
raise SystemExit(0 if report["passed"] else 1)
