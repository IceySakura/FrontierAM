#!/usr/bin/env python3
import argparse
import os
import sys
import traceback
from problem import Problem, assess, write_result, cone_defect, constant_certificate


def solve_core(problem, validate=None):
    import numpy as np
    if validate is None:
        validate = lambda result, cones=True: assess(problem, result, cones=cones)
    if not problem.q.any() and cone_defect(problem.b, problem.cones) <= 8e-9:
        result = {"status": "solved", "x": np.zeros(problem.n),
                  "s": problem.b.copy(), "z": np.zeros(problem.m)}
        if validate(result)[0]:
            return result
    if problem.n == 0:
        result = constant_certificate(problem)
        if validate(result)[0]:
            return result
    types = {cone["type"] for cone in problem.cones}
    if types <= {"zero", "nonnegative"}:
        from poly_solver import solve
        return solve(problem, validate=validate)
    from structured import solve as structured
    candidate = structured(problem)
    if candidate is not None and validate(candidate)[0]:
        return candidate
    if problem.n >= 20000 or problem.m >= 200000:
        from relaxation import probe
        candidate = probe(problem,validate)
        if candidate is not None:
            return candidate
    from ipm import solve
    from presolve import prepare
    prep = prepare(problem,validate)
    if prep.result is not None:
        return prep.result
    reduced = prep.reduced
    check = lambda result, cones=True: validate(prep.recover(result), cones=cones)
    reduced_types = {cone["type"] for cone in reduced.cones}
    if reduced.n == 0:
        candidate = {"status": "solved", "x": np.empty(0),
                     "s": reduced.b.copy(), "z": np.zeros(reduced.m)}
        if not check(candidate)[0]:
            candidate = constant_certificate(reduced)
            if not check(candidate)[0]:
                candidate = solve(reduced, validate=check)
    elif reduced_types <= {"zero", "nonnegative"}:
        from poly_solver import solve as linear
        candidate = linear(reduced, validate=check)
    else:
        candidate = solve(reduced, validate=check)
    return prep.recover(candidate)


def solve_problem(problem):
    types = {cone["type"] for cone in problem.cones}
    if not types <= {"zero", "nonnegative"}:
        from dualize import Dualized
        dual = Dualized.create(problem)
        if dual is not None:
            polished_result = {}

            def check(result, cones=True):
                mapped = dual.recover(result)
                passed, errors = assess(problem, mapped, cones=cones)
                if (cones and not passed and mapped["status"] == "solved"
                        and max(errors) < 1e-5):
                    from refine import project_solution
                    improved = project_solution(problem, mapped)
                    passed, errors = assess(problem, improved)
                    if passed:
                        polished_result["result"] = improved
                return passed, errors

            raw = solve_core(dual.problem, validate=check)
            candidate = polished_result.get("result", dual.recover(raw))
            if assess(problem, candidate, tolerance=1e-8)[0]:
                return candidate
            if candidate["status"] == "solved":
                from refine import project_solution
                candidate = project_solution(problem, candidate)
                if assess(problem, candidate, tolerance=1e-8)[0]:
                    return candidate
    return solve_core(problem)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = {"status": "unresolved"}
    try:
        problem = Problem.load(args.input)
        if not problem.q.any() and cone_defect(problem.b, problem.cones) <= 8e-9:
            import numpy as np
            result = {"status": "solved", "x": np.zeros(problem.n),
                      "s": problem.b.copy(), "z": np.zeros(problem.m)}
            write_result(args.output, result)
            return
        result = solve_problem(problem)
        passed, errors = assess(problem, result)
        if not passed and result["status"] == "solved":
            from refine import project_solution
            result = project_solution(problem, result)
            passed, errors = assess(problem, result, tolerance=1e-8)
        elif not passed:
            passed, errors = assess(problem, result, tolerance=1e-8)
        if os.environ.get("CONIC_VERBOSE"):
            print("Final:", result["status"], (passed, errors), flush=True)
        if not passed:
            result = {"status": "unresolved"}
    except Exception:
        if os.environ.get("CONIC_VERBOSE"):
            traceback.print_exc()
        result = {"status": "unresolved"}
    write_result(args.output, result)


if __name__ == "__main__":
    main()
