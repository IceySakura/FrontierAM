"""Focused checks for conditional resplitting and baseline-record validation."""

from pathlib import Path
import copy
import hashlib
import json
import tempfile
import unittest

from rebalance_split import METHODS, NoFeasibleSplit, collect_baselines, rebalance


def fixtures():
    rows = []
    for dataset in range(3):
        for profile in range(5):
            group = dataset * 5 + profile
            for option in range(3):
                split = "public" if option == 0 else "private"
                index = group if option == 0 else group * 2 + option - 1
                case = f"{split}_{index:02d}"
                loss = (1 + dataset * .1 + profile * .05) * (.8 + option * .2)
                if group == option == 0:
                    loss = 12.0  # A genuine tail; still admits balanced allocations.
                for method in METHODS:
                    rows.append({
                        "case": case, "split": split, "method": method,
                        "dataset": f"dataset_{dataset}", "profile": f"profile_{profile}",
                        "data_id": hashlib.sha256(case.encode()).hexdigest(),
                        "dimension": 16384, "steps": 50, "completed_steps": 50,
                        "valid": True, "resource_compliant": True,
                        "wall_seconds": 28.0, "peak_rss_mib": 40.0,
                        "mean_energy_loss": loss * (1.5 if method == "warm_cg" else 1),
                    })
    return rows


def write_rows(root, rows):
    root.mkdir(parents=True, exist_ok=True)
    for row in rows:
        (root / f"{row['case']}__{row['method']}.json").write_text(json.dumps(row))


class RebalanceChecks(unittest.TestCase):
    def test_reproducible_stratified_identity_preserving_split(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_rows(root, fixtures())
            # Other submissions are neither parsed nor used for allocation.
            (root / "public_00__agent_sol_iter.json").write_text("not valid JSON")
            cases = collect_baselines([root])
            first, diagnostics = rebalance(cases, candidates=2000)
            second, repeated = rebalance(list(reversed(cases)), candidates=2000)
            self.assertEqual(first, second)
            self.assertEqual(diagnostics, repeated)
            public = {c["case"] for c in first["cases"] if c["split"] == "public"}
            private = {c["case"] for c in first["cases"] if c["split"] == "private"}
            self.assertEqual((len(public), len(private)), (15, 30))
            self.assertFalse(public & private)
            self.assertEqual(public | private, {c["case"] for c in cases})
            self.assertTrue(all(g["public"] == 1 and g["private"] == 2 for g in diagnostics["group_counts"]))
            self.assertFalse(first["unseen_holdout_claim"])
            self.assertTrue(first["post_hoc"])
            by_case = {c["case"]: c for c in cases}
            for row in first["cases"]:
                self.assertEqual(row["data_id"], by_case[row["case"]]["data_id"])
                self.assertEqual(row["original_split"], by_case[row["case"]]["original_split"])
            for method in METHODS:
                self.assertLessEqual(diagnostics["after"][method]["larger_to_smaller_mean_ratio"], 1.25)

    def test_reject_mixed_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            rows = fixtures()
            rows[1]["data_id"] = "0" * 64
            write_rows(Path(temp), rows)
            with self.assertRaisesRegex(ValueError, "mixed data versions"):
                collect_baselines([Path(temp)])

    def test_reject_duplicate_and_missing_records(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = fixtures()
            write_rows(root, rows)
            write_rows(root / "duplicate", [rows[0]])
            with self.assertRaisesRegex(ValueError, "duplicate"):
                collect_baselines([root])
        with tempfile.TemporaryDirectory() as temp:
            write_rows(Path(temp), fixtures()[:-1])
            with self.assertRaisesRegex(ValueError, "same 45 cases"):
                collect_baselines([Path(temp)])

    def test_reject_resource_or_scale_contradiction(self):
        for field, value in (("wall_seconds", 30.1), ("peak_rss_mib", 257),
                             ("dimension", 64), ("completed_steps", 49),
                             ("mean_energy_loss", float("nan"))):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                rows = fixtures()
                rows[0][field] = value
                write_rows(Path(temp), rows)
                with self.assertRaises(ValueError):
                    collect_baselines([Path(temp)])

    def test_failed_search_does_not_claim_infeasibility(self):
        with tempfile.TemporaryDirectory() as temp:
            write_rows(Path(temp), fixtures())
            cases = collect_baselines([Path(temp)])
            cases = copy.deepcopy(cases)
            for row in cases:
                row["losses"] = {method: 1.0 for method in METHODS}
            cases[0]["losses"] = {method: 1e9 for method in METHODS}
            with self.assertRaises(NoFeasibleSplit) as caught:
                rebalance(cases, candidates=100)
            diagnostics = caught.exception.diagnostics
            self.assertEqual(diagnostics["selection"]["feasible_candidate_count"], 0)
            self.assertFalse(diagnostics["mathematical_infeasibility_claim"])
            self.assertIsNone(diagnostics["after"])


if __name__ == "__main__":
    unittest.main()
