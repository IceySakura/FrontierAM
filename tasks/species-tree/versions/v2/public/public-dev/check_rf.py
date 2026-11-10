"""Unrooted normalized RF for two fully resolved Newick trees on the same taxa."""

import re
from pathlib import Path


TOKEN = re.compile(r"""\s+|\[[^\[\]]*\]|'(?:[^']|'')*'|[^\s\[\]'(),:;]+|.""", re.DOTALL)
LENGTH = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")


def _splits(text):
    tokens = (m[0] for m in TOKEN.finditer(text)
              if not m[0].isspace() and not (m[0].startswith("[") and m[0].endswith("]")))
    current = next(tokens, None)
    taxa = set()
    branches = []

    def advance():
        nonlocal current
        current = next(tokens, None)

    def expect(token):
        if current != token:
            raise ValueError(f"Expected {token!r}, found {current!r}")
        advance()

    def is_label(token):
        return token is not None and token not in {"(", ")", ",", ":", ";", "[", "]", "'"}

    def subtree(at_root=False):
        if current == "(":
            advance()
            leaves = set(subtree())
            children = 1
            while current == ",":
                advance()
                leaves.update(subtree())
                children += 1
            expect(")")
            if children != 2 and not (at_root and children == 3):
                raise ValueError("Expected a fully resolved binary tree")
            if is_label(current):
                advance()  # Internal labels and support do not affect topology.
            if not at_root:
                branches.append(frozenset(leaves))
        else:
            if not is_label(current):
                raise ValueError(f"Expected a leaf label, found {current!r}")
            name = current[1:-1].replace("''", "'") if current.startswith("'") else current
            if not name or name in taxa:
                raise ValueError(f"Empty or duplicate leaf label: {name!r}")
            taxa.add(name)
            leaves = {name}
            advance()
        if current == ":":
            advance()
            if not current or not LENGTH.fullmatch(current):
                raise ValueError(f"Invalid branch length: {current!r}")
            advance()
        return leaves

    subtree(at_root=True)
    expect(";")
    if current is not None or len(taxa) < 4:
        raise ValueError("Expected one tree with at least four taxa")
    # Orient each split away from one fixed taxon. This also collapses a
    # degree-two serialization root whose children describe the same edge.
    anchor = min(taxa)
    splits = {side if anchor not in side else frozenset(taxa - side)
              for side in branches if 2 <= len(side) <= len(taxa) - 2}
    return taxa, splits


def score_files(truth_path, pred_path):
    """Return (taxon count, missed true splits, normalized RF). Raise ValueError on invalid trees."""
    taxa, truth = _splits(Path(truth_path).read_text())
    predicted_taxa, observed = _splits(Path(pred_path).read_text())
    if predicted_taxa != taxa:
        raise ValueError(f"Taxa differ: missing {sorted(taxa - predicted_taxa)}, "
                         f"unexpected {sorted(predicted_taxa - taxa)}")
    missed = len(truth - observed)
    return len(taxa), missed, missed / (len(taxa) - 3)


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="计算单个开发实例的归一化 RF 误差。")
    parser.add_argument("truth", help="开发实例的真值树")
    parser.add_argument("prediction", help="待检查的预测树")
    args = parser.parse_args()
    try:
        n, missed, error = score_files(args.truth, args.prediction)
    except (OSError, ValueError, RecursionError) as exc:
        parser.exit(2, f"无法评分：{exc}\n")
    print(json.dumps({"n_leaves": n, "missed_splits": missed, "normalized_rf": error}))
