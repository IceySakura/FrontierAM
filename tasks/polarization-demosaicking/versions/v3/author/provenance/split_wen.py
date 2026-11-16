#!/usr/bin/env python3
"""Create the fixed, performance-independent Wen5/Wen100 split."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def key(scene: str) -> str:
    return hashlib.sha256(("wen-vnext-dev-v1\0" + scene).encode()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("inventory", type=Path)
    p.add_argument("output_dir", type=Path)
    p.add_argument("--dev-count", type=int, default=5)
    args = p.parse_args()
    inventory = json.loads(args.inventory.read_text())
    scenes = [str(x) for x in inventory["scene_ids"]]
    if len(scenes) != 105 or len(set(scenes)) != 105:
        raise SystemExit("expected 105 unique scenes")
    ordered = sorted(scenes, key=key)
    dev = sorted(ordered[: args.dev_count], key=int)
    evaluation = sorted(ordered[args.dev_count :], key=int)
    if len(dev) != 5 or len(evaluation) != 100 or set(dev) & set(evaluation):
        raise SystemExit("invalid split")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "wen_scenes_dev.txt").write_text("\n".join(dev) + "\n")
    (args.output_dir / "wen_scenes_eval.txt").write_text("\n".join(evaluation) + "\n")
    manifest = {
        "rule": "sha256('wen-vnext-dev-v1\\0' + decimal_scene_id), ascending; first 5 are dev",
        "seed": "wen-vnext-dev-v1",
        "all_scene_ids": scenes,
        "dev_scene_ids": dev,
        "eval_scene_ids": evaluation,
        "dev_count": len(dev),
        "eval_count": len(evaluation),
    }
    (args.output_dir / "wen_split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"dev": dev, "eval_count": len(evaluation)}, indent=2))


if __name__ == "__main__":
    main()
