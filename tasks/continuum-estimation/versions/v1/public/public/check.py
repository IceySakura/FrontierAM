#!/usr/bin/env python3
"""Infrastructure data errors raise; submitted output errors are penalized."""
import json
import math
from pathlib import Path
import sys

def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result

def load(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)

def number(x):
    return type(x) in (int, float) and math.isfinite(x)

def score(inp_path, pred_path, truth_path):
    inp, truth_obj = load(inp_path), load(truth_path)
    if not isinstance(inp, dict) or set(inp) != {"x", "y"}:
        raise ValueError("invalid evaluator input schema")
    x, y = inp["x"], inp["y"]
    if (not isinstance(x,list) or not isinstance(y,list) or len(x)!=len(y)
        or len(x) not in (4,5) or not all(number(v) and v>0 for v in x)
        or len(set(x))!=len(x) or not all(number(v) for v in y)):
        raise ValueError("invalid evaluator input arrays")
    if not isinstance(truth_obj,dict) or not number(truth_obj.get("truth")):
        raise ValueError("invalid evaluator truth")
    truth = truth_obj["truth"]
    try:
        if Path(pred_path).is_symlink() or Path(pred_path).stat().st_size>65536:
            raise ValueError("output must be a bounded regular file")
        pred = load(pred_path)
        if not isinstance(pred,dict) or set(pred)!={"estimate"} or not number(pred["estimate"]):
            raise ValueError("output requires exactly one finite numeric estimate")
        loss=abs(pred["estimate"]-truth)/max(1.0,abs(truth))
        if not math.isfinite(loss):
            raise ValueError("nonfinite loss")
        return {"status":"ok", "loss":loss}
    except (ValueError, TypeError, OSError, UnicodeError, OverflowError) as exc:
        return {"status":"invalid", "loss":1.0, "reason":str(exc)}

if __name__=="__main__":
    if len(sys.argv)!=4:
        raise SystemExit("usage: score_case.py INPUT OUTPUT TRUTH")
    print(json.dumps(score(*sys.argv[1:]), allow_nan=False))
