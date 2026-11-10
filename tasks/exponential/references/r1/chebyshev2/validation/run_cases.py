import sys, json, os, time
sys.path.insert(0, "/pub/public")
import importlib.util
spec=importlib.util.spec_from_file_location("rl","/pub/public/tools/run_local.py")
rl=importlib.util.module_from_spec(spec); spec.loader.exec_module(rl)
from pathlib import Path
manifest=json.load(open("/cases/manifest.json"))
base=Path("/cases")
cmd=sys.argv[1]
ids=sys.argv[2].split(",") if len(sys.argv)>2 and sys.argv[2]!="ALL" else [c["id"] for c in manifest["cases"]]
maxse=0; fails=[]; total=0.0
for c in manifest["cases"]:
    if c["id"] not in ids: continue
    r=rl.run_session(cmd, base/c["file"], repeat=1)
    se=max([e["scaled_error"] or 0 for e in r["errors"]], default=0) if r["errors"] else 999
    maxse=max(maxse,se)
    if r["time_seconds"]: total+=r["time_seconds"]
    ok = r["valid"] and se<=0.1
    if not ok: fails.append((c["id"], r["status"], se))
    print(c["id"], r["status"], f"scaled={se:.6g}", f"t={r['time_seconds']}", flush=True)
print("RESULT", json.dumps({"max_scaled":maxse,"fails":fails,"n":len(ids),"sum_time":total}), flush=True)
