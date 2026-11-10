import sys, json, statistics, os
sys.path.insert(0, "/pub/public")
import importlib.util
spec=importlib.util.spec_from_file_location("rl","/pub/public/tools/run_local.py")
rl=importlib.util.module_from_spec(spec); spec.loader.exec_module(rl)
from pathlib import Path
manifest=json.load(open("/cases/manifest.json"))
base=Path("/cases")
B="bash /panel/chebyshev2/run.sh"
configs=["", "lanczos","fused","adaptive","sell","rcm","reuse"]
rounds=2
out={}
for c in manifest["cases"]:
    out[c["id"]]={"family":c["family"]}
    for cfg in configs:
        os.environ["CHEB2_DISABLE"]=cfg
        ts=[]; mx=0
        for r in range(rounds):
            rec=rl.run_session(B, base/c["file"], repeat=1)
            ts.append(rec["time_seconds"])
            se=max([e["scaled_error"] or 0 for e in rec["errors"]], default=999) if rec["errors"] else 999
            mx=max(mx,se)
        out[c["id"]][cfg or "default"]={"median":statistics.median(ts),"times":ts,"maxscaled":mx}
    print(c["id"], " ".join(f"{(cfg or 'def')}={out[c['id']][cfg or 'default']['median']:.4f}" for cfg in configs), flush=True)
os.environ.pop("CHEB2_DISABLE",None)
Path("/out/ablation_timing.json").write_text(json.dumps(out,indent=1))
print("WROTE /out/ablation_timing.json", flush=True)
