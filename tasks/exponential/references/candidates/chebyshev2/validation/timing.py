import sys, json, statistics
sys.path.insert(0, "/pub/public")
import importlib.util
spec=importlib.util.spec_from_file_location("rl","/pub/public/tools/run_local.py")
rl=importlib.util.module_from_spec(spec); spec.loader.exec_module(rl)
from pathlib import Path
manifest=json.load(open("/cases/manifest.json"))
base=Path("/cases")
A="bash /panel/chebyshev/run.sh"   # v1
B="bash /panel/chebyshev2/run.sh"  # v2
rounds=int(sys.argv[1]) if len(sys.argv)>1 else 3
out={}
for c in manifest["cases"]:
    times={"v1":[],"v2":[]}; ses={"v1":0,"v2":0}
    for r in range(rounds):
        order=[("v1",A),("v2",B)] if r%2==0 else [("v2",B),("v1",A)]
        for name,cmd in order:
            rec=rl.run_session(cmd, base/c["file"], repeat=1)
            times[name].append(rec["time_seconds"])
            se=max([e["scaled_error"] or 0 for e in rec["errors"]], default=999) if rec["errors"] else 999
            ses[name]=max(ses[name],se)
    m1=statistics.median(times["v1"]); m2=statistics.median(times["v2"])
    out[c["id"]]={"family":c["family"],"role":c["role"],"scenario":c["scenario"],"precision":c["precision"],
                  "v1_median":m1,"v2_median":m2,"ratio_v2_over_v1":m2/m1 if m1 else None,
                  "v1_times":times["v1"],"v2_times":times["v2"],"v1_maxscaled":ses["v1"],"v2_maxscaled":ses["v2"]}
    print(c["id"], f"v1={m1:.4f} v2={m2:.4f} ratio={m2/m1:.3f}", flush=True)
Path("/out/timing.json").write_text(json.dumps(out,indent=1))
print("WROTE /out/timing.json", flush=True)
