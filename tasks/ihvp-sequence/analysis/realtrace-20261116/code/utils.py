from pathlib import Path
import json

def savejson(p,a):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(a,indent=2,allow_nan=False)+'\n');tmp.replace(p)
