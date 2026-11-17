"""Build two distinct archives by explicit allowlist; no evaluation data are packaged."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1]
MODEL=['README.md','statement.md','interface.md','solver.py','smoke.json']

def make_archive(out,files):
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path,name in files:
            if path.is_symlink():raise ValueError('symlink rejected: '+str(path))
            z.write(path,name)
    with zipfile.ZipFile(out) as z:assert z.testzip() is None
    return {'file':out.name,'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest()}

if __name__=='__main__':
    dist=ROOT/'dist';dist.mkdir(exist_ok=True)
    model=make_archive(dist/'MODEL_ONLY_ihvp_realtrace_20261116.zip',[(ROOT/'model_materials'/n,n) for n in MODEL])
    files=[]
    for folder in ['code','runtime','submissions','reports','provenance','tools','model_materials']:
        for path in sorted((ROOT/folder).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix in {'.py','.cpp','.sh','.md','.json','.csv','.txt'}:
                if path.name in {'plans.json','reference.npz'}:raise ValueError('private data rejected')
                files.append((path,str(path.relative_to(ROOT))))
    for name in ['README.md','TESTER_PROMPT.md','PROTOCOL.md','USAGE_CURATOR.md','requirements-eval.txt']:
        files.append((ROOT/name,name))
    curator=make_archive(dist/'EVALUATOR_ONLY_ihvp_realtrace_tools_20261116.zip',files)
    receipt={'statement_revision':'neutral-statement-20261117','calibration_revision':'cg50-warm0-20261117','model_only':model,'evaluator_only':curator,'full_data_archive':json.loads((ROOT/'reports/REMOTE_DATA_PACKAGE.json').read_text())}
    (dist/'artifacts.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))
