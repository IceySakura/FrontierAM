"""Evaluator-only replay; never expose this process/data root to the solver model."""
from pathlib import Path
import argparse,json,sys,sysconfig
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from benchmark import main,METHODS

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--split',choices=['public','private'])
    p.add_argument('--split-manifest',type=Path)
    p.add_argument('--case')
    p.add_argument('--methods',nargs='+',choices=[m[0] for m in METHODS])
    a=p.parse_args();a.data=a.data.resolve();a.out=a.out.resolve();a.wall=30.
    if not a.data.is_dir():p.error('data root does not exist')
    if not list(a.data.glob('*/COMPLETE.json')):p.error('no complete cases; use the unpacked data/ root')
    native=ROOT/'submissions/agent_sol_single'/('spd_core'+sysconfig.get_config_var('EXT_SUFFIX'))
    if (not a.methods or 'agent_sol_single' in a.methods) and not native.exists():
        p.error('build the native solver first with: python code/build_extension.py; silent fallback is not an equivalent replay')
    var=ROOT/'var';var.mkdir(exist_ok=True)
    receipt=var/'freeze.json'
    if not receipt.exists():receipt.write_bytes((ROOT/'provenance/evaluation_freeze.json').read_bytes())
    main(a)
