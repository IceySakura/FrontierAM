import argparse,json,socket
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--input');p.add_argument('--output');a=p.parse_args()
checks={}
try:
    Path('/tests/private/score-anchor.json').read_bytes()
except PermissionError:
    checks['hidden_read_denied']=True
try:
    Path('/logs/verifier/overwrite.json').write_text('tamper')
except PermissionError:
    checks['grader_write_denied']=True
try:
    Path(__file__).with_name('altered').write_text('tamper')
except PermissionError:
    checks['source_write_denied']=True
for kind in (socket.SOCK_STREAM,socket.SOCK_DGRAM):
    try:
        socket.socket(socket.AF_INET,kind)
    except PermissionError:
        checks[f'network_socket_{kind}_denied']=True
checks['current_input_readable']=bool(Path(a.input).read_bytes())
assert len(checks)==6 and all(checks.values()),checks
print(json.dumps({'boundary_fixture':checks}),flush=True)
np.savez(a.output,metadata_json=np.array(json.dumps({'schema_version':1,'status':'unresolved'})))
