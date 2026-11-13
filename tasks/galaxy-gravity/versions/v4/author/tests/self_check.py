"""Small non-Docker checks for B.4 limits, manifest shape and raw-only failures."""
import copy
import os
from pathlib import Path
import shutil
import struct
import tempfile

import grade


def reject(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('Expected rejection')


assert grade.solve_limit_us({'cpu_limit_s': 600}, {'cpu_limit_s': 61.2}) == 61_200_000
assert grade.solve_wall_limit_s({}, {'cpu_limit_s': 61.2}) == 194
assert grade.solve_wall_limit_s({'cpu_limit_s': 2}) == 30
for value in (0, -1, 601, float('inf'), float('nan'), True, 0.0000001):
    reject(lambda: grade.solve_limit_us({'cpu_limit_s': value}))

with tempfile.TemporaryDirectory(prefix='galaxy-b4-self-check-') as folder:
    root = Path(folder)
    cfg = dict(mode='timing', cpu_limit_s=600, image='sha256:'+'0'*64, repeats=3, cases=[])
    for n in grade.SCALES:
        ip, ref = root/f'{n}.in', root/f'{n}.ref'
        with ip.open('wb') as f:
            f.truncate(n*32)
        with ref.open('wb') as f:
            f.truncate(n*24)
        for family in grade.FAMILIES:
            for slot in range(4):
                cfg['cases'].append(dict(case=f'{n}-{family}-{slot}', n=n, family=family,
                    input=str(ip), truth=str(ref), input_sha256='0'*64, truth_sha256='0'*64))
    original_digest = grade.digest
    grade.digest = lambda path: '0'*64  # Grid test uses sparse files; real digest is checked below.
    try:
        assert len(grade.validate(cfg)[0]) == 80
        assert grade.batch_timeout_s(cfg) <= 460000
        missing = copy.deepcopy(cfg); missing['cases'].pop()
        reject(lambda: grade.validate(missing))
        hard = copy.deepcopy(cfg); hard['mode']='precalibration'
        reject(lambda: grade.validate(hard))
        for case in hard['cases']:
            case['cpu_limit_s']=10.5
        assert len(grade.validate(hard)[0]) == 80
        invalid = copy.deepcopy(hard); invalid['cases'][0]['n']=100
        reject(lambda: grade.validate(invalid))
        reject(lambda: grade.validate({**hard, 'anchors':[0.1,0.01]}))
        reject(lambda: grade.validate({**hard, 'mode':'formal'}))
    finally:
        grade.digest=original_digest

    ip, ref = root/'toy.in', root/'toy.ref'
    ip.write_bytes(struct.pack('<8d',.5,0,0,0,.5,1,0,0))
    ref.write_bytes(struct.pack('<6d',1,0,0,-1,0,0))
    cases=[dict(case=k,n=2,family=k,input=str(ip),truth=str(ref),
                input_sha256=grade.digest(ip),truth_sha256=grade.digest(ref),cpu_limit_s=1.2)
           for k in ('good','bad')]
    diagnostic=dict(mode='diagnostic', image='sha256:'+'0'*64,repeats=3,cases=cases)
    assert grade.error(ref,ref)==0
    src=root/'source'; src.mkdir(); (src/'solve').write_text('placeholder')
    calls=[]
    def fake_execute(mode, source, target, config, entry=None, repeat=0):
        target.mkdir(parents=True)
        if mode=='build':
            (target/'handoff/built').mkdir(parents=True)
            (target/'handoff/built/solve').write_text('placeholder')
            return dict(status='ok')
        calls.append((entry['case'],repeat))
        if (entry['case'],repeat)==('bad',1):
            return dict(status='cpu_timeout',cpu_s=1.201,cpu_us=1201000,peak_bytes=1024)
        (target/'handoff').mkdir()
        shutil.copyfile(ref,target/'handoff/output.bin')
        return dict(status='ok',cpu_s=.1,cpu_us=100000,peak_bytes=1024)
    execute=grade.execute
    grade.execute=fake_execute
    try:
        result=grade.grade(src,root/'result',diagnostic)
    finally:
        grade.execute=execute
    assert len(calls)==6 and result['status']=='submission_failure'
    assert result['raw_error'] is None and result['project_score'] is None
    assert result['strata']['by_case']['good']['raw_error']==0
    assert result['strata']['by_case']['bad']['raw_error'] is None
    assert result['strata']['by_scale']['2']['raw_error'] is None
    assert result['strata']['by_scale']['2']['complete_cases']==1

    # A completed solve may lose its run.json during interruption: never repeat it.
    modes = grade.RECOVERY_MODES
    recovery = os.environ.pop('GALAXY_CALIBRATION_RECOVERY', None)
    def interrupted(mode, source, target, config, entry=None, repeat=0):
        if mode == 'build':
            return fake_execute(mode, source, target, config, entry, repeat)
        calls.append((entry['case'],repeat))
        target.mkdir(parents=True)
        raise RuntimeError('simulated lost terminal receipt')
    grade.execute=interrupted
    grade.RECOVERY_MODES=modes+('diagnostic',)
    try:
        first=grade.grade(src,root/'interrupted',diagnostic)
        assert first['status']=='infrastructure_failure'
        calls.clear()
        os.environ['GALAXY_CALIBRATION_RECOVERY']=str(root/'interrupted')
        resumed=grade.grade(src,root/'resumed',diagnostic)
        assert resumed['status']=='infrastructure_failure' and 'refusing to rerun' in resumed['reason']
        assert calls==[]
    finally:
        grade.execute=execute
        grade.RECOVERY_MODES=modes
        os.environ.pop('GALAXY_CALIBRATION_RECOVERY', None)
        if recovery is not None:
            os.environ['GALAXY_CALIBRATION_RECOVERY']=recovery

print('Passed: per-case limits, four-scale manifest, formal rejection, raw-only strata, failure continuation and partial-recovery rejection.')
