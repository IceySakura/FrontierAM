"""Native 0/5/10 mapping and FrontierAM's final 0/50/100 activation gates."""
from __future__ import annotations
import math

RELEASE_GATES = ('amd64_offline','worker_isolation','reference_headroom_25pct',
                 'two_run_stability','non_saturation')


def _finite(value):
    return not isinstance(value,bool) and isinstance(value,(int,float)) and math.isfinite(value)


def native_score(raw_error, E0, E5):
    if not all(_finite(value) for value in (raw_error,E0,E5)) or raw_error<0 or not E0>E5>0:
        raise ValueError('finite raw_error>=0 and E0>E5>0 are required')
    if raw_error>=E0:return 0.
    if raw_error>=E5:return 5.*(E0-raw_error)/(E0-E5)
    return 10.-5.*raw_error/E5


def final_score(raw_error,E0,E5):
    return 10.*native_score(raw_error,E0,E5)


def score_result(result,calibration,*,runtime_fingerprint,data_fingerprint,release_required=True):
    """Unavailable evidence is unmeasured, even if a candidate also failed.

    Active platform/data calibration must be established before assigning a
    candidate a formal zero. Local diagnostics can explicitly bypass release
    gates, but do not turn a draft task into a published benchmark.
    """
    raw=result.get('raw_error')
    def unmeasured(reason):
        return {'status':'unmeasured','score':None,'native_score':None,'raw_error':raw,'reason':reason}
    status=result.get('status')
    if status in (None,'infrastructure_failure','missing_data','unmeasured'):
        return unmeasured(result.get('reason') or 'platform/data evaluation unavailable')
    if not calibration:return unmeasured('local calibration is missing')
    if calibration.get('runtime_fingerprint')!=runtime_fingerprint:
        return unmeasured('calibration belongs to a different runtime/platform')
    if calibration.get('data_fingerprint')!=data_fingerprint:
        return unmeasured('calibration belongs to a different data split')
    E0,E5=calibration.get('E0'),calibration.get('E5')
    if not (_finite(E0) and _finite(E5) and E0>E5>0):
        return unmeasured('local anchors are missing or unordered')
    if release_required:
        gates=calibration.get('release_gates') or {}
        missing=[name for name in RELEASE_GATES if gates.get(name) is not True]
        if calibration.get('status')!='active' or missing:
            return unmeasured('release calibration/gates pending: '+', '.join(missing))
    elif calibration.get('status') not in ('measured_pending_activation','active'):
        return unmeasured('reference measurement is incomplete')
    if status in ('submission_failure','invalid_submission'):
        return {'status':'submission_failure','score':0.,'native_score':0.,'raw_error':raw,
                'reason':result.get('reason'),'mapping':'final = 10 * native piecewise-linear score'}
    if status!='ok':return unmeasured('unknown grader status: '+str(status))
    if not _finite(raw) or raw<0:
        return unmeasured('invalid raw-loss field in an allegedly successful grader record')
    native=native_score(raw,E0,E5)
    return {'status':'ok','score':10.*native,'native_score':native,'raw_error':raw,
            'anchors':{'cold_cg_0':E0,'r1_lbfgs_pcg_50':E5,'exact_100':0.},
            'mapping':'E>=E0:0; E5<=E<E0:50*(E0-E)/(E0-E5); E<E5:100-50*E/E5',
            'diagnostic_only':not release_required}
