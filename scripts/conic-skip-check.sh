#!/bin/bash
# Equivalence check of the conic skip-failed-repeats grader (scripts/prepare.py --conic-skip-failed-repeats) against the
# unmodified grader: the same 16 held-out cases of an existing submission (8 solved with margin, 8 failing on short
# limits), graded on the grading slots by each runtime; per-case success must match. Writes
# results/conic-fast-grading-check.json, which scripts/v62-final.sh waits for before the conic suites.
set -uo pipefail
cd "$(dirname "$0")/.."
R=$PWD
LOG=store/reruns/v62-final.log
STORE=store-conic-skipcheck
SUB=conic-20261106T084017Z-a0cb85
until grep -q '"label": "opus-single-trajectory"\|"no_submission", "label": "opus-single-trajectory"' "$LOG" 2>/dev/null; do sleep 60; done
mkdir -p $STORE/submissions $STORE/jobs/{pending,running,done,failed,held} $STORE/suites $STORE/results
[ -d $STORE/submissions/$SUB ] || cp -r store-conic2/submissions/$SUB $STORE/submissions/
cp tasks/conic/local_recal/skip-check-cases.json $STORE/cases.json 2>/dev/null || python3 - <<'PY'
import json, glob, random
fail, ok = [], []
for f in glob.glob('store-conic2/results/job-20261106T104650Z-*/grade/verifier/cases/*/repeat-1/record.json'):
    r = json.load(open(f)); lim = r['time_limit_seconds']; e = r['elapsed_seconds'] or 0
    if r['outcome'] == 'accepted' and lim <= 5 and e < 0.6 * lim: ok.append(r['case_id'])
    elif r['outcome'] != 'accepted' and lim <= 2: fail.append(r['case_id'])
random.seed(7)
json.dump(sorted(random.sample(ok, 8)) + sorted(random.sample(fail, 8)), open('store-conic-skipcheck/cases.json', 'w'))
PY
declare -A SUITE
for name in solver1105 solver1105-skip; do
    rt=$R/.runtime-conic-$name
    SUITE[$name]=$(FRONTIERAM_RUNTIME=$rt ./bench --store $STORE suite --task conic --submission $SUB --cases $STORE/cases.json \
                   --shards 4 --slots grade-06,grade-14,grade-22,grade-30 --isolation shared \
                   | python3 -c "import json,sys;print(json.loads(sys.stdin.readlines()[-1])['id'])")
    FRONTIERAM_RUNTIME=$rt ./bench --store $STORE grade-worker --task conic --workers 4 --drain >> store/reruns/conic-skip-check.log 2>&1
    FRONTIERAM_RUNTIME=$rt ./bench --store $STORE suite-report --suite "${SUITE[$name]}" > /dev/null 2>&1
done
python3 - "${SUITE[solver1105]}" "${SUITE[solver1105-skip]}" <<'PY'
import json, sys, glob
base, skip = (json.load(open(f'store-conic-skipcheck/suites/{s}.report.json')) for s in sys.argv[1:3])
a = {c['case_id']: c['success'] for c in base['cases']}
b = {c['case_id']: c['success'] for c in skip['cases']}
skipped = sum(1 for f in glob.glob('store-conic-skipcheck/results/*/grade/verifier/cases/*/repeat-*/record.json')
              if json.load(open(f)).get('reason') == 'skipped_after_failed_repeat')
result = {'unmodified_suite': sys.argv[1], 'skip_suite': sys.argv[2], 'cases': len(a),
          'complete': base['complete'] and skip['complete'], 'per_case_match': a == b,
          'mismatches': sorted(k for k in a if a.get(k) != b.get(k)), 'successes': sum(a.values()),
          'skipped_repeat_records': skipped}
result['equivalent'] = result['complete'] and result['per_case_match'] and len(a) == 16 and skipped > 0
json.dump(result, open('results/conic-fast-grading-check.json', 'w'), indent=1)
print(json.dumps(result))
PY
