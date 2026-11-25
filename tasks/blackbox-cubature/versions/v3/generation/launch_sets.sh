#!/usr/bin/env bash
# Build the v3 samples, public and private sets in parts, each part under its own slot-broker lease (pinned to the
# lease's cores), writing to var/raw/blackbox-cubature-v3/gen/<part>/; merge_sets.py then assembles data/<set>/.
# The public and private namespaces are read from mode-0600 files under data/<set>/ (created on first use).
# Usage: launch_sets.sh [PART ...]   (default: every part)
set -uo pipefail
cd "$(dirname "$0")/../../../../.."
R=$PWD
GEN=tasks/blackbox-cubature/versions/v3/generation
OUT=var/raw/blackbox-cubature-v3/gen
mkdir -p "$OUT/logs"
ns() { echo "--namespace-file $R/tasks/blackbox-cubature/versions/v3/data/$1/namespace.txt"; }
declare -A PARTS=(
  [samples]="dev 2 --set samples"
  [public-basket]="dev 1 --set public $(ns public) --only basket_asian_call"
  [public-rest]="dev 2 --set public $(ns public) --only H1 H2 H3 H3t asian_call digital_asian rainbow_asian_min barrier_do_call barrier_dko_call"
  [private-basket-m16]="dev 2 --set private $(ns private) --match basket_asian_call-L2-m16"
  [private-basket-m32-s0]="dev 1 --set private $(ns private) --match basket_asian_call-L2-m32-s0"
  [private-basket-m32-s1]="dev 1 --set private $(ns private) --match basket_asian_call-L2-m32-s1"
  [private-rainbow]="grade 2 --set private $(ns private) --only rainbow_asian_min"
  [private-rest]="grade 2 --set private $(ns private) --only H1 H2 H3 H3t asian_call digital_asian barrier_do_call barrier_dko_call"
)
for part in "${@:-${!PARTS[@]}}"; do
  read -r kind workers rest <<< "${PARTS[$part]}"
  mkdir -p "$OUT/$part"
  PYTHONPATH=$R nohup python3 scripts/ops/with_lease.py --kind "$kind" --pin --label "v3-gen-$part" --timeout 7200 -- \
    bash -c "cd $R/$GEN && exec python3 -u generate.py $R/$OUT/$part --workers $workers $rest" \
    > "$OUT/logs/$part.log" 2>&1 &
  echo "$part: pid $!"
  sleep 1
done
