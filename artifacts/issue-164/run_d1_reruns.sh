#!/bin/sh
# D1 re-runs of the #122, #128 (+ follow-up) and #131 ablations on the
# post-#137 population (#164). Run from the repository root on a clean tree:
#   tmux new -s issue164 'caffeinate -dimsu sh artifacts/issue-164/run_d1_reruns.sh 2>&1 | tee logs/issue164_d1_reruns.log'
set -eu

R=data/results_issue164
PY=venv/bin/python

test -z "$(git status --porcelain)" || { echo "Refusing to run on a dirty tree." >&2; exit 1; }

run() {
    dir=$1; shift
    "$PY" -m kgsemembed.pipeline.ppas_ablation --datasets D1 --conditions "$@" \
        --data_dir data/ --results_dir "$R/$dir/"
}

run ppas_ablation PA-det-48 PA-ppas-48 PA-rand-48-s0 PA-rand-48-s1 PA-rand-48-s2 \
    PA-rand-48-s3 PA-rand-48-s4 PA-full
run ppas_controlled PA-all-base-det-48 PA-all-base-ppas-48 PA-cov-base-det-48 \
    PA-cov-base-ppas-48 PA-all-fmax-ppas-48 PA-cov-fmax-det-48 PA-cov-fmax-ppas-48
run ppas_validation PA-all-base-det-32 PA-all-base-ppas-32 PA-cov-base-det-32 \
    PA-cov-base-ppas-32 PA-all-base-det-64 PA-all-base-ppas-64 PA-cov-base-det-64 \
    PA-cov-base-ppas-64
run predicate_roles $("$PY" -c "from kgsemembed.pipeline.ablation_conditions import ABLATION_CONDITIONS as A; print(' '.join(c.condition_id for c in A if c.is_role_ablation and 'D1' in c.datasets))")
