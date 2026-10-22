# Primary metrics and metric conventions (#139)

The paper's primary metrics are **re-ranking MRR** and **source-level Hits@1**.
F1 is secondary. The definitions (unit, denominator, 1:N handling, ceilings)
are in the module docstring of `src/kgsemembed/evaluation/metrics.py` and in
CLAUDE.md under "Metrics (#139)".

## Contents

- `metric_ceilings.json` / `.md`: for each of the 26 pairs, the validation and
  test reference counts, the number of test reference sources (the MRR/Hits@k
  denominator), the sources with several gold targets, and three top-20
  ceilings (ranking, candidate recall@20, F1 recall). Written by
  `scripts/metric_ceilings.py`, which loads no model.
- `tables/{mrr,hits_at_1,f1,recall_at_1}.tex`: output of
  `scripts/generate_results_table.py --baseline_dir data/results_ngram_baseline/
  --metric <m>` on the stored C1–C20 and B0 results. They compile with
  `threeparttable`.

## Hits@1 for stored results

Stored results record only reference-level `recall_at_1`. Each source has
exactly one rank-1 candidate, so Recall@1 and Hits@1 have the same numerator
(the number of test reference sources whose top candidate is gold), and

    Hits@1 = Recall@1 × n_test_refs / n_test_reference_sources.

This holds exactly, which is why no C1–C20 condition was re-run. It was
checked on all 26 B0 pairs against the per-source best gold ranks in their
`_entities.json` sidecars (0 mismatches; ranking ceilings and recall@20
match too). Hits@5 and Hits@10 cannot be recovered this way. New runs record
`hits_at_1/5/10` directly.

Hits@1 and Recall@1 differ only on pairs with multi-target test sources:
D1 (563 of 4,605 sources; B0 0.6126 vs 0.5398), D2 (16 of 1,198) and D3
cmt-conference (1 of 12).

## Effect on the paper's figures (descriptive)

> Superseded by `artifacts/issue-149/` (#149): the values below predate the
> #145 V8 re-runs, and both figures' claims are now judged against #140
> paired intervals.

With the primary metric, the story behind both figures shifts:

- **Figure 1** (D3, C3 vs C10): under MRR, C10 wins 3 of 21 pairs (3 of 11
  CMT/EDAS pairs). Under Hits@1 it wins 4/21 (4/11), and under F1 8/21
  (5/11). Neither metric supports "V8's advantage concentrates on the
  annotation-sparse pairs".
- **Figure 2** (D1/D2, V2+V8 per model): under MRR, M5 is the best V2+V8
  model on D1 (0.7365, against M4's 0.6825), and M4 is best on D2 (0.7833).
  C17 (V1/M4) is highest on both (0.7578 / 0.8605).
