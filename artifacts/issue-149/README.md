# Figure 1 and Figure 2 claims under MRR (#149)

Both figures plot re-ranking MRR (the primary metric, #139) from the results
re-run at c6931db (#140, after #145). Their claims are judged against the
#140 paired intervals, computed by `scripts/figure_claims.py` into
`figure_claims.md` / `figure_claims.json`. The paper text is
`paper_figures_section.tex`.

```bash
python scripts/figure_claims.py --results_dir data/results/ --output_dir artifacts/issue-149/  # ~1.5 min
python scripts/generate_figure1_d3_scatter.py --results_dir data/results/ --output_dir figures/
python scripts/generate_figure2_model_comparison.py --results_dir data/results/ --output_dir figures/
```

## Figure 1 (D3, C3 V2+V6/M2 vs C10 V2+V8/M2): keep, reframed

- **Old claim, dropped**: "V8's advantage over V6 concentrates on the
  annotation-sparse CMT/EDAS pairs". There is no V8 advantage: C3 is ahead
  on D3 (ΔMRR C10 − C3 −0.078 [−0.114, −0.042], Holm p 0.0006 in
  `report.md`), and no pair's own interval favours C10 on MRR or Hits@1.
  CMT and EDAS are also not the sparsest ontologies: by the #91 density
  measure (`scripts/analyse_annotation_sparsity.py`; mean of seven
  annotation predicates present per entity) EKAW is 0.019, CMT 0.064, EDAS
  0.083, IASTED 0.090, SIGKDD 0.106, Conference 0.114, ConfOf 0.627.
- **New claim**: C3's lead sits on the pairs without CMT or EDAS.

  | MRR, C10 − C3 | Δ | 95% CI |
  |---|---|---|
  | all 21 pairs | −0.078 | [−0.114, −0.042] |
  | 11 CMT/EDAS pairs | −0.030 | [−0.075, +0.016] |
  | 10 other pairs | −0.131 | [−0.191, −0.074] |
  | CMT/EDAS − other | +0.101 | sources [+0.030, +0.176]; pairs [+0.023, +0.182] |

  Pair intervals favouring C3: conference-sigkdd, confof-ekaw, ekaw-sigkdd
  (other) and confof-edas (CMT/EDAS); C10 is ahead on three CMT/EDAS pairs
  (cmt-edas, edas-ekaw, edas-sigkdd), none with an interval excluding 0.
- **Caveats**: exploratory; the cause is not established (the sparsity
  explanation is not supported); each ontology is in six pairs, so the
  pair-level interval is optimistic; 3–20 test reference sources per pair.
- **Hits@1** agrees (difference +0.143 [+0.037, +0.253]). **F1** differs:
  overall −0.054 [−0.111, +0.005] and the group difference +0.097
  [−0.017, +0.212] both include 0; cmt-edas is the one pair whose F1
  interval favours C10.
- **Script changes**: the docstring no longer asserts the finding; the
  legend says "CMT or EDAS pair" (no "annotation-sparse"); callouts name the
  three pairs where C10 is ahead (offsets scale with the axis, so the
  Hits@1/F1 variants stay in frame); "wins" regions read "ahead".

## Figure 2 (D1/D2, V2+V8 per model, plus C17 V1/M4): keep, claim restated

All 15 (D1) and 10 (D2) pairwise differences, Holm within each dataset
(exploratory, separate from the registered families in `report.md`):

- The domain-specific model does not lead with V2+V8 fixed. D1: M5 − M4
  +0.026 [+0.018, +0.034], M5 − M3 +0.019 [+0.012, +0.025], M3 vs M4
  −0.0075 [−0.0148, −0.0003] (Holm p 0.079). D2: M5 vs M4 −0.0002
  [−0.012, +0.011]. M1 ≈ M2 on both (D1 +0.0015 [−0.005, +0.008], D2 +0.010
  [−0.002, +0.022]), both below every other model.
- No V2+V8 model reaches C17 (V1/M4): C17 − best V2+V8 D1 +0.045
  [+0.036, +0.054] (C13), D2 +0.089 [+0.071, +0.107] (C11); with M4 fixed,
  V1 − V2+V8 +0.071 [+0.061, +0.080] / +0.089 [+0.071, +0.107]. V1 omits
  synonyms, definitions and the relational signature together, so this does
  not isolate V8 (consistent with #131/#136: synonyms and definitions hurt).
- The #139 README's figures (M5 0.7365 on D1, M4 0.7833 on D2, …) predate
  the #145 V8 re-runs; current values are M5 0.7132 / 0.7717, M4 0.6872 /
  0.7719, C17 0.7578 / 0.8605.
- **Hits@1** agrees except D1 M3 below M4 (−0.015 [−0.026, −0.004]).
  **F1** separates two MRR ties: M2 above M1 on D1 (+0.024 [+0.016, +0.032])
  and M5 above M4 on D2 (+0.044 [+0.025, +0.063]).
- **Script change**: the docstring states these claims and points here.
