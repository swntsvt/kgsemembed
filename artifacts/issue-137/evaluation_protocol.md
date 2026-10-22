# Evaluation protocol, stage-1 baseline and scope (#137)

This document states the protocol every Phase 2 result is produced under, so
that the comparisons in the paper can be read correctly. All numbers are from
the stored results (`data/results/`, protocol version 2) and the B0 run
(`data/results_ngram_baseline/`). The companion files are:

- `baseline_comparison.md`: every condition against B0, per dataset column
  and metric, plus the population check and the D5 section
  (`scripts/generate_baseline_comparison.py`);
- `dataset_provenance.md`: SHA-256 and correspondence counts of every input
  file, and a byte comparison with the published OAEI releases
  (`scripts/dataset_provenance.py`).

## 1. What is compared

Every Phase 2 condition (C1–C20) is a **candidate re-ranking** method. Stage 1
ranks all target entities of the pair's entity type for each source with a
character 3-gram cosine over entity labels, and keeps the top 20
(`candidates/ngram.py`, `NgramConfig(n=3, metric="cosine", top_k=20)`,
`data/candidates/`). Stage 2 verbalises the source and its 20 candidates and
re-orders the candidates by embedding cosine. Ranking is **local**: each
source is ranked only against its own 20 candidates, never against the whole
target ontology. No result here is full matching or all-target retrieval.

**B0** (`pipeline/ngram_baseline.py`) is the stage-1 order of those same
lists. Its scores are recomputed with the configuration that produced the
candidate files, and the run refuses to write anything unless the recomputed
order equals the stored order for every source of every pair. It passed with 0
mismatches on D1, D2, all 21 D3 pairs, D4_schema, D4_instance and D5. B0 is then
evaluated by the same function as every condition
(`run_experiment._evaluate_pair`), so the following are identical by
construction:

- candidate lists (the same `candidates_sha256` in every result);
- validation and test populations;
- threshold selection;
- metric definitions and denominators.

`baseline_comparison.md` checks this for every stored result: 271 of 271
C-results match B0 on candidates, population block and protocol version.

A condition's difference from B0 is therefore what embedding re-ranking adds
or costs on identical inputs. Neither B0 nor the conditions can exceed the
stage-1 candidate recall@20 (section 4).

## 2. Datasets and reference alignments

| Column | Pair | Source / target | Release | Reference |
|---|---|---|---|---|
| D1 | SNOMED CT body → FMA body (classes) | SNOMED version IRI `…/version/20220317`; FMA 4.14.0 | OAEI Bio-ML 2022, equivalence matching, **unsupervised** setting, `snomed.body-fma.body` | 6,530 `=` cells (the setting's test mappings) |
| D2 | Mouse anatomy (MA) → NCI human anatomy (classes) | OAEI Anatomy track | `anatomy_track/anatomy_track-default/mouse-human-suite` | 1,516 cells |
| D3 | 21 Conference pairs (classes and properties) | OAEI Conference track | `conference/conference-v1/{pair}` | 4–25 cells per pair, 305 in all |
| D4 | memoryalpha → stexpanded (schema and instances) | OAEI Knowledge Graph track | `knowledgegraph/v4` | 1,779 cells, partial gold standard |
| D5 | DBpedia → Wikidata, 15K, English (instances) | OpenEA benchmark v2.0 | `D_W_15K_V2`, fold `721_5fold/1` | 15,000 links; fold 1: 3,000 train / 1,500 valid / 10,500 test |

**Integrity.** Every D1–D4 file the loaders read is byte-identical to its
counterpart in a local copy of the OAEI MELT track repository
(`oaei.webdatacommons.org/`). That covers the 4 D1 files, the 3 D2 files, all
63 D3 files and the 3 D4 files; the SHA-256 values are in
`dataset_provenance.md`. D5 is not distributed through that repository. Its
files are recorded by SHA-256 only, and their identity with the OpenEA v2.0
archive has not been rechecked here.

**D1 provenance.** All four local D1 files (`source.rdf`, `target.rdf`,
`reference.rdf`, `input.rdf`) are byte-identical to
`bio-ml-equiv-unsupervised/2022/snomed.body-fma.body/` in the MELT copy. In
that release the pair has 7,256 equivalence mappings:

| Setting | `input.rdf` | `reference.rdf` |
|---|---|---|
| Unsupervised | 726 (validation) | 6,530 (test) |
| Supervised | 2,177 | 5,079 |

D1 therefore uses the unsupervised setting's **test** mappings, with the
SNOMED CT and FMA versions named in the ontology headers.

The 726 validation mappings (`input.rdf`) are loaded as **known pairs**
(`AlignmentPair.train_refs`, the slot D5's training links use) and are never
scored; see section 3. Before this was fixed in #137 they were not read at
all, so their sources counted as unmatched and a correct prediction for one
was a false positive.

## 3. Splits and evaluation populations

- **D1–D4.** The pair's reference is split 0/20/80 (train/validation/test)
  **by source entity**. Distinct source URIs are sorted, shuffled with seed 42
  (`loader._split_val_test_20_80`), and 20% (at least one) go to validation.
  Every reference of a 1:N source lands in the same slice. D4 splits its
  schema and instance references separately. Nothing is trained, so there is
  no training slice.
- **D5.** OpenEA fold 1 as published: `train_links` (3,000), `valid_links`
  (1,500), `test_links` (10,500). Training links are never trained on.
  Their sources are excluded from scoring.
- **Populations** (`evaluation/population.py`). Every source is scored in
  exactly one population, using these rules in order:
  1. a source with a test reference is test;
  2. otherwise, a source with a validation reference is validation;
  3. otherwise, a source with a training reference is excluded.

  Training references are D5's `train_links` and D1's Bio-ML validation
  mappings (`input.rdf`). A known pair is also removed from every ranked list
  before scoring (`population.judgeable_ranked_lists`), so for a scored
  source it is neither a distractor ranked above the gold target nor a
  prediction. Bio-ML does the same: when Precision, Recall and F1 are
  computed on one reference set, "reference mappings that are not in this
  set" are excluded from the system output (He et al., 2022, *Machine
  Learning-Friendly Biomedical Datasets for Equivalence and Subsumption
  Ontology Matching*, arXiv:2205.03447, section 3.3).
  On D1, the 726 mappings cover 713 distinct sources, all of them D1 source
  entities: 556 of those sources have no other reference and are excluded
  (137 from validation, 419 from test); the other 157 (27 validation, 130
  test) keep their population, minus 122 known candidate pairs. D2–D4 have no
  known pairs, and on D5 no scored source has one, so their results are
  unchanged.

  Sources with no reference anywhere are *unmatched*. They are divided
  between validation and test in proportion to the reference sources, using a
  stable `crc32` draw per URI. Under a complete reference (D1, D2, D3, D5),
  an unmatched source's prediction is a false positive.
- **D4 (partial reference).** An unmatched source's prediction is judged only
  when its target is a reference target. It is then a false positive, and is
  otherwise ignored, following the KG track's own evaluation. The number of
  judgeable sources therefore depends on each method's top-ranked target. It
  is the only population count that differs between methods, and the
  population check leaves it out for D4.

| Column | Source entities | Target pool (stage 1 ranks all of these) | Val refs | Test refs | Test reference sources |
|---|---|---|---|---|---|
| D1 | 24,182 | 64,726 FMA classes | 1,304 | 5,226 | 4,605 |
| D2 | 2,744 | 3,304 NCI classes | 302 | 1,214 | 1,198 |
| D3 | per pair | 2,521 classes and properties over 21 pairs | per pair | per pair | 250 over 21 pairs |
| D4_schema | 4,131 | 1,947 | 10 | 44 | 44 |
| D4_instance | 156,581 | 39,305 | 345 | 1,380 | 1,380 |
| D5 | 14,762 (with triples) | 14,992 Wikidata entities | 1,500 | 10,500 | 10,500 |

D5 has 167 test links whose DBpedia source has no triples. Those sources are
never ranked but stay in every denominator, contributing 0 to MRR and Recall@k
and a false negative to F1.

## 4. Metrics

**Source-level ranking metrics** (no threshold):

- **MRR** is the mean, over test reference sources, of 1 / rank of the
  source's best-ranked gold target in its 20-candidate list. A source with
  several gold targets (1:N) is credited with the best one. A reference source
  with no ranked list, or with no gold target among its candidates, scores 0.
  Ranked sources without a reference are not in the mean.
- **Recall@k** (k = 1, 5, 10) is the number of test reference *pairs* whose
  target appears in the source's top k, divided by the number of test
  references. Its denominator is therefore pairs, while MRR's is sources.
- **Candidate recall@20** is Recall@20 of the stage-1 lists. It is the
  ceiling no re-ranking can exceed. `candidate_recall_at_20` in every B0
  result equals `scripts/compute_recall_at_20.py` (the same
  `compute_recall_at_k`).

**Pair-level matching metrics** (threshold-dependent):

- Each source predicts only its top-ranked candidate, and keeps it if the
  score reaches the threshold (one mapping per source).
- The threshold is chosen on the **validation population only**. It is the
  F1-maximising value on a grid from 0.10 to 0.99 in steps of 0.01, with ties
  resolved to the lower median, and is then applied unchanged to the test
  population.
- Precision, recall and F1 are computed over test predictions against test
  references, and include the unmatched-source false positives and the D5
  unreachable false negatives above.
- B0's scores are n-gram cosines in [0, 1], tuned on the same grid.
  Candidates padded in at score 0 can never be predicted.

**Aggregation.**

- Each column is one pair, except D3, which is the unweighted mean over its 21
  pairs. That covers the main table, its ranking companion, the comparison and
  the report. The Section 4.3 entity-type table instead weights D3 by
  `n_refs`.
- Mean* in the main table averages D1–D4 (both D4 columns) and leaves out D5.
- Since #140 the primary tests are per-dataset paired tests over test
  reference sources (`evaluation/paired_tests.py`; D3 pairs are strata). The
  cross-dataset Wilcoxon summary (`evaluation/stats.py`) treats dataset
  columns, not pairs, as observations: the 21 D3 pairs share seven
  ontologies, so they are not independent.
- B0 is not part of the C1–C20 ablation groups or their Wilcoxon families. Its
  comparisons are descriptive: one pair per column, no intervals.

## 5. Results against B0

The full tables are in `baseline_comparison.md`. Since #139 the main table
is `--metric mrr`, with `--metric hits_at_1` as the other primary table and
`--metric f1` and `--metric recall_at_1` as secondary ones; each carries B0
as its first row (`scripts/generate_results_table.py --baseline_dir`).

D1 numbers were first produced by the re-run after the D1 known-pair fix
(section 3; all 15 D1 conditions and B0 at commit 65b0d33, clean tree);
D2–D5 were unaffected by the fix.
Every stored C1–C20 and B0 result was re-run at c6931db in #141 with every
metric unchanged, so that is the commit the stored files now record.

B0 per column:

| Metric | D1 | D2 | D3 | D4_schema | D4_instance | D5 |
|---|---|---|---|---|---|---|
| MRR | 0.7055 | 0.8175 | 0.7200 | 0.8879 | 0.9049 | 0.1808 |
| Recall@1 | 0.5398 | 0.7446 | 0.6528 | 0.8636 | 0.8696 | 0.1544 |
| F1 | 0.3062 | 0.6626 | 0.5140 | 0.8090 | 0.8802 | 0.1730 |
| Candidate recall@20 | 0.8578 | 0.9465 | 0.8621 | 0.9318 | 0.9725 | 0.2667 |

The effect of re-ranking depends on the dataset. Descriptively, on one pair
per column:

Number of conditions above / below / equal to B0, per metric (as regenerated
in #143: 271 C-results, after #142 retired C12 and #145 re-ran V8):

| Metric | D1 | D2 | D3 | D4_schema | D4_instance | D5 |
|---|---|---|---|---|---|---|
| MRR | 7 / 8 / 0 | 6 / 7 / 0 | 6 / 4 / 0 | 10 / 1 / 0 | 8 / 3 / 0 | 0 / 11 / 0 |
| Recall@1 | 7 / 8 / 0 | 5 / 8 / 0 | 5 / 5 / 0 | 8 / 1 / 2 | 8 / 3 / 0 | 0 / 11 / 0 |
| F1 | 9 / 6 / 0 | 7 / 6 / 0 | 1 / 9 / 0 | 9 / 1 / 1 | 0 / 11 / 0 | 0 / 11 / 0 |

- **Ranking improves on D1, D2 and D4, but not uniformly.** The best
  condition's MRR gain over B0 is:
  - D1: +0.074 (C4, V3/M2);
  - D2: +0.085 (C4);
  - D4_schema: +0.033 (C3, V2+V6/M2; C8 and C10 tie);
  - D4_instance: +0.039 (C3).

  Only about half the conditions beat B0 on D1 and D2 MRR. Conditions without
  lexical text collapse: C9 (V8 only) loses 0.44–0.57 MRR on D1–D3.
- **Better ranking does not always give better matching F1.**
  - On D3, 6 of 10 conditions beat B0 on MRR, but only 1 of 10 on F1 (C1,
    +0.033).
  - On D4_instance, 8 of 11 beat it on MRR, and none on F1. The best F1 is
    C20's 0.8036 against B0's 0.8802.
  - On D1, by contrast, 9 of 15 beat B0 on F1 while 7 do on MRR.

  F1 depends on the top-1 score clearing a threshold tuned on validation. How
  well that threshold separates correct from incorrect top-1 predictions
  depends on the score scale (n-gram cosine against embedding cosine) as much
  as on the order. D3's validation sets are small (20% of 4–25 references per
  pair), and D4_instance's F1 counts only judgeable unmatched sources.
- **Re-ranking reduces every metric on D5.** All 11 D5 conditions are below
  B0 on MRR, Recall@1, Recall@10 and F1 (D5 section of
  `baseline_comparison.md`).

### The D5 regression

On D5, stage 1 puts the gold target first for 1,621 of 10,500 test
references. It puts it in ranks 2–20 for 1,179, outside the top 20 for 7,533,
and 167 sources are unreachable. Every embedding condition then orders those
20 candidates worse than the n-gram did:

- The best, C14 (V4+V6/M3), reaches MRR 0.1550 against B0's 0.1808 (−0.026).
  The other conditions that read attribute text are next: C5 (V4/M2) −0.030
  and C3 (V2+V6/M2) −0.045.
- Conditions whose text is labels, annotations or relational signatures (C1,
  C2, C9, C10, C13, C18, C20) score MRR 0.034–0.049. That is about what a
  random order of the same lists would score: 2,800 of 10,500 test references
  have their gold target in the list, and a uniformly random position scores
  H₂₀/20 ≈ 0.180 on average, giving 0.048.

An interpretation, consistent with the dataset notes (CLAUDE.md, D5), but not
tested here:

- OpenEA removed entity labels, and both sides have opaque local names. Label-
  and annotation-based verbalisations therefore carry almost no signal.
- Stage-1 text uses attribute values, including dates exempted from the
  bare-quantity filter. An exact shared value, such as a birth or release
  date, then gives a large n-gram overlap.
- A sentence embedding of the verbalised attributes, by contrast, dilutes that
  exact-match signal.

The regression is reported as found. D5 stays in every table and is excluded
only from Mean*, as before.

## 6. Scope and limitations

- **Controlled protocol, not a benchmark reproduction.** Every number here is
  a comparison under the protocol above: re-ranking 20 stage-1 candidates per
  source, with the split, populations and threshold rule of section 3. None of
  them reproduces the evaluation of OAEI Anatomy, Conference, KG, Bio-ML or
  OpenEA. Those evaluate global matching over the full ontologies or KGs, use
  their own candidate sets or all-target ranking, and do not split references
  by source as here. The numbers therefore must not be put side by side with
  published system scores (LogMap, AML, BERTMap, BERTMapLt, OpenEA models) as
  head-to-head results. If an external number is quoted, give its source and
  protocol and the differences above.
- **D1 validation mappings (fixed, re-run).** Results produced before the fix
  treated Bio-ML's 726 validation mappings as unmatched sources, counting a
  correct prediction for one as a false positive. They are now known pairs
  (section 3). On B0 the fix moves D1 FP from 7,576 to 7,261, F1 from 0.2986
  to 0.3062 and MRR from 0.7035 to 0.7055. Across the 15 re-run D1
  conditions, MRR rises by 0.000–0.002 and F1 by 0.004–0.012, and no
  above/below count against B0 changes. Most D1 false positives come from
  the 17,870 genuinely unmatched SNOMED sources (3,615 validation, 14,255
  test), so absolute D1 precision stays low. D1 results without
  `"n_train_refs": 726` in their population record predate the fix and are
  not comparable; the report, the results table and the ablation runners
  refuse to combine them with post-fix results. The ablation and
  serialization results of #122–#134 were not re-run, so their D1 numbers
  remain on the pre-fix population.
- **D5 release identity.** Checksums are recorded but not compared with the
  OpenEA v2.0 archive.
- **One pair per column.** D1, D2, D4 and D5 are single pairs, and D3's 21
  pairs share seven ontologies. The comparisons with B0 are descriptive. No
  claim is made beyond these five datasets.
- **No external systems.** BERTMapLt, LogMap, AML and other systems were not
  run or reproduced (#137, scope item 4).

## Reproduce

```bash
python -m kgsemembed.pipeline.ngram_baseline --datasets D1 D2 D3 D4 D5 \
    --data_dir data/ --results_dir data/results_ngram_baseline/
python scripts/generate_results_table.py --results_dir data/results/ \
    --baseline_dir data/results_ngram_baseline/ --metric mrr  # also hits_at_1, f1, recall_at_1
python scripts/generate_baseline_comparison.py
python scripts/dataset_provenance.py --data_dir data/ \
    --release_cache ~/oaei_track_cache/oaei.webdatacommons.org
```
