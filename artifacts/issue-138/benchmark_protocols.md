# Dataset provenance and evaluation protocols (#138)

This document states, for each of the five datasets, where its data and
reference alignment come from, the benchmark protocol it was published with,
the protocol this repository actually applies, and how the two differ. It
describes the implementation as it stands at the commits recorded in
`results_inventory.md`, not an idealised protocol.

Companion files (all regenerated from code; none edits `data/`):

| File | Content | Script |
|---|---|---|
| `reference_statistics.md` | reference files, cardinality, entity kinds, populations, known pairs, stage-1 search space | `scripts/reference_statistics.py` |
| `release_checks.md` | D3 against Conference ra1, D5 against OpenEA v2.0 | `scripts/release_checks.py` |
| `results_inventory.md` | commit, protocol and digests behind every stored result | `scripts/results_inventory.py` |
| `../issue-137/dataset_provenance.md` | SHA-256 of every input file; D1–D4 against the OAEI MELT repository | `scripts/dataset_provenance.py` |
| `../issue-137/evaluation_protocol.md` | the shared protocol (splits, populations, metrics), B0 | — |
| `../issue-137/baseline_comparison.md` | every condition against the stage-1 baseline B0 | `scripts/generate_baseline_comparison.py` |

## 1. What every result is

Every Phase 2 number (C1–C20, B0) is a **controlled re-ranking of 20 stage-1
candidates per source**, scored under one protocol:

1. **Stage 1** ranks every target of the pair's entity type against each
   source with a character 3-gram cosine over labels (D5: attribute text) and
   stores the top 20 (`data/candidates/`). D3 and D4_schema rank classes and
   properties in one pool, so a class source can receive property candidates.
2. **Stage 2** (a condition) verbalises the source and its 20 candidates and
   re-orders them by embedding cosine. B0 keeps the stage-1 order.
3. **Evaluation** (`run_experiment._evaluate_pair`, protocol version 2):
   - the reference is split 0/20/80 by source (D5: OpenEA fold 1);
   - every source is scored in exactly one population (validation or test);
   - known (training) pairs are removed from every list;
   - the threshold is tuned on validation, per pair;
   - test metrics are computed from the re-ranked lists.

No result is full matching over the ontologies or KGs, all-target retrieval,
or a reproduction of any benchmark's official evaluation. Section 8 states
what this permits.

### 1.1 Metrics: unit, numerator, denominator, aggregation

| Metric | Unit | Numerator | Denominator | Notes |
|---|---|---|---|---|
| MRR | test reference source | Σ 1 / rank of the source's best-ranked gold target | test reference sources | 1:N sources count once, credited with their best gold. A source whose gold is not in its list, or that has no list, scores 0. Ranked sources without a reference are ignored. |
| Hits@k (1, 5, 10) (#139) | test reference source | sources with any gold target in their top k | test reference sources | Same unit and 1:N rule as MRR. Hits@1 and Recall@1 share their numerator (one rank-1 slot per source), so Hits@1 = Recall@1 × refs / sources exactly; results stored before #139 get it that way. |
| Recall@k (1, 5, 10) | test reference pair | pairs whose target is in the source's top k | test reference pairs | Pairs, not sources: a 1:N source contributes one pair per gold target. |
| Candidate recall@20 | test reference pair | pairs whose target is in the stage-1 list | test reference pairs | Ceiling of Recall@k; equal for all conditions of a pair. MRR and Hits@k are capped by the ranking ceiling instead (§1.2). |
| Precision | prediction | correct predictions | test predictions | One prediction per test source: its top candidate if its score ≥ threshold. Includes unmatched sources (complete references) or only judgeable ones (D4). |
| Recall | test reference pair | correct predictions | test reference pairs | Includes unreachable references (D5). A 1:N source can recover at most one of its pairs. |
| F1 | — | — | — | Harmonic mean of the two above. |
| Threshold | pair | — | — | F1-maximising on the **validation** population, grid 0.10–0.99 step 0.01, ties to the lower median. Applied unchanged to test. Per-entity-type rows (D3, D4_schema) reuse the pair's threshold. |

Aggregation:

- **Column value** = the pair's value, except D3 = unweighted mean over its
  21 pairs (main table, ranking tables, report, B0 comparison). The Section
  4.3 entity-type table and the report's entity-type section (#166) use the
  same unweighted mean over the D3 pairs reporting each entity type, plus
  micro F1 pooled over them.
- **Mean\*** averages D1–D4 (both D4 columns), never D5.
- **Significance tests** (#140) are per-dataset paired tests within each
  column: units are test reference sources (MRR, Hits@1; F1 adds other
  predicted sources), 95% percentile bootstrap within pairs (D3's pairs are
  strata), paired randomisation p, 10,000 replicates, seed 42
  (`evaluation/paired_tests.py`). Only MRR gets a decision, Holm within each
  ablation group × column; Hits@1 and F1 show interval and unadjusted p.
  Every condition is also tested against B0 per column the same way (#165),
  on MRR with Holm across the conditions compared with B0 on one column.
- **Wilcoxon** over dataset columns (`evaluation/stats.py`) is a consistency
  summary on MRR and F1, not the primary test: D3 is one column, since its
  pairs share seven ontologies and are not independent.

### 1.2 Ceilings set by stage 1 and by one prediction per source

| Column | Test refs | Test ref sources | Candidate recall@20 | Ranking ceiling | Sources with no gold in list | Recall ceiling (top-1 rule) |
|---|---|---|---|---|---|---|
| D1 | 5,226 | 4,605 | 0.8578 | 0.8860 | 525 | 0.7807 |
| D2 | 1,214 | 1,198 | 0.9465 | 0.9499 | 60 | 0.9374 |
| D3 (21 pairs pooled) | 251 | 250 | 0.8566 | 0.8560 | 36 | 0.8526 |
| D4_schema | 44 | 44 | 0.9318 | 0.9318 | 3 | 0.9318 |
| D4_instance | 1,380 | 1,380 | 0.9725 | 0.9725 | 38 | 0.9725 |
| D5 | 10,500 | 10,500 | 0.2667 | 0.2667 | 7,700 (167 with no list) | 0.2667 |

*Ranking ceiling* = test reference sources with a gold target in their list ÷
test reference sources: the cap of MRR and Hits@k
(`../issue-139/metric_ceilings.md`). Candidate recall@20 caps Recall@k only;
the two differ under 1:N references (D1 0.8860 against 0.8578).

*Recall ceiling* = test reference sources with at least one gold target in
their list ÷ test references: the best recall any condition could reach with
a perfect order and threshold. It combines two limits. One prediction
per source alone caps recall at test reference sources ÷ test references
(D1: 4,605 / 5,226 = 0.881). The stage-1 lists then remove every source whose
gold targets are all missing (D1: 525), giving 0.781. On D1 the ceiling is
0.077 below candidate recall@20 because 563 test sources have two or more gold
targets but can predict only one. D3's column value is a mean of per-pair values (candidate recall@20
0.8621, ranking ceiling 0.8618), so its pooled figures here are for
orientation only. Counts are from
`reference_statistics.md` §4.

## 2. D1 — SNOMED CT ↔ FMA (body), Bio-ML

**Data and release.**

- Bio-ML 2022 equivalence matching, task `snomed.body-fma.body`, as
  redistributed in the OAEI MELT track repository under
  `bio-ml-equiv-unsupervised/2022/snomed.body-fma.body/`. All four local
  files (`source.rdf`, `target.rdf`, `reference.rdf`, `input.rdf`) are
  byte-identical to it.
- SNOMED CT (UMLS US.2021.09.01 release, ontology version IRI
  `…/version/20220317`) pruned to "Body"; FMA 4.14.0 (BioPortal).
- The ontology sizes match the published task: 24,182 SNOMED and 64,726 FMA
  classes [He et al. 2022, Table 2]; locally, 24,182 stage-1 sources and a
  64,726-class target pool.

**Reference alignment.**

- The task has 7,256 equivalence mappings [ibid.]. In the unsupervised
  setting they are split 10% validation / 90% test [§3.3].
- The local files are exactly that split:
  - `input.rdf`: 726 cells (validation);
  - `reference.rdf`: 6,530 cells (test);
  - 726 + 6,530 = 7,256, with no shared pair.
- Every cell is `=` with measure 1.0. There are no malformed or duplicate
  cells.
- The 6,530 test mappings cover 5,756 sources and 6,104 targets:

  | Cardinality | Pairs |
  |---|---|
  | 1:1 | 4,656 |
  | 1:N | 1,046 |
  | N:1 | 397 |
  | N:M | 431 |

  Every end is a class in its graph.

**Explaining the difference from the published count.** 6,530 is not a
filtered or damaged 7,256. It is the unsupervised setting's test set, and the
missing 726 are its validation set, shipped as `input.rdf`. Nothing was
dropped, deduplicated or normalised.

**Official protocol** [He et al. 2022, §3]:

- **Global matching:** P/R/F over the system's full output, on the test
  mappings. Mappings outside the evaluated set are removed from the output
  first.
- **Local ranking:** each test mapping (c, c′) is ranked against 100 negative
  candidates (50 IDF-sampled, 50 neighbour-sampled, random fill) plus c′.
  Metrics are Hits@K and MRR over mappings. The gold target is always among
  the candidates.

**This study:**

- `reference.rdf` is re-split 20/80 by source into validation (1,304 refs,
  1,151 sources) and test (5,226 refs, 4,605 sources).
- `input.rdf` is loaded as **known pairs** (since #137):
  - 556 sources with no other reference are excluded;
  - 157 keep their population;
  - 122 known candidate pairs are removed from scored lists.
- Unmatched SNOMED sources (17,870) are scored as false positives.
- Candidates are the 20 stage-1 n-gram candidates from all 64,726 FMA classes.

| Aspect | Bio-ML | This study |
|---|---|---|
| Evaluated mappings | 6,530 test (unsupervised) | 5,226 of the 6,530 (80% of sources); 1,304 tune the threshold |
| Validation mappings | 726, for tuning | 726 known pairs, excluded and never scored; tuning uses 20% of the test mappings instead |
| Ranking candidates | 100 sampled hard negatives + gold, per mapping | 20 n-gram candidates per source; gold missing for 525 test sources |
| Ranking unit | mapping | source (best gold) for MRR; mapping for Recall@k |
| Matching | system's global output | top-1 per source over the 20 candidates, validation threshold |
| Output outside the evaluated set | excluded from P/R/F | known pairs removed from lists; validation-population sources are not scored on test |

**Implications.**

- D1 MRR/Hits are not Bio-ML local-ranking scores: different candidates, a
  different unit, and the gold can be missing.
- D1 F1 is not Bio-ML global-matching F1: a subset of the test mappings,
  top-1 over 20 candidates, and 14,255 unmatched test sources whose
  predictions are false positives (so precision is low for every method).
- Published Bio-ML numbers (e.g. BERTMap, LogMap) are context only (§8).

## 3. D2 — Mouse anatomy ↔ NCI anatomy, OAEI Anatomy

**Data and release.**

- OAEI Anatomy track: the Adult Mouse Anatomy and the anatomy part of the NCI
  Thesaurus.
- MELT `anatomy_track/anatomy_track-default/mouse-human-suite`; all three
  files are byte-identical.
- 2,744 MA source classes; 3,304 NCI classes in the target pool.

**Reference alignment.**

- Initial mapping by Ringwald and Hayamizu (MGI), improved by Beisswanger
  [OAEI Anatomy; Bodenreider et al. 2005].
- 1,516 `=` cells; no malformed or duplicate cells.
- 1,497 sources and 1,509 targets:

  | Cardinality | Pairs |
  |---|---|
  | 1:1 | 1,465 |
  | 1:N | 37 |
  | N:1 | 13 |
  | N:M | 1 |

**Official protocol.** Systems match the full ontologies, with no training
data and no reference split. Precision, recall, F-measure and recall+ (recall
on non-trivial correspondences, whose entities do not share a normalised
label) are computed against all 1,516 correspondences, and coherence is
checked [OAEI 2024 Anatomy results].

**This study.** The reference is split 20/80 by source:

- validation: 302 refs, 299 sources;
- test: 1,214 refs, 1,198 sources.

Unmatched MA classes are drawn into both populations (257 validation, 990
test) as false-positive sources. Candidates are 20 per source from all 3,304
NCI classes, and the gold is missing for 60 test sources.

| Aspect | OAEI Anatomy | This study |
|---|---|---|
| Evaluated reference | all 1,516 | 1,214 (80% of sources) |
| Tuning data | none | 302 validation references |
| Matching | global, any arity | top-1 per source over 20 candidates |
| Measures | P, R, F, recall+, coherence | P, R, F, MRR, Recall@k; no recall+ or coherence |

**Implications.** D2 F1 is over 80% of the reference, with a threshold tuned
on the other 20%, which no OAEI participant has. Not comparable with track
results.

## 4. D3 — Conference (21 pairs), OAEI Conference

**Data and release.**

- Seven OntoFarm ontologies (cmt, conference, confOf, edas, ekaw, iasted,
  sigkdd) [Zamazal & Svátek 2017]. All 21 pairs are the track's test cases.
- Every file is byte-identical to MELT `conference/conference-v1`.
- **Reference = ra1, confirmed** (`release_checks.md`): all 21
  `reference.rdf` files hold exactly the cells, in order, of the Conference
  track's `reference-alignment.zip`. 20 are byte-identical. `conference-confof`
  has the same cells and header elements, but in the original its first
  `<map>` is opened before the `onto1`/`onto2` declarations and encloses them.
  The MELT copy used here moves that `<map>` after them; nothing else changes.
- ra1 is the original reference. ra2 is its conflict-resolved transitive
  closure, and rar2 additionally resolves consistency and conservativity
  violations [OAEI Conference evaluation].
- The local files contain classes and properties, i.e. the **M3** variant
  (M1 = classes, M2 = properties).

**Reference alignment.**

- 305 `=` cells over 21 pairs (4–25 per pair): 259 class pairs and 46 property
  pairs (`reference_statistics.md` §3).
- Two N:1 pairs (confof-ekaw) and two 1:N pairs (cmt-conference).

**Official protocol** [OAEI 2024 Conference results]:

- Systems match each pair globally, with no training data.
- Main results use **rar2-M3**; ra1 underlies the uncertain-reference
  evaluation.
- Precision and recall are **micro-averaged**: TP/FP/FN are summed over the
  21 test cases. The highest average F1 under thresholding is also reported,
  alongside M1/M2 breakdowns and logical-coherence analysis.

**This study:**

- Each pair is split 20/80 by source: 1–5 validation references per pair, 54
  in all; 251 test references over 250 sources.
- The threshold is tuned **per pair** on those 1–5 references.
- Each pair also scores its unmatched classes and properties (334 validation,
  1,653 test) as false-positive sources.
- Candidates are 20 per source from a mixed class+property pool. Because
  every pool holds 74–181 entities, 20 candidates are 11–27% of the targets.
- The D3 column is the **unweighted mean** over pairs (a macro average).
- Class/predicate rows (Section 4.3 table) are weighted by `n_refs`.

| Aspect | OAEI Conference | This study |
|---|---|---|
| Reference | rar2-M3 (main); ra1, ra2 also published | ra1-M3 |
| Evaluated reference | all of it | 80% of sources per pair (251 refs) |
| Threshold | the system's own output; organisers also report the best F1 under thresholding | per pair, from 1–5 validation references |
| Aggregation | micro (summed counts) | macro (mean of pair values) |
| Ranking metrics | none | MRR, Recall@k within 20 candidates |

**Implications.** D3 F1 is a macro mean of 21 small, noisy per-pair F1s,
each with a threshold tuned on very few references; this is why ranking gains
on D3 often fail to carry to F1 (`baseline_comparison.md`). Not comparable
with Conference track results, which use a different reference (rar2),
micro-averaging and global matching.

## 5. D4 — memoryalpha ↔ stexpanded, OAEI Knowledge Graph track

**Data and release.**

- OAEI KG track v4 (MELT `knowledgegraph/v4`): Memory Alpha and Star Trek
  Expanded Universe wiki KGs [Hertling & Paulheim 2020]. All three files are
  byte-identical.
- Stage 1:
  - D4_schema: 4,131 class+property sources against 1,947 targets;
  - D4_instance: 156,581 instance sources against 39,305 targets.

**Reference alignment.**

- 1,779 `=` cells, all 1:1. The gold standard is **partial** [ibid.].
- Partitioned by the source's `rdf:type`:
  - D4_schema: 54 (13 class + 41 property);
  - D4_instance: 1,725.
- These are exactly the published sizes for this test case (13 class, 41
  property, 1,725 instance matches [ibid., Table 3]). No reference end is
  untyped or absent from its graph.

**Official protocol** [ibid., §3.2]:

- Systems match the whole KGs, schema and instances at once, with no training
  data.
- Because the gold standard is partial and 1:1, a system correspondence is a
  false positive only if it maps a gold-standard entity to a different
  partner. Correspondences between entities outside the gold standard are
  ignored.
- Results are reported per test case, overall and separately at schema and
  instance level [OAEI KG track].

**This study:**

- Schema and instance references are split 20/80 by source separately:
  - D4_schema: 10 validation / 44 test;
  - D4_instance: 345 / 1,380.
- `reference_complete = False`: an unmatched source's prediction is judged
  only when its top target is a reference target (then FP), otherwise
  ignored. This is the track's rule seen from the target side; reference
  sources' wrong predictions are FPs as in the track.
- The number of judgeable unmatched sources depends on each method's top
  target, so it is the only population count that varies between methods.
- D4_schema records per-entity-type metrics (test: 12 class, 32 property
  references, one shared threshold), but the entity-type analysis is D3 only
  (#166): one pair from another benchmark is not pooled with D3's 21.

| Aspect | OAEI KG track | This study |
|---|---|---|
| Task | one joint schema+instance matching | two separate pairs, schema and instance |
| Evaluated reference | all 1,779 | 44 + 1,380 (80% of sources) |
| FP rule | correspondence conflicting with a gold mapping | same rule, applied to top-1 predictions |
| Tuning | none | 10 / 345 validation references |

**Implications.** D4_schema has 44 test references, so its F1 and MRR move by
≈0.02 per source. Not comparable with KG track results: different task
structure, subset, and one-prediction-per-source matching.

## 6. D5 — DBpedia ↔ Wikidata 15K (English), OpenEA

**Data and release.**

- OpenEA benchmark v2.0, dataset `D_W_15K_V2` [Sun et al. 2020].
- **Release identity confirmed** (`release_checks.md`): the downloaded
  `OpenEA_dataset_v2.0.zip` matches the SHA-512 that pykeen pins for it, and
  all eight files the loader reads (relation and attribute triples,
  `ent_links`, fold-1 splits) are byte-identical to it.
- OpenEA deleted entity labels during dataset construction [ibid.], and
  Wikidata URIs are opaque.

**Reference alignment.**

- 15,000 links in `ent_links`, all 1:1.
- Fold 1 of the 5-fold split: 3,000 train / 1,500 valid / 10,500 test. The
  three files are disjoint and together equal `ent_links`.
- 188 reference sources (21 validation, 167 test) are the subject of no
  triple, so stage 1 never ranks them. 6 targets are likewise absent from the
  Wikidata graph.

**Official protocol** [ibid., §5.1; OpenEA library]:

- Supervised embedding-based entity alignment. Each of 5 folds trains on 20%
  of the links, validates on 10% and tests on 70%; results are reported per
  fold or averaged over folds.
- Metrics are Hits@1, Hits@5, MR and MRR, aligning DBpedia → Wikidata.
- At test time the OpenEA library ranks each test source against **all test
  targets** (10,500 per fold). With one candidate list per source, P = R = F1
  = Hits@1 [ibid., §6.3].

**This study:**

- Fold 1 only.
- Training links are **known pairs**, never trained on: their 3,000 sources
  are excluded from scoring.
- Validation tunes the threshold (1,500 references); test is scored on
  10,500.
- Candidates are 20 per source from all 14,992 Wikidata entities with
  triples. Stage-1 text is attribute values (labels are absent), with dates
  exempted from the bare-quantity filter.
- The gold is in the list for only 2,800 of 10,500 test references, so every
  metric is capped at 0.2667. The 167 unreachable sources stay in every
  denominator.

| Aspect | OpenEA | This study |
|---|---|---|
| Learning | supervised on 20% of links | none; training links are only excluded |
| Folds | 5, averaged | fold 1 |
| Ranking space | all 10,500 test targets | 20 n-gram candidates from 14,992 targets; gold present for 26.7% |
| Matching | top-1 = Hits@1 | top-1 above a validation threshold |

**Implications.**

- D5 MRR/Hits@1 are **within-candidate** ranking scores, capped by the
  ranking ceiling (0.2667, equal to candidate recall@20 on D5's 1:1
  reference). They cannot be compared with OpenEA's all-target Hits@1/MRR,
  which also come from supervised models.
- Every embedding condition is below B0 on D5 (`baseline_comparison.md`).
- D5 is excluded from Mean\*.

## 7. Provenance status

| Column | Data files | Reference | Checked against | Open uncertainty |
|---|---|---|---|---|
| D1 | identical | identical; 7,256 = 726 + 6,530 explained | MELT copy of Bio-ML 2022 | The Bio-ML Zenodo archive itself was not compared; MELT is a redistribution. Ontology versions are taken from the files' headers and the Bio-ML paper. |
| D2 | identical | identical | MELT anatomy track | none |
| D3 | identical | ra1-M3, cell-identical (20 byte-identical) | MELT conference-v1; Conference ra1 archive | The ontologies were checked against MELT only (the ra1 archive holds alignments only). |
| D4 | identical | identical; sizes equal the published ones | MELT KG track v4 | none |
| D5 | identical | identical | OpenEA v2.0 archive (pinned SHA-512) | none |

No reference cell was filtered, normalised or deduplicated. The loader's only
filters are `=` relations and resource-bearing cells, and they drop nothing on
these files. The D4 schema/instance partition and the 20/80 split are the
only transformations.

## 8. Rules for reporting

- Call the results *controlled comparisons under a re-ranking protocol*, never
  benchmark results, OAEI results, Bio-ML results or OpenEA results.
- Do not put published system scores (LogMap, AML, BERTMap, BERTMapLt,
  OpenEA models, track participants) in the same table as these numbers, or
  rank against them. If one is quoted for context, give its source, its
  setting (e.g. Bio-ML unsupervised global matching; Conference rar2-M3
  micro; OpenEA 5-fold all-target Hits@1) and the differences in §2–6.
- BERTMapLt contributes only the one-mapping-per-source prediction rule.
  BERTMapLt itself was not run, and nothing here is compared with its
  published scores.
- Compare conditions only within a column and on matched populations;
  `baseline_comparison.md` verifies 277/277 results (the sanity check C19
  left out) share candidates, population and protocol with B0.
- Report ranking metrics (MRR and Hits@1, primary since #139, and Recall@k:
  threshold-free, within candidates)
  and matching metrics (P/R/F1: top-1 above a validation-tuned threshold)
  separately; they can disagree (D3, D4_instance).
- Report D5 and D4_schema with their caps and sizes (§1.2, §5).
- Reproducing an official protocol (e.g. Bio-ML local ranking with its 100
  negatives, or OpenEA all-target ranking) is a separate extension, not part
  of this study.

## References

- Bodenreider, O., Hayamizu, T. F., Ringwald, M., De Coronado, S., Zhang, S.
  (2005). Of mice and men: Aligning mouse and human anatomies. *AMIA Annual
  Symposium Proceedings*, 61–65.
- He, Y., Chen, J., Dong, H., Jiménez-Ruiz, E., Hadian, A., Horrocks, I.
  (2022). Machine Learning-Friendly Biomedical Datasets for Equivalence and
  Subsumption Ontology Matching. *ISWC 2022*. arXiv:2205.03447.
- Hertling, S., Paulheim, H. (2020). The Knowledge Graph Track at OAEI: Gold
  Standards, Baselines, and the Golden Hammer Bias. *ESWC 2020*.
  arXiv:2002.10283.
- Sun, Z., Zhang, Q., Hu, W., Wang, C., Chen, M., Akrami, F., Li, C. (2020).
  A Benchmarking Study of Embedding-based Entity Alignment for Knowledge
  Graphs. *PVLDB* 13(11), 2326–2340. doi:10.14778/3407790.3407828.
- Zamazal, O., Svátek, V. (2017). The Ten-Year OntoFarm and its Fertilization
  within the Onto-Sphere. *Journal of Web Semantics* 43, 46–53.
- OAEI Anatomy track: https://oaei.ontologymatching.org/2024/anatomy/ and
  results https://oaei.ontologymatching.org/2024/results/anatomy/
- OAEI Conference track: evaluation variants
  https://oaei.ontologymatching.org/2016/conference/eval.html, results
  https://oaei.ontologymatching.org/2024/results/conference/, ra1
  https://oaei.ontologymatching.org/2024/conference/data/reference-alignment.zip
- OpenEA v2.0 archive: https://figshare.com/ndownloader/files/34234391;
  library https://github.com/nju-websoft/OpenEA
- Hertling, S., Portisch, J., Paulheim, H. (2019). MELT — Matching EvaLuation
  Toolkit. *SEMANTiCS 2019*. The MELT track repository is served from
  `oaei.webdatacommons.org`; a local copy of it is what
  `dataset_provenance.py --release_cache` compares against.
- OAEI Knowledge Graph track: https://oaei.ontologymatching.org/2023/knowledgegraph/
