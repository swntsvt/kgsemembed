# Controlled PPAS ablation (#122, #128)

## Reading this report

- Each dataset contributes one alignment pair, so there is no replication at
  the dataset level and no significance test is reported.  Bootstrap intervals
  resample test sources of that pair; sources share candidate targets, so the
  intervals are approximate and conditional on this dataset and candidate set.
- The random policy's t-interval is over five seeds and describes selection
  variability only.
- Entities whose full text fits the budget are byte-identical, and share one
  cached embedding, under every policy; differences arise only from sampled
  entities.  The "sampling not invoked" stratum can still move when a sampled
  candidate competes in an unsampled source's list.
- F1 depends on a threshold tuned on the validation population; MRR and
  Recall@k do not.  Read the ranking metrics first.
- The full reference differs from the capped runs in input length, and inputs
  beyond 512 tokens are truncated by the model ("truncated" columns).

## Reading the controlled-policy section (#128)

- Selected items are always rendered in canonical order, so two policies that
  select the same item set produce byte-identical text.  An ordering policy
  can therefore change *which* items are kept but never their presentation
  order: "same set, different order" is zero by construction, and the
  selection-change counts capture the whole effect of a policy.
- Bootstrap intervals resample the test sources of one pair (the unit of
  resampling) with their paired per-source differences, 2,000 resamples, one
  fixed-seed generator per contrast.  They quantify uncertainty across source
  entities within that one evaluation, not generalisation to other KG pairs.
- "Source and gold targets unchanged" is not an unaffected stratum: a
  source's rank still moves when a policy changes the text of competing
  (non-gold) candidates in its list.  Read it as the distractor effect.
- Runs are combined only within one run signature; a smoke run is never
  combined with a full-population run.

## D1 / d1_snomed_fma

### Controlled policies (#128)

#### Run group 1

- Model `BAAI/bge-large-en-v1.5` revision `d4aa6901d3a41ba39fb536a557fa166f842b0e09`; tokenizer revision `d4aa6901d3a41ba39fb536a557fa166f842b0e09`
- Candidates sha256 `a65eb2480a75f9ec…`; evaluation protocol 2; budget 48
- Population: full; test-source set `b7e3e2233cf3cf77`; test references 5226; known pairs 726
- Git commits: ['66aabc3b6c']; dirty: ['False']
- Missing policies: none

##### Execution matrix

| Policy | Condition | Dataset | Budget | Filter | Vocabulary | Config hash | Order | Status |
|---|---|---|---|---|---|---|---|---|
| P1 | `PA-all-base-det-48` | D1 | 48 | all | base | `eb299c6ac2d9` | deterministic | completed |
| P2 | `PA-all-base-ppas-48` | D1 | 48 | all | base | `eb299c6ac2d9` | ppas | completed |
| P3 | `PA-cov-base-det-48` | D1 | 48 | covered | base | `eb299c6ac2d9` | deterministic | completed |
| P4 | `PA-cov-base-ppas-48` | D1 | 48 | covered | base | `eb299c6ac2d9` | ppas | completed |
| P5 | `PA-all-fmax-ppas-48` | D1 | 48 | all | fma_extended | `1ebca7d18c8f` | ppas | completed |
| P6 | `PA-cov-fmax-det-48` | D1 | 48 | covered | fma_extended | `1ebca7d18c8f` | deterministic | completed |
| P7 | `PA-cov-fmax-ppas-48` | D1 | 48 | covered | fma_extended | `1ebca7d18c8f` | ppas | completed |

##### Matching quality (test population)

| Policy | mrr | recall_at_1 | recall_at_5 | recall_at_10 | f1 | precision | recall | threshold |
|---|---|---|---|---|---|---|---|---|
| P1 | 0.3567 | 0.2277 | 0.3827 | 0.5119 | 0.1206 | 0.0931 | 0.1709 | 0.8100 |
| P2 | 0.4941 | 0.3243 | 0.6150 | 0.7491 | 0.1450 | 0.0953 | 0.3035 | 0.7700 |
| P3 | 0.6992 | 0.5440 | 0.7710 | 0.8243 | 0.2757 | 0.2088 | 0.4059 | 0.9000 |
| P4 | 0.6862 | 0.5302 | 0.7620 | 0.8209 | 0.2368 | 0.1592 | 0.4619 | 0.8600 |
| P5 | 0.4984 | 0.3243 | 0.6169 | 0.7539 | 0.1550 | 0.1132 | 0.2455 | 0.8200 |
| P6 | 0.5514 | 0.3858 | 0.6370 | 0.7453 | 0.1901 | 0.1354 | 0.3190 | 0.8000 |
| P7 | 0.5118 | 0.3354 | 0.6322 | 0.7643 | 0.1538 | 0.1013 | 0.3194 | 0.7800 |

##### Representation cost and item accounting

| Policy | Side | Mean tok | Median tok | Max tok | Over budget | Utilisation | Sampled | Items | Filtered | Eligible | Selected | Skipped | Tiered | Vocab coverage | Predicate coverage |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P1 | source | 36.3 | 37 | 48 | 0 | 0.755 | 24.6% | 98880 | 0 | 98880 | 90757 | 8123 | 98880 | 100.0% | 1.000 |
| P1 | candidate | 42.3 | 43 | 48 | 0 | 0.882 | 81.3% | 306359 | 0 | 306359 | 221845 | 84514 | 150044 | 49.0% | 1.000 |
| P2 | source | 36.8 | 37 | 48 | 0 | 0.767 | 24.6% | 98880 | 0 | 98880 | 90514 | 8366 | 98880 | 100.0% | 1.000 |
| P2 | candidate | 42.7 | 43 | 48 | 0 | 0.889 | 81.3% | 306359 | 0 | 306359 | 226453 | 79906 | 150044 | 49.0% | 1.000 |
| P3 | source | 36.3 | 37 | 48 | 0 | 0.755 | 24.6% | 98880 | 0 | 98880 | 90757 | 8123 | 98880 | 100.0% | 1.000 |
| P3 | candidate | 29.9 | 29 | 48 | 0 | 0.624 | 1.3% | 306359 | 156315 | 150044 | 149392 | 652 | 150044 | 49.0% | 0.527 |
| P4 | source | 36.8 | 37 | 48 | 0 | 0.767 | 24.6% | 98880 | 0 | 98880 | 90514 | 8366 | 98880 | 100.0% | 1.000 |
| P4 | candidate | 29.9 | 29 | 48 | 0 | 0.624 | 1.3% | 306359 | 156315 | 150044 | 149378 | 666 | 150044 | 49.0% | 0.527 |
| P5 | source | 36.8 | 37 | 48 | 0 | 0.767 | 24.6% | 98880 | 0 | 98880 | 90514 | 8366 | 98880 | 100.0% | 1.000 |
| P5 | candidate | 44.3 | 45 | 48 | 0 | 0.923 | 81.3% | 306359 | 0 | 306359 | 222319 | 84040 | 231384 | 91.9% | 1.000 |
| P6 | source | 36.3 | 37 | 48 | 0 | 0.755 | 24.6% | 98880 | 0 | 98880 | 90757 | 8123 | 98880 | 100.0% | 1.000 |
| P6 | candidate | 41.9 | 42 | 48 | 0 | 0.872 | 71.4% | 306359 | 24872 | 281487 | 225575 | 55912 | 231384 | 91.9% | 0.941 |
| P7 | source | 36.8 | 37 | 48 | 0 | 0.767 | 24.6% | 98880 | 0 | 98880 | 90514 | 8366 | 98880 | 100.0% | 1.000 |
| P7 | candidate | 44.1 | 45 | 48 | 0 | 0.919 | 71.4% | 306359 | 24872 | 281487 | 221321 | 60166 | 231384 | 91.9% | 0.941 |

##### Selection changes between policies

| Contrast | Effect | Sources changed | Candidates changed | Same set, order changed |
|---|---|---|---|---|
| P3 vs P1 | filtering, deterministic order | 0 / 24182 (0.0%) | 50103 / 50103 (100.0%) | 0 (canonical rendering) |
| P4 vs P2 | filtering, PPAS order | 0 / 24182 (0.0%) | 46975 / 50103 (93.8%) | 0 (canonical rendering) |
| P4 vs P3 | ordering, eligibility fixed (base) | 4785 / 24182 (19.8%) | 13 / 50103 (0.0%) | 0 (canonical rendering) |
| P2 vs P1 | ordering, all predicates eligible | 4785 / 24182 (19.8%) | 37985 / 50103 (75.8%) | 0 (canonical rendering) |
| P5 vs P2 | vocabulary, PPAS order | 0 / 24182 (0.0%) | 37554 / 50103 (75.0%) | 0 (canonical rendering) |
| P6 vs P3 | vocabulary, covered eligibility | 0 / 24182 (0.0%) | 50103 / 50103 (100.0%) | 0 (canonical rendering) |
| P7 vs P6 | ordering, eligibility fixed (fma_extended) | 4785 / 24182 (19.8%) | 23428 / 50103 (46.8%) | 0 (canonical rendering) |
| P7 vs P5 | filtering, fma_extended PPAS | 0 / 24182 (0.0%) | 1015 / 50103 (2.0%) | 0 (canonical rendering) |

##### Paired differences over test sources (95% paired bootstrap, unit = test source)

| Contrast | Effect | Test sources | n | Δ MRR [95% CI] | Δ R@1 [95% CI] |
|---|---|---|---|---|---|
| P3 − P1 | filtering, deterministic order | all test sources | 4605 | 0.3425 [0.3296, 0.3548] | 0.3590 [0.3429, 0.3750] |
| P3 − P1 | filtering, deterministic order | selection differs for source or a gold target | 4568 | 0.3453 [0.3315, 0.3578] | 0.3619 [0.3457, 0.3787] |
| P3 − P1 | filtering, deterministic order | source and gold targets unchanged | 37 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| P4 − P2 | filtering, PPAS order | all test sources | 4605 | 0.1921 [0.1814, 0.2035] | 0.2337 [0.2178, 0.2482] |
| P4 − P2 | filtering, PPAS order | selection differs for source or a gold target | 4530 | 0.1957 [0.1846, 0.2067] | 0.2377 [0.2232, 0.2530] |
| P4 − P2 | filtering, PPAS order | source and gold targets unchanged | 75 | -0.0257 [-0.0526, -0.0056] | -0.0133 [-0.0400, 0.0000] |
| P4 − P3 | ordering, eligibility fixed (base) | all test sources | 4605 | -0.0130 [-0.0166, -0.0098] | -0.0156 [-0.0213, -0.0100] |
| P4 − P3 | ordering, eligibility fixed (base) | selection differs for source or a gold target | 1276 | -0.0469 [-0.0589, -0.0345] | -0.0564 [-0.0760, -0.0376] |
| P4 − P3 | ordering, eligibility fixed (base) | source and gold targets unchanged | 3329 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| P2 − P1 | ordering, all predicates eligible | all test sources | 4605 | 0.1374 [0.1265, 0.1476] | 0.1097 [0.0966, 0.1233] |
| P2 − P1 | ordering, all predicates eligible | selection differs for source or a gold target | 3817 | 0.1807 [0.1692, 0.1931] | 0.1514 [0.1375, 0.1669] |
| P2 − P1 | ordering, all predicates eligible | source and gold targets unchanged | 788 | -0.0726 [-0.0845, -0.0602] | -0.0926 [-0.1130, -0.0736] |
| P5 − P2 | vocabulary, PPAS order | all test sources | 4605 | 0.0043 [-0.0067, 0.0155] | 0.0000 [-0.0156, 0.0143] |
| P5 − P2 | vocabulary, PPAS order | selection differs for source or a gold target | 3562 | 0.0357 [0.0219, 0.0493] | 0.0427 [0.0241, 0.0598] |
| P5 − P2 | vocabulary, PPAS order | source and gold targets unchanged | 1043 | -0.1033 [-0.1187, -0.0880] | -0.1457 [-0.1707, -0.1198] |
| P6 − P3 | vocabulary, covered eligibility | all test sources | 4605 | -0.1478 [-0.1595, -0.1364] | -0.1796 [-0.1957, -0.1633] |
| P6 − P3 | vocabulary, covered eligibility | selection differs for source or a gold target | 4568 | -0.1490 [-0.1605, -0.1365] | -0.1810 [-0.1981, -0.1642] |
| P6 − P3 | vocabulary, covered eligibility | source and gold targets unchanged | 37 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| P7 − P6 | ordering, eligibility fixed (fma_extended) | all test sources | 4605 | -0.0396 [-0.0501, -0.0292] | -0.0571 [-0.0721, -0.0428] |
| P7 − P6 | ordering, eligibility fixed (fma_extended) | selection differs for source or a gold target | 2774 | 0.0615 [0.0474, 0.0758] | 0.0649 [0.0451, 0.0833] |
| P7 − P6 | ordering, eligibility fixed (fma_extended) | source and gold targets unchanged | 1831 | -0.1928 [-0.2057, -0.1799] | -0.2419 [-0.2616, -0.2223] |
| P7 − P5 | filtering, fma_extended PPAS | all test sources | 4605 | 0.0134 [0.0103, 0.0164] | 0.0126 [0.0091, 0.0163] |
| P7 − P5 | filtering, fma_extended PPAS | selection differs for source or a gold target | 324 | 0.2040 [0.1689, 0.2395] | 0.1944 [0.1481, 0.2438] |
| P7 − P5 | filtering, fma_extended PPAS | source and gold targets unchanged | 4281 | -0.0011 [-0.0018, -0.0004] | -0.0012 [-0.0026, 0.0000] |
