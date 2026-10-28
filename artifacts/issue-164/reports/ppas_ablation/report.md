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

## D5 / D_W_15K_V2

### Policy ablation (#122)

- Model `BAAI/bge-large-en-v1.5` revision `d4aa6901d3a41ba39fb536a557fa166f842b0e09`; tokenizer `BAAI/bge-large-en-v1.5`
- Candidates `data/candidates/D5/D_W_15K_V2_candidates.json` sha256 `2dbf0204239800ca…` (identical across runs: True)
- Git commits: ['be06539e3c']; dirty: ['False']
- Representation: Structured key-value items (V6 rendering) from _collect_triples; literal datatype suffix stripped; empty and duplicate items dropped; selected items joined in canonical order
- Budget counting: model tokenizer on the final joined text, special tokens included, no truncation
- Deterministic order: lexicographic (predicate URI, object term); PPAS tier config sha256 `34f65847c146167e…`
- Threshold: tuned on the validation population only, grid 0.10-0.99 step 0.01, ties to the lower median; one top-ranked prediction per source
- Test references: 10500; validation references: 1500

### Budget 420 tokens

#### Coverage and cost

| Condition | Src sampled | Src mean tok | Src truncated | Cand sampled | Cand mean tok | Cand truncated | Model input tokens | Oversized items | Selection s | New texts encoded | Encode s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PA-full | 0 (0.0%) | 69.7 | 0 | 0 (0.0%) | 170.0 | 75 | 3338159 | 0 | 2.1 | 28381 | 747.0 |
| PA-det-420 | 2 (0.0%) | 69.7 | 0 | 241 (1.8%) | 168.1 | 0 | 3324502 | 0 | 3.9 | 0 | 0.0 |
| PA-ppas-420 | 2 (0.0%) | 69.7 | 0 | 241 (1.8%) | 168.1 | 0 | 3324502 | 0 | 3.9 | 243 | 18.5 |
| PA-rand-420-s0 | 2 (0.0%) | 69.7 | 0 | 241 (1.8%) | 168.1 | 0 | 3324448 | 0 | 5.2 | 242 | 18.4 |
| PA-rand-420-s1 | 2 (0.0%) | 69.7 | 0 | 241 (1.8%) | 168.2 | 0 | 3324499 | 0 | 3.8 | 238 | 18.3 |
| PA-rand-420-s2 | 2 (0.0%) | 69.7 | 0 | 241 (1.8%) | 168.1 | 0 | 3324612 | 0 | 3.9 | 238 | 18.1 |
| PA-rand-420-s3 | 2 (0.0%) | 69.7 | 0 | 241 (1.8%) | 168.1 | 0 | 3324543 | 0 | 3.9 | 239 | 18.1 |
| PA-rand-420-s4 | 2 (0.0%) | 69.7 | 0 | 241 (1.8%) | 168.1 | 0 | 3324510 | 0 | 3.9 | 238 | 18.0 |

#### Effectiveness (test population)

| Policy | f1 | precision | recall | threshold | mrr | recall_at_1 | recall_at_5 | recall_at_10 |
|---|---|---|---|---|---|---|---|---|
| Full (uncapped reference) | 0.1515 | 0.1560 | 0.1471 | 0.5800 | 0.1777 | 0.1502 | 0.2085 | 0.2372 |
| PPAS | 0.1515 | 0.1560 | 0.1471 | 0.5800 | 0.1777 | 0.1502 | 0.2087 | 0.2372 |
| Deterministic | 0.1515 | 0.1560 | 0.1471 | 0.5800 | 0.1777 | 0.1502 | 0.2087 | 0.2372 |
| Random mean ± SD (seeds 0,1,2,3,4) | 0.1512 ± 0.0002 | 0.1558 ± 0.0002 | 0.1469 ± 0.0002 | 0.5800 ± 0.0000 | 0.1776 ± 0.0001 | 0.1500 ± 0.0002 | 0.2085 ± 0.0002 | 0.2373 ± 0.0001 |
| Random 95% t-interval | [0.1509, 0.1514] | [0.1555, 0.1560] | [0.1466, 0.1471] | [0.5800, 0.5800] | [0.1774, 0.1777] | [0.1497, 0.1502] | [0.2083, 0.2087] | [0.2371, 0.2374] |
| Random seed 0 | 0.1511 | 0.1556 | 0.1468 | 0.5800 | 0.1775 | 0.1499 | 0.2087 | 0.2373 |
| Random seed 1 | 0.1514 | 0.1559 | 0.1470 | 0.5800 | 0.1776 | 0.1501 | 0.2084 | 0.2371 |
| Random seed 2 | 0.1511 | 0.1557 | 0.1468 | 0.5800 | 0.1774 | 0.1498 | 0.2087 | 0.2373 |
| Random seed 3 | 0.1514 | 0.1560 | 0.1471 | 0.5800 | 0.1777 | 0.1503 | 0.2083 | 0.2371 |
| Random seed 4 | 0.1510 | 0.1555 | 0.1467 | 0.5800 | 0.1775 | 0.1498 | 0.2086 | 0.2373 |

#### Sampled selections: PPAS vs deterministic

| Side | Sampled entities | PPAS ≠ deterministic | Share |
|---|---|---|---|
| source | 2 | 0 | 0.0% |
| candidate | 241 | 0 | 0.0% |

#### Paired differences over test sources (95% paired bootstrap)

| Stratum | Contrast | Test sources | Δ MRR [95% CI] | Δ R@1 [95% CI] |
|---|---|---|---|---|
| all test sources | PPAS − Deterministic | 10500 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| all test sources | PPAS − Random (seed mean) | 10500 | 0.0002 [-0.0000, 0.0004] | 0.0002 [-0.0001, 0.0005] |
| all test sources | Deterministic − Random (seed mean) | 10500 | 0.0002 [-0.0000, 0.0004] | 0.0002 [-0.0001, 0.0005] |
| all test sources | PPAS − Full | 10500 | 0.0000 [-0.0001, 0.0002] | 0.0000 [-0.0003, 0.0003] |
| sampling invoked | PPAS − Deterministic | 169 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| sampling invoked | PPAS − Random (seed mean) | 169 | 0.0076 [-0.0027, 0.0194] | 0.0083 [-0.0083, 0.0260] |
| sampling invoked | Deterministic − Random (seed mean) | 169 | 0.0076 [-0.0030, 0.0192] | 0.0083 [-0.0083, 0.0260] |
| sampling invoked | PPAS − Full | 169 | -0.0031 [-0.0092, 0.0000] | -0.0059 [-0.0178, 0.0000] |
| sampling not invoked | PPAS − Deterministic | 10331 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| sampling not invoked | PPAS − Random (seed mean) | 10331 | 0.0001 [-0.0000, 0.0002] | 0.0001 [-0.0001, 0.0003] |
| sampling not invoked | Deterministic − Random (seed mean) | 10331 | 0.0001 [-0.0000, 0.0002] | 0.0001 [-0.0001, 0.0003] |
| sampling not invoked | PPAS − Full | 10331 | 0.0001 [0.0000, 0.0002] | 0.0001 [0.0000, 0.0003] |

#### MRR by source degree (eligible items)

| Source degree | Test sources | full | deterministic | ppas | random (seed mean) |
|---|---|---|---|---|---|
| 0 | 170 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 1–5 | 1532 | 0.3270 | 0.3271 | 0.3271 | 0.3268 |
| 6–10 | 6726 | 0.1549 | 0.1549 | 0.1549 | 0.1549 |
| 11–20 | 1983 | 0.1549 | 0.1549 | 0.1549 | 0.1542 |
| 21+ | 89 | 0.1819 | 0.1819 | 0.1819 | 0.1823 |

### Budget 80 tokens

#### Coverage and cost

| Condition | Src sampled | Src mean tok | Src truncated | Cand sampled | Cand mean tok | Cand truncated | Model input tokens | Oversized items | Selection s | New texts encoded | Encode s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PA-full | 0 (0.0%) | 69.7 | 0 | 0 (0.0%) | 170.0 | 75 | 3338159 | 0 | 2.1 | 28381 | 747.0 |
| PA-det-80 | 4124 (27.9%) | 62.1 | 0 | 11190 (81.9%) | 73.7 | 0 | 1923863 | 18 | 19.2 | 1 | 0.1 |
| PA-ppas-80 | 4124 (27.9%) | 62.1 | 0 | 11190 (81.9%) | 73.7 | 0 | 1923867 | 18 | 19.2 | 15314 | 277.8 |
| PA-rand-80-s0 | 4124 (27.9%) | 62.1 | 0 | 11190 (81.9%) | 74.0 | 0 | 1927529 | 18 | 18.8 | 14885 | 269.9 |
| PA-rand-80-s1 | 4124 (27.9%) | 62.1 | 0 | 11190 (81.9%) | 74.1 | 0 | 1928092 | 18 | 18.8 | 14553 | 263.8 |
| PA-rand-80-s2 | 4124 (27.9%) | 62.1 | 0 | 11190 (81.9%) | 74.1 | 0 | 1928569 | 18 | 19.0 | 14302 | 259.4 |
| PA-rand-80-s3 | 4124 (27.9%) | 62.1 | 0 | 11190 (81.9%) | 74.0 | 0 | 1927915 | 18 | 18.8 | 14032 | 255.6 |
| PA-rand-80-s4 | 4124 (27.9%) | 62.1 | 0 | 11190 (81.9%) | 74.0 | 0 | 1928139 | 18 | 18.8 | 13789 | 250.1 |

#### Effectiveness (test population)

| Policy | f1 | precision | recall | threshold | mrr | recall_at_1 | recall_at_5 | recall_at_10 |
|---|---|---|---|---|---|---|---|---|
| Full (uncapped reference) | 0.1515 | 0.1560 | 0.1471 | 0.5800 | 0.1777 | 0.1502 | 0.2085 | 0.2372 |
| PPAS | 0.1180 | 0.1190 | 0.1170 | 0.4800 | 0.1497 | 0.1170 | 0.1842 | 0.2224 |
| Deterministic | 0.1180 | 0.1190 | 0.1170 | 0.4800 | 0.1497 | 0.1170 | 0.1842 | 0.2224 |
| Random mean ± SD (seeds 0,1,2,3,4) | 0.1028 ± 0.0011 | 0.1042 ± 0.0009 | 0.1015 ± 0.0014 | 0.5360 ± 0.0152 | 0.1355 ± 0.0009 | 0.1021 ± 0.0011 | 0.1690 ± 0.0023 | 0.2108 ± 0.0011 |
| Random 95% t-interval | [0.1014, 0.1042] | [0.1030, 0.1054] | [0.0997, 0.1032] | [0.5172, 0.5548] | [0.1343, 0.1367] | [0.1008, 0.1034] | [0.1661, 0.1718] | [0.2095, 0.2122] |
| Random seed 0 | 0.1046 | 0.1057 | 0.1036 | 0.5200 | 0.1370 | 0.1036 | 0.1724 | 0.2122 |
| Random seed 1 | 0.1029 | 0.1046 | 0.1012 | 0.5500 | 0.1357 | 0.1026 | 0.1690 | 0.2115 |
| Random seed 2 | 0.1025 | 0.1038 | 0.1011 | 0.5400 | 0.1352 | 0.1017 | 0.1672 | 0.2094 |
| Random seed 3 | 0.1026 | 0.1036 | 0.1016 | 0.5200 | 0.1352 | 0.1016 | 0.1695 | 0.2102 |
| Random seed 4 | 0.1015 | 0.1033 | 0.0997 | 0.5500 | 0.1344 | 0.1009 | 0.1666 | 0.2108 |

#### Sampled selections: PPAS vs deterministic

| Side | Sampled entities | PPAS ≠ deterministic | Share |
|---|---|---|---|
| source | 4124 | 0 | 0.0% |
| candidate | 11190 | 1 | 0.0% |

#### Paired differences over test sources (95% paired bootstrap)

| Stratum | Contrast | Test sources | Δ MRR [95% CI] | Δ R@1 [95% CI] |
|---|---|---|---|---|
| all test sources | PPAS − Deterministic | 10500 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| all test sources | PPAS − Random (seed mean) | 10500 | 0.0142 [0.0113, 0.0172] | 0.0150 [0.0111, 0.0187] |
| all test sources | Deterministic − Random (seed mean) | 10500 | 0.0142 [0.0112, 0.0171] | 0.0150 [0.0113, 0.0188] |
| all test sources | PPAS − Full | 10500 | -0.0280 [-0.0310, -0.0253] | -0.0331 [-0.0372, -0.0293] |
| sampling invoked | PPAS − Deterministic | 8379 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| sampling invoked | PPAS − Random (seed mean) | 8379 | 0.0179 [0.0143, 0.0216] | 0.0190 [0.0144, 0.0236] |
| sampling invoked | Deterministic − Random (seed mean) | 8379 | 0.0179 [0.0142, 0.0216] | 0.0190 [0.0143, 0.0236] |
| sampling invoked | PPAS − Full | 8379 | -0.0354 [-0.0388, -0.0320] | -0.0419 [-0.0469, -0.0371] |
| sampling not invoked | PPAS − Deterministic | 2121 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| sampling not invoked | PPAS − Random (seed mean) | 2121 | -0.0003 [-0.0017, 0.0013] | -0.0008 [-0.0036, 0.0020] |
| sampling not invoked | Deterministic − Random (seed mean) | 2121 | -0.0003 [-0.0018, 0.0012] | -0.0008 [-0.0035, 0.0020] |
| sampling not invoked | PPAS − Full | 2121 | 0.0011 [-0.0004, 0.0027] | 0.0014 [-0.0014, 0.0042] |

#### MRR by source degree (eligible items)

| Source degree | Test sources | full | deterministic | ppas | random (seed mean) |
|---|---|---|---|---|---|
| 0 | 170 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 1–5 | 1532 | 0.3270 | 0.2794 | 0.2794 | 0.2532 |
| 6–10 | 6726 | 0.1549 | 0.1336 | 0.1336 | 0.1179 |
| 11–20 | 1983 | 0.1549 | 0.1186 | 0.1186 | 0.1166 |
| 21+ | 89 | 0.1819 | 0.1171 | 0.1171 | 0.1186 |

## D1 / d1_snomed_fma

### Policy ablation (#122)

- Model `BAAI/bge-large-en-v1.5` revision `d4aa6901d3a41ba39fb536a557fa166f842b0e09`; tokenizer `BAAI/bge-large-en-v1.5`
- Candidates `data/candidates/D1/d1_snomed_fma_candidates.json` sha256 `a65eb2480a75f9ec…` (identical across runs: True)
- Git commits: ['66aabc3b6c']; dirty: ['False']
- Representation: Structured key-value items (V6 rendering) from _collect_triples; literal datatype suffix stripped; empty and duplicate items dropped; selected items joined in canonical order
- Budget counting: model tokenizer on the final joined text, special tokens included, no truncation
- Deterministic order: lexicographic (predicate URI, object term); PPAS tier config sha256 `34f65847c146167e…`
- Threshold: tuned on the validation population only, grid 0.10-0.99 step 0.01, ties to the lower median; one top-ranked prediction per source
- Test references: 5226; validation references: 1304

### Budget 48 tokens

#### Coverage and cost

| Condition | Src sampled | Src mean tok | Src truncated | Cand sampled | Cand mean tok | Cand truncated | Model input tokens | Oversized items | Selection s | New texts encoded | Encode s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PA-full | 0 (0.0%) | 41.4 | 0 | 0 (0.0%) | 65.3 | 0 | 4273491 | 0 | 4.3 | 74285 | 1075.0 |
| PA-det-48 | 5938 (24.6%) | 36.3 | 0 | 40740 (81.3%) | 42.3 | 0 | 2996744 | 133 | 22.4 | 42770 | 379.3 |
| PA-ppas-48 | 5938 (24.6%) | 36.8 | 0 | 40740 (81.3%) | 42.7 | 0 | 3027595 | 133 | 23.2 | 46678 | 411.2 |
| PA-rand-48-s0 | 5938 (24.6%) | 36.7 | 0 | 40740 (81.3%) | 43.5 | 0 | 3069377 | 133 | 23.4 | 31583 | 278.2 |
| PA-rand-48-s1 | 5938 (24.6%) | 36.7 | 0 | 40740 (81.3%) | 43.6 | 0 | 3070166 | 133 | 23.4 | 27443 | 243.1 |
| PA-rand-48-s2 | 5938 (24.6%) | 36.7 | 0 | 40740 (81.3%) | 43.5 | 0 | 3068871 | 133 | 23.4 | 24174 | 214.5 |
| PA-rand-48-s3 | 5938 (24.6%) | 36.7 | 0 | 40740 (81.3%) | 43.5 | 0 | 3068689 | 133 | 23.5 | 21808 | 192.0 |
| PA-rand-48-s4 | 5938 (24.6%) | 36.7 | 0 | 40740 (81.3%) | 43.5 | 0 | 3067532 | 133 | 23.5 | 19429 | 171.8 |

#### Effectiveness (test population)

| Policy | f1 | precision | recall | threshold | mrr | recall_at_1 | recall_at_5 | recall_at_10 |
|---|---|---|---|---|---|---|---|---|
| Full (uncapped reference) | 0.1432 | 0.1078 | 0.2134 | 0.8100 | 0.4529 | 0.2954 | 0.5364 | 0.6759 |
| PPAS | 0.1450 | 0.0953 | 0.3035 | 0.7700 | 0.4941 | 0.3243 | 0.6150 | 0.7491 |
| Deterministic | 0.1206 | 0.0931 | 0.1709 | 0.8100 | 0.3567 | 0.2277 | 0.3827 | 0.5119 |
| Random mean ± SD (seeds 0,1,2,3,4) | 0.1080 ± 0.0007 | 0.0737 ± 0.0015 | 0.2023 ± 0.0108 | 0.7920 ± 0.0084 | 0.4055 ± 0.0013 | 0.2358 ± 0.0022 | 0.5165 ± 0.0049 | 0.6721 ± 0.0045 |
| Random 95% t-interval | [0.1071, 0.1089] | [0.0719, 0.0756] | [0.1890, 0.2157] | [0.7816, 0.8024] | [0.4039, 0.4070] | [0.2331, 0.2385] | [0.5104, 0.5226] | [0.6665, 0.6776] |
| Random seed 0 | 0.1092 | 0.0740 | 0.2080 | 0.7900 | 0.4055 | 0.2348 | 0.5159 | 0.6743 |
| Random seed 1 | 0.1081 | 0.0752 | 0.1921 | 0.8000 | 0.4072 | 0.2380 | 0.5144 | 0.6755 |
| Random seed 2 | 0.1072 | 0.0727 | 0.2044 | 0.7900 | 0.4039 | 0.2327 | 0.5184 | 0.6747 |
| Random seed 3 | 0.1076 | 0.0749 | 0.1910 | 0.8000 | 0.4060 | 0.2359 | 0.5235 | 0.6646 |
| Random seed 4 | 0.1078 | 0.0718 | 0.2162 | 0.7800 | 0.4048 | 0.2377 | 0.5103 | 0.6713 |

#### Sampled selections: PPAS vs deterministic

| Side | Sampled entities | PPAS ≠ deterministic | Share |
|---|---|---|---|
| source | 5938 | 4785 | 80.6% |
| candidate | 40740 | 37985 | 93.2% |

#### Paired differences over test sources (95% paired bootstrap)

| Stratum | Contrast | Test sources | Δ MRR [95% CI] | Δ R@1 [95% CI] |
|---|---|---|---|---|
| all test sources | PPAS − Deterministic | 4605 | 0.1374 [0.1265, 0.1476] | 0.1097 [0.0966, 0.1233] |
| all test sources | PPAS − Random (seed mean) | 4605 | 0.0886 [0.0810, 0.0965] | 0.1005 [0.0902, 0.1117] |
| all test sources | Deterministic − Random (seed mean) | 4605 | -0.0487 [-0.0563, -0.0412] | -0.0092 [-0.0188, 0.0007] |
| all test sources | PPAS − Full | 4605 | 0.0412 [0.0323, 0.0500] | 0.0328 [0.0206, 0.0450] |
| sampling invoked | PPAS − Deterministic | 3917 | 0.1749 [0.1635, 0.1868] | 0.1460 [0.1310, 0.1611] |
| sampling invoked | PPAS − Random (seed mean) | 3917 | 0.1030 [0.0937, 0.1118] | 0.1124 [0.1009, 0.1249] |
| sampling invoked | Deterministic − Random (seed mean) | 3917 | -0.0719 [-0.0802, -0.0636] | -0.0336 [-0.0437, -0.0232] |
| sampling invoked | PPAS − Full | 3917 | 0.0598 [0.0498, 0.0703] | 0.0531 [0.0393, 0.0666] |
| sampling not invoked | PPAS − Deterministic | 688 | -0.0760 [-0.0902, -0.0623] | -0.0974 [-0.1207, -0.0756] |
| sampling not invoked | PPAS − Random (seed mean) | 688 | 0.0068 [-0.0047, 0.0179] | 0.0326 [0.0137, 0.0520] |
| sampling not invoked | Deterministic − Random (seed mean) | 688 | 0.0828 [0.0712, 0.0945] | 0.1299 [0.1096, 0.1509] |
| sampling not invoked | PPAS − Full | 688 | -0.0648 [-0.0786, -0.0512] | -0.0828 [-0.1047, -0.0625] |

#### MRR by source degree (eligible items)

| Source degree | Test sources | full | deterministic | ppas | random (seed mean) |
|---|---|---|---|---|---|
| 1–5 | 3886 | 0.4587 | 0.3690 | 0.4975 | 0.4104 |
| 6–10 | 694 | 0.4214 | 0.2943 | 0.4774 | 0.3819 |
| 11–20 | 25 | 0.4333 | 0.1837 | 0.4267 | 0.2963 |
