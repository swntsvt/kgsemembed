# #183 re-run against the stored results

New results from commit(s) f08addf. A cell *expected to change* is touched by a #183 correction; every other cell must reproduce every metric and count exactly. Budgeted results must report no text over the budget and none truncated; C19 must equal C10.

**Checks: all passed.**

| Cond. | Column | Pair | Expected change | Result | MRR stored | MRR new | Δ | Budget |
|---|---|---|---|---|---:|---:|---:|---:|
| B0 | D2 | d2_anatomy | — | identical | 0.8175 | 0.8175 | +0.0000 | — |
| C1 | D2 | d2_anatomy | — | identical | 0.8505 | 0.8505 | +0.0000 | — |
| C10 | D1 | d1_snomed_fma | — | identical | 0.6492 | 0.6492 | +0.0000 | — |
| C10 | D2 | d2_anatomy | — | identical | 0.6978 | 0.6978 | +0.0000 | — |
| C11 | D2 | d2_anatomy | M4 512 | changed (4 values) | 0.7719 | 0.7731 | +0.0012 | — |
| C13 | D2 | d2_anatomy | — | identical | 0.7717 | 0.7717 | +0.0000 | — |
| C15 | D2 | d2_anatomy | — | identical | 0.7252 | 0.7252 | +0.0000 | — |
| C16 | D2 | d2_anatomy | budget, M4 512 | changed (13 values) | 0.7933 | 0.7829 | -0.0104 | 420 |
| C17 | D2 | d2_anatomy | M4 512 | identical | 0.8605 | 0.8605 | +0.0000 | — |
| C18 | D2 | d2_anatomy | — | identical | 0.6877 | 0.6877 | +0.0000 | — |
| C19 | D1 | d1_snomed_fma | — | identical | 0.6492 | 0.6492 | +0.0000 | — |
| C2 | D2 | d2_anatomy | — | identical | 0.8179 | 0.8179 | +0.0000 | — |
| C20 | D2 | d2_anatomy | — | identical | 0.7723 | 0.7723 | +0.0000 | — |
| C3 | D2 | d2_anatomy | budget | changed (3 values) | 0.8490 | 0.8497 | +0.0007 | 420 |
| C4 | D2 | d2_anatomy | budget | identical | 0.9021 | 0.9021 | +0.0000 | 420 |
| C6 | D2 | d2_anatomy | budget, M4 512 | changed (14 values) | 0.8191 | 0.8024 | -0.0167 | 420 |
| C7 | D2 | d2_anatomy | budget | changed (5 values) | 0.8495 | 0.8499 | +0.0004 | 420 |
| C9 | D2 | d2_anatomy | — | identical | 0.2511 | 0.2511 | +0.0000 | — |
