# #183 re-run against the stored results

New results from commit(s) 3b23a8b. A cell *expected to change* is touched by a #183 correction; every other cell must reproduce every metric and count exactly. Budgeted results must report no text over the budget and none truncated; C19 must equal C10.

**Checks: all passed.**

| Cond. | Column | Pair | Expected change | Result | MRR stored | MRR new | Δ | Budget |
|---|---|---|---|---|---:|---:|---:|---:|
| B0 | D1 | d1_snomed_fma | — | identical | 0.7055 | 0.7055 | +0.0000 | — |
| B0 | D2 | d2_anatomy | — | identical | 0.8175 | 0.8175 | +0.0000 | — |
| B0 | D3 | cmt-conference | — | identical | 0.8083 | 0.8083 | +0.0000 | — |
| B0 | D3 | cmt-confof | — | identical | 0.5821 | 0.5821 | +0.0000 | — |
| B0 | D3 | cmt-edas | — | identical | 0.5766 | 0.5766 | +0.0000 | — |
| B0 | D3 | cmt-ekaw | — | identical | 0.4741 | 0.4741 | +0.0000 | — |
| B0 | D3 | cmt-iasted | — | identical | 1.0000 | 1.0000 | +0.0000 | — |
| B0 | D3 | cmt-sigkdd | — | identical | 0.9143 | 0.9143 | +0.0000 | — |
| B0 | D3 | conference-confof | — | identical | 0.8958 | 0.8958 | +0.0000 | — |
| B0 | D3 | conference-edas | — | identical | 0.7619 | 0.7619 | +0.0000 | — |
| B0 | D3 | conference-ekaw | — | identical | 0.7167 | 0.7167 | +0.0000 | — |
| B0 | D3 | conference-iasted | — | identical | 0.3652 | 0.3652 | +0.0000 | — |
| B0 | D3 | conference-sigkdd | — | identical | 0.8036 | 0.8036 | +0.0000 | — |
| B0 | D3 | confof-edas | — | identical | 0.6278 | 0.6278 | +0.0000 | — |
| B0 | D3 | confof-ekaw | — | identical | 0.8021 | 0.8021 | +0.0000 | — |
| B0 | D3 | confof-iasted | — | identical | 0.6250 | 0.6250 | +0.0000 | — |
| B0 | D3 | confof-sigkdd | — | identical | 0.9167 | 0.9167 | +0.0000 | — |
| B0 | D3 | edas-ekaw | — | identical | 0.5614 | 0.5614 | +0.0000 | — |
| B0 | D3 | edas-iasted | — | identical | 0.5420 | 0.5420 | +0.0000 | — |
| B0 | D3 | edas-sigkdd | — | identical | 0.6369 | 0.6369 | +0.0000 | — |
| B0 | D3 | ekaw-iasted | — | identical | 0.7604 | 0.7604 | +0.0000 | — |
| B0 | D3 | ekaw-sigkdd | — | identical | 0.7500 | 0.7500 | +0.0000 | — |
| B0 | D3 | iasted-sigkdd | — | identical | 1.0000 | 1.0000 | +0.0000 | — |
| B0 | D4_instance | memoryalpha-stexpanded-instance | — | identical | 0.9049 | 0.9049 | +0.0000 | — |
| B0 | D4_schema | memoryalpha-stexpanded-schema | — | identical | 0.8879 | 0.8879 | +0.0000 | — |
| B0 | D5 | D_W_15K_V2 | — | identical | 0.1808 | 0.1808 | +0.0000 | — |
| C1 | D1 | d1_snomed_fma | — | identical | 0.7186 | 0.7186 | +0.0000 | — |
| C1 | D2 | d2_anatomy | — | identical | 0.8505 | 0.8505 | +0.0000 | — |
| C1 | D3 | cmt-conference | — | identical | 0.7639 | 0.7639 | +0.0000 | — |
| C1 | D3 | cmt-confof | — | identical | 0.5513 | 0.5513 | +0.0000 | — |
| C1 | D3 | cmt-edas | — | identical | 0.6667 | 0.6667 | +0.0000 | — |
| C1 | D3 | cmt-ekaw | — | identical | 0.6667 | 0.6667 | +0.0000 | — |
| C1 | D3 | cmt-iasted | — | identical | 1.0000 | 1.0000 | +0.0000 | — |
| C1 | D3 | cmt-sigkdd | — | identical | 0.9333 | 0.9333 | +0.0000 | — |
| C1 | D3 | conference-confof | — | identical | 0.8542 | 0.8542 | +0.0000 | — |
| C1 | D3 | conference-edas | — | identical | 0.8095 | 0.8095 | +0.0000 | — |
| C1 | D3 | conference-ekaw | — | identical | 0.8167 | 0.8167 | +0.0000 | — |
| C1 | D3 | conference-iasted | — | identical | 0.5000 | 0.5000 | +0.0000 | — |
| C1 | D3 | conference-sigkdd | — | identical | 0.9375 | 0.9375 | +0.0000 | — |
| C1 | D3 | confof-edas | — | identical | 0.6756 | 0.6756 | +0.0000 | — |
| C1 | D3 | confof-ekaw | — | identical | 0.8542 | 0.8542 | +0.0000 | — |
| C1 | D3 | confof-iasted | — | identical | 0.6250 | 0.6250 | +0.0000 | — |
| C1 | D3 | confof-sigkdd | — | identical | 0.8889 | 0.8889 | +0.0000 | — |
| C1 | D3 | edas-ekaw | — | identical | 0.6847 | 0.6847 | +0.0000 | — |
| C1 | D3 | edas-iasted | — | identical | 0.6062 | 0.6062 | +0.0000 | — |
| C1 | D3 | edas-sigkdd | — | identical | 0.5486 | 0.5486 | +0.0000 | — |
| C1 | D3 | ekaw-iasted | — | identical | 0.8750 | 0.8750 | +0.0000 | — |
| C1 | D3 | ekaw-sigkdd | — | identical | 0.8148 | 0.8148 | +0.0000 | — |
| C1 | D3 | iasted-sigkdd | — | identical | 0.9583 | 0.9583 | +0.0000 | — |
| C1 | D4_instance | memoryalpha-stexpanded-instance | — | identical | 0.9093 | 0.9093 | +0.0000 | — |
| C1 | D4_schema | memoryalpha-stexpanded-schema | — | identical | 0.9034 | 0.9034 | +0.0000 | — |
| C1 | D5 | D_W_15K_V2 | — | identical | 0.0420 | 0.0420 | +0.0000 | — |
| C10 | D1 | d1_snomed_fma | — | identical | 0.6492 | 0.6492 | +0.0000 | — |
| C10 | D2 | d2_anatomy | — | identical | 0.6978 | 0.6978 | +0.0000 | — |
| C10 | D3 | cmt-conference | — | identical | 0.4803 | 0.4803 | +0.0000 | — |
| C10 | D3 | cmt-confof | — | identical | 0.5231 | 0.5231 | +0.0000 | — |
| C10 | D3 | cmt-edas | — | identical | 0.6818 | 0.6818 | +0.0000 | — |
| C10 | D3 | cmt-ekaw | — | identical | 0.4964 | 0.4964 | +0.0000 | — |
| C10 | D3 | cmt-iasted | — | identical | 1.0000 | 1.0000 | +0.0000 | — |
| C10 | D3 | cmt-sigkdd | — | identical | 0.8750 | 0.8750 | +0.0000 | — |
| C10 | D3 | conference-confof | — | identical | 0.6375 | 0.6375 | +0.0000 | — |
| C10 | D3 | conference-edas | — | identical | 0.6339 | 0.6339 | +0.0000 | — |
| C10 | D3 | conference-ekaw | — | identical | 0.5847 | 0.5847 | +0.0000 | — |
| C10 | D3 | conference-iasted | — | identical | 0.3685 | 0.3685 | +0.0000 | — |
| C10 | D3 | conference-sigkdd | — | identical | 0.6417 | 0.6417 | +0.0000 | — |
| C10 | D3 | confof-edas | — | identical | 0.5829 | 0.5829 | +0.0000 | — |
| C10 | D3 | confof-ekaw | — | identical | 0.6101 | 0.6101 | +0.0000 | — |
| C10 | D3 | confof-iasted | — | identical | 0.5179 | 0.5179 | +0.0000 | — |
| C10 | D3 | confof-sigkdd | — | identical | 0.8889 | 0.8889 | +0.0000 | — |
| C10 | D3 | edas-ekaw | — | identical | 0.6440 | 0.6440 | +0.0000 | — |
| C10 | D3 | edas-iasted | — | identical | 0.5307 | 0.5307 | +0.0000 | — |
| C10 | D3 | edas-sigkdd | — | identical | 0.8333 | 0.8333 | +0.0000 | — |
| C10 | D3 | ekaw-iasted | — | identical | 0.6792 | 0.6792 | +0.0000 | — |
| C10 | D3 | ekaw-sigkdd | — | identical | 0.5540 | 0.5540 | +0.0000 | — |
| C10 | D3 | iasted-sigkdd | — | identical | 0.8056 | 0.8056 | +0.0000 | — |
| C10 | D4_instance | memoryalpha-stexpanded-instance | — | identical | 0.9174 | 0.9174 | +0.0000 | — |
| C10 | D4_schema | memoryalpha-stexpanded-schema | — | identical | 0.9205 | 0.9205 | +0.0000 | — |
| C10 | D5 | D_W_15K_V2 | — | identical | 0.0431 | 0.0431 | +0.0000 | — |
| C11 | D1 | d1_snomed_fma | M4 512 | changed (1 values) | 0.6872 | 0.6872 | +0.0000 | — |
| C11 | D2 | d2_anatomy | M4 512 | changed (4 values) | 0.7719 | 0.7731 | +0.0012 | — |
| C13 | D1 | d1_snomed_fma | — | identical | 0.7132 | 0.7132 | +0.0000 | — |
| C13 | D2 | d2_anatomy | — | identical | 0.7717 | 0.7717 | +0.0000 | — |
| C13 | D3 | cmt-conference | — | identical | 0.6258 | 0.6258 | +0.0000 | — |
| C13 | D3 | cmt-confof | — | identical | 0.5718 | 0.5718 | +0.0000 | — |
| C13 | D3 | cmt-edas | — | identical | 0.6818 | 0.6818 | +0.0000 | — |
| C13 | D3 | cmt-ekaw | — | identical | 0.5407 | 0.5407 | +0.0000 | — |
| C13 | D3 | cmt-iasted | — | identical | 0.8333 | 0.8333 | +0.0000 | — |
| C13 | D3 | cmt-sigkdd | — | identical | 1.0000 | 1.0000 | +0.0000 | — |
| C13 | D3 | conference-confof | — | identical | 1.0000 | 1.0000 | +0.0000 | — |
| C13 | D3 | conference-edas | — | identical | 0.6488 | 0.6488 | +0.0000 | — |
| C13 | D3 | conference-ekaw | — | identical | 0.6770 | 0.6770 | +0.0000 | — |
| C13 | D3 | conference-iasted | — | identical | 0.3639 | 0.3639 | +0.0000 | — |
| C13 | D3 | conference-sigkdd | — | identical | 0.9583 | 0.9583 | +0.0000 | — |
| C13 | D3 | confof-edas | — | identical | 0.7396 | 0.7396 | +0.0000 | — |
| C13 | D3 | confof-ekaw | — | identical | 0.6944 | 0.6944 | +0.0000 | — |
| C13 | D3 | confof-iasted | — | identical | 0.6250 | 0.6250 | +0.0000 | — |
| C13 | D3 | confof-sigkdd | — | identical | 0.8056 | 0.8056 | +0.0000 | — |
| C13 | D3 | edas-ekaw | — | identical | 0.6071 | 0.6071 | +0.0000 | — |
| C13 | D3 | edas-iasted | — | identical | 0.6146 | 0.6146 | +0.0000 | — |
| C13 | D3 | edas-sigkdd | — | identical | 0.8333 | 0.8333 | +0.0000 | — |
| C13 | D3 | ekaw-iasted | — | identical | 0.7054 | 0.7054 | +0.0000 | — |
| C13 | D3 | ekaw-sigkdd | — | identical | 0.6825 | 0.6825 | +0.0000 | — |
| C13 | D3 | iasted-sigkdd | — | identical | 0.8958 | 0.8958 | +0.0000 | — |
| C13 | D4_instance | memoryalpha-stexpanded-instance | — | identical | 0.9194 | 0.9194 | +0.0000 | — |
| C13 | D4_schema | memoryalpha-stexpanded-schema | — | identical | 0.9091 | 0.9091 | +0.0000 | — |
| C13 | D5 | D_W_15K_V2 | — | identical | 0.0485 | 0.0485 | +0.0000 | — |
| C14 | D5 | D_W_15K_V2 | D5 suffix | changed (14 values) | 0.1550 | 0.1717 | +0.0167 | — |
| C15 | D1 | d1_snomed_fma | — | identical | 0.6947 | 0.6947 | +0.0000 | — |
| C15 | D2 | d2_anatomy | — | identical | 0.7252 | 0.7252 | +0.0000 | — |
| C16 | D1 | d1_snomed_fma | budget, M4 512 | changed (14 values) | 0.7277 | 0.7237 | -0.0040 | 420 |
| C16 | D2 | d2_anatomy | budget, M4 512 | changed (13 values) | 0.7933 | 0.7829 | -0.0104 | 420 |
| C17 | D1 | d1_snomed_fma | M4 512 | identical | 0.7578 | 0.7578 | +0.0000 | — |
| C17 | D2 | d2_anatomy | M4 512 | identical | 0.8605 | 0.8605 | +0.0000 | — |
| C18 | D1 | d1_snomed_fma | — | identical | 0.6477 | 0.6477 | +0.0000 | — |
| C18 | D2 | d2_anatomy | — | identical | 0.6877 | 0.6877 | +0.0000 | — |
| C18 | D3 | cmt-conference | — | identical | 0.5084 | 0.5084 | +0.0000 | — |
| C18 | D3 | cmt-confof | — | identical | 0.3969 | 0.3969 | +0.0000 | — |
| C18 | D3 | cmt-edas | — | identical | 0.6818 | 0.6818 | +0.0000 | — |
| C18 | D3 | cmt-ekaw | — | identical | 0.5648 | 0.5648 | +0.0000 | — |
| C18 | D3 | cmt-iasted | — | identical | 1.0000 | 1.0000 | +0.0000 | — |
| C18 | D3 | cmt-sigkdd | — | identical | 0.9500 | 0.9500 | +0.0000 | — |
| C18 | D3 | conference-confof | — | identical | 0.5207 | 0.5207 | +0.0000 | — |
| C18 | D3 | conference-edas | — | identical | 0.6548 | 0.6548 | +0.0000 | — |
| C18 | D3 | conference-ekaw | — | identical | 0.6992 | 0.6992 | +0.0000 | — |
| C18 | D3 | conference-iasted | — | identical | 0.4104 | 0.4104 | +0.0000 | — |
| C18 | D3 | conference-sigkdd | — | identical | 0.9167 | 0.9167 | +0.0000 | — |
| C18 | D3 | confof-edas | — | identical | 0.6359 | 0.6359 | +0.0000 | — |
| C18 | D3 | confof-ekaw | — | identical | 0.5871 | 0.5871 | +0.0000 | — |
| C18 | D3 | confof-iasted | — | identical | 0.5208 | 0.5208 | +0.0000 | — |
| C18 | D3 | confof-sigkdd | — | identical | 0.8333 | 0.8333 | +0.0000 | — |
| C18 | D3 | edas-ekaw | — | identical | 0.6395 | 0.6395 | +0.0000 | — |
| C18 | D3 | edas-iasted | — | identical | 0.5299 | 0.5299 | +0.0000 | — |
| C18 | D3 | edas-sigkdd | — | identical | 0.7361 | 0.7361 | +0.0000 | — |
| C18 | D3 | ekaw-iasted | — | identical | 0.7083 | 0.7083 | +0.0000 | — |
| C18 | D3 | ekaw-sigkdd | — | identical | 0.6667 | 0.6667 | +0.0000 | — |
| C18 | D3 | iasted-sigkdd | — | identical | 0.8611 | 0.8611 | +0.0000 | — |
| C18 | D4_instance | memoryalpha-stexpanded-instance | — | identical | 0.8960 | 0.8960 | +0.0000 | — |
| C18 | D4_schema | memoryalpha-stexpanded-schema | — | identical | 0.9091 | 0.9091 | +0.0000 | — |
| C18 | D5 | D_W_15K_V2 | — | identical | 0.0338 | 0.0338 | +0.0000 | — |
| C19 | D1 | d1_snomed_fma | — | identical | 0.6492 | 0.6492 | +0.0000 | — |
| C2 | D1 | d1_snomed_fma | — | identical | 0.6838 | 0.6838 | +0.0000 | — |
| C2 | D2 | d2_anatomy | — | identical | 0.8179 | 0.8179 | +0.0000 | — |
| C2 | D3 | cmt-conference | — | identical | 0.7897 | 0.7897 | +0.0000 | — |
| C2 | D3 | cmt-confof | — | identical | 0.5432 | 0.5432 | +0.0000 | — |
| C2 | D3 | cmt-edas | — | identical | 0.6010 | 0.6010 | +0.0000 | — |
| C2 | D3 | cmt-ekaw | — | identical | 0.5714 | 0.5714 | +0.0000 | — |
| C2 | D3 | cmt-iasted | — | identical | 1.0000 | 1.0000 | +0.0000 | — |
| C2 | D3 | cmt-sigkdd | — | identical | 0.9333 | 0.9333 | +0.0000 | — |
| C2 | D3 | conference-confof | — | identical | 0.7694 | 0.7694 | +0.0000 | — |
| C2 | D3 | conference-edas | — | identical | 0.6988 | 0.6988 | +0.0000 | — |
| C2 | D3 | conference-ekaw | — | identical | 0.7005 | 0.7005 | +0.0000 | — |
| C2 | D3 | conference-iasted | — | identical | 0.5417 | 0.5417 | +0.0000 | — |
| C2 | D3 | conference-sigkdd | — | identical | 0.9375 | 0.9375 | +0.0000 | — |
| C2 | D3 | confof-edas | — | identical | 0.6641 | 0.6641 | +0.0000 | — |
| C2 | D3 | confof-ekaw | — | identical | 0.7693 | 0.7693 | +0.0000 | — |
| C2 | D3 | confof-iasted | — | identical | 0.6250 | 0.6250 | +0.0000 | — |
| C2 | D3 | confof-sigkdd | — | identical | 0.9167 | 0.9167 | +0.0000 | — |
| C2 | D3 | edas-ekaw | — | identical | 0.6857 | 0.6857 | +0.0000 | — |
| C2 | D3 | edas-iasted | — | identical | 0.6375 | 0.6375 | +0.0000 | — |
| C2 | D3 | edas-sigkdd | — | identical | 0.6424 | 0.6424 | +0.0000 | — |
| C2 | D3 | ekaw-iasted | — | identical | 0.8750 | 0.8750 | +0.0000 | — |
| C2 | D3 | ekaw-sigkdd | — | identical | 0.7778 | 0.7778 | +0.0000 | — |
| C2 | D3 | iasted-sigkdd | — | identical | 0.9028 | 0.9028 | +0.0000 | — |
| C2 | D4_instance | memoryalpha-stexpanded-instance | — | identical | 0.9111 | 0.9111 | +0.0000 | — |
| C2 | D4_schema | memoryalpha-stexpanded-schema | — | identical | 0.8977 | 0.8977 | +0.0000 | — |
| C2 | D5 | D_W_15K_V2 | — | identical | 0.0420 | 0.0420 | +0.0000 | — |
| C20 | D1 | d1_snomed_fma | — | identical | 0.6211 | 0.6211 | +0.0000 | — |
| C20 | D2 | d2_anatomy | — | identical | 0.7723 | 0.7723 | +0.0000 | — |
| C20 | D3 | cmt-conference | — | identical | 0.7280 | 0.7280 | +0.0000 | — |
| C20 | D3 | cmt-confof | — | identical | 0.4484 | 0.4484 | +0.0000 | — |
| C20 | D3 | cmt-edas | — | identical | 0.6667 | 0.6667 | +0.0000 | — |
| C20 | D3 | cmt-ekaw | — | identical | 0.4788 | 0.4788 | +0.0000 | — |
| C20 | D3 | cmt-iasted | — | identical | 1.0000 | 1.0000 | +0.0000 | — |
| C20 | D3 | cmt-sigkdd | — | identical | 0.9250 | 0.9250 | +0.0000 | — |
| C20 | D3 | conference-confof | — | identical | 0.8056 | 0.8056 | +0.0000 | — |
| C20 | D3 | conference-edas | — | identical | 0.7500 | 0.7500 | +0.0000 | — |
| C20 | D3 | conference-ekaw | — | identical | 0.6950 | 0.6950 | +0.0000 | — |
| C20 | D3 | conference-iasted | — | identical | 0.5000 | 0.5000 | +0.0000 | — |
| C20 | D3 | conference-sigkdd | — | identical | 0.9444 | 0.9444 | +0.0000 | — |
| C20 | D3 | confof-edas | — | identical | 0.7016 | 0.7016 | +0.0000 | — |
| C20 | D3 | confof-ekaw | — | identical | 0.7991 | 0.7991 | +0.0000 | — |
| C20 | D3 | confof-iasted | — | identical | 0.6250 | 0.6250 | +0.0000 | — |
| C20 | D3 | confof-sigkdd | — | identical | 1.0000 | 1.0000 | +0.0000 | — |
| C20 | D3 | edas-ekaw | — | identical | 0.7518 | 0.7518 | +0.0000 | — |
| C20 | D3 | edas-iasted | — | identical | 0.6062 | 0.6062 | +0.0000 | — |
| C20 | D3 | edas-sigkdd | — | identical | 0.7153 | 0.7153 | +0.0000 | — |
| C20 | D3 | ekaw-iasted | — | identical | 0.8750 | 0.8750 | +0.0000 | — |
| C20 | D3 | ekaw-sigkdd | — | identical | 0.7778 | 0.7778 | +0.0000 | — |
| C20 | D3 | iasted-sigkdd | — | identical | 0.8819 | 0.8819 | +0.0000 | — |
| C20 | D4_instance | memoryalpha-stexpanded-instance | — | identical | 0.9110 | 0.9110 | +0.0000 | — |
| C20 | D4_schema | memoryalpha-stexpanded-schema | — | identical | 0.8939 | 0.8939 | +0.0000 | — |
| C20 | D5 | D_W_15K_V2 | — | identical | 0.0417 | 0.0417 | +0.0000 | — |
| C3 | D1 | d1_snomed_fma | budget | identical | 0.7676 | 0.7676 | +0.0000 | 420 |
| C3 | D2 | d2_anatomy | budget | changed (3 values) | 0.8490 | 0.8497 | +0.0007 | 420 |
| C3 | D3 | cmt-conference | budget | identical | 0.6759 | 0.6759 | +0.0000 | 420 |
| C3 | D3 | cmt-confof | budget | identical | 0.5689 | 0.5689 | +0.0000 | 420 |
| C3 | D3 | cmt-edas | budget | identical | 0.6212 | 0.6212 | +0.0000 | 420 |
| C3 | D3 | cmt-ekaw | budget | identical | 0.5437 | 0.5437 | +0.0000 | 420 |
| C3 | D3 | cmt-iasted | budget | identical | 1.0000 | 1.0000 | +0.0000 | 420 |
| C3 | D3 | cmt-sigkdd | budget | identical | 0.8750 | 0.8750 | +0.0000 | 420 |
| C3 | D3 | conference-confof | budget | identical | 0.7292 | 0.7292 | +0.0000 | 420 |
| C3 | D3 | conference-edas | budget | identical | 0.6857 | 0.6857 | +0.0000 | 420 |
| C3 | D3 | conference-ekaw | budget | identical | 0.6064 | 0.6064 | +0.0000 | 420 |
| C3 | D3 | conference-iasted | budget | identical | 0.4563 | 0.4563 | +0.0000 | 420 |
| C3 | D3 | conference-sigkdd | budget | identical | 0.9167 | 0.9167 | +0.0000 | 420 |
| C3 | D3 | confof-edas | budget | identical | 0.7937 | 0.7937 | +0.0000 | 420 |
| C3 | D3 | confof-ekaw | budget | identical | 0.8516 | 0.8516 | +0.0000 | 420 |
| C3 | D3 | confof-iasted | budget | identical | 0.6250 | 0.6250 | +0.0000 | 420 |
| C3 | D3 | confof-sigkdd | budget | identical | 0.9167 | 0.9167 | +0.0000 | 420 |
| C3 | D3 | edas-ekaw | budget | identical | 0.6355 | 0.6355 | +0.0000 | 420 |
| C3 | D3 | edas-iasted | budget | identical | 0.5437 | 0.5437 | +0.0000 | 420 |
| C3 | D3 | edas-sigkdd | budget | identical | 0.6647 | 0.6647 | +0.0000 | 420 |
| C3 | D3 | ekaw-iasted | budget | identical | 0.8750 | 0.8750 | +0.0000 | 420 |
| C3 | D3 | ekaw-sigkdd | budget | identical | 0.6852 | 0.6852 | +0.0000 | 420 |
| C3 | D3 | iasted-sigkdd | budget | identical | 0.9375 | 0.9375 | +0.0000 | 420 |
| C3 | D4_instance | memoryalpha-stexpanded-instance | budget | changed (11 values) | 0.9443 | 0.9386 | -0.0057 | 420 |
| C3 | D4_schema | memoryalpha-stexpanded-schema | budget | identical | 0.9205 | 0.9205 | +0.0000 | 420 |
| C3 | D5 | D_W_15K_V2 | budget, D5 suffix | changed (13 values) | 0.1356 | 0.1616 | +0.0261 | 420 |
| C4 | D1 | d1_snomed_fma | budget | changed (5 values) | 0.7792 | 0.7792 | +0.0000 | 420 |
| C4 | D2 | d2_anatomy | budget | identical | 0.9021 | 0.9021 | +0.0000 | 420 |
| C4 | D3 | cmt-conference | budget | identical | 0.6187 | 0.6187 | +0.0000 | 420 |
| C4 | D3 | cmt-confof | budget | identical | 0.6603 | 0.6603 | +0.0000 | 420 |
| C4 | D3 | cmt-edas | budget | identical | 0.6818 | 0.6818 | +0.0000 | 420 |
| C4 | D3 | cmt-ekaw | budget | identical | 0.6889 | 0.6889 | +0.0000 | 420 |
| C4 | D3 | cmt-iasted | budget | identical | 0.6944 | 0.6944 | +0.0000 | 420 |
| C4 | D3 | cmt-sigkdd | budget | identical | 0.9500 | 0.9500 | +0.0000 | 420 |
| C4 | D3 | conference-confof | budget | identical | 0.8333 | 0.8333 | +0.0000 | 420 |
| C4 | D3 | conference-edas | budget | identical | 0.6159 | 0.6159 | +0.0000 | 420 |
| C4 | D3 | conference-ekaw | budget | identical | 0.7267 | 0.7267 | +0.0000 | 420 |
| C4 | D3 | conference-iasted | budget | identical | 0.4583 | 0.4583 | +0.0000 | 420 |
| C4 | D3 | conference-sigkdd | budget | identical | 0.8750 | 0.8750 | +0.0000 | 420 |
| C4 | D3 | confof-edas | budget | identical | 0.7865 | 0.7865 | +0.0000 | 420 |
| C4 | D3 | confof-ekaw | budget | identical | 0.9062 | 0.9062 | +0.0000 | 420 |
| C4 | D3 | confof-iasted | budget | identical | 0.5096 | 0.5096 | +0.0000 | 420 |
| C4 | D3 | confof-sigkdd | budget | identical | 0.9167 | 0.9167 | +0.0000 | 420 |
| C4 | D3 | edas-ekaw | budget | identical | 0.7333 | 0.7333 | +0.0000 | 420 |
| C4 | D3 | edas-iasted | budget | identical | 0.5286 | 0.5286 | +0.0000 | 420 |
| C4 | D3 | edas-sigkdd | budget | identical | 0.6944 | 0.6944 | +0.0000 | 420 |
| C4 | D3 | ekaw-iasted | budget | identical | 0.7596 | 0.7596 | +0.0000 | 420 |
| C4 | D3 | ekaw-sigkdd | budget | identical | 0.7037 | 0.7037 | +0.0000 | 420 |
| C4 | D3 | iasted-sigkdd | budget | identical | 0.9583 | 0.9583 | +0.0000 | 420 |
| C4 | D4_instance | memoryalpha-stexpanded-instance | budget | changed (12 values) | 0.9455 | 0.9446 | -0.0009 | 420 |
| C4 | D4_schema | memoryalpha-stexpanded-schema | budget | identical | 0.9091 | 0.9091 | +0.0000 | 420 |
| C4 | D5 | D_W_15K_V2 | budget, D5 suffix | changed (14 values) | 0.1147 | 0.1501 | +0.0354 | 420 |
| C5 | D4_instance | memoryalpha-stexpanded-instance | budget | changed (14 values) | 0.9251 | 0.9177 | -0.0074 | 420 |
| C5 | D4_schema | memoryalpha-stexpanded-schema | budget | identical | 0.9091 | 0.9091 | +0.0000 | 420 |
| C5 | D5 | D_W_15K_V2 | budget, D5 suffix | changed (14 values) | 0.1507 | 0.1777 | +0.0270 | 420 |
| C6 | D1 | d1_snomed_fma | budget, M4 512 | changed (13 values) | 0.7548 | 0.7650 | +0.0101 | 420 |
| C6 | D2 | d2_anatomy | budget, M4 512 | changed (14 values) | 0.8191 | 0.8024 | -0.0167 | 420 |
| C7 | D1 | d1_snomed_fma | budget | changed (9 values) | 0.7103 | 0.7109 | +0.0006 | 420 |
| C7 | D2 | d2_anatomy | budget | changed (5 values) | 0.8495 | 0.8499 | +0.0004 | 420 |
| C7 | D3 | cmt-conference | budget | identical | 0.8452 | 0.8452 | +0.0000 | 420 |
| C7 | D3 | cmt-confof | budget | identical | 0.6608 | 0.6608 | +0.0000 | 420 |
| C7 | D3 | cmt-edas | budget | identical | 0.6364 | 0.6364 | +0.0000 | 420 |
| C7 | D3 | cmt-ekaw | budget | identical | 0.6481 | 0.6481 | +0.0000 | 420 |
| C7 | D3 | cmt-iasted | budget | identical | 1.0000 | 1.0000 | +0.0000 | 420 |
| C7 | D3 | cmt-sigkdd | budget | identical | 0.8750 | 0.8750 | +0.0000 | 420 |
| C7 | D3 | conference-confof | budget | identical | 0.8611 | 0.8611 | +0.0000 | 420 |
| C7 | D3 | conference-edas | budget | identical | 0.6858 | 0.6858 | +0.0000 | 420 |
| C7 | D3 | conference-ekaw | budget | identical | 0.6884 | 0.6884 | +0.0000 | 420 |
| C7 | D3 | conference-iasted | budget | identical | 0.5278 | 0.5278 | +0.0000 | 420 |
| C7 | D3 | conference-sigkdd | budget | identical | 0.8750 | 0.8750 | +0.0000 | 420 |
| C7 | D3 | confof-edas | budget | identical | 0.7812 | 0.7812 | +0.0000 | 420 |
| C7 | D3 | confof-ekaw | budget | identical | 0.8490 | 0.8490 | +0.0000 | 420 |
| C7 | D3 | confof-iasted | budget | identical | 0.6250 | 0.6250 | +0.0000 | 420 |
| C7 | D3 | confof-sigkdd | budget | identical | 0.7917 | 0.7917 | +0.0000 | 420 |
| C7 | D3 | edas-ekaw | budget | identical | 0.7330 | 0.7330 | +0.0000 | 420 |
| C7 | D3 | edas-iasted | budget | identical | 0.5182 | 0.5182 | +0.0000 | 420 |
| C7 | D3 | edas-sigkdd | budget | identical | 0.7619 | 0.7619 | +0.0000 | 420 |
| C7 | D3 | ekaw-iasted | budget | identical | 0.8750 | 0.8750 | +0.0000 | 420 |
| C7 | D3 | ekaw-sigkdd | budget | identical | 0.7593 | 0.7593 | +0.0000 | 420 |
| C7 | D3 | iasted-sigkdd | budget | identical | 0.8611 | 0.8611 | +0.0000 | 420 |
| C7 | D4_instance | memoryalpha-stexpanded-instance | budget | changed (14 values) | 0.9231 | 0.9168 | -0.0063 | 420 |
| C7 | D4_schema | memoryalpha-stexpanded-schema | budget | identical | 0.9091 | 0.9091 | +0.0000 | 420 |
| C7 | D5 | D_W_15K_V2 | budget, D5 suffix | changed (14 values) | 0.1286 | 0.1687 | +0.0401 | 420 |
| C8 | D4_instance | memoryalpha-stexpanded-instance | budget | changed (11 values) | 0.8234 | 0.8237 | +0.0003 | 420 |
| C8 | D4_schema | memoryalpha-stexpanded-schema | budget | identical | 0.9205 | 0.9205 | +0.0000 | 420 |
| C8 | D5 | D_W_15K_V2 | budget | changed (13 values) | 0.0658 | 0.0655 | -0.0003 | 420 |
| C9 | D1 | d1_snomed_fma | — | identical | 0.1808 | 0.1808 | +0.0000 | — |
| C9 | D2 | d2_anatomy | — | identical | 0.2511 | 0.2511 | +0.0000 | — |
| C9 | D3 | cmt-conference | — | identical | 0.2132 | 0.2132 | +0.0000 | — |
| C9 | D3 | cmt-confof | — | identical | 0.3576 | 0.3576 | +0.0000 | — |
| C9 | D3 | cmt-edas | — | identical | 0.3894 | 0.3894 | +0.0000 | — |
| C9 | D3 | cmt-ekaw | — | identical | 0.4009 | 0.4009 | +0.0000 | — |
| C9 | D3 | cmt-iasted | — | identical | 0.1361 | 0.1361 | +0.0000 | — |
| C9 | D3 | cmt-sigkdd | — | identical | 0.3131 | 0.3131 | +0.0000 | — |
| C9 | D3 | conference-confof | — | identical | 0.4062 | 0.4062 | +0.0000 | — |
| C9 | D3 | conference-edas | — | identical | 0.3196 | 0.3196 | +0.0000 | — |
| C9 | D3 | conference-ekaw | — | identical | 0.2344 | 0.2344 | +0.0000 | — |
| C9 | D3 | conference-iasted | — | identical | 0.1079 | 0.1079 | +0.0000 | — |
| C9 | D3 | conference-sigkdd | — | identical | 0.4756 | 0.4756 | +0.0000 | — |
| C9 | D3 | confof-edas | — | identical | 0.4382 | 0.4382 | +0.0000 | — |
| C9 | D3 | confof-ekaw | — | identical | 0.2692 | 0.2692 | +0.0000 | — |
| C9 | D3 | confof-iasted | — | identical | 0.0535 | 0.0535 | +0.0000 | — |
| C9 | D3 | confof-sigkdd | — | identical | 0.3341 | 0.3341 | +0.0000 | — |
| C9 | D3 | edas-ekaw | — | identical | 0.3558 | 0.3558 | +0.0000 | — |
| C9 | D3 | edas-iasted | — | identical | 0.2443 | 0.2443 | +0.0000 | — |
| C9 | D3 | edas-sigkdd | — | identical | 0.3147 | 0.3147 | +0.0000 | — |
| C9 | D3 | ekaw-iasted | — | identical | 0.2354 | 0.2354 | +0.0000 | — |
| C9 | D3 | ekaw-sigkdd | — | identical | 0.1368 | 0.1368 | +0.0000 | — |
| C9 | D3 | iasted-sigkdd | — | identical | 0.1401 | 0.1401 | +0.0000 | — |
| C9 | D4_instance | memoryalpha-stexpanded-instance | — | identical | 0.4729 | 0.4729 | +0.0000 | — |
| C9 | D4_schema | memoryalpha-stexpanded-schema | — | identical | 0.8043 | 0.8043 | +0.0000 | — |
| C9 | D5 | D_W_15K_V2 | — | identical | 0.0454 | 0.0454 | +0.0000 | — |
