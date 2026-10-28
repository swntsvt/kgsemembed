# Paper scope of the #122–#134 ablations (#164)

Decided 2026-10-28, before any re-run. The ablations' D1 numbers were computed
before #137 made D1's 726 Bio-ML validation mappings known pairs
(`n_train_refs` 0); D2 and D5 numbers are unaffected.

## Decisions

| Study | Issue(s) | Where | Reason |
|---|---|---|---|
| PPAS ablation and controlled policies | #122, #128, #128 follow-up | Main text, one short section | The draft calls PPAS a novel contribution and lists "a genuine PPAS ablation" as future work. Both need correcting with the evidence: the #122 gain is eligibility (excluding untiered predicates), not prioritisation (P4 − P3 ≈ 0 once eligibility is fixed), and P2 − P1 is mostly the lexicographic order dropping `rdfs:label`. (Pre-#137 D1 numbers; re-checked on the re-run.) |
| Predicate-role ablation | #131 | Appendix | Explains the #136 finding that synonyms and definitions added to V2 text hurt (C+syn, C+def), without adding a main-text claim. |
| Serialization order | #134 | Follow-up paper; the paper cites none of its numbers | A large effect (D1 A0 MRR +0.19 under `label` vs `uri`) that needs its own treatment, and none of the C1–C20 verbalisers uses the orders it compares. A D1 re-run would reopen the frozen test stage; not doing it keeps the freeze rules intact. |

The draft's other future-work item, quantifying annotation density, is done
in #149 (`artifacts/issue-149/`); the paper should cite it rather than
propose it.

## Re-runs (post-#137 D1 population)

Every D1 run of the kept studies is re-run into `data/results_issue164/`,
one subtree per original tree, so the runners' known-pair check
(`check_known_pairs`) never sees two populations in one directory:

| Subtree | Replaces D1 of | D1 conditions | Copied unchanged |
|---|---|---|---|
| `ppas_ablation/` | `data/results_ppas_ablation/` (#122) | 8: `PA-det-48`, `PA-ppas-48`, `PA-rand-48-s0`…`s4`, `PA-full` | D5 (15 results) |
| `ppas_controlled/` | `data/results_ppas_controlled/` (#128) | 7: P1–P7 at 48 | — |
| `ppas_validation/` | `data/results_ppas_validation/` | 8: P1–P4 at 32 and 64 | D2 P1–P7 at 48 (7) |
| `predicate_roles/` | `data/results_predicate_roles/` (#131) | 20: every D1 arm of `ROLE_MATRIX` | D2 (18) |

The runs are `artifacts/issue-164/run_d1_reruns.sh` (refuses a dirty tree).
Copied D2/D5 results and their sidecars are byte-identical to the originals.
The original trees are kept unchanged as the pre-#137 record. The #131
report reads the D1 P1@48 link run from `ppas_controlled/` and the D2 one from
`ppas_validation/` (`--d1_link_dir`, `--d2_link_dir`).

## What the paper cites

Filled in after the re-run: the reports, their commit, and the numbers the
PPAS section and the appendix use. Every cited D1 number comes from a result
recording `population.n_train_refs == 726`.
