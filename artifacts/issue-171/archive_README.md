# kgsemembed v0.3.0-phase2-results — data archive

Companion data for tag `v0.3.0-phase2-results` of
https://github.com/swntsvt/kgsemembed (issue #171). It holds the results the
paper reports.

- `data/results/`: C1–C20 result JSONs, their per-source sidecars
  (`*_entities.json`) and `report.md`; 278 results (19 conditions, C12
  retired in #142).
- `data/results_ngram_baseline/B0/`: the stage-1 n-gram baseline B0, 26
  results and sidecars.
- `data/candidates/`: the top-20 stage-1 candidate lists every condition
  re-ranks.
- `data/release/MANIFEST.sha256`: SHA-256 of every file above;
  `requirements-lock.txt`: the environment (macOS arm64, Python 3.12.14).

**Provenance.** Every result comes from one run at commit `3b23a8b` (#183:
token budgets counted with each model's tokenizer on the encoder input, M4 at
512, V7's annotation written once, D5 datatype suffixes removed), from a
clean tree on Apple MPS with pinned model revisions. That commit was
squash-merged into `develop`, so it is kept reachable by the annotated tag
`results/issue-183`; each result records it as `git_commit`.

**Version numbers.** `pyproject.toml` says `0.1.0`, and every result records
`kgsemembed_version: 0.1.0`. The package version was deliberately not bumped
with the release tags (`v0.2.x`, `v0.3.0`): a bump would make results that
are otherwise identical differ in that field. The release tag, not the
package version, identifies this release.

**Content.** All files hold entity URIs, scores and metrics only; no dataset
text is redistributed. The datasets come from their original releases (see
"Data" in the repository README); D1 requires a SNOMED CT licence.

**Reproduce.** Extract at the repository root of a checkout of the tag, then
follow "Reproducing the paper tables and figures" in the README: the report,
every paper table, the figure claims, the comparisons and paired tests
against B0, the coverage table, the cost report and the results inventory
regenerate byte-identically, with no model or dataset needed.

**Scope.** A controlled comparison under a re-ranking protocol over 20
stage-1 candidates per source, not benchmark results; do not compare these
numbers with published OAEI, Bio-ML or OpenEA system scores.
