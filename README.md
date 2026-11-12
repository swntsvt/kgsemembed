# kgsemembed

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24162839.svg)](https://doi.org/10.5281/zenodo.24162839)

Phase 2 of a three-phase knowledge graph (KG) alignment study. Here, dense
sentence embeddings re-rank stage-1 candidates. Each KG entity is turned into
text by a verbalisation strategy (V1–V8) and encoded by a
`sentence-transformers` model (M1–M5). For every source entity, its 20
character n-gram candidates are then re-ranked by cosine similarity.
Experiments run on five OAEI/OpenEA datasets.

- **Phase 1** (complete): character n-gram candidate generation,
  [`kgcandgen`](https://github.com/swntsvt/kgcandgen).
- **Phase 2** (this repository): verbalisation × embedding model conditions
  C1–C20, compared with the stage-1 n-gram order (B0).
- **Phase 3** (planned): GraphRAG ablation over the Phase 2 verbalisations.

Every number this repository produces is a **controlled re-ranking of 20
stage-1 candidates per source**. It does not follow any benchmark's official
protocol, so never compare these numbers head-to-head with published OAEI,
Bio-ML or OpenEA system scores (see `artifacts/issue-138/benchmark_protocols.md`).

## Installation

Python 3.12. The project uses a `venv/` directory and calls its binaries
directly (no `activate`):

```bash
python3.12 -m venv venv
venv/bin/pip install --upgrade pip
venv/bin/pip install -r requirements.txt
venv/bin/pip install -e .
```

To reproduce the stored results exactly, install the locked environment
(macOS arm64, Python 3.12.14) instead:

```bash
venv/bin/pip install -r requirements-lock.txt && venv/bin/pip install -e . --no-deps
```

**Hardware.** The stored results were produced on an Apple M4 Pro (64 GB) with
the PyTorch MPS backend. `run_experiment` and `ppas_ablation` refuse any
device other than MPS unless it is named explicitly with `--allow_device cpu`
or `--allow_device cuda`, which is recorded in each result as
`device_override`. The #134 serialization runner and the #131 diagnostics
script require MPS and have no override. The model weights (M1–M5,
pinned revisions) are downloaded from the Hugging Face Hub on first use.

## Data

The datasets are not redistributed: the tracked artefacts and the release
archive hold only URIs, scores and metrics. Download each dataset from its
release, under its own terms, into `data/`. D1 requires a SNOMED CT licence.
Reference counts below are from `artifacts/issue-138/reference_statistics.md`.

| ID | Task | Release (input files byte-identical) | Reference used |
|---|---|---|---|
| D1 | SNOMED CT → FMA body, classes | Bio-ML 2022 `snomed.body-fma.body` (unsupervised) | 6,530 test mappings; the 726 validation mappings are known pairs, never scored |
| D2 | Mouse anatomy → NCI anatomy, classes | OAEI Anatomy | full reference (1,516) |
| D3 | 21 Conference pairs, classes and properties | OAEI Conference | ra1-M3 (not the track's main rar2) |
| D4 | memoryalpha → stexpanded, schema and instances | OAEI Knowledge Graph track v4 | partial gold standard (54 schema + 1,725 instance) |
| D5 | DBpedia → Wikidata 15K (V2), instances | OpenEA v2.0 | fold 1 (train links excluded, never trained on) |

```bash
venv/bin/python scripts/download_datasets.py --instructions        # where to get each file
venv/bin/python scripts/download_datasets.py --data_dir data/ --check
```

`artifacts/issue-138/` records provenance: checksums against each release,
reference statistics, and each benchmark's protocol compared with ours.

## Running

### Tests

```bash
venv/bin/pytest tests/ -q             # ~4 min; without the datasets in data/, the 6
                                      # tests in test_dataset_loader.py and 2 in
                                      # test_budget.py (D4_instance, D5) fail
venv/bin/pytest tests/ -q --runslow   # also slow model tests (downloads models)
```

### Candidates (stage 1)

`data/candidates/` holds the stored top-20 lists: character 3-grams, cosine
similarity, deterministic tie-break. They come with the release archive (see
"Reproducing the paper tables"). Regenerating them overwrites those files:

```bash
venv/bin/python scripts/generate_candidates.py --datasets D1 D2 D3 D4 D5 \
    --data_dir data/ --n 3 --metric cosine --top_k 20        # overwrites data/candidates/
venv/bin/python scripts/compute_recall_at_20.py --datasets D1 D2 D3 D4 D5 --data_dir data/
```

### Conditions C1–C20

The conditions are registered in `src/kgsemembed/pipeline/conditions.py`. That
makes 19 conditions; C12 was retired in #142 and the IDs were not renumbered.
Each condition pairs one verbalisation strategy with one model:

| Group | Conditions |
|---|---|
| A: lower bounds | C1 V1/M1, C2 V2/M1, C17 V1/M4 |
| B: V8 and V6 | C3 V2+V6/M2, C9 V8/M2, C10 V2+V8/M2, C18 V2+V8/M1 |
| C: strategies and models | C4 V3/M2, C5 V4/M2, C6 V2+V7/M4, C7 V6+V3/M2, C8 V5/M2, C11 V2+V8/M4, C13 V2+V8/M5, C16 V2+V8+V7/M4 |
| D: no token budget | C14 V4+V6/M3, C15 V2+V8/M3 |
| E: V8 control | C20 V2/M2 |
| S: sanity check | C19 V2+V8/M2_uncapped (must equal C10) |

Strategies: V1 label + synonyms, V2 annotations, V3 template natural
language, V4 structured key–value, V5 neighbourhood walk, V6 schema-aware,
V7 hierarchical context, V8 relational signature (`src/kgsemembed/verbalisation/`).
Models: M1 all-MiniLM-L6-v2, M2 bge-large-en-v1.5, M3 bge-m3, M4 BioLORD-2023,
M5 stella_en_1.5B_v5 (`src/kgsemembed/embeddings/models.py`).

**Token budgets (#183).** A strategy with a V3–V7 component is held to its
model's token budget (M1 200; M2, M4, M5 420; M3 has none), counted with the
model's own tokenizer on the exact text the encoder receives. Unbudgeted
texts (V1, V2, V8, and everything under M3) are truncated only at the
model's input limit (M1 256, M3 8192, the others 512), which each result
records in its `encoder_input` block.

```bash
# One condition on one dataset (C1 on D2: ~10 s; larger models and datasets take longer)
venv/bin/python -m kgsemembed.pipeline.run_experiment \
    --conditions C1 --datasets D2 --data_dir data/ --results_dir data/results_scratch/
```

Existing result files are skipped unless `--force_recompute` is passed.
`--results_dir` defaults to `data/results/`, the stored release tree, so
always pass a separate directory unless you mean to add to or replace the
stored results.

**Long-running:** omitting `--conditions` and `--datasets` runs all 19
conditions on every dataset each one covers (278 results). The #183 re-run
of all of them and B0 took about 24 hours on the M4 Pro. Run it in `tmux`:

```bash
venv/bin/python -m kgsemembed.pipeline.run_experiment \
    --data_dir data/ --results_dir data/results_rerun/
```

The stage-1 baseline B0 needs no model (~5 min) and refuses `data/results/`:

```bash
venv/bin/python -m kgsemembed.pipeline.ngram_baseline --datasets D1 D2 D3 D4 D5 \
    --data_dir data/ --results_dir data/results_ngram_baseline_scratch/
```

### Evaluation

Protocol version 2 (`src/kgsemembed/evaluation/`):

- **Splits.** Each reference is split 20/80 (validation/test) by source entity;
  D5 uses OpenEA fold 1.
- **Threshold.** Tuned on validation only.
- **Prediction.** Each source predicts only its top-ranked candidate
  (BERTMapLt's one-mapping-per-source rule).
- **Primary metrics.** Re-ranking MRR and source-level Hits@1, both computed
  within the top-20 lists. Every table prints its top-20 ceiling.
- **Secondary metric.** F1.
- **Statistics.** Per-dataset paired bootstrap intervals and randomisation
  tests come from the per-source sidecars (`*_entities.json`), with Holm
  correction within each group × column on MRR. A Wilcoxon test over the six
  dataset columns gives a cross-dataset consistency summary.

The full definitions are in `artifacts/issue-137/evaluation_protocol.md` and
the `metrics.py` docstring.

### Ablation studies

Each study has its own registry, runner and results directory. None of them
touches the C1–C20 tables. Commands, results and caveats are in `CLAUDE.md`.

- **#122/#128: controlled PPAS ablation.** Token-budget selection policies,
  `pipeline/ppas_ablation.py`.
- **#131: predicate-role ablation.** Removes or keeps predicate roles.
- **#134: serialization order.** The selected item set is fixed and only the
  item order changes. The order was frozen on D2 validation before the test
  run. Tracked URI-only artefacts are in `artifacts/issue-134/`.

The original ablation D1 runs predate #137's D1 known-pair fix, so compare
them only with each other. #164 re-ran the D1 runs of #122, #128, its
validation follow-up and #131 (not #134) on the current D1 population (`data/results_issue164/`, reports in
`artifacts/issue-164/`); those are the D1 numbers the paper cites. The
ablation trees are not in the release archive.

## Reproducing the paper tables and figures

The paper's results are those of the release `v0.3.0-phase2-results`
(created in #171; the earlier releases predate #183, see "Earlier
releases" below). Its data archive holds every stored result with its
per-source sidecar and `report.md` (`data/results/`), B0 (`data/results_ngram_baseline/`), the stage-1
candidates (`data/candidates/`), the locked requirements and a checksum
manifest (`data/release/`). Every stored C1–C20 and B0 result comes from one
run at commit `3b23a8b` (the #183 re-run, kept reachable by the tag
`results/issue-183`). Rebuilding from the archive needs neither the models
nor the datasets:

```bash
git clone --branch v0.3.0-phase2-results https://github.com/swntsvt/kgsemembed.git && cd kgsemembed
python3.12 -m venv venv
venv/bin/pip install -r requirements-lock.txt && venv/bin/pip install -e . --no-deps
gh release download v0.3.0-phase2-results
shasum -a 256 -c kgsemembed-v0.3.0-phase2-results-data.tar.gz.sha256
tar -xzf kgsemembed-v0.3.0-phase2-results-data.tar.gz && shasum -a 256 -c data/release/MANIFEST.sha256
```

Then, in that checkout, regenerate every output the paper uses and check
that it is unchanged. The scripts that write files rewrite tracked files, so
`git diff --exit-code` is the check:

```bash
# Results report (~6.5 min; paired tests from the sidecars, then every condition against B0)
venv/bin/python scripts/generate_report.py --results_dir data/results/ --output report.md
cmp report.md data/results/report.md

# Results tables (tab:main_results is mrr; hits_at_1, f1, recall_at_1) and tab:entity_type
for m in mrr hits_at_1 f1 recall_at_1; do
  venv/bin/python scripts/generate_results_table.py --results_dir data/results/ \
      --baseline_dir data/results_ngram_baseline/ --metric $m | cmp - artifacts/issue-139/tables/$m.tex
done
venv/bin/python scripts/generate_entity_type_table.py --results_dir data/results/ \
    | cmp - artifacts/issue-166/entity_type_table.tex

# Tracked outputs written in place
venv/bin/python scripts/generate_ablation_tests_table.py --results_dir data/results/   # tab:stats, a few min
venv/bin/python scripts/figure_claims.py --results_dir data/results/ \
    --output_dir artifacts/issue-149/                                                # ~1.5 min
venv/bin/python scripts/generate_baseline_comparison.py
venv/bin/python scripts/baseline_paired_tests.py                                     # ~3.5 min
venv/bin/python scripts/coverage_table.py --results_dir data/results/ \
    --output artifacts/issue-163/coverage.md
venv/bin/python scripts/generate_cost_report.py --column D1     # tab:cost_d1 and its figure
venv/bin/python scripts/results_inventory.py                    # reads git refs; seconds
git diff --exit-code
```

Checked in #171 exactly as written here: a fresh clone at the tag, a new
venv from `requirements-lock.txt`, the downloaded release (archive and all
636 manifest checksums OK), no datasets or models. Every command above
reproduced its output byte-identically, `git diff --exit-code` was clean,
and the figure loop below ran without error.

| Output | Paper | Generated by |
|---|---|---|
| `data/results/report.md` | results text: summary, paired tests, tests against B0 | `generate_report.py` |
| `artifacts/issue-139/tables/{mrr,hits_at_1,f1,recall_at_1}.tex` | `tab:main_results`, `tab:ranking_hits_at_1`, `tab:matching_f1`, `tab:ranking_recall_at_1` | `generate_results_table.py` |
| `artifacts/issue-166/entity_type_table.tex` | `tab:entity_type` | `generate_entity_type_table.py` |
| `artifacts/issue-188/tables/ablation_tests.tex` | `tab:stats` | `generate_ablation_tests_table.py` |
| `artifacts/issue-181/cost_D1.{tex,md}`, `cost_vs_mrr_D1.{pdf,png}` | `tab:cost_d1` and the cost figure | `generate_cost_report.py` |
| `artifacts/issue-149/figure_claims.{md,json}` | Figure 1 and 2 claims | `figure_claims.py` |
| `artifacts/issue-165/baseline_paired_tests.{md,json}` | paired tests against B0 | `baseline_paired_tests.py` |
| `artifacts/issue-137/baseline_comparison.md` | descriptive B0 comparison | `generate_baseline_comparison.py` |
| `artifacts/issue-163/coverage.md` | condition × dataset coverage | `coverage_table.py` |
| `artifacts/issue-138/results_inventory.md` | result provenance | `results_inventory.py` |

The inventory records, per source commit, only whether it is on
`develop`/`main` and which `v*` release tags contain it, so it changes only
when a new release tag contains a stored result's commit.

**Figures.** `figures/` is not tracked, so a local copy may be older than the
code (a Figure 1 rendered before #149 has the same points but different
callouts). Regenerate every figure from the release before exporting it to
the paper:

```bash
for m in mrr hits_at_1 f1; do
  venv/bin/python scripts/generate_figure1_d3_scatter.py \
      --results_dir data/results/ --output_dir figures/ --metric $m
  venv/bin/python scripts/generate_figure2_model_comparison.py \
      --results_dir data/results/ --output_dir figures/ --metric $m
done
```

**Outputs that need more than the archive.** These read the datasets (and,
where noted, the models), which are not redistributed (see "Data"):

| Output | Needs | Command |
|---|---|---|
| `artifacts/issue-139/metric_ceilings.{json,md}` (ceilings, reference counts) | datasets, ~2 min | `scripts/metric_ceilings.py --data_dir data/ --output_dir artifacts/issue-139/` |
| `artifacts/issue-181/encoded_input_size.*`, `artifacts/issue-183/{text_limitations,budget_binding}.*` | datasets, tokenizers | `encoded_input_size.py`, `text_limitations.py`, `budget_binding.py` |
| `artifacts/issue-183/m4_length_check.*` | datasets, M4, ~15 min | `m4_length_check.py` |
| `artifacts/issue-181/timing/` (the timed runs the cost table reads) | datasets, every model, ~11 h idle machine | `encode_timing.py` |
| `artifacts/issue-138/reference_statistics.md` | datasets, ~2 min | `reference_statistics.py` |
| `artifacts/issue-138/release_checks.md` | datasets and the original release archives | `release_checks.py` |
| the stored results themselves | datasets, every model, ~24 h | `run_experiment` and `ngram_baseline` (see "Running") |

The commands and their arguments are in `CLAUDE.md`. The paper's own text
was written from these tracked files: methods, protocol, metrics and
statistics `artifacts/issue-138/paper_methods_section.tex`; figures
`artifacts/issue-149/paper_figures_section.tex`; PPAS
`artifacts/issue-164/paper_ppas_section.tex` and the predicate-role appendix
`paper_appendix_predicate_roles.tex` beside it; cost
`artifacts/issue-181/paper_cost_section.tex`; limitations
`artifacts/issue-183/paper_limitations.tex`.

**Earlier releases.** `v0.2.0-phase2-results` holds the 271 C-results and
26 B0 results of the #141 re-run (`c6931db`). It lacks the 7 cells #163
added, and #183 replaced its results (budgets counted with each model's
tokenizer, M4 at 512, V7's annotation written once, D5 datatype suffixes
removed; `artifacts/issue-183/rerun_comparison.md` lists the 21 changed
cells). Use it only to reproduce those earlier numbers.
`v0.2.1-phase2-docs` changed documentation and figure claims only.

## Repository layout

```
src/kgsemembed/   datasets, candidates, verbalisation, embeddings, evaluation, pipeline
scripts/          CLI scripts: candidates, reports, tables, figures, inventories
artifacts/        tracked, URI-only outputs and paper text, one folder per issue
tests/            pytest suite
configs/          legacy Hydra config groups (see below)
data/, figures/   local only (gitignored), except the data/sample/ test fixture
```

### Legacy Hydra scaffold

`configs/` and the `@hydra.main` entry point in `pipeline/run_experiment.py`
come from the project's first issues (#4–#6). They are not the Phase 2
pipeline: `python -m kgsemembed.pipeline.run_experiment` uses the argparse
interface above and rejects Hydra overrides such as `model=...`. The scaffold
stays because tests import it. Its optional machine-specific settings (local
paths, device) go in a gitignored copy of the tracked template:

```bash
cp -n configs/local/runtime.example.yaml configs/local/runtime.yaml   # never overwrites
```

## Citation

No paper has been published yet. To cite the software and its data archive,
archived on Zenodo as `10.5281/zenodo.24162839` (this release; all versions:
`10.5281/zenodo.24162838`), use the entry below or `CITATION.cff`, which
GitHub offers under "Cite this repository":

```bibtex
@software{sawant_kgsemembed_2026,
  author    = {Sawant, Savita Kiran},
  title     = {kgsemembed: Verbalisation and Dense Embeddings for Knowledge Graph Alignment},
  year      = {2026},
  version   = {v0.3.0-phase2-results},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.24162839},
  url       = {https://doi.org/10.5281/zenodo.24162839}
}
```

## Licence

MIT (see `LICENSE`).
