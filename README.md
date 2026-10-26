# kgsemembed

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
venv/bin/pytest tests/ -q             # ~4 min; without data/ the 6 tests in
                                      # test_dataset_loader.py fail, the rest pass
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
conditions on every dataset each one covers (271 results). That took about
20 hours on the M4 Pro, most of it on D4 instances. Run it in `tmux`:

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

The ablation D1 numbers predate #137's D1 known-pair fix, so compare them only
with each other.

## Reproducing the paper tables and figures

The tagged release `v0.2.0-phase2-results` ships a data archive with all
stored results, B0, the candidates and a checksum manifest. Rebuilding from it
needs neither the models nor the datasets:

```bash
git clone --branch v0.2.0-phase2-results https://github.com/swntsvt/kgsemembed.git && cd kgsemembed
python3.12 -m venv venv
venv/bin/pip install -r requirements-lock.txt && venv/bin/pip install -e . --no-deps
gh release download v0.2.0-phase2-results
shasum -a 256 -c kgsemembed-v0.2.0-phase2-results-data.tar.gz.sha256
tar -xzf kgsemembed-v0.2.0-phase2-results-data.tar.gz && shasum -a 256 -c data/release/MANIFEST.sha256
```

Then, in that checkout (#141 verified the report and tables this way from a
fresh clone):

```bash
# Results report (a few minutes; per-dataset paired tests from the sidecars)
venv/bin/python scripts/generate_report.py --results_dir data/results/ --output report.md
cmp report.md data/results/report.md

# LaTeX results tables (mrr is the main table; hits_at_1, f1, recall_at_1)
for m in mrr hits_at_1 f1 recall_at_1; do
  venv/bin/python scripts/generate_results_table.py --results_dir data/results/ \
      --baseline_dir data/results_ngram_baseline/ --metric $m | cmp - artifacts/issue-139/tables/$m.tex
done

# Figures 1 and 2 (into figures/, gitignored)
venv/bin/python scripts/generate_figure1_d3_scatter.py \
    --results_dir data/results/ --output_dir figures/ --metric mrr
venv/bin/python scripts/generate_figure2_model_comparison.py \
    --results_dir data/results/ --output_dir figures/ --metric mrr
```

Three generated outputs postdate the tag: the paired intervals behind the
figures' claims (`figure_claims.py`, #149), the B0 comparison regenerated
after #142/#145 (#143; the tag's `baseline_comparison.md` predates both), and
the results inventory, which since #168 records only whether each source
commit is on `develop`/`main` and which release tags contain it, so it no
longer changes when a branch is created. It does change when a later release
tag contains the commit.
Reproducing them needs a checkout of `develop` that includes #143 (until the
next release), with the same archive extracted into it. The scripts rewrite
tracked files, so the last step checks that they regenerate unchanged. The
hand-written protocol notes in `artifacts/issue-137/` and
`artifacts/issue-138/` were also corrected in #143, so read them on `develop`
rather than at the tag.

```bash
git checkout develop
venv/bin/python scripts/figure_claims.py --results_dir data/results/ \
    --output_dir artifacts/issue-149/                       # ~1.5 min
venv/bin/python scripts/generate_baseline_comparison.py
venv/bin/python scripts/results_inventory.py                # reads git refs; seconds
git diff --exit-code artifacts/issue-149/figure_claims.md artifacts/issue-149/figure_claims.json \
    artifacts/issue-137/baseline_comparison.md artifacts/issue-138/results_inventory.md
```

`artifacts/issue-139/metric_ceilings.json` (the ceilings and reference counts
the tables read) is rebuilt by
`scripts/metric_ceilings.py --data_dir data/ --output_dir artifacts/issue-139/`.
That needs the datasets and takes about 2 minutes. The paper's methods text is
in `artifacts/issue-138/paper_methods_section.tex`, and its figures text in
`artifacts/issue-149/paper_figures_section.tex`.

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

No paper has been published yet. To cite the software:

```bibtex
@software{sawant_kgsemembed_2026,
  author  = {Sawant, Savita Kiran},
  title   = {kgsemembed: Verbalisation and Dense Embeddings for Knowledge Graph Alignment},
  year    = {2026},
  version = {v0.2.0-phase2-results},
  url     = {https://github.com/swntsvt/kgsemembed}
}
```

## Licence

MIT (see `LICENSE`).
