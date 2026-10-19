# kgsemembed

## Overview

`kgsemembed` is a research-oriented Python framework for semantic embedding-based Knowledge Graph matching.
This repository implements Phase 2 (dense retrieval + verbalisation study) with reproducible, config-driven experiments.

## Installation

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Configuration Management (Issue #4)

Hydra-based YAML config groups are used for reproducible runs:

- `configs/config.yaml` (root defaults)
- `configs/model/*.yaml`
- `configs/dataset/*.yaml`
- `configs/verbalisation/*.yaml`
- `configs/candidates/*.yaml`
- `configs/experiment/*.yaml`

### Local-only runtime config

Do not commit machine-specific config.

1. Copy sample file:
```bash
cp configs/local/runtime.example.yaml configs/local/runtime.yaml
```
2. Update local absolute paths and device settings in `configs/local/runtime.yaml`.

`configs/local/runtime.yaml` is gitignored, while `runtime.example.yaml` is tracked.

## Dataset Loader (Issue #5)

The dataset loader supports OAEI-style graph inputs for:

- source ontology graph
- target ontology graph
- alignment graph

Supported formats:

- RDF/XML (`.rdf`, `.xml`)
- Turtle (`.ttl`)

Label extraction priority:

1. `rdfs:label`
2. `skos:prefLabel`
3. URI local-name fallback

### English-only assumption

For this project phase, datasets are assumed to be exclusively English.
Label extraction therefore prefers English (`@en`) literals and falls back to URI local-name when needed.

## Candidate Input Pipeline (Issue #6)

Character n-gram candidate generation is available for Stage-1 style filtering before dense retrieval.

- deterministic preprocessing (lowercase, trim, whitespace normalize)
- optional punctuation stripping
- similarity metrics: `cosine`, `jaccard`
- top-`k` candidates per source entity
- deterministic tie-break: score desc, target URI asc
- optional CSV persistence for downstream embedding evaluation

Main candidate config keys:

- `candidates.method` (`char_ngram`)
- `candidates.n`
- `candidates.metric` (`cosine|jaccard`)
- `candidates.top_k`
- `candidates.persist`
- `candidates.output_file`

## Stage-1 n-gram baseline and evaluation protocol (Issue #137)

The stage-1 character n-gram order of the stored top-20 candidate lists (B0)
is scored under the same populations, validation-tuned threshold and metrics
as every Phase 2 condition:

```bash
python -m kgsemembed.pipeline.ngram_baseline --datasets D1 D2 D3 D4 D5 \
    --data_dir data/ --results_dir data/results_ngram_baseline/
```

D1 (Bio-ML 2022 `snomed.body-fma.body`, unsupervised setting) needs four
files in `data/d1_snomed_fma/`: `source.rdf`, `target.rdf`, `reference.rdf`
(the 6,530 test mappings) and `input.rdf` (the 726 validation mappings,
loaded as known pairs and never scored). A missing file is reported by:

```bash
python scripts/download_datasets.py --data_dir data/ --check
```

The protocol, provenance and B0 comparison are in `artifacts/issue-137/`.

## Datasets, provenance and evaluation protocol (Issue #138)

| Column | Task | Release (input files byte-identical) | Reference used |
|---|---|---|---|
| D1 | SNOMED CT → FMA body, classes | Bio-ML 2022 `snomed.body-fma.body` | unsupervised test mappings (6,530 of 7,256); the 726 validation mappings are known pairs |
| D2 | Mouse anatomy → NCI anatomy, classes | OAEI Anatomy | full reference (1,516) |
| D3 | 21 Conference pairs, classes and properties | OAEI Conference | ra1-M3 (305), not the track's main rar2 |
| D4 | memoryalpha → stexpanded, schema and instances | OAEI Knowledge Graph track v4 | partial gold standard (54 schema + 1,725 instance) |
| D5 | DBpedia → Wikidata 15K (V2), instances | OpenEA v2.0 | fold 1: 3,000 train / 1,500 valid / 10,500 test |

Every result is a **controlled re-ranking of 20 stage-1 candidates per
source**. The reference is split 20/80 by source (D5: OpenEA fold 1). The
threshold is tuned on validation, and test metrics are computed within the
candidate lists. This is not any benchmark's official protocol, so the
numbers must not be compared head-to-head with published system scores.

Details are in `artifacts/issue-138/`:
- `benchmark_protocols.md`: per-dataset protocol versus the official one,
  metric definitions and ceilings;
- `reference_statistics.md`, `release_checks.md`, `results_inventory.md`:
  provenance and reference statistics;
- `paper_methods_section.tex` and `references.bib`: the paper's text.

```bash
python scripts/reference_statistics.py --data_dir data/
python scripts/results_inventory.py
python scripts/release_checks.py --data_dir data/ \
    --ra1_zip reference-alignment.zip --openea_zip OpenEA_dataset_v2.0.zip
```

## Usage

Base run:
```bash
python3.12 -m kgsemembed.pipeline.run_experiment
```

Run with config-group overrides:
```bash
python3.12 -m kgsemembed.pipeline.run_experiment model=all-minilm-l6-v2 dataset=sample verbalisation=v1
```

Run with candidate overrides:
```bash
python3.12 -m kgsemembed.pipeline.run_experiment candidates.n=2 candidates.metric=jaccard candidates.top_k=25
```

Persist candidate pairs:
```bash
python3.12 -m kgsemembed.pipeline.run_experiment candidates.persist=true experiment.output_dir=outputs/run1 candidates.output_file=candidates.csv
```

## Logging

- Console + file logging
- Includes resolved experiment/model/dataset/verbalisation/candidate config at startup
- Logs stored in `logs/`

## Testing

```bash
pytest
```
