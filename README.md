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

## Usage

Base run:
```bash
python3.12 -m kgsemembed.pipeline.run_experiment
```

Run with config-group overrides:
```bash
python3.12 -m kgsemembed.pipeline.run_experiment model=all-minilm-l6-v2 dataset=sample verbalisation=v1
```

Run with value overrides:
```bash
python3.12 -m kgsemembed.pipeline.run_experiment model.device=mps experiment.seed=7 experiment.output_dir=outputs/dev
```

## Logging

- Console + file logging
- Includes resolved experiment/model/dataset/verbalisation at startup for traceability
- Logs stored in `logs/`

## Testing

```bash
pytest
```
