# CLAUDE.md

Guidance for Claude Code when working in the `kgsemembed` repository.

## Project Context

Three-phase Knowledge Graph (KG) alignment pipeline evaluated against OAEI
benchmark datasets.

- **Phase 1** (complete): Character n-gram candidate generation (`kgcandgen` repo).
- **Phase 2** (active): Dense embeddings via `sentence-transformers` with RDF
  verbalisation strategies. BERTMapLt is the primary baseline.
- **Phase 3** (planned): GraphRAG integration as an ablation study over Phase 2
  verbalisation, using Ollama for local generation.

Phase 2 datasets: D1 (SNOMED-FMA Body), D2 (Anatomy MA-NCI), D3 (Conference,
21 pairs), D4 (memoryalpha-stexpanded, schema and instance), D5
(DBpedia-Wikidata 15K EN, OpenEA). Five datasets total.

**Current focus**: <!-- Update this line when switching issues, e.g. "Issue #12 — verbalisation strategy ablation (Phase 2)" -->

## Git Workflow

### Branch Strategy
- `main` — stable snapshots only; tagged at paper milestones (e.g. `v0.2.0-phase2-baseline`)
- `develop` — integration branch; all feature/fix branches merge here via PR
- `feature/issue-n` — new capability tied to Issue n
- `fix/issue-n` — bug fix tied to Issue n

Never commit directly to `main` or `develop`.

### Starting Work on an Issue
```bash
git checkout develop
git pull origin develop
git checkout -b feature/issue-n   # or fix/issue-n for bug fixes
```

### Commit Messages
Reference the issue number on every commit:
```
feat: <description> (#n)
fix: <description> (#n)
```

The final commit before raising a PR does not need `Closes #n` — put that in
the PR description instead (see "Raising a Pull Request" below).

Never reference a coding agent, AI assistant, or any automated tool in commit
messages. Write commit messages as if a human authored them.

Never use `Co-Authored-By` trailers in commit messages.

### Raising a Pull Request
- Open as a **draft PR** immediately when the branch is created, so the
  Issue→Branch→PR chain is visible on GitHub from the start.
- Target branch: `develop` (never `main`).
- PR description must contain `Closes #n` to link the PR to the Issue.
- Merge strategy: **squash merge** to keep `develop` log clean — one
  meaningful commit per issue.
- **Close the Issue manually after merging** — see below.

### Closing the Issue

`Closes #n` does **not** close the Issue when the PR is squash-merged into
`develop`. GitHub only honours closing keywords for PRs merged into the
repository's default branch, which is `main`. Because every PR here targets
`develop`, closing is a manual step:

```bash
gh issue close n --comment "Merged into develop via #<pr> as <squash-sha>."
```

Keep `Closes #n` in the PR description regardless — it creates the
Issue→PR link on GitHub and closes the Issue automatically once `develop`
reaches `main` at the next milestone merge.

### Merging to Main
Merge `develop` → `main` only at stable milestones. Tag the merge commit:
```bash
git tag -a v0.x.0-<milestone> -m "Phase 2 verbalisation baseline"
git push origin --tags
```

## Development Commands

Always invoke Python, pip, and pytest via the venv binaries directly — never
use `source venv/bin/activate`. This avoids shell evaluation and works without
permission prompts in Claude Code.

### Installation & Setup
```bash
venv/bin/pip install -r requirements.txt
venv/bin/pip install -e .                    # editable install
```

### Running Experiments
```bash
# Run specific conditions and datasets
python -m kgsemembed.pipeline.run_experiment \
    --conditions C1 C10 --datasets D1 D2 \
    --data_dir data/ --results_dir data/results/

# Re-run a condition, overwriting existing results
python -m kgsemembed.pipeline.run_experiment \
    --conditions C1 --datasets D2 \
    --data_dir data/ --results_dir data/results/ \
    --force_recompute

# Generate candidate files
python scripts/generate_candidates.py --datasets D1 D2 D3 D4 D5 \
    --data_dir data/

# Generate the results report
python scripts/generate_report.py \
    --results_dir data/results/ --output data/results/report.md

# Check dataset presence
python scripts/download_datasets.py --data_dir data/ --check
```

Omitting `--conditions` runs every registered condition; omitting `--datasets`
runs every dataset configured for each condition. Conditions are grouped by
`model_key` so each `SentenceTransformer` is loaded once per group.

### Testing
```bash
venv/bin/pytest                          # full suite
venv/bin/pytest tests/test_filename.py  # single file
venv/bin/pytest tests/ -x -q            # stop on first failure (preferred in CI)
```

Always run `venv/bin/pytest tests/ -x -q` after changes to any module under
`kgsemembed/pipeline/` or `kgsemembed/evaluation/`.

## Coding Conventions

- Python 3.12. Type hints on all function signatures.
- Docstrings in NumPy style on all public functions (see example below).
- Functions should not exceed 20 lines and do one thing well.
- Prefer at most two or three arguments per function.
- Avoid inline comments; write self-explanatory code instead.
- Use exceptions rather than error codes for error handling.
- Do NOT refactor unrelated code. Do NOT rename existing functions unless required.
- Fixed random seed: `np.random.seed(42)` and `random.seed(42)` at the top of
  each evaluation script's `main()` entry point.
- Results written to `data/results/{condition_id}/{dataset_id}/{pair_name}_results.json`.
- New modules go under `kgsemembed/` following the existing package structure.

### NumPy Docstring Example
```python
def compute_recall(matches: list[str], gold: list[str]) -> float:
    """
    Compute recall of retrieved matches against gold standard.

    Parameters
    ----------
    matches : list[str]
        Retrieved candidate URIs.
    gold : list[str]
        Ground-truth URIs.

    Returns
    -------
    float
        Recall score in [0, 1].
    """
```

## Architecture

### Pipeline Flow

```
Data Loading → Candidate Generation → Embedding & Verbalisation → Evaluation
(datasets)      (candidates)           (embeddings, verbalisation)  (evaluation)
```

### Directory Structure

```
kgsemembed/                      # repo root
├── src/kgsemembed/              # main source package
│   ├── pipeline/                # experiment runner (run_experiment.py), conditions.py
│   ├── datasets/                # KG dataset loading — RDF/XML and OpenEA triple files
│   ├── candidates/              # candidate pair generation — char n-gram, cosine, jaccard
│   ├── embeddings/              # model registry (models.py) and encoder (encoder.py)
│   ├── verbalisation/           # entity-to-text strategies V1–V8, PPAS (Phase 2 active)
│   ├── evaluation/              # matching performance metrics, threshold tuning
│   └── utils/                   # logging and custom error types
├── scripts/                     # generate_candidates, generate_report, download_datasets
├── configs/                     # legacy Hydra YAML groups — see Configuration below
│   └── local/
│       └── runtime.yaml         # ⚠️ machine-specific — gitignored, DO NOT MODIFY
├── tests/                       # pytest suite
├── data/                        # OAEI KG datasets (do not regenerate or overwrite)
│   ├── candidates/              # generated candidate files
│   └── results/                 # output JSON — data/results/<condition>/<dataset>/
├── logs/                        # experiment execution logs
└── venv/                        # ⚠️ virtual environment — DO NOT MODIFY
```

### Key Technical Details

- **Configuration**: Experiment conditions are defined as `ExperimentCondition`
  dataclasses in `src/kgsemembed/pipeline/conditions.py`. There are currently
  **19 conditions (C1–C19)**, validated at import time; the count is asserted
  against `_EXPECTED_CONDITION_COUNT`. No external config file drives the
  Phase 2 pipeline.
- **Legacy Hydra scaffold**: `run_experiment.py` still holds a `@hydra.main`
  `main()` entry point and a `run_experiment(cfg)` candidate-generation
  scaffold, backed by `configs/`. It is *not* the Phase 2 pipeline —
  `python -m kgsemembed.pipeline.run_experiment` dispatches to `cli_main()`
  (argparse). Do not delete `configs/`, `hydra-core`, or `omegaconf`:
  `run_experiment.py` imports `hydra` at module scope, and
  `tests/test_configs.py`, `tests/test_candidates.py`, and
  `test_smoke_pipeline.py::test_hydra_entrypoints_still_importable` all depend
  on them.
- **Language**: English-only datasets assumed throughout.
- **Python**: 3.12+ required, always via `venv`.
- **Key dependencies**: `rdflib` (7.6.0 installed; unpinned in
  `requirements.txt` / `pyproject.toml`), `sentence-transformers`,
  `torch`, `transformers`, `scipy`, `pandas`, `numpy`, `orjson`, `tqdm`,
  plus `hydra-core` and `omegaconf` for the legacy scaffold above.

### Embedding Models

Registered in `src/kgsemembed/embeddings/models.py` as `MODEL_REGISTRY`.

| Key | Model ID | `max_tokens` | `ppas_budget` | `batch_size` |
|-----|----------|--------------|---------------|--------------|
| M1 | `sentence-transformers/all-MiniLM-L6-v2` | 256 | 200 | 64 |
| M2 | `BAAI/bge-large-en-v1.5` | 512 | 420 | 32 |
| M2_uncapped | `BAAI/bge-large-en-v1.5` | 512 | `None` | 64 |
| M3 | `BAAI/bge-m3` | 8192 | `None` | 16 |
| M4 | `FremyCompany/BioLORD-2023` | 512 | 420 | 32 |
| M5 | `dunzhang/stella_en_1.5B_v5` | 512 | 420 | 8 |

- Every registered model is a stock `transformers` architecture, so
  `SentenceTransformer()` is called with `model_id` and `device` only. No
  model executes custom modelling code at load time. Adding a model that
  would need to is a design decision, not a config change.
- M3 replaces `Alibaba-NLP/gte-large-en-v1.5` and
  `jinaai/jina-embeddings-v2-base-en`, both incompatible with transformers 5.x
  due to custom remote code. `bge-m3` is a stock XLM-RoBERTa architecture and
  reaches its 8192-token context without any.
- **M2_uncapped**: same weights as M2, `ppas_budget=None`. Used only for the
  controlled PPAS ablation (C19). V2+V8 does not invoke PPAS under any model,
  so C19 is bit-for-bit identical to C10.
- **M5 asymmetric**: source side gets the instruction prefix
  (`query_prefix="Instruct: Retrieve semantically similar text.\nQuery: {}"`),
  candidate side gets no prefix. Apply the prefix only via `encode_source()` /
  `encode_batch(role="source")`, never via `encode_candidate()`. The encoder
  applies it internally — never pre-prefix verbalised text yourself.
- `load_sentence_transformer()` returns a `(model, model_info)` tuple.
  `model_info` keys: `model_id`, `model_key`, `device`, `hf_revision`. Always
  assert `device` starts with `"mps"` after loading.

PPAS budgets are consulted independently by the verbalisers through
`PPAS_BUDGETS` in `src/kgsemembed/verbalisation/ppas.py`, keyed by the same
model keys. Registered verbalisation strategies (`VALID_STRATEGY_NAMES`):
V1, V2, V3, V4, V5, V6, V7, V8, V2+V6, V2+V8, V2+V7, V2+V8+V7, V6+V3, V4+V6,
V8+V6.

## Datasets

```
D1  SNOMED-FMA Body (Bio-ML, OAEI)        Classes only
    data/d1_snomed_fma/source.rdf, target.rdf, reference.rdf
    Split: 0/20/80, seeded shuffle, EDOAL reference

D2  Anatomy MA-NCI (OAEI Anatomy Track)    Classes only
    data/d2_anatomy/source.rdf, target.rdf, reference.rdf
    Split: 0/20/80, seeded shuffle, EDOAL reference

D3  Conference (OAEI Conference Track)     Classes + Predicates (mixed)
    data/d3_conference/{pair_name}/source.rdf, target.rdf, reference.rdf
    21 pair subdirectories; entity_type="mixed"
    Split: 0/20/80 per pair, seeded shuffle

D4_schema  memoryalpha-stexpanded schema  Classes + Predicates (mixed)
D4_instance memoryalpha-stexpanded instances  Instances only
    data/d4_kgtrack/ontologies/memoryalpha.rdf + stexpanded.rdf
    data/d4_kgtrack/references/memoryalpha-stexpanded.rdf (EDOAL)
    Schema/instance split: inspect rdf:type of source URI in source_graph
    D4_schema and D4_instance share the same Graph objects (identity, not copy)
    Parse all .rdf files with format="xml"

D5  DBpedia-Wikidata 15K EN (OpenEA)      Instances only
    data/d5_openea/D_W_15K_V2/
    Built from rel_triples_1/2 + attr_triples_1/2 via _graph_from_triple_files()
    Do NOT use rdflib.Graph.parse() for D5
    Val: 721_5fold/1/valid_links   Test: 721_5fold/1/test_links
    OpenEA deleted all entity labels, and both sides carry opaque local
    names (DBpedia E291085, Wikidata Q1108721). Falling back to those names
    gave candidate recall@20 of 0.0015 and pinned every D5 F1 at ≈0.0001.
    get_entity_label() in candidates/ngram.py now substitutes attribute
    text for opaque IDs, raising recall@20 from 0.0015 to 0.1339. Exempting
    YYYY-MM-DD values from the bare-quantity filter (DATE_PATTERN in
    ngram.py) raised it again to 0.2667; best F1 is 0.0609 (C14, V4+V6/M3).
    Dates are a strong cross-KG key here — an aligned pair carries the same
    birthDate or releaseDate on both sides — and account for 19.8% of
    DBpedia and 4.0% of Wikidata attribute values.
    D5 remains far weaker than D1-D4, but candidate generation is no longer
    the binding constraint: recall@20 is 0.2667 while C14 reaches R@10
    0.2260 and F1 0.0609, so gold pairs now enter the candidate set without
    being ranked to the top. The residual gap is a ranking and precision
    problem at the embedding stage, not a candidate-recall one.
    Treat a D5 recall@20 gain as an upper bound on the achievable F1 gain:
    dates are shared by construction (thousands of films release on one
    date), so some recovered pairs arrive via a non-discriminative key.
    Attribute text is capped at 10 literal values, sorted by predicate then
    value so the selection does not depend on store iteration order.
    OpenEA inlines the datatype into the value column, so a D5 literal's
    string form is "1955-03-02"^^<http://www.w3.org/2001/XMLSchema#date>.
    Strip that suffix before using D5 literal text anywhere: left intact it
    enters an n-gram index as boilerplate shared by nearly every entity on
    both sides, inflating gold-pair similarity while carrying no signal.
    Stripping is what makes the date exemption safe: _literal_text() yields
    a bare 1955-03-02, so DATE_PATTERN matches the value, not the datatype.
    WARNING: Do not use DBP2.0 — it is multilingual, has no attr_triples,
    and uses a different folder structure.
```

`load_dataset()` is keyed `"D1"`–`"D5"`; the `"D4"` loader returns two pairs
whose `dataset_id` values are `D4_schema` and `D4_instance`. Conditions
therefore list `"D4"`, while results are written under both directories.

## Result JSON schema

Every file written by `run_experiment.py` contains:

```
{
  "condition_id", "dataset_id", "pair_name",
  "strategy", "model_key", "model_id",
  "apply_ppas", "ppas_effective",
  "metrics": {
    "f1", "precision", "recall", "threshold",
    "mrr", "recall_at_1", "recall_at_5", "recall_at_10"
  },
  "per_entity_type": {
    "class":     {"f1", "precision", "recall", "threshold",
                  "mrr", "recall_at_1", "recall_at_5", "recall_at_10",
                  "n_refs"},
    "predicate": {...}
  },
  "n_source_entities", "n_candidates_per_entity",
  "hf_revision", "kgsemembed_version",
  "python_version", "run_timestamp",
  "versions": {"python", "torch", "transformers", "sentence_transformers"}
}
```

`per_entity_type` is non-empty only for mixed-type pairs (D3, D4_schema).
Old result files without `per_entity_type` are valid — the aggregator handles
both. A bucket with fewer than three test references is dropped with a warning
rather than reported.

`ppas_effective` records what verbalisation actually did (`PPAS_BUDGETS[model]
is not None`), independent of the condition's declared `apply_ppas`, so any
divergence between configured intent and executed behaviour is visible in the
result file.

## Critical pitfalls

```
rdflib graph.triples() returns a generator.
Collect to list before iterating more than once:
  triples = list(graph.triples((entity_uri, None, None)))

Never mutate module-level tier list constants (CLASS_TIER_LIST, etc).
Deep-copy before extending:
  tier = [list(t) for t in CLASS_TIER_LIST]

Do not use pytest caplog for kgsemembed logger assertions.
init_logging() sets propagate=False. Attach a handler directly
to the specific logger and restore it in teardown.

Dynamic batch size reduction: encode_batch() automatically reduces
batch_size for long sequences to prevent MPS OOM (observed on C14/D5
with bge-m3 at batch_size=16 and uncapped 8192-token texts).
Do not override batch_size manually in encoding loops.

V2+V8 does not invoke PPAS under any model. V2 (AnnotationVerbaliser)
and V8 (RelationalSignatureVerbaliser) have no budget logic.
apply_ppas=False on a V2+V8 condition has no effect on output.

zlib.crc32, not hash(), for per-entity RNG seeds in V5.
hash() is salted per process by PYTHONHASHSEED; crc32 is stable.

float64 arithmetic throughout ngram.py scoring.
float32 causes tie-breaking inconsistency on real data (observed on D3).

Threshold is tuned on val-source pairs only.
Never pass test-source scored pairs to tune_threshold().
```

## Constraints

- Always use `venv` for Python and pip — never the system Python.
- Never commit directly to `main` or `develop`.
- Do NOT modify `configs/local/runtime.yaml`.
- Do NOT overwrite files under `data/` unless explicitly asked; existing result
  files are skipped unless `--force_recompute` is passed.
- Do NOT add dependencies outside the key dependencies listed above without
  confirming first.
