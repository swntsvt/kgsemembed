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

**Current focus**: Issue #134 — text-serialization-order experiment on D1 and D2 (effect of item order with the selected set held fixed; order frozen on D2 validation before any test evaluation); #128, #130 and #131 merged (Phase 2)

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

# Generate the results report. The Statistical Significance section tests
# dataset columns, not alignment pairs (see "Statistical analysis"), and only
# accepts results written under the current evaluation protocol.
python scripts/generate_report.py \
    --results_dir data/results/ --output data/results/report.md

# Compute stage-1 candidate recall@20 (the ceiling re-ranking cannot exceed)
python scripts/compute_recall_at_20.py \
    --datasets D1 D2 D3 D4 D5 --data_dir data/

# Print the main results table (LaTeX, Table 6, Section 4.1): mean F1 per
# condition per dataset, bold best per column, rules between ablation groups.
# Mean$^*$ averages D1-D4 (both D4 columns, never D5) and is printed only for
# conditions run on all five of those columns; partial-coverage rows such as
# C17 (D1/D2 only) show a dash, so every reported mean is comparable.
# Output is wrapped in threeparttable with the Mean note in tablenotes, so the
# paper's preamble must load \usepackage{threeparttable}.
python scripts/generate_results_table.py --results_dir data/results/

# Print D3 class vs predicate F1 rows (LaTeX) for the Section 4.3 table.
# F1 is weighted by n_refs per pair; report.md uses the unweighted pair mean.
python scripts/generate_entity_type_table.py --results_dir data/results/

# Plot Figure 1: D3 per-pair F1, C3 vs C10, CMT/EDAS pairs highlighted.
# Writes figures/fig1_d3_scatter.pdf (+ .png) and prints win counts.
python scripts/generate_figure1_d3_scatter.py \
    --results_dir data/results/ --output_dir figures/

# Plot Figure 2: D1/D2 F1 per model with V2+V8 held constant, plus the C17
# (V1/M4) reference bar. C15 has no D2 run, so that bar is drawn as n/a.
# Writes figures/fig2_model_comparison.pdf (+ .png) and prints the values.
python scripts/generate_figure2_model_comparison.py \
    --results_dir data/results/ --output_dir figures/

# Check dataset presence
python scripts/download_datasets.py --data_dir data/ --check

# Controlled PPAS ablation (#122): pre-run sampling inventory (tokenizer only,
# ~4 min), the ablation run itself (loads M2 once; long — run in tmux), and
# its report. Results go to data/results_ppas_ablation/, never data/results/.
python scripts/ppas_ablation_inventory.py --datasets D5 D1 --data_dir data/
python -m kgsemembed.pipeline.ppas_ablation --datasets D5 D1 \
    --data_dir data/ --results_dir data/results_ppas_ablation/
python scripts/generate_ppas_ablation_report.py \
    --results_dir data/results_ppas_ablation/ \
    --output data/results_ppas_ablation/report.md

# Controlled policies P1-P7 (#128): D1 only, own results directory (the runner
# refuses to write them beside the #122 results). The inventory now also
# covers P1-P7; --results_dir switches it to a run-status table.
python -m kgsemembed.pipeline.ppas_ablation --datasets D1 \
    --conditions PA-all-base-det-48 PA-all-base-ppas-48 PA-cov-base-det-48 \
    PA-cov-base-ppas-48 PA-all-fmax-ppas-48 PA-cov-fmax-det-48 PA-cov-fmax-ppas-48 \
    --data_dir data/ --results_dir data/results_ppas_controlled/
python scripts/ppas_ablation_inventory.py --datasets D1 \
    --results_dir data/results_ppas_controlled/
python scripts/generate_ppas_ablation_report.py \
    --results_dir data/results_ppas_controlled/ \
    --output data/results_ppas_controlled/report.md

# Validation of the eligibility effect (#128 follow-up): P1-P4 on D1 at 32/64
# and P1-P7 on D2 at 48, into their own directory (which refuses D1 at 48: the
# #128 runs stay the reference). Two invocations, because P1-P4 at 48 are also
# registered on D1. The report reads the reference read-only and writes
# report.md plus figures/ into the validation directory.
python -m kgsemembed.pipeline.ppas_ablation --datasets D1 \
    --conditions PA-all-base-det-32 PA-all-base-ppas-32 PA-cov-base-det-32 \
    PA-cov-base-ppas-32 PA-all-base-det-64 PA-all-base-ppas-64 PA-cov-base-det-64 \
    PA-cov-base-ppas-64 --data_dir data/ --results_dir data/results_ppas_validation/
python -m kgsemembed.pipeline.ppas_ablation --datasets D2 \
    --conditions PA-all-base-det-48 PA-all-base-ppas-48 PA-cov-base-det-48 \
    PA-cov-base-ppas-48 PA-all-obox-ppas-48 PA-cov-obox-det-48 PA-cov-obox-ppas-48 \
    --data_dir data/ --results_dir data/results_ppas_validation/
python scripts/ppas_ablation_inventory.py --datasets D1 D2 \
    --results_dir data/results_ppas_validation/
python scripts/generate_ppas_validation_report.py \
    --reference_dir data/results_ppas_controlled/ \
    --results_dir data/results_ppas_validation/

# Predicate-role ablation (#131): tokenizer-only inventory (checks every
# predicate has a role), the run (PR-* conditions only, into their own
# directory), status, and the report (reads the #128/validation P1 at 48
# read-only for the ordering artefact).
python scripts/predicate_role_inventory.py --datasets D1 D2 --data_dir data/
python -m kgsemembed.pipeline.ppas_ablation --datasets D1 D2 \
    --conditions $(python -c "from kgsemembed.pipeline.ablation_conditions import ABLATION_CONDITIONS as A; print(' '.join(c.condition_id for c in A if c.is_role_ablation))") \
    --data_dir data/ --results_dir data/results_predicate_roles/
python scripts/predicate_role_inventory.py --datasets D1 D2 \
    --results_dir data/results_predicate_roles/
python scripts/generate_predicate_role_report.py \
    --results_dir data/results_predicate_roles/

# Diagnostics behind the #131 write-up and the #128 review: label loss and
# fingerprint reproduction per P1-P7 policy (tokenizer only); entity-level
# re-ranking of sampled test sources, checked against the sidecars; and
# rendering-position probes on validation sources only (both load M2).
python scripts/predicate_role_diagnostics.py label-loss --datasets D1 D2
python scripts/predicate_role_diagnostics.py examples --n_sample 300 \
    --pair PR-core-full PR-all-full PR-no-syn-full PR-core-plus-syn-full
python scripts/predicate_role_diagnostics.py probe --n_sample 400

# Serialization order (#134): tokenizer-only identity gates and diagnostics
# (~10 min); check that every stored #122/#128/#131 run still renders
# identically; the validation stage (loads M2; long — run in tmux); the freeze
# (validation results only; commit and push it before the test stage); the
# test stage (refuses to start without a committed, pushed freeze on a clean
# tree); the tracked bundle and reports (URI-only, rebuilt from the bundle alone).
python scripts/serialization_inventory.py gates --datasets D1 D2
python scripts/serialization_inventory.py existing --datasets D1 D2
python -m kgsemembed.pipeline.serialization_order --population validation \
    --datasets D1 D2 --data_dir data/ --results_dir data/results_serialization/
python scripts/freeze_serialization_order.py --results_dir data/results_serialization/
python scripts/freeze_serialization_order.py --verify   # recompute; must match the committed record
python -m kgsemembed.pipeline.serialization_order --population test \
    --datasets D1 D2 --data_dir data/ --results_dir data/results_serialization/
python scripts/generate_serialization_report.py export --stage validation
python scripts/generate_serialization_report.py report --stage validation
python scripts/generate_serialization_report.py verify-rebuild --stage validation
python scripts/generate_serialization_report.py check-text
python scripts/generate_serialization_report.py examples-local --stage validation  # local text only
python scripts/generate_serialization_report.py gate-ranks   # test stage; before report --stage test
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
├── figures/                     # generated paper figures — gitignored, rebuilt by scripts
├── logs/                        # experiment execution logs
└── venv/                        # ⚠️ virtual environment — DO NOT MODIFY
```

### Key Technical Details

- **Configuration**: Experiment conditions are defined as `ExperimentCondition`
  dataclasses in `src/kgsemembed/pipeline/conditions.py`. There are currently
  **20 conditions (C1–C20)**, validated at import time; the count is asserted
  against `_EXPECTED_CONDITION_COUNT`. No external config file drives the
  Phase 2 pipeline.
- **C20 / ablation group E**: C20 (V2/M2) is the controlled baseline for V8.
  It differs from C10 (V2+V8/M2) only by the V8 component, and sits alone in
  group E so its Wilcoxon test gets its own Bonferroni family (alpha / 1)
  instead of tightening the correction for the group B comparisons.
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
- **Key dependencies**: `rdflib` (pinned `==7.6.0` in `requirements.txt`,
  `>=7.6,<8` in `pyproject.toml`), `scipy` (`>=1.18`; see below),
  `sentence-transformers`, `torch`, `transformers`, `pandas`, `numpy`,
  `orjson`, `tqdm`, plus `hydra-core` and `omegaconf` for the legacy scaffold
  above. Installed on the development machine: scipy 1.18.1, numpy 2.5.3,
  torch 2.11.0, transformers 5.5.4, sentence-transformers 5.4.0.
- **scipy floor**: scipy 1.15.3 wheels fail to `dlopen` on macOS 26+ (Darwin
  27), aborting every import of `kgsemembed.candidates` with a
  ``__DATA/__thread_bss`` section error from `_propack`. 1.18.1 loads cleanly;
  the floor is set there because that is the version verified, not because
  intermediate releases were bisected. Upgrading from 1.15.3 leaves results
  unchanged: D3 candidate files regenerate byte-identical, and C1/D4 metrics
  reproduce exactly.

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

- `SentenceTransformer()` is called with `model_id` and `device` only, so no
  model executes custom modelling code at load time. Adding a model that
  would need to is a design decision, not a config change. Every model loads
  as a stock `transformers` architecture, but that is not the same as running
  as trained: M5 publishes custom code (`auto_map` in its `config.json`) that
  the loader never runs, and its trained behaviour is restored in-repo
  instead (see **M5 bidirectional attention** below).
- M3 replaces `Alibaba-NLP/gte-large-en-v1.5` and
  `jinaai/jina-embeddings-v2-base-en`, both incompatible with transformers 5.x
  due to custom remote code. `bge-m3` is a stock XLM-RoBERTa architecture and
  reaches its 8192-token context without any.
- **M2_uncapped**: same weights as M2, `ppas_budget=None`. Used only for the
  controlled PPAS ablation (C19). V2+V8 does not invoke PPAS under any model,
  so C19 is bit-for-bit identical to C10 (both record `ppas_effective: false`).
- **M5 bidirectional attention**: stella_en_1.5B_v5 is Qwen2 fine-tuned with
  the causal mask removed; its repository restores that via custom
  `modeling_qwen.Qwen2Model` (`is_causal=False`). Loaded without that code,
  the stock Qwen2 decoder is causal, so mean pooling sees each token only
  through earlier ones. That silently degraded every C13 result before
  #123: the model card's example scored 0.396 where 0.818 is published, and
  D3 R@1 was 0.130 instead of 0.529. `ModelConfig.bidirectional_attention=True`
  makes `load_sentence_transformer()` call
  `enable_bidirectional_attention()` (`embeddings/attention.py`), which marks
  every attention layer non-causal and passes the decoder a bidirectional
  padding mask. `tests/test_attention.py` checks the model card's
  similarities to 1e-3 (`--runslow`).
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

## Controlled PPAS ablation (#122)

A separate registry (`pipeline/ablation_conditions.py`, 64 conditions: 22 `PA-*` from #122, P1–P7 on D1 at 48 from #128, 11 validation conditions, and 24 `PR-*` predicate-role conditions from #131)
and runner (`pipeline/ppas_ablation.py`) keep the ablation out of C1–C20, so
`_EXPECTED_CONDITION_COUNT`, the main tables and the group statistics are
unchanged. All conditions use M2 and one representation; within a budget the
selection policy (and random seed) is the only difference.

- **Representation** (`verbalisation/selection.py`): V6 `key: value` items
  from `_collect_triples`, OpenEA datatype suffix stripped, empty and
  duplicate items dropped, selected items always joined in canonical order.
- **Budget**: real model tokenizer on the final joined text, `[CLS]`/`[SEP]`
  included. Greedy in policy order; an item that does not fit is skipped, not
  a stop. An entity whose full text fits is never sampled, so its text — and
  its cached embedding — is identical under every policy.
- **Policies**: `ppas` (existing tier lists, untiered last), `deterministic`
  (lexicographic predicate URI, object), `random` (seeds 0–4, per-entity RNG
  from `crc32(uri)`), `full` (uncapped; inputs over 512 are model-truncated).
- **Budgets**: 420 (M2's nominal budget) on D5 only, plus outcome-blind stress
  budgets fixed before any run: pooled median full length floored to a
  multiple of 8 — D5 80, D1 48. D1 is not run at 420: nothing exceeds it.
- **PPAS cannot act on D5**: no D5 source and 35 of 13,656 candidates carry a
  tier predicate (OpenEA dropped labels/types; Wikidata uses P31), so PPAS
  selections equal deterministic ones for every sampled D5 entity but one.
  D1 at 48 is the comparison where the tiers matter.
- `ppas_effective` in ablation results is `true` only when the policy is PPAS
  and sampling was invoked for at least one entity. In C1–C20 results it still
  means "strategy and model *can* sample", not that sampling occurred.
- **Results (be06539, one pair per dataset, no significance tests)**: D5 at
  420 — every policy equals the full reference (MRR 0.1777; 1.8% of
  candidates sampled). D5 at 80 — PPAS = deterministic (MRR 0.1497), above
  random (0.1355 ± 0.0009, 5 seeds), below full (0.1777). D1 at 48 — PPAS
  0.4932, random 0.4050 ± 0.0013, deterministic 0.3566, full 0.4525.
- **Read D1 with care**: the gain comes from what the tiers *exclude*. FMA's
  `preferred_name`/`synonym`/`FMAID` are untiered, so PPAS keeps candidates'
  `rdfs:label` + `subClassOf` and drops synonym lists that deterministic
  order keeps. PPAS beating full text says the full V6 key-value text is
  noisy, not that less text is better in general: this representation's full
  MRR (0.4525) is far below C3 V2+V6/M2 (0.8036) on D1.

### Controlled policies P1–P7 (#128)

Seven conditions vary one dimension at a time on D1 at 48 tokens
(`CONTROLLED_POLICIES` in `ablation_conditions.py`), identified by
`policy_label` and IDs `PA-{all|cov}-{base|fmax}-{det|ppas}-48`:

| Label | Filter | Vocabulary | Order |
|---|---|---|---|
| P1 | all | base | deterministic |
| P2 | all | base | ppas |
| P3 | covered | base | deterministic |
| P4 | covered | base | ppas |
| P5 | all | fma_extended | ppas |
| P6 | covered | fma_extended | deterministic |
| P7 | covered | fma_extended | ppas |

- **Vocabularies** (`verbalisation/predicate_config.py`): `base` is a deep
  copy of the `ppas.py` tier lists (its `tiers_hash()` equals
  `tier_config_digest()`); its covered set is exactly the tiered predicates.
  `fma_extended` adds `fma:preferred_name` and `fma:synonym` to class tier 0
  (lexical) and covers `fma:FMAID` without tiering it (identifier: no SNOMED
  counterpart). Roles come from predicate semantics, never from alignments.
- **FMA namespace**: annotation predicates are `http://purl.org/sig/ont/fma/`;
  FMA *class* URIs use `http://purl.org/sig/ont/fma.owl#`. Match predicates by
  full URI (`expand_predicate()`), never by prefix name. #126 used a wrong
  namespace, so its extension matched nothing.
- **Ordering uses the active vocabulary**: `PolicySelector._order()` reads
  `predicate_config.tier_list_for()`. Under #126 it read the global lists,
  so P5 selected exactly what P2 selected. Untiered predicates (unknown or
  covered-but-untiered) fall back after every tier in canonical order.
- **Filter "all" + deterministic order never consults a vocabulary**, so an
  extended deterministic control under "all" would equal P1; P6/P7 are the
  extended controls instead.
- Every ablation result now records `policy_label`, `filter_mode`,
  `order_mode`, the vocabulary (`config_hash`, `tiers_sha256`, covered
  predicates, roles) and per-side item accounting and budget utilisation;
  sidecars fingerprint every entity's selection (`selection_fingerprints`)
  and list each test source's `gold_targets`.
- The report groups controlled runs by run signature (model/tokenizer
  revision, candidate hash, protocol, budget, smoke restriction, test-source
  set) and never combines groups; missing labels are listed.
- Selected items are rendered canonically, so a policy changes which items
  are kept, never their order: "same set, different order" is 0 by design.
  Compare selections by text, never by `Selection.selected`: those are
  indices into the *filtered* item list, so they differ across filter modes.
- The contrast table splits test sources into "selection differs for source
  or a gold target" and its complement. The complement is not unaffected:
  it moves when a policy changes competing (non-gold) candidates.
- **Results (56b9f15, D1 at 48, one pair, paired bootstrap over 4605 test
  sources, no generalisation claim)**: MRR P1 0.3566, P2 0.4932, P3 0.6974,
  P4 0.6845, P5 0.4972, P6 0.5508, P7 0.5105. Filtering dominates (P3−P1
  +0.341); with eligibility fixed, PPAS order is slightly *worse* than
  deterministic (P4−P3 −0.013, all of it on 1276 changed sources, almost
  entirely SNOMED-side). So the #122 PPAS gain (P2−P1 +0.137) is exclusion of
  untiered FMA predicates, not prioritisation. The FMA extension does not
  help: P5−P2 +0.004 [−0.007, 0.015]; P6−P3 −0.147. Lexical items help gold
  targets but help distractors more (P7−P6: changed +0.061, complement −0.193).
  P1/P2 reproduce #122's det/PPAS MRR exactly.

### Predicate-role ablation (#131)

- **Role map** `roles_v1` (`verbalisation/predicate_roles.py`): one role per
  full predicate URI for every predicate in D1/D2 — core (`rdfs:label`,
  `rdf:type`, `rdfs:subClassOf`), preferred_name, synonym, definition,
  identifier, non_english, disjointness, other — fixed from property
  declarations, labels and sample values before any run. An unmapped
  predicate raises `UnmappedPredicateError`; the runner checks every pair
  before encoding.
- **Arms** (`ROLE_ARMS`, `ROLE_MATRIX`, 38 runs): A0 (nothing removed), R-x
  (remove role x), A-core (core only), C+x (core plus x); removal applies to
  both sides. D1: 8 arms at none and 48, 4 C+x at none. D2: 6 arms at none and
  48, A0/R-syn/R-def/A-core at 24, C+syn/C+def at none.
- **No budget** = `full` order: all remaining items, canonical order; only the
  model's 512-token limit applies, and results record `n_model_truncated` and
  `n_truncated_tokens` per side. **Budgeted** arms use `core_first` order
  (core items, then the rest, each canonical; no PPAS tiers), because
  lexicographic order drops `rdfs:label` for 43% (D1) / 48% (D2) of trimmed
  candidates at 48 — the `purl.org/fma` and `geneontology.org` predicates sort
  before `w3.org`. At 24 on D2, 39 candidates' labels do not fit at all, in
  every arm.
- `Selection.selected` indexes the *filtered* item list (vocabulary and role
  filters), so never look items up in the unfiltered list with it; compare
  texts.
- Results: `data/results_predicate_roles/` only; the runner refuses `PR-*`
  conditions in the historical trees and anything else there. Each result
  records `role_arm`, `excluded_roles`, the role map (hash and roles) and,
  per side, `role_removal` (entities, items and item tokens per role).
- The report gives arm − A0 per budget, the per-source difference-in-differences
  `(R−A0)@b − (R−A0)@none` (four runs resampled jointly), budget costs, C+x −
  A-core, and A0@48 core-first − lexicographic P1@48 (the ordering artefact).
- **Results (3d5c487; 38/38 full runs, one clean commit; no input exceeded
  512 tokens, so nothing was truncated; per-pair bootstrap, never pooled)**:
  A0 without a budget reproduces #122 `full` (D1 0.4525). Core only wins
  everywhere: A-core − A0 Δ MRR D1 +0.272 (none) / +0.238 (48), D2 +0.136 /
  +0.093 (48) / +0.070 (24). Single-role removals do not add up to it (sum
  of R-x − A0 without a budget: D1 +0.077, D2 +0.083), so the covered effect
  is not one role. D1, no budget: R-nonEn +0.137, R-syn +0.026, R-def +0.005,
  R-disj −0.001 (n.s.), R-pref −0.037, R-id −0.052; C+pref − A-core +0.028,
  C+id −0.056, C+def −0.027 (−0.434 on the 305 sources whose own or gold text
  has a definition), C+syn −0.249. D2, no budget: R-def +0.076 (+0.416 on 234
  affected sources), R-syn +0.007 [−0.007, 0.021], R-id/R-disj exactly 0;
  C+def − A-core −0.129, C+syn −0.060. Removing a role behaves differently
  under a budget: synonym removal gains under it on D2 (DiD +0.063 at 24 / +0.058 at 48)
  while definition removal loses (−0.076 / −0.066). A budget costs nothing
  with core only (A-core@48 − A-core@none D1 −0.001, D2 −0.001) and helps A0
  (D1 +0.033, D2 +0.042 at 48 / +0.053 at 24). Ordering artefact at 48:
  A0 core-first − P1 lexicographic D1 +0.129, D2 +0.028. On D2, A-core@48 =
  #128 P3 (0.8705): `base` covers only core there. On D1, A-core@48 0.723 >
  P3 0.697. One pair per dataset; not a generalisation claim.
- **Rendering position (investigation, `scripts/predicate_role_diagnostics.py`)**:
  selected items are *rendered* in canonical predicate-URI order whatever the
  selection order, so FMA annotations (`purl.org/sig/ont/fma/…`) precede
  `type: Class | label: …` in every candidate text, while SNOMED's SKOS
  annotations follow the label. The two odd D1 effects are mostly this:
  - **C+syn −0.249**: the loss is confined to sources whose gold target has
    synonyms (−0.39/−0.35 per source vs +0.03/+0.02 without). Probe on 400 D1
    *validation* sources (never test): C+syn − A-core −0.240 [−0.279, −0.204];
    the same items with synonyms rendered after core −0.045 [−0.074, −0.017].
    R-syn is small from A0 because A0's candidates already lead with FMAID,
    non-English and preferred-name items.
  - **C+id −0.056 vs R-id −0.052**: C+id is position only (FMAID last:
    −0.002 [−0.023, +0.017]; masking the digits to `FMAID: 0` is worse,
    −0.093). R-id is *not* position (with every annotation rendered last it is
    still −0.063 [−0.082, −0.043]) and not the digits (masked A0 +0.002);
    why the segment helps a full text is unexplained.
  - **Full text**: rendering all annotations after core lifts A0 by +0.165
    [0.136, 0.195] (0.478 → 0.642), leaving A-core +0.069 [0.038, 0.103]
    ahead instead of +0.234. Hypothesis for D1 only; D2 not probed; probes are
    exploratory renderings, not registered arms.
  - **Checks**: 3,600 recomputed test-source ranks (300 sources × 4 arms × 3
    runs) equal the stored sidecar ranks; all four contrasts share population,
    commit, candidates, data digests, model/tokenizer revision and role map.
- **Reproduce / access**: results (JSON + `_entities.json` sidecars) are in
  `data/results_predicate_roles/{condition}/{D1,D2}/`; `report.md`,
  `figures/` and `diagnostics/` sit beside them. All of `data/` is gitignored
  (local only); every artefact is regenerated from code at a pinned commit:
  - configurations: `ROLE_ARMS`/`ROLE_MATRIX` (`ablation_conditions.py`),
    `roles_v1` (`predicate_roles.py`, hash `d4c5bd86becb…`, recorded in every
    result); budgets and order in each result's `ablation` block;
  - runs: the run command above at commit `3d5c487` (~2–3 h; existing results
    are skipped unless `--force_recompute`);
  - report and figures: `generate_predicate_role_report.py` (reads results
    only; regenerates byte-identically apart from file timestamps);
  - diagnostics: `predicate_role_diagnostics.py label-loss | examples | probe`
    (label-loss is tokenizer only; examples and probe load M2, ~10–40 min).

### Serialization order (#134)

- **Question**: how much the order in which selected items are *written*
  changes matching when the selected item set is fixed. Orders
  (`verbalisation/serialization.py`, hash `9429a87c…`, pinned in tests):
  `uri` (canonical, every earlier condition), `core` (core items first),
  `label` (`rdfs:label` first), `role` (label → preferred_name → synonym →
  definition → subClassOf → type → disjointness → non_english → identifier →
  other). Within a group, canonical order.
- **Selector**: `PolicySelector(render_order=...)`, default `"uri"`, which
  renders through `join_items` exactly as before; non-canonical renders and
  the `label_first`/`role_priority` selection orders need a role filter.
  `ABLATION_CONDITIONS` refuses any non-`uri` render, so every #122/#128/#131
  condition is byte-identical (`serialization_inventory.py existing`). The
  main V2+V6 pipeline does not import `selection.py`.
- **Registry** (`pipeline/serialization_conditions.py`, separate from
  `ABLATION_CONDITIONS`): 62 logical conditions, 57 executions (37 `SO-*`
  condition IDs). Block S (no budget): D1 A0/A-core/C+syn/C+id, D2
  A0/A-core/C+syn/C+def × 4 renders = 32, A-core `core` ≡ `uri` → 30 runs.
  Block B (A0 at D1@48, D2@24, D2@48): F-* (core-first selection, each
  render), C-* (selection = render), X-label/X-role (selection, `uri`
  render) = 10 per budget, C-core ≡ F-core → 9 runs (B1–B9).
- **Gates** (tokenizer only, before any run): item-set fingerprints
  (`(predicate, item text)`, order-free) equal within every fixed-set group;
  shared conditions render their execution's exact text; every `uri`-rendered
  execution reproduces the stored #131/#128 text of every entity; nothing
  exceeds its budget. After the freeze, `gate-ranks` compares test ranks of
  those executions with the stored sidecars source by source.
- **Stages**: `--population validation` ranks only the validation
  population (D1 4,903 sources / 33,490 candidates; D2 556 / 2,687) and
  writes no test rank. The order is chosen on D2 validation (299 reference
  sources) by `evaluation/order_selection.choose_order` (10,000 shared
  replicates, seed 20261016, 95% percentile; E = best plus every candidate
  whose `MRR(best) − MRR(o)` interval has lower bound ≤ 0; precedence
  `label` > `core` > `role`). Those intervals are descriptive, not adjusted
  for selection. D1 validation is exploratory, split into the 400 sources the
  #131 probe used (sorted-URI SHA-256 `a773ea69dc13…`) and the 751 it did
  not. `--population test` verifies the freeze record
  (`artifacts/issue-134/serialization_freeze.json`) first.
- **Frozen**: `label` (2c4f63e, validation at 4809a98). It was chosen by the
  tie-breaking precedence, not by the highest D2 validation MRR: `core` was
  highest (0.8291 vs label 0.8257, uri 0.7899), but no interval of
  `MRR(core) − MRR(o)` excluded 0, so E = {label, core, role}.
- **Test-stage guards**: besides the freeze checks, `verify_freeze` refuses
  a changed rule (seed, replicates, precedence), validation files whose
  digests differ from the record, and any change under `src/kgsemembed/`
  since the validation commit except the runner itself
  (`FROZEN_IMPLEMENTATION`). The test stage refuses `--max_sources` and
  existing test results from another freeze or commit. `export` writes every
  bundle file once (a different byte refuses), and checks each result's
  population and, for test, freeze digest and run signature against
  `PROVENANCE_validation.json`; it also refuses a stage from several commits,
  a dirty tree, a smoke run, a sidecar that does not cover every counted
  reference, or executions over different sources. `report` refuses missing
  or empty executions. `gate-ranks` compares the 13 registered reproduction
  targets (`expected_rank_targets`) with their stored #131/#128 sidecars
  source by source (test reference sources matched by URI): best gold rank,
  gold-target set, and source and gold text fingerprints; a missing baseline,
  missing or extra source, empty comparison or any mismatch fails.
  `report --stage test` refuses to run unless `diagnostics/rank_gate.json`
  passed and the SHA-256 of every report input still matches it.
- **Primary contrasts** (test report only): frozen − uri on D1/S/A0 and
  D2/S/A0, MRR and Recall@1, 95% and Bonferroni 97.5% intervals (family:
  the two datasets of one metric, same 2,000 replicates). Examples:
  5 largest gains and losses of frozen over uri per dataset (ties by URI),
  `examples/<stage>_examples.csv` (URI-only); readable text via
  `examples-local` into the gitignored results directory.
- **Artifacts**: `artifacts/issue-134/` is tracked and URI-only (results
  JSON, per-source outcomes, conditions map, diagnostics, provenance,
  manifest, reports, figures). Readable entity text stays local; dataset
  labels go into public material only once redistribution is confirmed.

### Validation of the eligibility effect (#128 follow-up)

- **Matrix**: P1–P4 (base) on D1 at `VALIDATION_BUDGETS` (32, 64) beside the
  #128 reference at 48, and P1–P7 on D2 at 48, P5–P7 with `obo_extended`
  (`EXTENDED_VOCABULARIES`). A policy label is unique per (dataset, budget),
  not globally. Results: `data/results_ppas_validation/`; the runner refuses
  D1 at 48 there and anything but D1 at 48 in `data/results_ppas_controlled/`.
- **What "covered" means under `base`**: exactly the PPAS tier predicates.
  Its synonym entries (`obo:hasExactSynonym`, `obo:hasSynonym`) use
  `http://purl.obolibrary.org/obo/`, which no dataset uses (oboInOwl is
  `http://www.geneontology.org/formats/oboInOwl#`). So covered keeps SNOMED's
  SKOS synonyms (D1 sources) but drops FMA synonyms (D1 candidates) and NCI/MA
  `hasRelatedSynonym`/`hasDefinition` (D2). `base` is deliberately unchanged
  so P1–P4 keep their #128 definitions.
- **`base_v2`** (#130): `base` with exactly its two wrong synonym URIs remapped
  (`BASE_V2_CORRECTIONS`: `obo:hasExactSynonym` → `oboInOwl:hasExactSynonym`,
  `obo:hasSynonym` → `oboInOwl:hasSynonym`); no other scope added. `base`,
  `fma_extended`, `obo_extended` and `ppas.py` are unchanged and their hashes
  are pinned in `tests/test_predicate_config.py`. On D1/D2 it covers exactly
  the same items as `base` (neither uses those two predicates; D2 uses
  `hasRelatedSynonym`), so no rerun is implied. No condition uses it yet.
- **`obo_extended`** (D2 only): base plus the oboInOwl synonym family and
  `hasDefinition` in class tier 0 (lexical), `hasAlternativeId`/`hasDbXref`
  covered but untiered (identifier) — the D2 analogue of `fma_extended`.
- **D2 at 48 barely binds** under covered eligibility (0.9% of candidates
  sampled), so on D2 P4 ≡ P3: that contrast is not identifiable, which the
  validation report states instead of printing a zero effect. D2's own
  outcome-blind stress rule would give 24.
- Results now record `dataset_files_sha256` (source/target/reference digests);
  the #128 reference predates it, so the report compares current digests and
  file mtimes instead.
- **Results (4a42227; 19/19 full runs; per-pair bootstrap, never pooled)**:
  covered − all, deterministic (P3−P1) Δ MRR D1 +0.331 / +0.341 / +0.292 at
  32 / 48 / 64, D2 +0.121 [0.100, 0.141]; under PPAS order (P4−P2) D1 +0.228 /
  +0.191 / +0.214, D2 +0.093. PPAS − deterministic is positive only with every
  predicate eligible (P2−P1 D1 +0.089 / +0.137 / +0.076, D2 +0.028) and nil
  or slightly negative once eligibility is fixed (P4−P3 D1 −0.014 / −0.013 /
  −0.002; D2 not identifiable). P3 MRR is flat across D1 budgets
  (0.697 / 0.697 / 0.691). On D2, P5–P7 (`obo_extended`) select almost what P1
  selects (≤4 sources, ≤109 candidates differ) and score like P1, so
  re-admitting synonyms and definitions erases the covered gain. Same
  direction on two pairs; not a generalisation claim. No uncapped
  all-vs-covered control was run.
- **Review amendments (supersede parts of the two Results bullets above)**:
  deterministic (lexicographic) order drops `rdfs:label` for many trimmed
  candidates, because `purl.org/fma` and `geneontology.org` predicates sort
  before `w3.org`: P1 drops it for 17,647 D1 candidates at 48 (44,732 at 32;
  6,586 at 64) and 548 on D2; P2–P4 for ≤30. `fma_extended`/`obo_extended`
  put FMA/oboInOwl items in tier 0 *ahead of* the label (canonical order
  within a tier), so P5/P7 drop it for 5,360 D1 / 439 D2 candidates and P6
  for 14,369 / 548. Hence:
  - P3−P1 is eligibility *plus* label restoration; the eligibility estimate
    with the label kept in both arms is P4−P2 (D1 +0.228 / +0.191 / +0.214,
    D2 +0.093).
  - P2−P1 is mostly the label-dropping baseline, not evidence for the tier
    design: a tier-free core-first order (#131 A0@48) reaches P2 within 0.008
    on D1 and selects identically on D2.
  - P5−P2, P6−P3, P7−P6, the "lexical items help distractors more" reading
    and D2's "re-admitting synonyms erases the gain" all mix vocabulary with
    label loss; use #131's no-budget C+x arms instead.
  - Items are *rendered* in canonical order, so FMA annotations precede the
    label in every candidate text (#131 investigation); on D1 validation
    sources, rendering annotations after core lifts the full text by +0.165.
  Selections and contrasts themselves reproduce exactly (0 fingerprint
  mismatches; `scripts/predicate_role_diagnostics.py label-loss` on
  `feature/issue-131`).

## Datasets

```
D1  SNOMED-FMA Body (Bio-ML, OAEI)        Classes only
    data/d1_snomed_fma/source.rdf, target.rdf, reference.rdf
    Split: 0/20/80 by source entity, seeded shuffle, EDOAL reference

D2  Anatomy MA-NCI (OAEI Anatomy Track)    Classes only
    data/d2_anatomy/source.rdf, target.rdf, reference.rdf
    Split: 0/20/80 by source entity, seeded shuffle, EDOAL reference

D3  Conference (OAEI Conference Track)     Classes + Predicates (mixed)
    data/d3_conference/{pair_name}/source.rdf, target.rdf, reference.rdf
    21 pair subdirectories; entity_type="mixed"
    Split: 0/20/80 by source entity per pair, seeded shuffle

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
    Train: 721_5fold/1/train_links (loaded into train_refs; excluded from
    scoring, never trained on)
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

## Evaluation protocol (version 2)

`EVALUATION_PROTOCOL_VERSION` in `evaluation/metrics.py` is written into every
result. Version 1 results (no `evaluation_protocol` field) are not comparable
with version 2 and are rejected by `evaluation/stats.py`.

- **Splits** are grouped by source entity, so every reference of a 1:N source
  lands in the same slice.
- **Populations** (`evaluation/population.py`): each source is scored in
  exactly one of validation or test. A source with a test reference is test;
  else with a validation reference, validation; else with a training reference
  (D5 only), excluded. Unmatched sources (no reference anywhere) are split
  between validation and test at the slice's reference-source ratio by a
  stable `zlib.crc32` draw; their predictions are false positives (the OAEI
  complete-reference convention), so tuning sees the same FP pressure as test.
- **Partial references (D4)**: the KG track's gold standard is partial, so
  both D4 pairs set `AlignmentPair.reference_complete=False`. An unmatched
  source's prediction is then judged only when its target is a reference
  target (it contradicts a gold mapping → FP); otherwise it is ignored
  (`judgeable_ranked_lists()`), following the track's own evaluation.
- **D5 unreachable references**: 167 D5 test links have a source with no
  triples, so it is never ranked; they stay in the recall denominator as FNs.
- **Prediction**: each source predicts only its top-ranked candidate, kept if
  its score reaches the threshold (BERTMapLt's one-mapping-per-source rule).
- **Threshold**: tuned on the validation population over 0.10–0.99 in steps
  of 0.01; ties resolve to the lower median of the tied thresholds.
- **MRR** averages over test reference sources; a reference source without a
  ranked list contributes 0, a ranked source without a reference is ignored.
  Recall@k divides by the number of test references.

## Statistical analysis

`evaluation/stats.py` tests **dataset columns**, not alignment pairs: D1, D2,
D3 (mean of its 21 pairs), D4_schema, D4_instance, D5. The 21 D3 pairs share
seven ontologies (each one in six pairs), so they are not independent and must
not be pooled as separate observations. `"D4"` in a comparison expands to both
D4 columns.

- Wilcoxon signed-rank, two-sided, zero differences dropped, p-value by exact
  sign-flip permutation. Fewer than five columns → reported, not tested.
- `effect_size_r` is the signed matched-pairs rank-biserial
  `(R+ - R-)/(R+ + R-)`, positive when B beats A.
- With n columns the smallest attainable p is `2/2**n` (0.03125 at n=6), so
  under group B's alpha/4 no comparison can be significant; results carry
  `min_attainable_p` and `underpowered`. Only group E (alpha/1) can reach
  significance. Pair-level W/T/L counts are descriptive only.

## Result JSON schema

Every file written by `run_experiment.py` contains:

```
{
  "condition_id", "dataset_id", "pair_name",
  "strategy", "model_key", "model_id",
  "apply_ppas", "ppas_effective", "token_budget",
  "metrics": {
    "f1", "precision", "recall", "threshold",
    "mrr", "recall_at_1", "recall_at_5", "recall_at_10"
  },
  "counts": {"tp", "fp", "fn"},
  "per_entity_type": {
    "class":     {"f1", "precision", "recall", "threshold",
                  "mrr", "recall_at_1", "recall_at_5", "recall_at_10",
                  "tp", "fp", "fn", "n_refs"},
    "predicate": {...}
  },
  "population": {"n_train_refs", "n_val_refs", "n_test_refs",
                 "n_val_ranked_sources", "n_test_ranked_sources",
                 "n_val_unmatched_sources", "n_test_unmatched_sources",
                 "reference_complete"},
  "n_source_entities", "n_candidates_per_entity", "candidates_sha256",
  "hf_revision", "kgsemembed_version",
  "python_version", "run_timestamp",
  "versions": {"python", "torch", "transformers", "sentence_transformers",
               "numpy", "scipy", "rdflib"},
  "random_seed", "split_seed", "evaluation_protocol",
  "git_commit", "git_dirty"
}
```

`per_entity_type` is non-empty only for mixed-type pairs (D3, D4_schema).
Old result files without `per_entity_type` are valid — the aggregator handles
both. A bucket with fewer than three test references is dropped with a warning
rather than reported.

`ppas_effective` records whether verbalisation can actually sample with PPAS:
the strategy contains V3, V4, or V6 (`PPAS_SAMPLING_COMPONENTS` in
`verbalisation/ppas.py`) **and** the model has a budget
(`strategy_uses_ppas()`). Only C3, C4, C5, and C7 qualify; every V1/V2/V8
condition records `false` despite declaring `apply_ppas=True`, so the
divergence between configured intent and executed behaviour is visible.
`token_budget` records the model budget V5/V7 truncate to, which is not PPAS.

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

MPS keeps driver memory per distinct batch shape (~130 MB each on
bge-large) and torch.mps.empty_cache() does not release it, so a corpus
whose batches take hundreds of padded lengths (C3/D4_instance) exhausts
device memory mid-encode. load_sentence_transformer() therefore pads
every batch to a multiple of PAD_TO_MULTIPLE_OF (64) tokens via
bucket_sequence_padding(). Padded positions are masked: embeddings move
by at most 7.2e-7 across M1-M5.

V2+V8 does not invoke PPAS under any model. V2 (AnnotationVerbaliser)
and V8 (RelationalSignatureVerbaliser) have no budget logic.
apply_ppas=False on a V2+V8 condition has no effect on output.

zlib.crc32, not hash(), for per-entity RNG seeds in V5.
hash() is salted per process by PYTHONHASHSEED; crc32 is stable.

float64 arithmetic throughout ngram.py scoring.
float32 causes tie-breaking inconsistency on real data (observed on D3).

Before registering a model, check its config.json for auto_map.
A repository that maps AutoModel to custom code may depend on it for
trained behaviour (attention direction, pooling) that the stock
architecture silently lacks. Reproduce the model card's published
example output before trusting any result (see M5, #123).

estimate_tokens() (words x 1.3) is not a tokenizer count. On D5 the real
bge-large length is 2.6-3.5x the estimate, so V4/V6 under the "420" budget
emitted texts over 512 tokens (53 D5 sources, 124 candidates under V6) that
the model silently truncated. Use make_token_counter() in selection.py when
a budget must hold for the encoded input.

Threshold is tuned on the validation population only.
Never pass test-population pairs to tune_threshold(); pass
top_ranked_pairs() of the validation ranked lists, as _resolve_threshold does.
```

## Constraints

- Always use `venv` for Python and pip — never the system Python.
- Never commit directly to `main` or `develop`.
- Do NOT modify `configs/local/runtime.yaml`.
- Do NOT overwrite files under `data/` unless explicitly asked; existing result
  files are skipped unless `--force_recompute` is passed.
- Do NOT add dependencies outside the key dependencies listed above without
  confirming first.
