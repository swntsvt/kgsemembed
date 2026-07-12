# CLAUDE.md

Guidance for Claude Code when working in the `kgsemembed` repository.

## Project Context

Three-phase Knowledge Graph (KG) alignment pipeline evaluated against OAEI
benchmark datasets.

- **Phase 1** (complete): Character n-gram candidate generation (`kgcandgen` repo).
- **Phase 2** (active): Dense embeddings via `sentence-transformers` with RDF
  verbalisation strategies. BERTMapLt is the primary baseline.
- **Phase 3** (planned): GraphRAG integration as an ablation study over Phase 2
  verbalisation, using a custom `rdflib` + `leidenalg` + `sentence-transformers`
  stack.

OAEI tracks: anatomy, conference, biodiv, commonkg, largebio (34 datasets total).

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
the PR description instead (see below).

Never reference a coding agent, AI assistant, or any automated tool in commit
messages. Write commit messages as if a human authored them.

Never use `Co-Authored-By` trailers in commit messages.

### Raising a Pull Request
- Open as a **draft PR** immediately when the branch is created, so the
  Issue→Branch→PR chain is visible on GitHub from the start.
- Target branch: `develop` (never `main`).
- PR description must contain `Closes #n` to auto-close the Issue on merge.
- Merge strategy: **squash merge** to keep `develop` log clean — one
  meaningful commit per issue.

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
# Base run
venv/bin/python3.12 -m kgsemembed.pipeline.run_experiment

# With Hydra overrides
venv/bin/python3.12 -m kgsemembed.pipeline.run_experiment \
  model=all-minilm-l6-v2 dataset=sample verbalisation=v1

# Override candidate settings
venv/bin/python3.12 -m kgsemembed.pipeline.run_experiment \
  candidates.n=2 candidates.metric=jaccard candidates.top_k=25
```

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
- All OAEI results written to `results/<track>/<model>_results.csv`.
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
│   ├── pipeline/                # experiment entry point (run_experiment.py)
│   ├── datasets/                # KG dataset loading — RDF/XML and Turtle via rdflib
│   ├── candidates/              # candidate pair generation — char n-gram, cosine, jaccard
│   ├── embeddings/              # dense retrieval and sentence-transformers embedding
│   ├── verbalisation/           # entity-to-text conversion strategies (Phase 2 active)
│   ├── evaluation/              # matching performance metrics
│   └── utils/                   # logging and custom error types
├── configs/                     # Hydra YAML configuration groups
│   └── local/
│       └── runtime.yaml         # ⚠️ machine-specific — gitignored, DO NOT MODIFY
├── tests/                       # pytest suite
├── data/                        # OAEI KG datasets (do not regenerate or overwrite)
├── results/                     # output CSVs — results/<track>/<model>_results.csv
├── logs/                        # experiment execution logs
└── venv/                        # ⚠️ virtual environment — DO NOT MODIFY
```

### Key Technical Details

- **Configuration**: Hydra for hierarchical config. Local paths and device
  settings live in `configs/local/runtime.yaml` — never commit this file.
- **Language**: English-only datasets assumed throughout.
- **Python**: 3.12+ required, always via `venv`.
- **Key dependencies**: `rdflib`, `bm25s`, `sentence-transformers`, `scipy.stats`,
  `pandas`, `numpy`. GraphRAG stack (Phase 3): `leidenalg`, `igraph`.

## Constraints

- Always use `venv` for Python and pip — never the system Python.
- Never commit directly to `main` or `develop`.
- Do NOT modify `configs/local/runtime.yaml`.
- Do NOT overwrite files under `data/` or `results/` unless explicitly asked.
- Do NOT add dependencies outside the key dependencies listed above without
  confirming first.
