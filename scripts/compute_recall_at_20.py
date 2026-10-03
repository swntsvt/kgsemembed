import sys, json
from pathlib import Path

# Adjust these paths if running from a different working directory
SRC   = Path("src")
DATA  = Path("data")
CANDS = DATA / "candidates"

sys.path.insert(0, str(SRC))
from kgsemembed.datasets.loader import load_dataset


def recall_at_20(test_refs, candidates):
    """Fraction of gold (src, tgt) pairs where tgt appears in the candidate list."""
    if not test_refs:
        return float("nan"), 0, 0
    hits = sum(1 for src, tgt in test_refs if tgt in candidates.get(src, []))
    return hits / len(test_refs), hits, len(test_refs)


def load_cands(path):
    with open(path) as f:
        raw = json.load(f)
    return raw.get("candidates", raw)   # handles both JSON shapes


results = {}

# D1
pairs = load_dataset("D1", str(DATA))
cands = load_cands(CANDS / "D1" / "d1_snomed_fma_candidates.json")
r, hits, total = recall_at_20(pairs[0].test_refs, cands)
results["D1"] = (r, hits, total)

# D2
pairs = load_dataset("D2", str(DATA))
cands = load_cands(CANDS / "D2" / "d2_anatomy_candidates.json")
r, hits, total = recall_at_20(pairs[0].test_refs, cands)
results["D2"] = (r, hits, total)

# D3 — 21 pairs, report mean / min / max
pairs = load_dataset("D3", str(DATA))
d3_recalls = []
for pair in pairs:
    cand_file = CANDS / "D3" / f"{pair.pair_name}_candidates.json"
    if cand_file.exists():
        cands = load_cands(cand_file)
        r, hits, total = recall_at_20(pair.test_refs, cands)
        d3_recalls.append(r)
    else:
        print(f"  WARNING: missing {cand_file}")
if d3_recalls:
    results["D3_mean"] = (sum(d3_recalls) / len(d3_recalls), None, None)
    results["D3_min"]  = (min(d3_recalls), None, None)
    results["D3_max"]  = (max(d3_recalls), None, None)

# D4 — schema and instance returned as two pairs
pairs = load_dataset("D4", str(DATA))
for pair in pairs:
    cand_file = CANDS / pair.dataset_id / f"{pair.pair_name}_candidates.json"
    if cand_file.exists():
        cands = load_cands(cand_file)
        r, hits, total = recall_at_20(pair.test_refs, cands)
        results[pair.dataset_id] = (r, hits, total)
    else:
        print(f"  WARNING: missing {cand_file}")

# D5
pairs = load_dataset("D5", str(DATA))
cands = load_cands(CANDS / "D5" / "D_W_15K_V2_candidates.json")
r, hits, total = recall_at_20(pairs[0].test_refs, cands)
results["D5"] = (r, hits, total)

# Report
print(f"\n{'Dataset':<14} {'Recall@20':>10}  {'Hits':>7} / {'Total':>7}")
print("-" * 46)
for label, (r, hits, total) in results.items():
    if hits is None:
        print(f"{label:<14} {r:>10.4f}")
    else:
        print(f"{label:<14} {r:>10.4f}  {hits:>7} / {total:>7}")
