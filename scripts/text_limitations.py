"""Quantify two retained text characteristics, documented as limitations (#183).

1. **D5 opaque identifiers.** OpenEA deleted D5's labels, so an object with
   no label is written as its local name (DBpedia ``E…``, Wikidata ``Q…``).
   This measures the share of encoded tokens those identifiers take in every
   D5 condition, how many texts contain one, and how many identifiers the two
   graphs share (none would mean they carry no cross-graph signal).  It also
   counts the literals whose inlined datatype is a unit (e.g. DBpedia's
   ``usDollar``), which the suffix removal drops as stage 1 does.
2. **D4_instance metadata predicates in V3/V4/V6.**  Those strategies write
   every predicate by design; #145 excluded annotation and page-metadata
   predicates from V8 only.  This measures the share of encoded tokens that
   come from predicates V8 excludes (any family but ``type_and_label``), per
   family, in every D4_instance condition with a V3, V4 or V6 component.

Texts are the ones the pipeline encodes (budgets applied), on a fixed random
sample of each role.  Tokenizer only; nothing under ``data/`` is written.

Usage
-----
python scripts/text_limitations.py --data_dir data/ --output_dir artifacts/issue-183/
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
from rdflib import Literal, URIRef

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from encoded_input_size import compared_conditions  # noqa: E402
from kgsemembed.candidates import load_candidates  # noqa: E402
from kgsemembed.datasets import AlignmentPair, load_dataset  # noqa: E402
from kgsemembed.pipeline.run_experiment import _unique_candidate_uris  # noqa: E402
from kgsemembed.verbalisation.budget import encoder_input_counters  # noqa: E402
from kgsemembed.verbalisation.registry import build_verbaliser  # noqa: E402
from kgsemembed.verbalisation.v3 import TemplateNLVerbaliser  # noqa: E402
from kgsemembed.verbalisation.v8 import exclusion_family  # noqa: E402

SAMPLE = 2000
SEED = 183
OPAQUE_ID = re.compile(r"\b[EQ]\d{3,}\b")
INLINED_DATATYPE = re.compile(r'^"(?:.*)"\^\^<([^>]*)>$', re.DOTALL)
XSD = "http://www.w3.org/2001/XMLSchema#"
KV_COMPONENTS = {"V3", "V4", "V6"}


def sampled_entities(pair: AlignmentPair, data_dir: str) -> Dict[str, tuple]:
    """
    Draw the fixed sample of sources and of unique candidates of a pair.

    Parameters
    ----------
    pair : AlignmentPair
        The pair.
    data_dir : str
        Dataset and candidate root directory.

    Returns
    -------
    Dict[str, tuple]
        Per role, the graph and the sampled URIs.
    """
    rng = random.Random(SEED)
    candidates = _unique_candidate_uris(pair, load_candidates(pair.dataset_id, pair.pair_name, data_dir))
    pools = {"source": (pair.source_graph, sorted(pair.source_entities)), "candidate": (pair.target_graph, candidates)}
    return {role: (graph, rng.sample(uris, min(SAMPLE, len(uris)))) for role, (graph, uris) in pools.items()}


def role_texts(condition: object, role: str, graph: object, uris: List[str], entity_type: str) -> List[str]:
    """Verbalise sampled entities as the pipeline does for this condition and role."""
    verbaliser = build_verbaliser(condition.strategy_name, condition.model_key, role)
    return [verbaliser.verbalise(graph, URIRef(uri), entity_type) for uri in uris]


def opaque_share(texts: List[str], pieces: object) -> Dict[str, float]:
    """
    Measure the tokens opaque identifiers take in some texts.

    Parameters
    ----------
    texts : List[str]
        Encoded texts.
    pieces : object
        Bare token counter of the condition's model.

    Returns
    -------
    Dict[str, float]
        Total tokens, identifier tokens, their share and the share of texts
        containing an identifier.
    """
    total = sum(pieces(text) for text in texts)
    without = sum(pieces(OPAQUE_ID.sub("", text)) for text in texts)
    return {"tokens": total, "id_tokens": total - without, "id_token_share": (total - without) / max(total, 1),
            "texts_with_id_share": float(np.mean([bool(OPAQUE_ID.search(text)) for text in texts]))}


def shared_identifiers(pair: AlignmentPair) -> Dict[str, int]:
    """
    Count the object identifiers of each graph and those both share.

    Parameters
    ----------
    pair : AlignmentPair
        The D5 pair.

    Returns
    -------
    Dict[str, int]
        ``source``, ``target`` and ``shared`` distinct local names.
    """
    def names(graph):
        return {str(o).rsplit("/", 1)[-1] for o in graph.objects() if isinstance(o, URIRef)}

    source, target = names(pair.source_graph), names(pair.target_graph)
    return {"source": len(source), "target": len(target), "shared": len(source & target)}


def unit_datatypes(pair: AlignmentPair) -> Dict[str, int]:
    """
    Count inlined literal datatypes that are not XSD (units such as currencies).

    Parameters
    ----------
    pair : AlignmentPair
        The D5 pair.

    Returns
    -------
    Dict[str, int]
        Count per non-XSD datatype, both graphs together.
    """
    counts: Counter = Counter()
    for graph in (pair.source_graph, pair.target_graph):
        for obj in graph.objects():
            match = INLINED_DATATYPE.match(str(obj)) if isinstance(obj, Literal) else None
            if match and not match.group(1).startswith(XSD):
                counts[match.group(1).rsplit("/", 1)[-1]] += 1
    return dict(counts.most_common())


def d5_section(data_dir: str) -> dict:
    """
    Measure opaque identifiers in every D5 condition and the graphs' identifiers.

    Parameters
    ----------
    data_dir : str
        Dataset and candidate root directory.

    Returns
    -------
    dict
        Per condition and role, :func:`opaque_share`; shared identifiers; unit datatypes.
    """
    (pair,) = load_dataset("D5", data_dir)
    sample = sampled_entities(pair, data_dir)
    conditions = {}
    for condition in compared_conditions("D5"):
        conditions[condition.condition_id] = {"strategy": condition.strategy_name, "model_key": condition.model_key, **{
            role: opaque_share(role_texts(condition, role, graph, uris, "instance"),
                               encoder_input_counters(condition.model_key, role)[1])
            for role, (graph, uris) in sample.items()}}
    return {"conditions": conditions, "identifiers": shared_identifiers(pair), "unit_datatypes": unit_datatypes(pair)}


def metadata_items(graph: object, uri: str, strategy: str) -> Dict[str, List[str]]:
    """
    Render an entity's metadata-predicate items the way V3, V4 and V6 write them.

    Parameters
    ----------
    graph : object
        The entity's graph.
    uri : str
        The entity.
    strategy : str
        The condition's strategy (V3 sentences or V4/V6 pairs).

    Returns
    -------
    Dict[str, List[str]]
        Rendered items per #145 family, ``type_and_label`` left out.
    """
    renderer, entity = TemplateNLVerbaliser("M3"), URIRef(uri)
    name = renderer.get_label_or_local(graph, entity)
    items: Dict[str, List[str]] = {}
    for _subj, pred, obj in renderer._collect_triples(graph, entity):
        family = exclusion_family(str(pred))
        if family in (None, "type_and_label"):
            continue
        rendered = [renderer._format_pair(graph, pred, obj)]
        if "V3" in strategy:
            rendered.append(renderer._render(graph, name, pred, obj) or "")
        items.setdefault(family, []).extend(item for item in rendered if item)
    return items


def metadata_share(texts: List[str], graph: object, uris: List[str], strategy: str, pieces: object) -> dict:
    """
    Measure the encoded tokens that come from metadata predicates, per family.

    Parameters
    ----------
    texts, uris : List[str]
        Encoded texts and their entities.
    graph : object
        The role's graph.
    strategy : str
        The condition's strategy.
    pieces : object
        Bare token counter of the condition's model.

    Returns
    -------
    dict
        Total tokens, metadata tokens, their share, and tokens per family.
    """
    total, per_family = 0, Counter()
    for text, uri in zip(texts, uris):
        total += pieces(text)
        for family, items in metadata_items(graph, uri, strategy).items():
            per_family[family] += sum(pieces(item) for item in set(items) if item in text)
    metadata = sum(per_family.values())
    return {"tokens": total, "metadata_tokens": metadata, "metadata_token_share": metadata / max(total, 1),
            "per_family_share": {family: n / max(total, 1) for family, n in per_family.most_common()}}


def d4_section(data_dir: str) -> dict:
    """
    Measure metadata-predicate tokens in D4_instance conditions with V3, V4 or V6.

    Parameters
    ----------
    data_dir : str
        Dataset and candidate root directory.

    Returns
    -------
    dict
        Per condition and role, :func:`metadata_share`.
    """
    pair = next(p for p in load_dataset("D4", data_dir) if p.dataset_id == "D4_instance")
    sample = sampled_entities(pair, data_dir)
    out = {}
    for condition in compared_conditions("D4"):
        if not set(condition.strategy_name.split("+")) & KV_COMPONENTS:
            continue
        out[condition.condition_id] = {"strategy": condition.strategy_name, "model_key": condition.model_key, **{
            role: metadata_share(role_texts(condition, role, graph, uris, "instance"), graph, uris,
                                 condition.strategy_name, encoder_input_counters(condition.model_key, role)[1])
            for role, (graph, uris) in sample.items()}}
    return out


def render_markdown(report: dict) -> str:
    """
    Render both measurements as Markdown.

    Parameters
    ----------
    report : dict
        The JSON report.

    Returns
    -------
    str
        The Markdown document.
    """
    d5, d4 = report["d5"], report["d4_instance"]
    ids, units = d5["identifiers"], d5["unit_datatypes"]
    lines = [
        "# Retained text characteristics, quantified (#183)", "",
        f"Generated by `scripts/text_limitations.py` (tokenizer only) on a random sample of {SAMPLE:,} sources "
        f"and {SAMPLE:,} unique candidates per column (seed {SEED}), as each condition encodes them (budgets "
        "applied, D5 datatype suffixes removed). Shares are of the condition model's tokens, special tokens and "
        "prompt prefix excluded.", "",
        "## D5: opaque identifiers", "",
        f"OpenEA deleted D5's labels, so an object without one is written as its local name. The graphs have "
        f"{ids['source']:,} and {ids['target']:,} distinct object identifiers and share **{ids['shared']:,}** of "
        "them, so an identifier cannot match across the graphs.", "",
        "| Cond. | Strategy | Model | Identifier share of tokens (src / cand) | Texts with an identifier (src / cand) |",
        "|---|---|---|---:|---:|",
    ]
    for cid, row in d5["conditions"].items():
        lines.append(f"| {cid} | {row['strategy']} | {row['model_key']} "
                     f"| {row['source']['id_token_share']:.1%} / {row['candidate']['id_token_share']:.1%} "
                     f"| {row['source']['texts_with_id_share']:.1%} / {row['candidate']['texts_with_id_share']:.1%} |")
    unit_list = ", ".join(f"{name} {n:,}" for name, n in units.items())
    lines += ["", f"Literals whose inlined datatype is a unit rather than an XSD type ({sum(units.values()):,}: "
              f"{unit_list}) keep their number but lose the unit when the suffix is removed, as in stage 1.", "",
              "## D4_instance: annotation and page-metadata predicates in V3/V4/V6", "",
              "Predicates that #145 excludes from V8 (every family but `type_and_label`), which V3, V4 and V6 "
              "keep by design.", "",
              "| Cond. | Strategy | Model | Metadata share of tokens (src / cand) | Largest families (cand) |",
              "|---|---|---|---:|---|"]
    for cid, row in d4.items():
        top = ", ".join(f"{f} {s:.1%}" for f, s in list(row["candidate"]["per_family_share"].items())[:3])
        lines.append(f"| {cid} | {row['strategy']} | {row['model_key']} "
                     f"| {row['source']['metadata_token_share']:.1%} / {row['candidate']['metadata_token_share']:.1%} "
                     f"| {top} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Quantify both characteristics and write JSON and Markdown."""
    np.random.seed(42)
    random.seed(42)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data_dir", default="data/")
    parser.add_argument("--output_dir", default="artifacts/issue-183/")
    args = parser.parse_args()
    report = {"sample": SAMPLE, "seed": SEED, "d5": d5_section(args.data_dir), "d4_instance": d4_section(args.data_dir)}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "text_limitations.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "text_limitations.md").write_text(render_markdown(report))


if __name__ == "__main__":
    main()
