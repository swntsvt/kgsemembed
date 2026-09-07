"""Dataset loading utilities for OAEI-style experiments."""

from kgsemembed.datasets.loader import (
    AlignmentPair,
    EntityPair,
    LoadedDatasetBundle,
    LoadedGraphData,
    extract_class_uris,
    extract_entity_uris,
    extract_instance_uris,
    extract_predicate_uris,
    load_dataset,
    load_graph,
    load_oaei_dataset,
    load_pair_from_dir,
    load_tsv_alignment,
)

__all__ = [
    "AlignmentPair",
    "EntityPair",
    "LoadedGraphData",
    "LoadedDatasetBundle",
    "extract_class_uris",
    "extract_entity_uris",
    "extract_instance_uris",
    "extract_predicate_uris",
    "load_dataset",
    "load_graph",
    "load_oaei_dataset",
    "load_pair_from_dir",
    "load_tsv_alignment",
]
