"""Pipeline entrypoint scaffold for Phase 2 experiments."""

from __future__ import annotations

import time
from pathlib import Path

import hydra
from omegaconf import DictConfig

from kgsemembed.candidates import build_candidate_table, generate_candidates
from kgsemembed.datasets import load_oaei_dataset
from kgsemembed.utils.errors import DataError
from kgsemembed.utils.logging import RunContext, get_logger, init_logging

_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"


def run_experiment(cfg: DictConfig) -> int:
    logger, run_id = init_logging(level=str(cfg.experiment.log_level))
    context = RunContext(
        run_id=run_id,
        experiment_name=str(cfg.experiment.name),
        stage="ingest",
        dataset=str(cfg.dataset.name),
        model=str(cfg.model.name),
        device=str(cfg.model.device),
    )
    log = get_logger("kgsemembed.pipeline", context)
    logger.info("Pipeline bootstrap complete", extra={"run_id": run_id, "stage": "init"})
    log.info(
        "Resolved config: experiment=%s model=%s dataset=%s verbalisation=%s candidates=%s(n=%s,metric=%s,k=%s) device=%s ks=%s",
        cfg.experiment.name,
        cfg.model.name,
        cfg.dataset.name,
        cfg.verbalisation.strategy,
        cfg.candidates.method,
        cfg.candidates.n,
        cfg.candidates.metric,
        cfg.candidates.top_k,
        cfg.model.device,
        list(cfg.experiment.evaluation_ks),
    )

    try:
        bundle = load_oaei_dataset(
            source_path=cfg.dataset.source_rdf,
            target_path=cfg.dataset.target_rdf,
            alignment_path=cfg.dataset.alignment_rdf,
        )
    except DataError:
        raise
    except Exception as exc:
        raise DataError(f"Dataset loading failed: {exc}") from exc

    log.info(
        "Loaded source graph: format=%s entities=%d triples=%d",
        bundle.source.format,
        len(bundle.source.entities),
        len(bundle.source.triples),
    )
    log.info(
        "Loaded target graph: format=%s entities=%d triples=%d",
        bundle.target.format,
        len(bundle.target.entities),
        len(bundle.target.triples),
    )
    log.info(
        "Loaded alignment graph: format=%s entities=%d triples=%d",
        bundle.alignment.format,
        len(bundle.alignment.entities),
        len(bundle.alignment.triples),
    )

    t0 = time.perf_counter()
    candidates = generate_candidates(
        source_labels=bundle.source.labels,
        target_labels=bundle.target.labels,
        n=int(cfg.candidates.n),
        metric=str(cfg.candidates.metric),
        top_k=int(cfg.candidates.top_k),
        strip_punctuation=bool(cfg.candidates.strip_punctuation),
    )
    elapsed = time.perf_counter() - t0
    table = build_candidate_table(candidates)

    log.info(
        "Generated candidates: source=%d target=%d pairs=%d metric=%s n=%d top_k=%d runtime_sec=%.4f",
        len(bundle.source.labels),
        len(bundle.target.labels),
        len(candidates),
        cfg.candidates.metric,
        int(cfg.candidates.n),
        int(cfg.candidates.top_k),
        elapsed,
    )

    if bool(cfg.candidates.persist):
        out_dir = Path(str(cfg.experiment.output_dir))
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / str(cfg.candidates.output_file)
        table.to_csv(out_path, index=False)
        log.info("Persisted candidate pairs to %s", out_path)

    return 0


@hydra.main(version_base=None, config_path=str(_CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    raise SystemExit(run_experiment(cfg))


if __name__ == "__main__":
    main()
