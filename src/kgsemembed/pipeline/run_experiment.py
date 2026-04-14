"""Pipeline entrypoint scaffold for Phase 2 experiments."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

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
        "Resolved config: experiment=%s model=%s dataset=%s verbalisation=%s device=%s ks=%s",
        cfg.experiment.name,
        cfg.model.name,
        cfg.dataset.name,
        cfg.verbalisation.strategy,
        cfg.model.device,
        list(cfg.experiment.evaluation_ks),
    )
    return 0


@hydra.main(version_base=None, config_path=str(_CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    raise SystemExit(run_experiment(cfg))


if __name__ == "__main__":
    main()
