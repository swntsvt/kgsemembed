"""Centralized logging utilities for kgsemembed."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, cast

Stage = str


@dataclass(frozen=True)
class RunContext:
    run_id: str
    experiment_name: str
    stage: Stage
    dataset: str
    model: str
    device: str


class ContextLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        extra = cast(dict[str, Any], kwargs.setdefault("extra", {}))
        adapter_extra = cast(dict[str, Any], self.extra or {})
        context = cast(Optional[RunContext], adapter_extra.get("context"))
        if context is not None:
            extra.setdefault("run_id", context.run_id)
            extra.setdefault("stage", context.stage)
            extra.setdefault("experiment", context.experiment_name)
            extra.setdefault("dataset", context.dataset)
            extra.setdefault("model", context.model)
            extra.setdefault("device", context.device)
        return msg, kwargs


class DefaultContextFilter(logging.Filter):
    """Populate default context fields so plain log records do not break formatting."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "run_id"):
            record.run_id = "-"
        if not hasattr(record, "stage"):
            record.stage = "-"
        if not hasattr(record, "experiment"):
            record.experiment = "-"
        if not hasattr(record, "dataset"):
            record.dataset = "-"
        if not hasattr(record, "model"):
            record.model = "-"
        if not hasattr(record, "device"):
            record.device = "-"
        return True


def _formatter() -> logging.Formatter:
    fmt = (
        "%(asctime)s | %(levelname)s | %(name)s | run=%(run_id)s | "
        "stage=%(stage)s | %(message)s"
    )
    return logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S")


def init_logging(
    log_dir: Path = Path("logs"),
    level: str = "INFO",
    run_id: Optional[str] = None,
    console: bool = True,
    file: bool = True,
) -> tuple[logging.Logger, str]:
    logger = logging.getLogger("kgsemembed")
    resolved_run_id = run_id or uuid.uuid4().hex[:8]
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    has_handlers = bool(logger.handlers)

    if getattr(logger, "_configured", False) and has_handlers:
        active_run_id = cast(Optional[str], getattr(logger, "_run_id", None))
        return logger, active_run_id or resolved_run_id

    logger.propagate = False
    formatter = _formatter()
    context_filter = DefaultContextFilter()

    if console:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        ch.addFilter(context_filter)
        logger.addHandler(ch)

    if file:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / f"run_{resolved_run_id}.log")
        fh.setFormatter(formatter)
        fh.addFilter(context_filter)
        logger.addHandler(fh)

    setattr(logger, "_configured", has_handlers or bool(logger.handlers))
    setattr(logger, "_run_id", resolved_run_id)
    return logger, resolved_run_id


def get_logger(name: str, context: Optional[RunContext] = None) -> logging.LoggerAdapter:
    base = logging.getLogger(name)
    base_adapter = ContextLoggerAdapter(
        base,
        {
            "context": context,
        },
    )
    return base_adapter
