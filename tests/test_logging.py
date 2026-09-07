import logging
from pathlib import Path

from kgsemembed.utils.logging import RunContext, get_logger, init_logging


def _reset_logger() -> None:
    logger = logging.getLogger("kgsemembed")
    logger.handlers.clear()
    if hasattr(logger, "_configured"):
        delattr(logger, "_configured")
    if hasattr(logger, "_run_id"):
        delattr(logger, "_run_id")


def test_init_logging_creates_file(tmp_path: Path) -> None:
    _reset_logger()
    _, run_id = init_logging(log_dir=tmp_path, run_id="testrun", console=False, file=True)
    assert run_id == "testrun"
    assert (tmp_path / "run_testrun.log").exists()


def test_context_logger_includes_fields(tmp_path: Path) -> None:
    _reset_logger()
    _, run_id = init_logging(log_dir=tmp_path, run_id="run123", console=False, file=True)
    context = RunContext(
        run_id=run_id,
        experiment_name="exp",
        stage="retrieve",
        dataset="d1",
        model="m1",
        device="mps",
    )
    log = get_logger("kgsemembed.test", context)
    log.info("hello")

    log_text = (tmp_path / "run_run123.log").read_text()
    assert "hello" in log_text
    assert "stage=retrieve" in log_text


def test_repeated_init_logging_reuses_active_run_id(tmp_path: Path) -> None:
    _reset_logger()
    _, run_id_1 = init_logging(log_dir=tmp_path, run_id="first", console=False, file=True)
    _, run_id_2 = init_logging(log_dir=tmp_path, run_id="second", console=False, file=True)

    assert run_id_1 == "first"
    assert run_id_2 == "first"
    assert (tmp_path / "run_first.log").exists()
    assert not (tmp_path / "run_second.log").exists()


def test_plain_logger_record_does_not_fail_formatter(tmp_path: Path) -> None:
    _reset_logger()
    init_logging(log_dir=tmp_path, run_id="plain", console=False, file=True)
    plain_logger = logging.getLogger("kgsemembed")
    plain_logger.info("plain message")

    log_text = (tmp_path / "run_plain.log").read_text()
    assert "plain message" in log_text
    assert "run=-" in log_text
    assert "stage=-" in log_text


def test_init_logging_can_enable_handlers_after_disabled_start(tmp_path: Path) -> None:
    _reset_logger()
    init_logging(log_dir=tmp_path, run_id="nohandlers", console=False, file=False)
    _, run_id = init_logging(log_dir=tmp_path, run_id="enabled", console=False, file=True)

    assert run_id == "enabled"
    logging.getLogger("kgsemembed").info("late start")
    log_text = (tmp_path / "run_enabled.log").read_text()
    assert "late start" in log_text


def test_get_logger_honors_debug_level_from_init(tmp_path: Path) -> None:
    _reset_logger()
    init_logging(log_dir=tmp_path, run_id="debug", level="DEBUG", console=False, file=True)
    context = RunContext(
        run_id="debug",
        experiment_name="exp",
        stage="embed",
        dataset="d1",
        model="m1",
        device="cpu",
    )
    log = get_logger("kgsemembed.test.debug", context)
    log.debug("debug message")

    log_text = (tmp_path / "run_debug.log").read_text()
    assert "debug message" in log_text
