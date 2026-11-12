"""Shared pytest configuration.

Registers the ``slow`` and ``integration`` markers, and deselects slow
integration tests by default. Run them explicitly with ``pytest --runslow``.
"""

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "slow: slow integration test (deselected by default; run with --runslow)",
    )
    config.addinivalue_line(
        "markers",
        "integration: tests requiring real data/ files on disk",
    )


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run slow integration tests",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture
def word_budget():
    """Build a token budget that counts whitespace-separated words (#183).

    Budgeted verbalisers take their budget from the model's tokenizer; tests
    replace it with this one so a budget of ``n`` keeps exactly ``n`` words.
    """
    from kgsemembed.verbalisation.budget import TokenBudget

    def words(text: str) -> int:
        return len(text.split())

    return lambda tokens: TokenBudget(tokens, words, words)
