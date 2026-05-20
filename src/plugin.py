"""pytest-fakellm — pytest fixtures for the fakellm mock LLM server.

Installing this package makes the fixtures below available in any test, with no
imports or registration required (pytest auto-discovers the plugin via the
``pytest11`` entry point declared in pyproject.toml).

Typical use::

    def test_agent(fakellm):
        fakellm.set_config_text(MY_RULES_YAML)
        result = run_my_agent(fakellm.openai_client(), prompt="research fakellm")
        assert "found what you were looking for" in result

Fixtures
--------
fakellm
    Session-scoped server, automatically reset before each test that uses it.
fakellm_openai
    Convenience: a ready ``openai.OpenAI`` client pointed at the server.
fakellm_anthropic
    Convenience: a ready ``anthropic.Anthropic`` client pointed at the server.

Configuration
-------------
The server's starting config file can be set with the ``--fakellm-config``
command-line option or the ``fakellm_config`` ini option. If neither is set, a
temporary empty config is created so that ``set_config_text`` / ``load_rules``
work out of the box.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import pytest

from .server import FakellmServer

__all__ = ["FakellmServer", "fakellm", "fakellm_openai", "fakellm_anthropic"]


# -- options ---------------------------------------------------------------


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("fakellm")
    group.addoption(
        "--fakellm-config",
        action="store",
        default=None,
        help="Path to a fakellm YAML config to start the server with.",
    )
    group.addoption(
        "--fakellm-startup-timeout",
        action="store",
        type=float,
        default=10.0,
        help="Seconds to wait for the fakellm server to become ready.",
    )
    parser.addini(
        "fakellm_config",
        help="Path to a fakellm YAML config to start the server with.",
        default=None,
    )


def _resolve_config(request: pytest.FixtureRequest) -> str | None:
    cli = request.config.getoption("--fakellm-config")
    if cli:
        return cli
    ini = request.config.getini("fakellm_config")
    return ini or None


# -- fixtures --------------------------------------------------------------


@pytest.fixture(scope="session")
def fakellm_server(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[FakellmServer]:
    """Session-scoped, *unreset* server handle.

    Starts one fakellm process for the whole test session and tears it down at
    the end. Most tests should use the ``fakellm`` fixture instead, which layers
    automatic per-test state reset on top of this.
    """
    config = _resolve_config(request)
    if config is None:
        # Give the server a writable config file so set_config_text/load_rules
        # have somewhere to write, even when the user didn't supply one.
        config_path = tmp_path_factory.mktemp("fakellm") / "fakellm.yaml"
        config_path.write_text("version: 1\nrules: []\n")
        config = str(config_path)

    timeout = request.config.getoption("--fakellm-startup-timeout")
    server = FakellmServer(config=config, startup_timeout=timeout)
    server.start()
    try:
        yield server
    finally:
        server.stop()


@pytest.fixture
def fakellm(fakellm_server: FakellmServer) -> Iterator[FakellmServer]:
    """The main fixture: a server handle with fresh state for each test.

    The underlying process is shared across the session (fast), but conversation
    state is cleared before the test body runs, so tests are isolated from one
    another regardless of order.
    """
    fakellm_server.reset()
    yield fakellm_server
    # No teardown reset needed: the next test resets before it runs. Leaving
    # state intact after a failure can also aid debugging.


@pytest.fixture
def fakellm_openai(fakellm: FakellmServer) -> Any:
    """A ready-to-use ``openai.OpenAI`` client pointed at the reset server."""
    return fakellm.openai_client()


@pytest.fixture
def fakellm_anthropic(fakellm: FakellmServer) -> Any:
    """A ready-to-use ``anthropic.Anthropic`` client pointed at the reset server."""
    return fakellm.anthropic_client()
