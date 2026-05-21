"""pytest-fakellm — pytest fixtures for the fakellm mock LLM server."""

from __future__ import annotations

import threading
from typing import Any, Iterator

import pytest

from .server import FakellmServer

__all__ = [
    "FakellmServer",
    "fakellm",
    "fakellm_openai",
    "fakellm_anthropic",
    "fakellm_logs",
]


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


# -- report hook -----------------------------------------------------------


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo) -> Any:
    """Stash each phase's report on the item so fixtures can read the outcome.

    The ``fakellm_logs`` fixture uses this to decide, at teardown, whether the
    test actually failed (and logs should therefore be emitted).
    """
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"_fakellm_rep_{rep.when}", rep)


# -- log drain -------------------------------------------------------------


class _LogDrain:
    """Continuously drains a process's combined stdout/stderr into a buffer.

    The fakellm server is session-scoped, so its stdout pipe is shared across
    every test. Reading directly from that pipe in a per-test fixture would
    consume bytes other tests/mechanisms expect and risks blocking. Instead a
    single background thread reads the pipe to EOF and appends to an in-memory
    buffer; tests snapshot byte offsets into that buffer to capture only the
    output produced during their own run.
    """

    def __init__(self, server: FakellmServer) -> None:
        self._server = server
        self._buf = bytearray()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        proc = self._server._proc
        if proc is None or proc.stdout is None:
            return
        self._thread = threading.Thread(
            target=self._pump, args=(proc.stdout,), daemon=True
        )
        self._thread.start()

    def _pump(self, stream: Any) -> None:
        # Read line-by-line rather than in fixed-size blocks. The server is
        # long-lived and low-volume, so a block read (stream.read(4096)) would
        # block until 4096 bytes accumulate or the process exits — meaning the
        # buffer would stay empty for the entire session and teardown would
        # capture nothing. readline() returns as each line is flushed, which is
        # exactly the granularity we want for log capture.
        try:
            for line in iter(stream.readline, b""):
                with self._lock:
                    self._buf.extend(line)
        except (ValueError, OSError):
            # Stream closed while we were reading (e.g. server torn down).
            pass

    def offset(self) -> int:
        with self._lock:
            return len(self._buf)

    def slice_from(self, start: int) -> bytes:
        with self._lock:
            return bytes(self._buf[start:])


def _get_drain(server: FakellmServer) -> _LogDrain:
    """Return the (single) log drain for a server, creating it on first use."""
    drain = getattr(server, "_fakellm_log_drain", None)
    if drain is None:
        drain = _LogDrain(server)
        # Stash on the server instance so it is shared across tests, matching
        # the session scope of the underlying process.
        setattr(server, "_fakellm_log_drain", drain)
        drain.start()
    return drain


# -- fixtures --------------------------------------------------------------


@pytest.fixture(scope="session")
def fakellm_server(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[FakellmServer]:
    """Session-scoped, *unreset* server handle."""
    config = _resolve_config(request)
    if config is None:
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

    Before each test this restores the starting config (undoing any rule or
    error-simulation changes a previous test made) and clears conversation
    state, so tests are isolated regardless of order.
    """
    fakellm_server.restore_original_config()
    fakellm_server.reset()
    fakellm_server.mark_request_baseline()
    yield fakellm_server


@pytest.fixture
def fakellm_openai(fakellm: FakellmServer) -> Any:
    """A ready-to-use ``openai.OpenAI`` client pointed at the reset server."""
    return fakellm.openai_client()


@pytest.fixture
def fakellm_anthropic(fakellm: FakellmServer) -> Any:
    """A ready-to-use ``anthropic.Anthropic`` client pointed at the reset server."""
    return fakellm.anthropic_client()


# -- FEATURE 3: Log Capture Fixture ---------------------------------------


@pytest.fixture
def fakellm_logs(
    request: pytest.FixtureRequest, fakellm_server: FakellmServer
) -> Iterator[None]:
    """Dump the fakellm server's output for this test *if the test fails*.

    The background server's combined stdout/stderr is drained continuously by a
    single session-lived thread (see :class:`_LogDrain`). This fixture records
    the buffer offset before the test body and, during teardown, emits only the
    bytes produced while the test ran — and only when the test's call phase
    failed, as determined via the ``pytest_runtest_makereport`` hook above.

    The captured text is attached to the report through pytest's section
    mechanism so it appears under the failure in the terminal report, and is
    also printed so it is picked up by pytest's stdout capture as a fallback.
    """
    drain = _get_drain(fakellm_server)
    start_offset = drain.offset()

    yield

    rep = getattr(request.node, "_fakellm_rep_call", None)
    failed = rep is not None and rep.failed
    if not failed:
        return

    captured = drain.slice_from(start_offset)
    if not captured:
        return

    text = captured.decode(errors="replace")
    # Attach to the report as a named section; it renders under the failure in
    # the terminal report. (We deliberately do not also print() it: that would
    # duplicate the output under "Captured stdout teardown".)
    rep.sections.append(("fakellm server logs", text))
