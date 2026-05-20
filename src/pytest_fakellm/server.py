"""Server lifecycle management and the user-facing control handle.

This module knows how to launch a ``fakellm`` server as a subprocess, wait for
it to become healthy, and talk to its documented admin endpoints
(``/_fakellm/reset``, ``/_fakellm/reload``, ``/_fakellm/stats``). It deliberately
treats fakellm as a black box driven through its public CLI and HTTP surface,
so the plugin does not couple to fakellm's internals.
"""

from __future__ import annotations

import contextlib
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx


def _free_port() -> int:
    """Ask the OS for an unused TCP port, then release it for fakellm to claim.

    There is a small race between releasing the port and fakellm binding it,
    but in practice it is negligible for a test fixture and far simpler than
    handing an open socket to a subprocess.
    """
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


class FakellmServer:
    """A running fakellm instance plus helpers for using it from a test.

    Instances are created by the plugin's fixtures; you normally interact with
    one through the ``fakellm`` fixture rather than constructing it yourself.

    Attributes:
        host: Host the server is bound to.
        port: Port the server is bound to.
        base_url: Root URL of the server, e.g. ``http://127.0.0.1:9999``.
    """

    def __init__(
        self,
        *,
        config: str | Path | None = None,
        host: str = "127.0.0.1",
        port: int | None = None,
        startup_timeout: float = 10.0,
        executable: str | None = None,
    ) -> None:
        self.host = host
        self.port = port or _free_port()
        self._config = Path(config) if config is not None else None
        self._startup_timeout = startup_timeout
        # Launch via the module entry point by default so the plugin works even
        # when the ``fakellm`` console script is not on PATH (e.g. some CI venvs).
        self._executable = executable
        self._proc: subprocess.Popen[bytes] | None = None
        self._client = httpx.Client(base_url=self.base_url, timeout=10.0)

    # -- URLs ---------------------------------------------------------------

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def openai_base_url(self) -> str:
        """Base URL to hand to an OpenAI client (note the ``/v1`` suffix)."""
        return f"{self.base_url}/v1"

    @property
    def anthropic_base_url(self) -> str:
        """Base URL to hand to an Anthropic client."""
        return self.base_url

    # -- lifecycle ----------------------------------------------------------

    def _command(self) -> list[str]:
        if self._executable:
            cmd = [self._executable, "serve"]
        else:
            # `python -m fakellm serve` is the most portable invocation.
            cmd = [sys.executable, "-m", "fakellm", "serve"]
        cmd += ["--port", str(self.port)]
        if self._config is not None:
            cmd += ["--config", str(self._config)]
        return cmd

    def start(self) -> None:
        """Launch the server subprocess and block until it is accepting requests."""
        if self._proc is not None:
            return
        self._proc = subprocess.Popen(
            self._command(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self._wait_until_ready()

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self._startup_timeout
        last_err: Exception | None = None
        while time.monotonic() < deadline:
            # If the process died during startup, surface its output immediately
            # rather than waiting out the full timeout.
            if self._proc is not None and self._proc.poll() is not None:
                output = b""
                if self._proc.stdout is not None:
                    output = self._proc.stdout.read()
                raise RuntimeError(
                    "fakellm server exited during startup "
                    f"(code {self._proc.returncode}). Command: "
                    f"{' '.join(self._command())}\n"
                    f"Output:\n{output.decode(errors='replace')}"
                )
            try:
                resp = self._client.get("/_fakellm/stats")
                if resp.status_code < 500:
                    return
            except httpx.HTTPError as exc:  # not up yet
                last_err = exc
            time.sleep(0.05)
        raise RuntimeError(
            f"fakellm server did not become ready within {self._startup_timeout}s "
            f"at {self.base_url}. Last error: {last_err!r}"
        )

    def stop(self) -> None:
        """Terminate the server subprocess and close the admin client."""
        self._client.close()
        if self._proc is None:
            return
        self._proc.terminate()
        try:
            self._proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait(timeout=5.0)
        self._proc = None

    # -- admin operations ---------------------------------------------------

    def reset(self) -> None:
        """Clear all conversation state (POST ``/_fakellm/reset``).

        Stats and request history are preserved, matching fakellm's documented
        behavior. Called automatically between tests by the ``fakellm`` fixture.
        """
        self._client.post("/_fakellm/reset").raise_for_status()

    def reload(self) -> None:
        """Re-read the YAML config from disk (POST ``/_fakellm/reload``)."""
        self._client.post("/_fakellm/reload").raise_for_status()

    def load_rules(self, config: str | Path) -> None:
        """Point the server at a new config file and reload it.

        This rewrites the path the server was started with by writing the given
        config into place is *not* done here; instead it relies on fakellm's
        ``--config`` having been set to a path the test controls. For ad-hoc
        rules use :meth:`set_config_text`.
        """
        path = Path(config)
        if self._config is None:
            raise RuntimeError(
                "This server was not started with a config path, so load_rules "
                "cannot point it at a file. Use the `fakellm_config` fixture to "
                "set a config path, or call set_config_text()."
            )
        # Copy the provided file's contents into the active config path.
        Path(self._config).write_text(path.read_text())
        self.reload()

    def set_config_text(self, yaml_text: str) -> None:
        """Write raw YAML into the active config file and reload.

        Requires the server to have been started with a config path (the
        ``fakellm_config`` fixture does this for you).
        """
        if self._config is None:
            raise RuntimeError(
                "This server was not started with a config path. Use the "
                "`fakellm_config` fixture (or pass config=...) so there is a "
                "file to write to."
            )
        Path(self._config).write_text(yaml_text)
        self.reload()

    def stats(self) -> dict[str, Any]:
        """Return the server's stats JSON (GET ``/_fakellm/stats``)."""
        resp = self._client.get("/_fakellm/stats")
        resp.raise_for_status()
        return resp.json()

    def conversations(self) -> dict[str, Any]:
        """Return per-conversation turn/tool-result info (GET ``/_fakellm/conversations``)."""
        resp = self._client.get("/_fakellm/conversations")
        resp.raise_for_status()
        return resp.json()

    @property
    def request_count(self) -> int:
        """Convenience accessor: number of requests seen, from stats."""
        data = self.stats()
        # Be tolerant of the exact stats shape; fall back to len(recent).
        for key in ("request_count", "requests", "total_requests"):
            if isinstance(data.get(key), int):
                return data[key]
        recent = data.get("recent_requests") or data.get("recent") or []
        return len(recent) if isinstance(recent, list) else 0

    # -- client helpers -----------------------------------------------------

    def openai_client(self, **kwargs: Any) -> Any:
        """Return an ``openai.OpenAI`` client pointed at this server.

        Requires the ``openai`` package to be installed. Extra kwargs are passed
        through to the ``OpenAI(...)`` constructor.
        """
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on user env
            raise RuntimeError(
                "openai is not installed. Install it (e.g. `pip install openai`) "
                "to use fakellm.openai_client()."
            ) from exc
        kwargs.setdefault("base_url", self.openai_base_url)
        kwargs.setdefault("api_key", "not-used")
        return OpenAI(**kwargs)

    def anthropic_client(self, **kwargs: Any) -> Any:
        """Return an ``anthropic.Anthropic`` client pointed at this server.

        Requires the ``anthropic`` package to be installed.
        """
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover - depends on user env
            raise RuntimeError(
                "anthropic is not installed. Install it (e.g. `pip install "
                "anthropic`) to use fakellm.anthropic_client()."
            ) from exc
        kwargs.setdefault("base_url", self.anthropic_base_url)
        kwargs.setdefault("api_key", "not-used")
        return Anthropic(**kwargs)
