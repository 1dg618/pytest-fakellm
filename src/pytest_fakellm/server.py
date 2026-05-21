"""Server lifecycle management and the user-facing control handle.

This module knows how to launch a ``fakellm`` server as a subprocess, wait for
it to become healthy, and talk to its documented admin endpoints
(``/_fakellm/reset``, ``/_fakellm/reload``, ``/_fakellm/stats``). It deliberately
treats fakellm as a black box driven through its public CLI and HTTP surface,
so the plugin does not couple to fakellm's internals.
"""

from __future__ import annotations

import contextlib
import os
import shutil
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
        # Snapshot the starting config so per-test isolation can restore it.
        # set_config_text()/set_error_simulation()/load_rules() overwrite the
        # active config file, and reset() only clears conversation state (it
        # does not reload the config), so without this a config change in one
        # test would leak into later tests. restore_original_config() undoes it.
        self._original_config_text: str | None = None
        if self._config is not None:
            try:
                self._original_config_text = self._config.read_text()
            except OSError:
                self._original_config_text = None
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

    def _command(self) -> tuple[list[str], dict[str, str]]:
        """Build the launch argv and any extra environment variables.

        Returns ``(argv, env_overrides)``. Three launch strategies, in order:

        1. An explicit ``executable`` (used as ``<exe> serve --port ... [--config ...]``).
        2. A ``fakellm`` console script on PATH, if one exists.
        3. Otherwise, run fakellm's ASGI app directly with uvicorn
           (``python -m uvicorn fakellm.server:app``). This is the reliable
           fallback: as of fakellm 0.3.x the published wheel ships no console
           script and the package has no ``__main__``, so ``python -m fakellm``
           does not work. The uvicorn entrypoint takes the config through the
           ``FAKELLM_CONFIG`` environment variable rather than ``--config``,
           which is how fakellm's own CLI passes it.
        """
        env: dict[str, str] = {}

        if self._executable:
            cmd = [self._executable, "serve", "--port", str(self.port)]
            if self._config is not None:
                cmd += ["--config", str(self._config)]
            return cmd, env

        script = shutil.which("fakellm")
        if script:
            cmd = [script, "serve", "--port", str(self.port)]
            if self._config is not None:
                cmd += ["--config", str(self._config)]
            return cmd, env

        # Fallback: drive the ASGI app with uvicorn the same way fakellm's CLI
        # does. Config is supplied via the environment, not a CLI flag.
        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "fakellm.server:app",
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]
        if self._config is not None:
            env["FAKELLM_CONFIG"] = str(self._config)
        return cmd, env

    def start(self) -> None:
        """Launch the server subprocess and block until it is accepting requests."""
        if self._proc is not None:
            return
        argv, env_overrides = self._command()
        env = {**os.environ, **env_overrides} if env_overrides else None
        self._proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
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
                argv, _ = self._command()
                raise RuntimeError(
                    "fakellm server exited during startup "
                    f"(code {self._proc.returncode}). Command: "
                    f"{' '.join(argv)}\n"
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

    def restore_original_config(self) -> None:
        """Rewrite the config file with its starting contents and reload.

        Undoes any ``set_config_text`` / ``set_error_simulation`` / ``load_rules``
        done during a test. The ``fakellm`` fixture calls this before each test
        so config changes don't leak between tests (``reset()`` only clears
        conversation state, not the config). A no-op if the server was started
        without a config path or its original contents couldn't be read.
        """
        if self._config is None or self._original_config_text is None:
            return
        self._config.write_text(self._original_config_text)
        self.reload()

    def load_rules(self, config: str | Path) -> None:
        """Point the server at a new config file and reload it."""
        path = Path(config)
        if self._config is None:
            raise RuntimeError(
                "This server was not started with a config path, so load_rules "
                "cannot point it at a file. Use the `fakellm_config` fixture to "
                "set a config path, or call set_config_text()."
            )
        Path(self._config).write_text(path.read_text())
        self.reload()

    def set_config_text(self, yaml_text: str) -> None:
        """Write raw YAML into the active config file and reload."""
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
        """Total requests the server has seen this session, from stats.

        fakellm reports this as ``total_requests`` and it is **cumulative for
        the whole server process** — ``reset()`` does *not* zero it (fakellm
        preserves stats across resets, and exposes no way to clear them). For
        per-test counting use :attr:`requests_since_reset` or
        :meth:`assert_request_count`, which both measure from a baseline the
        ``fakellm`` fixture records at the start of each test.

        The extra keys are tolerated only as a fallback in case the stats shape
        ever changes.
        """
        data = self.stats()
        for key in ("total_requests", "request_count", "requests"):
            if isinstance(data.get(key), int):
                return data[key]
        recent = data.get("recent") or data.get("recent_requests") or []
        return len(recent) if isinstance(recent, list) else 0

    def mark_request_baseline(self) -> None:
        """Record current request totals as the baseline for delta counting.

        Called by the ``fakellm`` fixture at the start of each test so that
        :attr:`requests_since_reset`, :meth:`assert_request_count`, and
        :meth:`assert_rule_matched` count only what happened during that test,
        despite fakellm's stats (both ``total_requests`` and ``by_rule``) being
        session-cumulative.
        """
        data = self.stats()
        total = data.get("total_requests")
        self._request_baseline = total if isinstance(total, int) else self.request_count
        by_rule = data.get("by_rule", {})
        self._by_rule_baseline = dict(by_rule) if isinstance(by_rule, dict) else {}

    @property
    def requests_since_reset(self) -> int:
        """Requests seen since the last :meth:`mark_request_baseline` call.

        This is the per-test count. If no baseline was recorded (e.g. you are
        driving the server outside the fixture), the baseline is treated as 0,
        so this equals :attr:`request_count`.
        """
        baseline = getattr(self, "_request_baseline", 0)
        return self.request_count - baseline

    def _rule_matches_since_reset(self, rule_name: str) -> int:
        """How many times ``rule_name`` matched since the baseline."""
        current = self.stats().get("by_rule", {}).get(rule_name, 0)
        baseline = getattr(self, "_by_rule_baseline", {}).get(rule_name, 0)
        return current - baseline

    # -- FEATURE 1: Assertion Helpers ----------------------------------------

    def assert_request_count(self, expected: int) -> None:
        """Assert exactly ``expected`` LLM requests were made *during this test*.

        This measures the delta from the baseline the ``fakellm`` fixture records
        at the start of each test (via :meth:`mark_request_baseline`), not the
        server's session-cumulative ``total_requests`` — fakellm never zeroes
        that, so an absolute count would only be correct for the first test in a
        session. Outside the fixture, with no baseline, this equals the absolute
        count.
        """
        actual = self.requests_since_reset
        if actual != expected:
            raise AssertionError(
                f"Expected {expected} LLM request(s) during this test, but "
                f"{actual} were made."
            )

    def assert_rule_matched(self, rule_name: str, min_times: int = 1) -> None:
        """Assert a named config rule matched at least ``min_times`` *this test*.

        fakellm tracks per-rule match counts (the ``by_rule`` map in
        ``/_fakellm/stats``); requests that matched no rule are counted under
        ``"<fallthrough>"``. Like :meth:`assert_request_count`, this measures the
        delta from the per-test baseline, since ``by_rule`` is also cumulative
        across the session.
        """
        actual = self._rule_matches_since_reset(rule_name)
        if actual < min_times:
            by_rule = self.stats().get("by_rule", {})
            known = ", ".join(sorted(by_rule)) or "(none yet)"
            raise AssertionError(
                f"Expected rule {rule_name!r} to match at least {min_times} "
                f"request(s) this test, but it matched {actual}. Rules seen "
                f"this session: {known}."
            )

    def assert_tool_results_seen(self, min_results: int = 1) -> None:
        """Assert the server observed at least ``min_results`` tool results.

        Summed across all active conversations.

        Important:
            fakellm records only a *count* of tool results per conversation
            (``tool_results_seen``); it does not retain or expose tool *names*.
            So this can confirm that an agent fed tool output back to the model,
            but it cannot tell you *which* tool. There is deliberately no
            ``assert_tool_called(name)`` helper, because the server provides no
            data to implement it against — see :meth:`tool_results_seen`.
        """
        actual = self.tool_results_seen()
        if actual < min_results:
            raise AssertionError(
                f"Expected at least {min_results} tool result(s) to have been "
                f"seen by the server, but found {actual}."
            )

    def tool_results_seen(self) -> int:
        """Total tool results seen across all conversations (from the server).

        Reads ``tool_results_seen`` for every conversation in
        ``/_fakellm/conversations`` and sums them. Returns 0 if the payload has
        no recognizable per-conversation entries.
        """
        total = 0
        for info in self.conversations().values():
            if isinstance(info, dict) and isinstance(info.get("tool_results_seen"), int):
                total += info["tool_results_seen"]
        return total


    # -- FEATURE 2: Error Simulation -----------------------------------------

    def set_error_simulation(
        self,
        status: int,
        error_message: str = "Simulated error",
        *,
        when: dict[str, Any] | None = None,
        name: str = "fakellm_error_simulation",
    ) -> None:
        """Make the server respond with an HTTP error for matching requests.

        Useful for exercising an agent's retry/back-off and error-handling paths.

        Args:
            status: The HTTP status to return (must be >= 400 for fakellm to
                treat it as an error; lower values produce a normal response).
            error_message: The error string placed in the response body. It is
                emitted via a YAML serializer, so quotes, colons, newlines, and
                other metacharacters are escaped correctly rather than breaking
                or injecting structure into the config.
            when: Optional fakellm ``when:`` matcher dict (e.g.
                ``{"messages_contain": "search"}``). Omit to match every request.
            name: The rule name (surfaces in the server's stats/dashboard).

        This builds the rule in fakellm's actual config schema, verified against
        fakellm 0.3.x: a rule with a ``when`` matcher and a ``respond`` block
        carrying ``status`` and a string ``error``. (An earlier draft used a
        top-level ``prompt`` key and a nested ``error`` object; fakellm silently
        ignores unknown top-level keys, so that produced a normal 200 response
        instead of an error — the failure this signature avoids.)

        The reload performed here raises if the server rejects the config.
        """
        if not isinstance(status, int):
            raise TypeError("status must be an int")
        if status < 400:
            raise ValueError(
                f"status must be >= 400 to simulate an error; got {status}. "
                "fakellm only treats responses with status >= 400 as errors."
            )

        rule: dict[str, Any] = {
            "name": name,
            "when": when or {},
            "respond": {"status": status, "error": error_message},
        }
        config: dict[str, Any] = {"version": 1, "rules": [rule]}
        self.set_config_text(_dump_yaml(config))

    # -- client helpers -----------------------------------------------------

    def openai_client(self, **kwargs: Any) -> Any:
        """Return an ``openai.OpenAI`` client pointed at this server."""
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai is not installed. Install it (e.g. `pip install openai`) "
                "to use fakellm.openai_client()."
            ) from exc
        kwargs.setdefault("base_url", self.openai_base_url)
        kwargs.setdefault("api_key", "not-used")
        return OpenAI(**kwargs)

    def anthropic_client(self, **kwargs: Any) -> Any:
        """Return an ``anthropic.Anthropic`` client pointed at this server."""
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise RuntimeError(
                "anthropic is not installed. Install it (e.g. `pip install "
                "anthropic`) to use fakellm.anthropic_client()."
            ) from exc
        kwargs.setdefault("base_url", self.anthropic_base_url)
        kwargs.setdefault("api_key", "not-used")
        return Anthropic(**kwargs)


# -- module helpers --------------------------------------------------------


def _dump_yaml(data: Any) -> str:
    """Serialize ``data`` to YAML, preferring PyYAML and falling back to JSON.

    YAML is a superset of JSON, so a JSON document is always valid YAML. This
    keeps error simulation working even in minimal environments without PyYAML,
    while still escaping metacharacters correctly in both cases.
    """
    try:
        import yaml  # type: ignore

        return yaml.safe_dump(data, sort_keys=False)
    except ImportError:
        import json

        return json.dumps(data)
