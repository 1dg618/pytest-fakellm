"""Tests for the fakellm_logs fixture.

The fixture only does anything when a test *fails*, so the tests that exercise
it must themselves fail. We use @pytest.mark.xfail(strict=True): the test is
expected to fail, and the run stays green. If the body ever stops failing,
strict=True turns that into a failure, so the guard can't silently rot.

We can't assert on the captured report text from inside the failing test
itself (the capture happens at teardown, after the body). These tests therefore
verify the mechanism end-to-end by failing on purpose; the manual/CI check that
logs actually appear is to run one of them with `-rA` and look for the
"fakellm server logs" section. The non-failing test below verifies the quiet
path: a passing test using the fixture must not error in setup/teardown.
"""
import httpx
import pytest


def _chat(server, text):
    httpx.post(
        f"{server.openai_base_url}/chat/completions",
        json={"model": "gpt-4", "messages": [{"role": "user", "content": text}]},
        timeout=10.0,
    ).raise_for_status()


def test_logs_fixture_quiet_on_pass(fakellm, fakellm_logs):
    # The fixture must be harmless on a passing test (no logs dumped, no error).
    _chat(fakellm, "passing traffic")
    assert True


@pytest.mark.xfail(strict=True, reason="exercises fakellm_logs dump-on-failure")
def test_logs_fixture_dumps_on_failure(fakellm, fakellm_logs):
    # Generate a recognizable server log line, then fail. On failure the
    # fakellm_logs fixture attaches the server's output to the report.
    _chat(fakellm, "FAILING_TRAFFIC_MARKER")
    assert False, "intentional failure to exercise fakellm_logs"
