"""Tests for the assertion helpers, error simulation, and log capture.

Like test_plugin_usage.py, these talk to the server with raw httpx where
possible so the core suite needs no SDK installs. The error-simulation tests
that assert on client-side exception types are guarded with importorskip so
they run only when the openai SDK is present.
"""
import httpx
import pytest


def _chat(server, text, model="gpt-4"):
    resp = httpx.post(
        f"{server.openai_base_url}/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": text}]},
        timeout=10.0,
    )
    return resp


def _chat_content(server, text):
    resp = _chat(server, text)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# -- assert_request_count -------------------------------------------------


def test_assert_request_count_passes(fakellm):
    _chat_content(fakellm, "one")
    _chat_content(fakellm, "two")
    fakellm.assert_request_count(2)


def test_assert_request_count_fails_with_message(fakellm):
    _chat_content(fakellm, "only one")
    with pytest.raises(AssertionError, match="Expected 5 LLM request"):
        fakellm.assert_request_count(5)


# -- assert_rule_matched --------------------------------------------------


def test_assert_rule_matched(fakellm):
    fakellm.set_config_text(
        """
        version: 1
        rules:
          - name: greet
            when: { messages_contain: "hello" }
            respond: { content: "hi" }
        """
    )
    _chat_content(fakellm, "hello there")
    fakellm.assert_rule_matched("greet")
    fakellm.assert_rule_matched("greet", min_times=1)


def test_assert_rule_matched_counts_fallthrough(fakellm):
    fakellm.set_config_text("version: 1\nrules: []\n")
    _chat_content(fakellm, "nothing matches this")
    # Unmatched requests are bucketed under "<fallthrough>".
    fakellm.assert_rule_matched("<fallthrough>")


def test_assert_rule_matched_fails_for_unknown_rule(fakellm):
    fakellm.set_config_text("version: 1\nrules: []\n")
    _chat_content(fakellm, "x")
    with pytest.raises(AssertionError, match="never_defined"):
        fakellm.assert_rule_matched("never_defined")


# -- tool results ---------------------------------------------------------


def test_assert_tool_results_seen(fakellm):
    fakellm.set_config_text("version: 1\nrules: []\n")
    # A request whose history contains a tool result (OpenAI shape).
    httpx.post(
        f"{fakellm.openai_base_url}/chat/completions",
        json={
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "search please"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "web_search", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "the result"},
            ],
        },
        timeout=10.0,
    ).raise_for_status()
    fakellm.assert_tool_results_seen(1)
    assert fakellm.tool_results_seen() >= 1


def test_tool_results_seen_is_zero_without_tools(fakellm):
    _chat_content(fakellm, "no tools here")
    assert fakellm.tool_results_seen() == 0


# -- error simulation -----------------------------------------------------


def test_set_error_simulation_returns_error_status(fakellm):
    fakellm.set_error_simulation(429, "slow down (mock)")
    resp = _chat(fakellm, "anything")
    assert resp.status_code == 429
    assert "slow down (mock)" in resp.json()["error"]["message"]


def test_set_error_simulation_anthropic_shape(fakellm):
    fakellm.set_error_simulation(503, "unavailable (mock)")
    resp = httpx.post(
        f"{fakellm.anthropic_base_url}/v1/messages",
        json={
            "model": "claude-3",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
        },
        timeout=10.0,
    )
    assert resp.status_code == 503
    body = resp.json()
    assert body["type"] == "error"
    assert "unavailable (mock)" in body["error"]["message"]


def test_set_error_simulation_scoped_by_when(fakellm):
    # Only requests containing "boom" should error; others pass.
    fakellm.set_error_simulation(500, "kaboom", when={"messages_contain": "boom"})
    assert _chat(fakellm, "please boom now").status_code == 500
    assert _chat(fakellm, "stay calm").status_code == 200


def test_set_error_simulation_handles_yaml_metacharacters(fakellm):
    nasty = 'broke: "it", line1\nline2: nested'
    fakellm.set_error_simulation(503, nasty)  # must not corrupt the YAML
    resp = _chat(fakellm, "x")
    assert resp.status_code == 503
    assert nasty in resp.json()["error"]["message"]


def test_set_error_simulation_rejects_non_error_status(fakellm):
    with pytest.raises(ValueError, match=">= 400"):
        fakellm.set_error_simulation(200, "not an error")


# -- error simulation via the SDK (skipped if openai not installed) -------


def test_error_simulation_raises_sdk_exception(fakellm):
    openai = pytest.importorskip("openai")
    fakellm.set_error_simulation(429, "rate limited (mock)")
    client = fakellm.openai_client()
    with pytest.raises(openai.RateLimitError):
        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello"}],
        )
