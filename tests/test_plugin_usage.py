"""Tests that use the plugin exactly as an end user would.

No imports from pytest_fakellm here on purpose — the `fakellm` fixture must be
auto-discovered via the entry point, proving the plugin is wired correctly.
We talk to the server with raw httpx (rather than the openai SDK) so the suite
needs no extra SDK installs.
"""
import httpx


RULES = """
version: 1
rules:
  - name: summarize
    when: { messages_contain: "research" }
    respond: { content: "I found what you were looking for." }
"""


def _chat(server, text):
    """Make an OpenAI-style chat call against the server using raw httpx."""
    resp = httpx.post(
        f"{server.openai_base_url}/chat/completions",
        json={"model": "gpt-4", "messages": [{"role": "user", "content": text}]},
        timeout=10.0,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def test_fixture_is_autodiscovered(fakellm):
    # If this argument resolves at all, the entry point worked.
    assert fakellm.base_url.startswith("http://127.0.0.1:")


def test_inline_config_drives_response(fakellm):
    fakellm.set_config_text(RULES)
    content = _chat(fakellm, "please research fakellm")
    assert content == "I found what you were looking for."


def test_request_count_increments(fakellm):
    start = fakellm.request_count
    _chat(fakellm, "hello")
    _chat(fakellm, "hello again")
    assert fakellm.request_count == start + 2


def test_state_is_reset_between_tests_part1(fakellm):
    # Make some requests; the NEXT test should not see this conversation state.
    _chat(fakellm, "first test traffic")
    convs = fakellm.conversations()
    # Stub tracks conversations dict; just assert the call works and is JSON.
    assert isinstance(convs, dict)


def test_state_is_reset_between_tests_part2(fakellm):
    # Because `fakellm` resets before each test, conversation state is empty
    # even though the previous test made requests.
    convs = fakellm.conversations()
    assert convs == {}


def test_raw_base_urls_are_exposed(fakellm):
    assert fakellm.openai_base_url.endswith("/v1")
    assert fakellm.anthropic_base_url == fakellm.base_url


def test_default_config_returns_something(fakellm):
    # With no rules set this run, the stub returns its default content.
    fakellm.set_config_text("version: 1\nrules: []\n")
    content = _chat(fakellm, "anything")
    assert isinstance(content, str) and content        # got some response
    assert "mock response" in content                  # it's the fallback echo
