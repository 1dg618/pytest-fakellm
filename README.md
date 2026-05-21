# pytest-fakellm

Pytest fixtures for [fakellm](https://github.com/1dg618/fakellm), the mock
OpenAI/Anthropic server. Spin up a server, get a clean state per test, and
assert on what your code sent — with zero boilerplate.

```bash
pip install pytest-fakellm
```

Once installed, the fixtures are available automatically — no imports, no
`conftest.py` setup.

## The point

Without the plugin, using fakellm in a test means starting the server, wiring a
client to its URL, resetting state, and tearing it all down yourself, in every
test. With the plugin, that becomes:

```python
def test_agent_handles_search(fakellm):
    fakellm.set_config_text("""
    version: 1
    rules:
      - name: summarize
        when: { messages_contain: "research" }
        respond: { content: "Based on the search, I found what you were looking for." }
    """)
    result = run_my_agent(fakellm.openai_client(), prompt="Please research fakellm")
    assert "found what you were looking for" in result
    fakellm.assert_request_count(1)
```

The server starts once per session, state is reset before each test, and
everything is torn down at the end. You never touch a port number or a
subprocess.

## Fixtures

| Fixture | What you get |
|---|---|
| `fakellm` | A `FakellmServer` handle with fresh conversation state for the test. |
| `fakellm_openai` | A ready `openai.OpenAI` client pointed at the (reset) server. |
| `fakellm_anthropic` | A ready `anthropic.Anthropic` client pointed at the (reset) server. |
| `fakellm_logs` | Opt-in. Dumps the server's output into the failure report **only if the test fails** — handy for debugging without cluttering passing runs. |

### `FakellmServer` handle

Clients and URLs:

- `openai_client(**kwargs)` / `anthropic_client(**kwargs)` — clients pointed at the server.
- `openai_base_url` / `anthropic_base_url` — raw URLs if you build your own client.

Configuring rules:

- `set_config_text(yaml)` — write rules inline and reload.
- `load_rules(path)` — load rules from a file and reload.
- `reset()` — clear conversation state (done for you between tests).
- `reload()` — re-read the config from disk.

Inspecting what happened:

- `stats()` / `conversations()` — the admin JSON, for assertions.
- `request_count` — absolute session total of requests seen.
- `requests_since_reset` — requests made during the current test (per-test count).
- `tool_results_seen()` — total tool results the server observed across all conversations.

Assertions (raise `AssertionError` with a readable message on failure):

- `assert_request_count(expected)` — exactly `expected` requests were seen.
- `assert_rule_matched(rule_name, min_times=1)` — a named config rule matched at least `min_times`.
- `assert_tool_results_seen(min_results=1)` — at least `min_results` tool results were fed back.

Error injection:

- `set_error_simulation(status, error_message="...", *, when=None, name="...")` — make the server return an HTTP error for matching requests.

See [Assertions and error simulation](#assertions-and-error-simulation) for details.

## Assertions and error simulation

### Asserting on traffic

After your code runs, assert on what the server saw:

```python
def test_agent_makes_one_call(fakellm):
    fakellm.set_config_text("""
    version: 1
    rules:
      - name: answer
        when: { messages_contain: "weather" }
        respond: { content: "It is sunny." }
    """)

    run_my_agent(fakellm.openai_client(), prompt="what is the weather?")

    fakellm.assert_request_count(1)
    fakellm.assert_rule_matched("answer")
```

`assert_rule_matched` reads the per-rule match counts the server keeps in
`stats()["by_rule"]`. Requests that matched no rule are counted under
`"<fallthrough>"`, so you can assert on those too.

Both `assert_request_count` and `assert_rule_matched` count only what happened
**during the current test**. fakellm's stats are cumulative for the whole server
process (a `reset()` clears conversations but not stats), so the `fakellm`
fixture records a baseline at the start of each test and these helpers measure
the delta from it. If you want the raw numbers, `request_count` is the absolute
session total and `requests_since_reset` is the per-test count.

### Tool results

If your agent calls a tool and feeds the result back to the model, the server
counts those tool results:

```python
def test_agent_used_a_tool(fakellm):
    run_my_tool_using_agent(fakellm.openai_client(), prompt="search for X")
    fakellm.assert_tool_results_seen(1)
```

**A deliberate limitation worth knowing:** fakellm records only a *count* of
tool results per conversation — it does not retain or expose tool *names*. So
you can confirm that a tool result came back, but not *which* tool produced it.
There is intentionally no `assert_tool_called("search")`, because the server
transmits no data to implement it against. If you need to assert on a specific
tool, match on it in a rule (`when: { tools_include: "search" }`) and then use
`assert_rule_matched` on that rule's name.

### Simulating errors

To exercise your retry/back-off and error-handling paths, make the server
return an HTTP error:

```python
import openai

def test_agent_retries_on_rate_limit(fakellm):
    fakellm.set_error_simulation(429, "slow down")
    client = fakellm.openai_client()

    with pytest.raises(openai.RateLimitError):
        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello"}],
        )
```

`set_error_simulation` works for both the OpenAI and Anthropic endpoints,
emitting the error in each API's native shape. `status` must be `>= 400`
(fakellm only treats those as errors). Pass a `when=` matcher dict to scope the
error to specific requests, e.g. `set_error_simulation(503, "down", when={"messages_contain": "search"})`;
omit it to fail every request. The error message is YAML-serialized safely, so
quotes, colons, and newlines in the message won't corrupt the config.

### Surfacing server logs on failure

Add the `fakellm_logs` fixture to a test and, **if that test fails**, the
server's output for that test is attached to the failure report. Passing tests
stay quiet:

```python
def test_something_tricky(fakellm, fakellm_logs):
    ...
    assert result == expected   # on failure, server logs appear in the report
```

## Configuration

Set a starting config file via the command line:

```bash
pytest --fakellm-config=tests/fixtures/rules.yaml
```

or in `pyproject.toml` / `pytest.ini`:

```toml
[tool.pytest.ini_options]
fakellm_config = "tests/fixtures/rules.yaml"
```

If you don't set one, a temporary empty config is created so `set_config_text`
and `load_rules` work immediately.

`--fakellm-startup-timeout` (default `10.0`) controls how long the fixture waits
for the server to come up.

## Client extras

`openai_client()` and `anthropic_client()` require the respective SDKs. Install
what you need:

```bash
pip install "pytest-fakellm[openai]"      # adds openai
pip install "pytest-fakellm[anthropic]"   # adds anthropic
```

## License

MIT
