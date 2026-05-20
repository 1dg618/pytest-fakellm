# pytest-fakellm

Pytest fixtures for [fakellm](https://github.com/dgregor/fakellm), the mock
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
    assert fakellm.request_count >= 1
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

### `FakellmServer` handle

- `openai_client(**kwargs)` / `anthropic_client(**kwargs)` — clients pointed at the server.
- `openai_base_url` / `anthropic_base_url` — raw URLs if you build your own client.
- `set_config_text(yaml)` — write rules inline and reload.
- `load_rules(path)` — load rules from a file and reload.
- `reset()` — clear conversation state (done for you between tests).
- `reload()` — re-read the config from disk.
- `stats()` / `conversations()` — the admin JSON, for assertions.
- `request_count` — convenience count of requests seen.

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
