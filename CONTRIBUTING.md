# Contributing to pytest-fakellm

Thanks for your interest in contributing! This document covers how to set up a
development environment, run the tests, and submit changes.

## Getting started

1. Fork the repository and clone your fork:

   ```bash
   git clone https://github.com/YOUR_USERNAME/pytest-fakellm.git
   cd pytest-fakellm
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

3. Install the package in editable mode with the development extras. This pulls
   in the `openai` and `anthropic` clients used by the test suite:

   ```bash
   pip install -e ".[dev]"
   ```

## Running the tests

The test suite uses pytest and lives in the `tests/` directory:

```bash
pytest
```

To run a single test file or test:

```bash
pytest tests/test_plugin_usage.py
pytest tests/test_plugin_usage.py::test_some_specific_case -v
```

Because this package *is* a pytest plugin, its own fixtures (`fakellm`,
`fakellm_openai`, `fakellm_anthropic`) are available automatically once the
package is installed in editable mode — no imports or `conftest.py` setup
needed.

## Making changes

- Keep changes focused. One logical change per pull request makes review easier.
- Add or update tests for any behavior you change. New fixtures or options
  should come with a test that exercises them.
- Update the `README.md` and docstrings if you change public behavior.
- Follow the existing code style. The project targets Python 3.10+ and uses
  type hints throughout.

## Submitting a pull request

1. Create a branch for your change:

   ```bash
   git checkout -b my-change
   ```

2. Commit with a clear message describing what changed and why.

3. Push to your fork and open a pull request against `main`.

4. In the PR description, explain the motivation and link any related issues.

The maintainer will review as soon as they can. Continuous, friendly
back-and-forth on a PR is normal — it's how we get the change right together.

## Reporting bugs and requesting features

Open an issue on the
[issue tracker](https://github.com/1dg618/pytest-fakellm/issues). For bugs,
please include:

- What you expected to happen and what actually happened
- A minimal test or code snippet that reproduces the problem
- Your Python version and the versions of `pytest-fakellm` and `fakellm`
  (`pip show pytest-fakellm fakellm`)

## Code of Conduct

By participating in this project, you agree to abide by its Code of Conduct.
Please be respectful and constructive in all interactions.

## License

By contributing, you agree that your contributions will be licensed under the
same [MIT License](LICENSE) that covers the project.
