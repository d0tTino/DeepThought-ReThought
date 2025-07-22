# Contributing to DeepThought-ReThought

Welcome to DeepThought-ReThought! We appreciate your interest in contributing. These guidelines are here to help you get started.

## Reporting Issues

If you encounter a bug, have a feature request, or find an issue with the documentation, please report it using GitHub Issues.

When reporting an issue, please include:
*   A clear and descriptive title.
*   Steps to reproduce the issue, if applicable.
*   What you expected to happen and what actually happened.
*   Your environment details (e.g., OS, Python version, NATS version) if relevant.

## Pull Requests

We welcome contributions via Pull Requests (PRs). To make a PR:

1.  **Fork the repository** to your own GitHub account.
2.  **Create a new branch** for your changes (e.g., `feature/my-new-feature` or `fix/issue-123`).
3.  **Make your changes** in your branch.
4.  **Install development dependencies** so that local checks mirror CI:
    ```bash
    pip install -r requirements-ci.txt
    ```
    This installs `flake8`, `pytest`, and other tools pinned to the same versions
    used in the continuous integration workflow.
5.  **Ensure tests pass** if your changes affect code that is covered by tests. (See `README.md` for how to run tests).
6.  **Maintain test coverage.** The CI workflow fails if coverage drops below 80%, so run `pytest --cov --cov-fail-under=80` locally to verify.
7.  **Keep your PRs focused.** Submit separate PRs for separate features or fixes.
8.  **Write a clear commit message** for your changes.
9.  **Push your changes** to your fork and then **submit a Pull Request** to the main DeepThought-ReThought repository.
10.  Provide a clear description of your PR, explaining the changes and why they are being made.

## Code Style

*   **Python:** Follow PEP 8 guidelines.
*   Use [`pre-commit`](https://pre-commit.com/) for lint checks. See
    [Pre-commit Hooks](#pre-commit-hooks) for setup instructions.
*   More detailed style guides or linters may be introduced later. For now, aim for clarity and consistency with the existing codebase.

## Pre-commit Hooks

Running the pre-commit hooks locally mirrors the checks performed in CI. Install
the tool and set up the hooks:

```bash
pip install pre-commit
pre-commit install
pre-commit run --files <changed files>
```

To replicate the exact CI process, including environment setup and conditional
test execution, use the helper script:

```bash
./scripts/codex_setup.sh
```

## Questions

If you have questions about the project, how to use it, or how to contribute, please feel free to open an issue on GitHub with the `question` label.

Thank you for contributing!
