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
6.  **Keep your PRs focused.** Submit separate PRs for separate features or fixes.
7.  **Write a clear commit message** for your changes.
8.  **Push your changes** to your fork and then **submit a Pull Request** to the main DeepThought-ReThought repository.
9.  Provide a clear description of your PR, explaining the changes and why they are being made.

## Code Style

*   **Python:** Follow PEP 8 guidelines.
*   Run `flake8` to check style before submitting a PR:
    ```bash
    flake8 src tests
    ```
    The configuration lives in [.flake8](.flake8).
*   Install `pre-commit` if you want flake8 and other checks to run automatically:
    ```bash
    pip install pre-commit
    pre-commit install
    ```
    This installs the hooks defined in
    [.pre-commit-config.yaml](.pre-commit-config.yaml). You can run them
    manually with `pre-commit run --files <changed files>`.
*   More detailed style guides or linters may be introduced later. For now, aim for clarity and consistency with the existing codebase.

## Questions

If you have questions about the project, how to use it, or how to contribute, please feel free to open an issue on GitHub with the `question` label.

Thank you for contributing!
