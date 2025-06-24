# Continuous Integration

This project uses GitHub Actions for its CI workflow. All jobs now run exclusively on a self-hosted runner, ensuring a consistent environment for linting, tests, and deployment tasks. Workflows are limited to the `main` and `develop` branches and old runs are cancelled automatically via GitHub's concurrency feature.

## Registering a Self-Hosted Runner

1. Open your repository on GitHub and go to **Settings** > **Actions** > **Runners**.
2. Click **New self-hosted runner** and choose the operating system and architecture of your machine.
3. Download the runner package and extract it on the host you want to use.
4. From the runner directory, run:
   ```bash
   ./config.sh --url https://github.com/<OWNER>/<REPO> --token <TOKEN>
   ```
   Replace `<OWNER>` and `<REPO>` with your repository information. The token is generated on the GitHub page and is time limited.
5. Start the runner with `./run.sh` or install it as a service using `./svc.sh install` followed by `./svc.sh start`.

Once the runner is online, update `.github/workflows/ci.yml` to use the self-hosted runner by setting:

```yaml
runs-on: ["self-hosted", "linux"]
```

## Runner Maintenance

- Ensure the service remains active by periodically checking `./svc.sh status`.
- Keep the operating system and dependencies up to date.
- Monitor disk space and clean up old build artifacts.

## Local Test Workflow

To replicate the CI steps locally or on your self-hosted runner:

1. Install the pinned dependencies:
   ```bash
   pip install -r requirements-ci.txt
   ```
2. Start a NATS server with JetStream enabled:
   ```bash
   ./scripts/start_nats.sh
   ```
3. Initialize the required JetStream streams:
   ```bash
   python setup_jetstream.py
   ```
4. Run the tests with the project root on `PYTHONPATH`:
   ```bash
   PYTHONPATH=src pytest
   ```

Tests that require NATS will be skipped automatically if the server isn't
running, but full coverage requires JetStream to be available.

## Detecting Code Changes

The CI workflow relies on `scripts/check_code_changes.py` to decide when
linters and tests should run. The script examines the commit diff and returns
`true` only when a non-comment code line has changed. Documentation files and
blank or commented lines are ignored. When the output is `false`, the workflow
skips the expensive steps.

## Example: codex_setup.sh

To mirror the CI process locally, execute:

```bash
./scripts/codex_setup.sh
```

This script installs dependencies, starts a temporary NATS server, initializes
JetStream, and runs the checks above only when code has actually changed.
