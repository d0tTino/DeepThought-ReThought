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
linters and tests should run.

### How `check_code_changes.py` Works

1. The script compares the current commit with the base reference provided by
   `GITHUB_BASE_REF` and `GITHUB_SHA`.
2. If those variables are missing or the merge base cannot be determined, it
   falls back to diffing against the previous commit (`HEAD^`).
3. If every changed file is documentation (either under `docs/` or with a
   common doc extension) it prints `false` and exits.
4. Otherwise it inspects the unified diff, ignoring blank and commented lines
   (`#`, `//`, `/*`, `*`, or triple-quoted blocks). If a real code line changed
   it prints `true`; if not, `false`.

The workflow runs expensive steps only when the output is `true`.

## Example: codex_setup.sh

Run the helper script from the repository root to mimic the CI workflow
locally:

```bash
./scripts/codex_setup.sh
```

The script installs dependencies, spins up a temporary NATS server, initializes
JetStream, and then executes the linting and test steps only when
`check_code_changes.py` reports that code has changed. The NATS container is
removed automatically when the run completes.

## Launching the Metrics Dashboard

The dashboard script aggregates metrics files from replay runs or training sessions and visualizes BLEU, ROUGE-L and latency trends.

1. Install the optional dependency:
   ```bash
   pip install matplotlib
   ```
2. Run the dashboard pointing to one or more metrics files or a directory containing them:
   ```bash
   python tools/dashboard.py path/to/metrics --show
   ```
   The plot is also saved as `dashboard.png` in the current directory.
