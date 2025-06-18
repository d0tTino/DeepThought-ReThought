# Continuous Integration

This project uses GitHub Actions for its CI workflow. By default, jobs run on GitHub-hosted runners. If you want to use your own machine, you can register a self-hosted runner and update the workflow accordingly.

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

Once the runner is online, you can change `.github/workflows/ci.yml` to target it by setting the job's `runs-on` value to:

```yaml
runs-on: ["self-hosted", "linux"]
```

