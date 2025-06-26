import subprocess
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "check_code_changes.py"


def _init_repo(repo_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=repo_path, check=True)
    subprocess.run(["git", "config", "user.email", "tester@example.com"], cwd=repo_path, check=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=repo_path, check=True)
    (repo_path / "code.py").write_text("print('hi')\n")
    subprocess.run(["git", "add", "code.py"], cwd=repo_path, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo_path, check=True)


def _run_script(repo_path: Path) -> str:
    result = subprocess.run(["python", str(SCRIPT_PATH)], cwd=repo_path, stdout=subprocess.PIPE, text=True, check=True)
    return result.stdout.strip()


def test_only_docs_change(tmp_path):
    repo = tmp_path / "repo_docs"
    repo.mkdir()
    _init_repo(repo)
    (repo / "README.md").write_text("docs\n")
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "docs"], cwd=repo, check=True)
    assert _run_script(repo) == "false"


def test_only_comment_change(tmp_path):
    repo = tmp_path / "repo_comment"
    repo.mkdir()
    _init_repo(repo)
    (repo / "code.py").write_text("print('hi')\n# comment\n")
    subprocess.run(["git", "add", "code.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "comment"], cwd=repo, check=True)
    assert _run_script(repo) == "false"


def test_actual_code_change(tmp_path):
    repo = tmp_path / "repo_code"
    repo.mkdir()
    _init_repo(repo)
    (repo / "code.py").write_text("print('hi there')\n")
    subprocess.run(["git", "add", "code.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "code"], cwd=repo, check=True)
    assert _run_script(repo) == "true"


def test_mixed_docs_and_code_change(tmp_path):
    repo = tmp_path / "repo_mixed"
    repo.mkdir()
    _init_repo(repo)
    (repo / "README.md").write_text("docs\n")
    (repo / "code.py").write_text("print('hi there')\n")
    subprocess.run(["git", "add", "README.md", "code.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "docs and code"], cwd=repo, check=True)
    assert _run_script(repo) == "true"
