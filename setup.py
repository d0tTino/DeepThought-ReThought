from pathlib import Path

from setuptools import find_packages, setup

# Read requirements from the provided requirements.txt
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
    requirements = []

# Import version from the package if available
version = "0.0.0"
try:
    from src.deepthought import __version__ as package_version

    version = package_version
except Exception:
    pass

setup(
    name="deepthought-rethought",
    version=version,
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=requirements,
    description="DeepThought reThought - experimental AI framework",
    license="MIT",
    url="https://github.com/d0tTino/DeepThought-ReThought",
    python_requires=">=3.8",
)
