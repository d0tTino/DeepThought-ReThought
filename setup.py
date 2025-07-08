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
    packages=find_packages("src") + ["tools"],
    package_dir={"": "src", "tools": "tools"},
    package_data={
        "tools": ["template_service/*"],
        "deepthought.templates": ["bus_service/*"],
    },
    include_package_data=True,
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "dtrt=deepthought.cli:main",
            "dtrt-finetune=deepthought.cli.finetune:main",
        ]
    },
    description="DeepThought reThought - experimental AI framework",
    license="MIT",
    url="https://github.com/d0tTino/DeepThought-ReThought",
    python_requires=">=3.8",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
