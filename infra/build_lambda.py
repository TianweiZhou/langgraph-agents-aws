"""Builds the Lambda bundle (Linux arm64 wheels + our source) with uv, no Docker needed."""

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
BUILD_DIR = ROOT / "build" / "lambda"
LAMBDA_PLATFORM = "aarch64-manylinux2014"  # matches the arm64 Lambda architecture
PYTHON_VERSION = "3.12"


def _run(*args: str) -> None:
    subprocess.run(args, cwd=ROOT, check=True)


def build_lambda_bundle() -> Path:
    """Return a fresh directory ready to zip into the Lambda function."""
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    BUILD_DIR.mkdir(parents=True)

    requirements = BUILD_DIR.parent / "requirements.txt"
    _run(
        "uv", "export", "--frozen", "--no-dev", "--no-emit-project", "--no-hashes",
        "--format", "requirements.txt", "--output-file", str(requirements), "--quiet",
    )  # fmt: skip
    _run(
        "uv", "pip", "install", "--quiet",
        "--requirement", str(requirements),
        "--target", str(BUILD_DIR),
        "--python-platform", LAMBDA_PLATFORM,
        "--python-version", PYTHON_VERSION,
        "--only-binary", ":all:",
    )  # fmt: skip

    shutil.copytree(
        SRC_DIR,
        BUILD_DIR,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    return BUILD_DIR
