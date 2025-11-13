#!/usr/bin/env python3
"""Utility script for validating dependencies and running the API server."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = ROOT / "plate_detector_v1" / "weights" / "best.pt"


def check_python_version() -> None:
    if sys.version_info < (3, 10):
        raise RuntimeError(
            f"Python 3.10 or newer is required, but {sys.version.split()[0]} is installed."
        )


def ensure_requirements() -> None:
    try:
        import fastapi  # noqa: F401
        import ultralytics  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        print("Installing Python dependencies from requirements.txt ...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", "pip"]
        )
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        )


def assert_model_weights() -> None:
    if WEIGHTS_PATH.exists():
        return
    alternatives = sorted(WEIGHTS_PATH.parent.glob("*.pt"))
    if alternatives:
        return
    raise FileNotFoundError(
        "Model weights were not found. Please place ao menos um arquivo *.pt dentro de "
        f"{WEIGHTS_PATH.parent}"
    )


def run_server() -> None:
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "app:app",
        "--host",
        "0.0.0.0",
        "--port",
        os.getenv("PORT", "8000"),
    ]
    subprocess.check_call(command)


def main() -> None:
    os.chdir(ROOT)
    check_python_version()
    ensure_requirements()
    assert_model_weights()
    print("All checks passed. Starting the API server...")
    run_server()


if __name__ == "__main__":
    main()
