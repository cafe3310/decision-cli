"""
Common test utilities for decision-cli test suite.
"""

import os
import socket
import subprocess
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def get_cli_command() -> List[str]:
    cli_bin = os.environ.get("DECISION_CLI_BIN")
    if cli_bin:
        return [cli_bin]
    compiled_bin = REPO_ROOT / "bin" / "decision-cli"
    if compiled_bin.exists():
        return [str(compiled_bin)]
    return ["go", "run", "./cmd/decision-cli"]


def run_cli(
    args: List[str],
    check: bool = True,
    input_str: Optional[str] = None,
    timeout: int = 120,
) -> subprocess.CompletedProcess:
    cmd = get_cli_command() + args
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        input=input_str,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}:\n"
            f"Command: {' '.join(cmd)}\n"
            f"Stdout: {result.stdout}\n"
            f"Stderr: {result.stderr}"
        )
    return result


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def is_model_ready(model_name: str) -> bool:
    try:
        res = run_cli(["models"], check=False)
        for line in res.stdout.splitlines():
            if model_name in line and "[READY" in line:
                return True
    except Exception:
        pass
    return False
