from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_BUNDLE_PATTERN = re.compile(r"Run bundle written to (?P<path>.+)$")


def run_and_capture_bundle(command: list[str]) -> Path:
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    bundle_path: Path | None = None
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        match = RUN_BUNDLE_PATTERN.search(line.strip())
        if match:
            bundle_path = (REPO_ROOT / match.group("path")).resolve()

    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    if bundle_path is None:
        raise RuntimeError("Inference completed but no run bundle path was found in the output")
    if not bundle_path.is_dir():
        raise RuntimeError(f"Inference reported a run bundle that does not exist: {bundle_path}")
    return bundle_path


def main() -> None:
    inference_command = [sys.executable, "main.py", "--experimental"]
    bundle_path = run_and_capture_bundle(inference_command)

    analysis_command = [
        sys.executable,
        "analyze.py",
        "--results",
        str(bundle_path),
        "--experimental",
    ]
    subprocess.run(analysis_command, cwd=REPO_ROOT, check=True)

    print(f"Experimental run bundle: {bundle_path}")


if __name__ == "__main__":
    main()
