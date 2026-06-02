"""Repository quality gate for source-only validation.

The gate intentionally checks the code owned by this repository and skips
generated outputs plus the embedded external Q16 checkout.
"""

from __future__ import annotations

import argparse
import compileall
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ["fcf", "fcf-novel-methods", "lsse"]
TEST_DIRS = ["fcf/tests", "fcf-novel-methods/tests", "lsse/tests"]
PORTABLE_SUFFIXES = {".py", ".yaml", ".yml"}
FORBIDDEN_LOCAL_PATHS = ("C:/Users/", "c:/Users/", "c:\\Users\\", "/mnt/c/Users/")
SKIP_RE = re.compile(
    r"([\\/](outputs|__pycache__|\.pytest_cache|\.git)[\\/])|"
    r"([\\/]fcf[\\/]Q16[\\/])"
)


def run_compile() -> bool:
    ok = True
    for rel in SOURCE_DIRS:
        path = ROOT / rel
        print(f"[compile] {rel}", flush=True)
        if not compileall.compile_dir(
            str(path),
            quiet=1,
            rx=SKIP_RE,
            workers=0,
        ):
            ok = False
    return ok


def run_pytest(extra_args: list[str]) -> int:
    for rel in TEST_DIRS:
        cmd = [sys.executable, "-m", "pytest", rel, *extra_args]
        print("[pytest] " + " ".join(cmd), flush=True)
        code = subprocess.call(cmd, cwd=ROOT)
        if code != 0:
            return code
    return 0


def iter_owned_files():
    for rel in SOURCE_DIRS:
        base = ROOT / rel
        for path in base.rglob("*"):
            if path.is_file() and path.suffix in PORTABLE_SUFFIXES:
                path_text = str(path)
                if not SKIP_RE.search(path_text):
                    yield path


def run_portability_scan() -> bool:
    print("[portability] local absolute path scan", flush=True)
    offenders: list[tuple[Path, int, str]] = []
    for path in iter_owned_files():
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(lines, start=1):
            if any(pattern in line for pattern in FORBIDDEN_LOCAL_PATHS):
                offenders.append((path.relative_to(ROOT), lineno, line.strip()))

    for path, lineno, line in offenders:
        print(f"  {path}:{lineno}: {line}", flush=True)
    return not offenders


def main() -> int:
    parser = argparse.ArgumentParser(description="Run repository quality checks.")
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--skip-portability", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Optional arguments passed to pytest after '--'.",
    )
    args = parser.parse_args()

    pytest_args = args.pytest_args
    if pytest_args and pytest_args[0] == "--":
        pytest_args = pytest_args[1:]

    if not args.skip_compile and not run_compile():
        return 1

    if not args.skip_portability and not run_portability_scan():
        return 1

    if not args.skip_tests:
        code = run_pytest(pytest_args or ["-q"])
        if code != 0:
            return code

    print("[ok] quality gate passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
