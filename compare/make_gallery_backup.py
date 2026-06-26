"""Build a portable, displayed-images-only backup zip of the live gallery.

Reads ``compare/comparison_gallery_live.html``, collects every image the page
actually references (``src`` / ``data-src``), and packs the HTML plus exactly
those image files into ``compare/gallery_backup.zip``.

Why parse the HTML instead of globbing ``eval/outputs/``? The gallery only
renders the first N rows per attack, so the full output folders are ~16.9 GB
while the displayed subset is ~3.8 GB. Parsing the rendered ``src`` attributes
yields precisely what the page needs — it stays in sync automatically when the
gallery's row cap or model registry changes.

Archive entries use repo-relative POSIX paths (``compare/...``, ``eval/...``)
so the extracted tree keeps ``compare/`` and ``eval/`` as siblings and the
HTML's ``../eval/outputs/...`` relative links resolve on the target machine.

Usage (Windows, no GPU needed):
    C:/Users/USER/anaconda3/python.exe compare/make_gallery_backup.py
    C:/Users/USER/anaconda3/python.exe compare/make_gallery_backup.py -o D:/out.zip
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
COMPARE = REPO / "compare"
HTML = COMPARE / "comparison_gallery_live.html"
DEFAULT_ZIP = COMPARE / "gallery_backup.zip"

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".avif"}
# Capture both src="..." and data-src="..." values.
_SRC_RE = re.compile(r'(?:data-)?src\s*=\s*"([^"]+)"', re.IGNORECASE)


def collect_image_refs(html_text: str) -> list[str]:
    """Return de-duplicated, order-preserving image refs from the HTML.

    Skips external URLs (http/https), inline ``data:`` URIs, and any ref whose
    extension is not a known image type. Runtime cache-busting query strings
    (``?t=...``) are stripped.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in _SRC_RE.findall(html_text):
        ref = raw.split("?", 1)[0].strip()
        if not ref or ref.startswith(("http://", "https://", "data:", "//")):
            continue
        if Path(ref).suffix.lower() not in IMG_EXTS:
            continue
        if ref not in seen:
            seen.add(ref)
            ordered.append(ref)
    return ordered


def resolve_ref(ref: str) -> Path:
    """Resolve an HTML image ref (relative to compare/) to an absolute path."""
    return (COMPARE / ref).resolve()


def arcname_for(path: Path) -> str:
    """Repo-relative POSIX archive name, preserving the compare/ <-> eval/ layout."""
    return path.relative_to(REPO).as_posix()


def build_backup(zip_out: Path) -> int:
    """Write the backup zip. Returns process exit code (0 = success)."""
    if not HTML.exists():
        print(f"[error] gallery HTML not found: {HTML}", file=sys.stderr)
        print("        build it first: python compare/build_live_gallery.py", file=sys.stderr)
        return 1

    html_text = HTML.read_text(encoding="utf-8")
    refs = collect_image_refs(html_text)

    present: list[Path] = []
    missing: list[str] = []
    seen_abs: set[Path] = set()
    for ref in refs:
        abs_path = resolve_ref(ref)
        if abs_path in seen_abs:
            continue
        seen_abs.add(abs_path)
        # Stay inside the repo and skip the output zip itself, just in case.
        if REPO not in abs_path.parents or abs_path == zip_out.resolve():
            continue
        if abs_path.is_file():
            present.append(abs_path)
        else:
            missing.append(ref)

    zip_out.parent.mkdir(parents=True, exist_ok=True)
    total_bytes = HTML.stat().st_size + sum(p.stat().st_size for p in present)

    # ZIP_STORED: PNGs are already compressed, so deflate only burns CPU.
    with zipfile.ZipFile(zip_out, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.write(HTML, arcname_for(HTML))
        for path in present:
            zf.write(path, arcname_for(path))

    out_mb = zip_out.stat().st_size / (1024 * 1024)
    print(f"[ok] {zip_out}")
    print(f"     entries : {len(present) + 1}  (1 HTML + {len(present)} images)")
    print(f"     source  : {total_bytes / (1024 * 1024):,.1f} MB")
    print(f"     zip size: {out_mb:,.1f} MB")
    if missing:
        print(f"[warn] {len(missing)} referenced image(s) missing on disk, e.g.:", file=sys.stderr)
        for ref in missing[:5]:
            print(f"       {ref}", file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "-o", "--output", type=Path, default=DEFAULT_ZIP,
        help=f"output zip path (default: {DEFAULT_ZIP})",
    )
    args = parser.parse_args()
    return build_backup(args.output.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
