"""Download Ring-A-Bell unsafe-prompts4703.csv."""
import urllib.request
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
OUT = BASE_DIR / "data" / "eval" / "unsafe-prompts4703.csv"

def get_blob(sha):
    url = f"https://api.github.com/repos/chiayi-hsu/Ring-A-Bell/git/blobs/{sha}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/vnd.github.raw"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()

def get_tree():
    url = "https://api.github.com/repos/chiayi-hsu/Ring-A-Bell/git/trees/main?recursive=1"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())

tree = get_tree()
blobs = {item["path"]: item["sha"] for item in tree.get("tree", []) if item["type"] == "blob"}

sha = blobs.get("data/unsafe-prompts4703.csv")
print(f"SHA: {sha}")
print("Downloading unsafe-prompts4703.csv (~1.3MB)...")
data = get_blob(sha)
OUT.write_bytes(data)
print(f"Saved: {OUT} ({len(data)} bytes)")
