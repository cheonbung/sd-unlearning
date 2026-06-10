"""Download Ring-A-Bell files: Nudity_vector.npy + InversePrompt.ipynb."""
import urllib.request
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
EVAL_DIR = BASE_DIR / "data" / "eval"

def get_tree():
    url = "https://api.github.com/repos/chiayi-hsu/Ring-A-Bell/git/trees/main?recursive=1"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())

def get_blob(sha):
    url = f"https://api.github.com/repos/chiayi-hsu/Ring-A-Bell/git/blobs/{sha}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/vnd.github.raw"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()

def download_raw(url, out_path):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = r.read()
    out_path.write_bytes(data)
    print(f"  -> {out_path} ({len(data)} bytes)")

print("Fetching repo tree...")
tree = get_tree()
blobs = {item["path"]: item["sha"] for item in tree.get("tree", []) if item["type"] == "blob"}

targets = {
    "Concept Vectors/Nudity_vector.npy": EVAL_DIR / "Nudity_vector.npy",
    "Concept Vectors/VanGogh_vector.npy": EVAL_DIR / "VanGogh_vector.npy",
    "InversePrompt.ipynb": BASE_DIR / "scripts" / "InversePrompt.ipynb",
    "data/Prompts_For_ConceptVector/Nudity_prompt.csv": EVAL_DIR / "Nudity_prompt.csv",
}

EVAL_DIR.mkdir(parents=True, exist_ok=True)

for path, out in targets.items():
    sha = blobs.get(path)
    if not sha:
        print(f"NOT FOUND: {path}")
        continue
    print(f"Downloading {path} (sha={sha[:8]})...")
    try:
        data = get_blob(sha)
        out.write_bytes(data)
        print(f"  -> {out} ({len(data)} bytes)")
    except Exception as e:
        print(f"  blob API failed: {e}")
        # fallback: raw URL
        raw_url = "https://raw.githubusercontent.com/chiayi-hsu/Ring-A-Bell/main/" + path.replace(" ", "%20")
        try:
            download_raw(raw_url, out)
        except Exception as e2:
            print(f"  raw URL also failed: {e2}")

print("\nDone.")
