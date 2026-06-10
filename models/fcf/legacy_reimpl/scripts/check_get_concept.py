"""Download and print Get_Concept_Vector.ipynb from Ring-A-Bell repo."""
import urllib.request
import json
from pathlib import Path

def get_blob(sha):
    url = f"https://api.github.com/repos/chiayi-hsu/Ring-A-Bell/git/blobs/{sha}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/vnd.github.raw"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()

def get_tree():
    url = "https://api.github.com/repos/chiayi-hsu/Ring-A-Bell/git/trees/main?recursive=1"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())

tree = get_tree()
blobs = {item["path"]: item["sha"] for item in tree.get("tree", []) if item["type"] == "blob"}
data = get_blob(blobs["Get_Concept_Vector.ipynb"])
nb = json.loads(data)
for cell in nb.get("cells", []):
    src = "".join(cell.get("source", []))
    if src.strip():
        print("---")
        print(src[:500])
