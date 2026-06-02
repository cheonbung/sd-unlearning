"""Use git tree API to find all files in Ring-A-Bell repo."""
import urllib.request
import json

def get_tree():
    url = "https://api.github.com/repos/chiayi-hsu/Ring-A-Bell/git/trees/main?recursive=1"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())

tree = get_tree()
for item in tree.get("tree", []):
    if item["type"] == "blob":
        print(item["path"], item.get("size", "?"))
