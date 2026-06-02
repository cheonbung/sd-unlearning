"""Check Ring-A-Bell GitHub repo structure via API."""
import urllib.request
import json

def check_github_api(path=""):
    url = f"https://api.github.com/repos/chiayi-hsu/Ring-A-Bell/contents/{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except Exception as e:
        return str(e)

# Check root
root = check_github_api()
if isinstance(root, list):
    print("Root:", [f["name"] for f in root])
else:
    print("Root error:", root)

# Check concept_vectors folder
cv = check_github_api("concept_vectors")
if isinstance(cv, list):
    print("concept_vectors:", [f["name"] for f in cv])
elif isinstance(cv, str):
    # Try other possible locations
    for folder in ["", "vectors", "data", "assets"]:
        r = check_github_api(folder)
        if isinstance(r, list):
            print(f"{folder or 'root'}:", [f["name"] for f in r])
