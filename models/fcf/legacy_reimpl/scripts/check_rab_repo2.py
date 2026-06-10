"""Check Ring-A-Bell repo subfolders."""
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

for folder in ["Concept Vectors", "data/InvPrompt", "data/Prompts_For_ConceptVector"]:
    r = check_github_api(folder)
    if isinstance(r, list):
        print(f"\n{folder}:")
        for f in r:
            print(f"  {f['name']} ({f.get('size', '?')} bytes) - {f.get('download_url', '')}")
    else:
        print(f"\n{folder}: {r}")
