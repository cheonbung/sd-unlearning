"""Check Ring-A-Bell Concept Vectors folder and Violence InvPrompt."""
import urllib.request
import json

def check_github_api(path=""):
    encoded = urllib.request.pathname2url(path) if hasattr(urllib.request, 'pathname2url') else path.replace(" ", "%20")
    url = f"https://api.github.com/repos/chiayi-hsu/Ring-A-Bell/contents/{encoded}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except Exception as e:
        return str(e)

# Concept Vectors folder (URL encode space)
for folder in ["Concept%20Vectors", "data/InvPrompt/Violence"]:
    r = check_github_api(folder)
    if isinstance(r, list):
        print(f"\n{folder}:")
        for f in r:
            print(f"  {f['name']} ({f.get('size','?')} bytes) - {f.get('download_url','')}")
    else:
        print(f"\n{folder}: {r}")
