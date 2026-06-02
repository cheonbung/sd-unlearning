"""Download Nudity_vector.npy from Ring-A-Bell GitHub repo."""
import urllib.request
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
OUT_PATH = BASE_DIR / "data" / "eval" / "Nudity_vector.npy"

URLS = [
    "https://github.com/chiayi-hsu/Ring-A-Bell/raw/main/concept_vectors/Nudity_vector.npy",
    "https://raw.githubusercontent.com/chiayi-hsu/Ring-A-Bell/main/concept_vectors/Nudity_vector.npy",
]

def download():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    for url in URLS:
        try:
            print(f"Trying: {url}")
            urllib.request.urlretrieve(url, OUT_PATH)
            size = OUT_PATH.stat().st_size
            print(f"Downloaded Nudity_vector.npy ({size} bytes) -> {OUT_PATH}")
            return True
        except Exception as e:
            print(f"  Failed: {e}")
    return False

if __name__ == "__main__":
    success = download()
    if not success:
        print("\nManual download required:")
        print("  https://github.com/chiayi-hsu/Ring-A-Bell/tree/main/concept_vectors")
