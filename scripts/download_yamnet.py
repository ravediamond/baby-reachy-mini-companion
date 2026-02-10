from pathlib import Path

import requests


CACHE_DIR = Path("cache/models--yamnet")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

URL_MODEL = "https://huggingface.co/jafet21/yamnetonnx/resolve/main/yamnet.onnx"
URL_MAP = "https://huggingface.co/jafet21/yamnetonnx/resolve/main/yamnet_class_map.csv"

def download_file(url, path):
    """Download a file from a URL to the given path."""
    if path.exists():
        print(f"File already exists: {path}")
        return

    print(f"Downloading {url} to {path}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download complete.")

if __name__ == "__main__":
    download_file(URL_MODEL, CACHE_DIR / "yamnet.onnx")
    download_file(URL_MAP, CACHE_DIR / "yamnet_class_map.csv")
