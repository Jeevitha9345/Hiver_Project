import os
import sys
import time
import urllib.request

DATA_URL = "https://huggingface.co/datasets/SunidhiSriram/twcs/resolve/main/twcs.csv"
DEST_PATH = os.path.join("data", "raw", "twcs.csv")

def download_file(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 500_000_000:
        print(f"File already exists at {dest} ({os.path.getsize(dest)} bytes). Skipping download.")
        return

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    temp_dest = dest + ".tmp"
    
    print(f"Starting download from {url} to {dest}...")
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)
    
    with urllib.request.urlopen(req) as resp, open(temp_dest, "wb") as f:
        total_size = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        block_size = 1024 * 1024  # 1MB
        start_time = time.time()
        last_log = start_time

        while True:
            chunk = resp.read(block_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            
            now = time.time()
            if now - last_log >= 5:  # log every 5 seconds
                speed = (downloaded / (now - start_time)) / (1024 * 1024)
                pct = (downloaded / total_size * 100) if total_size else 0
                mb_down = downloaded / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                print(f"Progress: {mb_down:.1f}/{total_mb:.1f} MB ({pct:.1f}%) | Speed: {speed:.2f} MB/s")
                last_log = now

    os.replace(temp_dest, dest)
    print(f"Download complete: {dest} ({os.path.getsize(dest)} bytes)")

if __name__ == "__main__":
    download_file(DATA_URL, DEST_PATH)
