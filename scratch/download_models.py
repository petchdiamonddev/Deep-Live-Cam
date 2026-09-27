import os
import urllib.request
import sys

MODELS = {
    "inswapper_128_fp16.onnx": "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx",
    "GFPGANv1.4.onnx": "https://huggingface.co/hacksider/deep-live-cam/resolve/main/GFPGANv1.4.onnx"
}

models_dir = os.path.join(os.getcwd(), "models")
os.makedirs(models_dir, exist_ok=True)

def download_file(url, target_path):
    print(f"Downloading {os.path.basename(target_path)} from {url}...")
    def reporthook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size) if total_size > 0 else 0
        mb_downloaded = (count * block_size) / (1024 * 1024)
        mb_total = total_size / (1024 * 1024) if total_size > 0 else 0
        sys.stdout.write(f"\rDownloading: {percent}% ({mb_downloaded:.1f} MB / {mb_total:.1f} MB)")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, target_path, reporthook)
    print("\nDownload finished!")

for filename, url in MODELS.items():
    path = os.path.join(models_dir, filename)
    if os.path.exists(path) and os.path.getsize(path) > 1000000:
        print(f"[SKIP] {filename} already exists ({os.path.getsize(path) / (1024*1024):.1f} MB).")
    else:
        download_file(url, path)

print("All models downloaded successfully.")
