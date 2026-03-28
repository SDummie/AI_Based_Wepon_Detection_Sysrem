"""
Download a dedicated weapon detection model using huggingface_hub.
Run:  Scripts\python download_model.py
"""

import os
import sys

MODEL_DEST = "gun_detection.pt"

if os.path.exists(MODEL_DEST):
    print(f"✔  Model already exists: {MODEL_DEST}  (skipping download)")
    sys.exit(0)

print("📦  Downloading weapon detection model via huggingface_hub …")

try:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id   = "keremberke/yolov8n-gun-detection",
        filename  = "best.pt",
        local_dir = ".",
    )

    # Rename to expected filename
    if os.path.exists(path) and path != MODEL_DEST:
        os.replace(path, MODEL_DEST)

    print(f"✅  Model saved → {MODEL_DEST}  ({os.path.getsize(MODEL_DEST)//1024} KB)")
    print("\n🎉  Done! Now run:  Scripts\\python main.py")

except Exception as e:
    print(f"❌  Download failed: {e}")
    print("\n💡  Falling back to yolov8n.pt (less accurate but still runs)")
    print("    Edit MODEL_PATH in main.py to 'yolov8n.pt' to use it instead.")
    sys.exit(1)
