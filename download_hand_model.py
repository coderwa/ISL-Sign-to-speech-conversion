"""Download Google's official MediaPipe Hand Landmarker task model."""
from pathlib import Path
from urllib.request import urlopen

URL="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
OUTPUT=Path(__file__).resolve().parent/"models"/"hand_landmarker.task"

def main():
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    if OUTPUT.exists() and OUTPUT.stat().st_size>1_000_000:
        print(f"Already present: {OUTPUT}"); return
    temp=OUTPUT.with_suffix(".task.part")
    print(f"Downloading {URL}")
    try:
        with urlopen(URL,timeout=60) as response, temp.open("wb") as out:
            while True:
                chunk=response.read(1024*1024)
                if not chunk: break
                out.write(chunk)
        if temp.stat().st_size<1_000_000: raise RuntimeError("Downloaded file is unexpectedly small.")
        temp.replace(OUTPUT); print(f"Saved {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)")
    except Exception:
        temp.unlink(missing_ok=True); raise

if __name__=="__main__": main()
