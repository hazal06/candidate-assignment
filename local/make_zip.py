"""
Two details baked in: ZIP_STORED (no compression) because MP3s don't compress and it's much faster to build and for Kaggle to unpack; and arcname constructed with forward slashes explicitly, so the archive contains clips/... and subsets/... regardless of what OS built it.
"""

import zipfile, time
from pathlib import Path

start = time.time()
n = 0
with zipfile.ZipFile("nixi-cv26-de-subset.zip", "w", zipfile.ZIP_STORED) as z:
    for folder in ["data/clips", "data/subsets"]:
        for f in Path(folder).iterdir():
            # arcname with explicit forward slashes, dropping the leading "data/"
            z.write(f, arcname=f"{Path(folder).name}/{f.name}")
            n += 1
print(f"Zipped {n:,} files in {(time.time()-start)/60:.1f} min")