"""
Second and final pass through the tarball. Same streaming trick, but now extracting the MP3s my three subsets reference — roughly 400 MB–1 GB instead of 35 GB.
"""

import csv, tarfile, time
import pandas as pd
from pathlib import Path

TAR_PATH = r"C:\Users\hazal\Masaüstü\nixi-asr\data\1781721683073-cv-corpus-26.0-2026-06-12-de.tar.gz"
PREFIX = "cv-corpus-26.0-2026-06-12/de/clips/"   # from Step 4's FOUND lines
SUB = Path("data/subsets")
CLIPS_OUT = Path("data/clips")
CLIPS_OUT.mkdir(parents=True, exist_ok=True)

# collect every clip filename the three subsets reference
needed = set()
for name in ["train_subset", "dev_subset", "test_subset"]:
    df = pd.read_csv(SUB / f"{name}.tsv", sep="\t", quoting=csv.QUOTE_NONE, dtype=str)
    needed.update(df["path"])
print(f"Clips needed: {len(needed):,}")

start = time.time()
extracted, scanned = 0, 0
with tarfile.open(TAR_PATH, mode="r|gz") as tar:
    for member in tar:
        scanned += 1
        if scanned % 100_000 == 0:
            print(f"  scanned {scanned:,}, extracted {extracted:,} ({time.time()-start:.0f}s)")
        if member.name.startswith(PREFIX):
            fname = member.name[len(PREFIX):]
            if fname in needed:
                with tar.extractfile(member) as src, open(CLIPS_OUT / fname, "wb") as dst:
                    dst.write(src.read())
                extracted += 1
                if extracted == len(needed):
                    print("All needed clips found - stopping early.")
                    break

n_files = sum(1 for _ in CLIPS_OUT.iterdir())
size_mb = sum(f.stat().st_size for f in CLIPS_OUT.iterdir()) / 1e6
print(f"\nDone in {(time.time()-start)/60:.1f} min. Extracted {extracted:,} clips "
      f"({n_files:,} files on disk, {size_mb:.0f} MB).")
if extracted != len(needed):
    print(f"WARNING: {len(needed)-extracted} clips were NOT found in the archive!")