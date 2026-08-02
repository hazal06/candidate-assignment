"""
The tarball contains a huge clips/ folder (a million MP3s) plus a handful of small .tsv spreadsheet files — including the three that matter: train.tsv, dev.tsv, test.tsv. This script streams through the archive once and saves only the TSVs. It never unpacks the audio.
"""
import tarfile, time
from pathlib import Path

TAR_PATH = r"C:\Users\hazal\Masaüstü\nixi-asr\data\1781721683073-cv-corpus-26.0-2026-06-12-de.tar.gz"
OUT_DIR = Path("data/tsv")
OUT_DIR.mkdir(parents=True, exist_ok=True)

start = time.time()
found = []
scanned = 0

# "r|gz" = streaming mode: reads the archive front-to-back without loading it
with tarfile.open(TAR_PATH, mode="r|gz") as tar:
    for member in tar:
        scanned += 1
        if scanned % 50000 == 0:
            print(f"  scanned {scanned:,} files... ({time.time()-start:.0f}s)")
        if member.name.endswith(".tsv"):
            out_path = OUT_DIR / Path(member.name).name
            with tar.extractfile(member) as src, open(out_path, "wb") as dst:
                dst.write(src.read())
            found.append((member.name, member.size))
            print(f"FOUND: {member.name}  ({member.size/1e6:.1f} MB)")

print(f"\nDone in {(time.time()-start)/60:.1f} min. {scanned:,} files scanned, {len(found)} TSVs saved to {OUT_DIR}/")