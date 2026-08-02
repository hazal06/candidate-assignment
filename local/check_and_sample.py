"""
This script does the trust-but-verify work: row counts vs. the dataset page, speaker disjointness, sentence disjointness, then the duration filter and seeded sampling — producing my final subset files including test_ids.txt.
"""

import csv
import pandas as pd
from pathlib import Path

TSV = Path("data/tsv")
OUT = Path("data/subsets")
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
N_TRAIN, N_DEV, N_TEST = 10_000, 1_000, 1_000
MIN_S, MAX_S = 1.0, 30.0

def load(name):
    # quoting=QUOTE_NONE: sentences may contain quote chars; default parsing would break rows
    return pd.read_csv(TSV / f"{name}.tsv", sep="\t", quoting=csv.QUOTE_NONE,
                       dtype=str, na_filter=False)

train, dev, test = load("train"), load("dev"), load("test")
print("== Row counts (page says 623,289 / 16,208 / 16,208) ==")
print(f"train: {len(train):,}   dev: {len(dev):,}   test: {len(test):,}")

print("\n== Speaker disjointness (all intersections must be 0) ==")
s_tr, s_de, s_te = set(train.client_id), set(dev.client_id), set(test.client_id)
print(f"train∩dev: {len(s_tr & s_de)}   train∩test: {len(s_tr & s_te)}   dev∩test: {len(s_de & s_te)}")

print("\n== Sentence disjointness of dev/test vs train (expect 0 or near 0) ==")
t_tr, t_de, t_te = set(train.sentence_id), set(dev.sentence_id), set(test.sentence_id)
print(f"train∩dev: {len(t_tr & t_de)}   train∩test: {len(t_tr & t_te)}   dev∩test: {len(t_de & t_te)}")

# --- duration filter ---
dur = pd.read_csv(TSV / "clip_durations.tsv", sep="\t", quoting=csv.QUOTE_NONE)
dur.columns = ["path", "ms"]
dur["s"] = pd.to_numeric(dur["ms"], errors="coerce") / 1000.0

def add_duration(df):
    m = df.merge(dur[["path", "s"]], on="path", how="left")
    missing = m["s"].isna().sum()
    return m, missing

train, miss_tr = add_duration(train)
dev,   miss_de = add_duration(dev)
test,  miss_te = add_duration(test)
print(f"\n== Duration merge (clips with no duration entry) ==")
print(f"train: {miss_tr}   dev: {miss_de}   test: {miss_te}")

def dfilter(df, name):
    ok = df[(df["s"] >= MIN_S) & (df["s"] <= MAX_S)]
    print(f"{name}: {len(df):,} -> {len(ok):,} after {MIN_S}-{MAX_S}s filter "
          f"({len(df)-len(ok):,} removed)")
    return ok

print("\n== Duration filter ==")
train_f, dev_f, test_f = dfilter(train, "train"), dfilter(dev, "dev"), dfilter(test, "test")

# --- seeded sampling (sort first: TSV row order is not a guaranteed-stable thing to cite) ---
def sample(df, n, name):
    out = df.sort_values("path").sample(n=min(n, len(df)), random_state=SEED)
    out.to_csv(OUT / f"{name}_subset.tsv", sep="\t", index=False, quoting=csv.QUOTE_NONE)
    return out

train_s = sample(train_f, N_TRAIN, "train")
dev_s   = sample(dev_f,   N_DEV,   "dev")
test_s  = sample(test_f,  N_TEST,  "test")

(OUT / "test_ids.txt").write_text("\n".join(sorted(test_s["path"])), encoding="utf-8")

print(f"\n== Subsets written to {OUT}/ ==")
print(f"train: {len(train_s):,}   dev: {len(dev_s):,}   test: {len(test_s):,}   + test_ids.txt")
print(f"train audio hours (approx): {train_s['s'].astype(float).sum()/3600:.1f}")
print(f"test audio minutes (approx): {test_s['s'].astype(float).sum()/60:.1f}")