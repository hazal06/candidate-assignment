"""
Paired bootstrap for WER deltas. Answers: "is the improvement real, or
could it be sampling luck of these particular 1000 clips?"

Method: compute per-clip error counts once, then resample the SAME clip
indices for both systems 1000 times, recomputing the WER difference each
time. Because each resample applies to both systems identically, clip
difficulty cancels out - this tests the DELTA, which is much stronger
than comparing two separate confidence intervals.
Runs on CPU in seconds from the raw JSONLs - part of my repro package.
"""
import json, re, sys
import numpy as np
import jiwer
from scoring.score import normalize   # the exact same normalization as my WER table

RNG = np.random.default_rng(42)
N_BOOT = 1000

def per_clip_counts(path):
    """For each clip: (edit errors, reference word count) under my
    standard normalization. Corpus WER = sum(errors)/sum(words)."""
    errs, words = [], []
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        ref, hyp = normalize(d["ref"]), normalize(d["hyp"])
        m = jiwer.process_words([ref], [hyp])
        errs.append(m.substitutions + m.deletions + m.insertions)
        words.append(m.substitutions + m.deletions + m.hits)  # = ref length
    return np.array(errs), np.array(words)

def paired_bootstrap(base_path, ft_path):
    e0, w0 = per_clip_counts(base_path)
    e1, w1 = per_clip_counts(ft_path)
    assert len(e0) == len(e1), "JSONLs must cover identical clips"
    n = len(e0)
    deltas = []
    for _ in range(N_BOOT):
        idx = RNG.integers(0, n, n)          # same clips for both systems
        deltas.append(e1[idx].sum()/w1[idx].sum() - e0[idx].sum()/w0[idx].sum())
    deltas = np.array(deltas)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    print(f"{base_path} -> {ft_path}")
    print(f"  WER delta: {e1.sum()/w1.sum() - e0.sum()/w0.sum():+.2%}")
    print(f"  95% CI: [{lo:+.2%}, {hi:+.2%}]"
          f"  -> {'excludes zero: improvement is real' if hi < 0 else 'INCLUDES ZERO: not distinguishable from noise'}")

if __name__ == "__main__":
    paired_bootstrap(sys.argv[1], sys.argv[2])

# One prerequisite: both JSONLs list clips in identical order 
# (they do — the harness sorts by path; the assert guards the count anyway)
