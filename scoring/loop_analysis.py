"""
Supplementary analysis: hallucination-loop clips and their effect on my
headline numbers. Criterion: errors > 2x reference length - no ordinary
misrecognition produces that (worst normal clips are ~1.3x); only
repetition loops do. Flagged clips are listed, and I report WER/CI both
with and without them. Nothing is silently dropped from the main table.
"""
import json, sys
import numpy as np
import jiwer
from score import normalize

RNG = np.random.default_rng(42)

def load_counts(path):
    out = []
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        m = jiwer.process_words([normalize(d["ref"])], [normalize(d["hyp"])])
        errs = m.substitutions + m.deletions + m.insertions
        ref_len = m.substitutions + m.deletions + m.hits
        out.append((d["path"], errs, ref_len))
    return out

base, ft = load_counts(sys.argv[1]), load_counts(sys.argv[2])

# Flag loops in EITHER system, exclude the union from both - symmetric,
# so the comparison stays paired and fair.
loops = {p for p, e, r in base + ft if e > 2 * r}
print(f"loop clips (errors > 2x ref length): {sorted(loops) or 'none'}")

keep = [i for i, (p, _, _) in enumerate(base) if p not in loops]
e0 = np.array([base[i][1] for i in keep]); w0 = np.array([base[i][2] for i in keep])
e1 = np.array([ft[i][1] for i in keep]);   w1 = np.array([ft[i][2] for i in keep])

print(f"retained {len(keep)}/{len(base)} clips")
print(f"WER without loops: base {e0.sum()/w0.sum():.1%}, ft {e1.sum()/w1.sum():.1%}")

deltas = []
for _ in range(1000):
    idx = RNG.integers(0, len(keep), len(keep))
    deltas.append(e1[idx].sum()/w1[idx].sum() - e0[idx].sum()/w0[idx].sum())
lo, hi = np.percentile(deltas, [2.5, 97.5])
print(f"delta without loops: {e1.sum()/w1.sum()-e0.sum()/w0.sum():+.2%}, 95% CI [{lo:+.2%}, {hi:+.2%}]")