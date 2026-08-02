"""
Diagnostic: which clips contribute the most errors? Written to explain the
suspiciously wide bootstrap CI for tiny - my hypothesis is a few
hallucination-loop clips (insertions >> reference length) dominate it.
"""
import json, sys
from score import normalize
import jiwer

rows = []
for line in open(sys.argv[1], encoding="utf-8"):
    d = json.loads(line)
    ref, hyp = normalize(d["ref"]), normalize(d["hyp"])
    m = jiwer.process_words([ref], [hyp])
    errs = m.substitutions + m.deletions + m.insertions
    ref_len = m.substitutions + m.deletions + m.hits
    rows.append((errs, ref_len, d["path"], hyp[:120]))

rows.sort(reverse=True)
total_e = sum(r[0] for r in rows); total_w = sum(r[1] for r in rows)
print(f"corpus: {total_e} errors / {total_w} words = {total_e/total_w:.1%}")
print(f"top-5 clips' share of all errors: {sum(r[0] for r in rows[:5])/total_e:.1%}\n")
for errs, ref_len, path, hyp in rows[:5]:
    print(f"{errs} errors vs {ref_len} ref words  ({path})\n  HYP: {hyp}\n")