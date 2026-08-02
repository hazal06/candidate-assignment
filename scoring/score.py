"""
My scoring script. Reads raw prediction JSONLs (written by the frozen GPU
harness) and computes WER under explicit normalization rules. Runs on CPU
in seconds - this file plus the JSONLs is the full reproduction path for
every number in my results table.

score.py runs without PyTorch installed (transformers is imported only for Whisper's text normalizer)
"""
import json, re, sys
from pathlib import Path

import jiwer
from num2words import num2words
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

# Whisper's language-agnostic normalizer: lowercases, strips punctuation
# and symbols. I deliberately do NOT use the English variant (it rewrites
# English-specific contractions and would be wrong for German).
basic = BasicTextNormalizer()

def spell_out_numbers(text):
    """
    Replace each digit sequence with German words ("20" -> "zwanzig").
    Whisper often outputs digits while Common Voice references spell
    numbers out - without this rule, those count as errors even when the
    model heard correctly. Known limitation: num2words reads years wrong
    ("1990" -> "eintausend..." instead of "neunzehnhundert..."), so I
    report WER with AND without this rule, plus a digit-free subset.
    """
    return re.sub(r"\d+", lambda m: num2words(int(m.group()), lang="de"), text)

def normalize(text, numbers=True):
    # One shared function applied to BOTH reference and hypothesis,
    # so neither side gets an advantage. Order matters: spell out numbers
    # first (while digits still exist), then lowercase/strip punctuation.
    # ss/umlauts: I keep them as-is - BasicTextNormalizer doesn't touch
    # them, and rewriting valid German spelling would hide real errors.
    if numbers:
        text = spell_out_numbers(text)
    return basic(text).strip()

def wer_from_jsonl(path, numbers=True, digit_free_only=False):
    refs, hyps = [], []
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        # digit-free subset: my cross-check that sidesteps the num2words
        # year problem entirely (only clips where neither side has digits)
        if digit_free_only and (re.search(r"\d", d["ref"]) or re.search(r"\d", d["hyp"])):
            continue
        refs.append(normalize(d["ref"], numbers))
        hyps.append(normalize(d["hyp"], numbers))
    return jiwer.wer(refs, hyps), len(refs)

if __name__ == "__main__":
    for path in sys.argv[1:]:
        name = Path(path).stem
        w_full, n_full = wer_from_jsonl(path, numbers=True)
        w_nonum, _     = wer_from_jsonl(path, numbers=False)
        w_df, n_df     = wer_from_jsonl(path, numbers=True, digit_free_only=True)
        print(f"{name}:")
        print(f"  WER (number-normalized):   {w_full:.1%}  on {n_full} clips")
        print(f"  WER (no number rule):      {w_nonum:.1%}")
        print(f"  WER (digit-free subset):   {w_df:.1%}  on {n_df} clips")