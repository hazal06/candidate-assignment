# decisions log (mostly kept for myself, roughly chronological)

## setup
- task A
- whisper tiny + small -> clean comparison. hypothesis: gains shrink as base model gets stronger
- NOT wav2vec2: german checkpoints on HF mostly already fine-tuned on CV -> test set contamination, before/after would be meaningless. also needs char vocab + KenLM on top, more moving parts
- kaggle over colab: guaranteed ~30 GPU-h/week, background execution, T4 has tensor cores (fp16)
- dev decides everything, test touched exactly 2x per model

## data
- CV german, official splits unmodified. re-splitting = how you leak speakers
- !! CV left HF, official repos are empty stubs. options considered:
  - fsicoli mirror: needs datasets 2.x pin + trust_remote_code (loading scripts dead in datasets v3). works but not trustuble enough
  - flozi00/asr-german-mixed: no. references re-transcribed/corrected by someone else's pipeline + custom re-split, can't verify speaker disjointness. also 136 GB
  - MDC official: winner. treated like a company task -> provenance matters
- CV 26.0 de, 34.77 GB, released 2026-06
- terms read at download: CC0 BUT no re-sharing + no speaker identification. so: repo ships no audio, upload as private kaggle dataset that stays private forever
- meaning: repro package = raw prediction jsonls + reference text + test_ids.txt + scoring code. full eval reproducible on cpu without audio. decode-once/score-offline design

## dataset checks (measured)
- row counts match published stats exactly: 623289 / 16208 / 16208
- speaker disjointness train/dev/test: all intersections 0
- sentence disjointness: also all 0 -> test = unseen voices AND unseen text
- duration filter 1-30s: removed 0 clips. corpus min is 72ms but none of the super short clips are in the official splits. verified durations file parses right (mean duration of 5.27s). kept filter in code as guard
- subsets: 10k train (15.9h) / 1k dev / 1k test, seed 42, sorted before sampling, test_ids.txt committed

## build
- powershell Compress-Archive makes zips with backslash separators -> kaggle upload was rejected. rebuilt with python zipfile, forward slashes, ZIP_STORED (mp3 doesn't compress)
- harness frozen before any training: one decode function, same clips/order/greedy/language token for all 4 evals. raw jsonl out, no normalization on gpu side
- smoke test 50 clips: german out, tiny-typical errors. 22s -> full eval ~1min, cheap
- tiny base 46.9 (expected 35-50, ok)
- small base 15.3 (expected 11-18, ok)
- num2words rule: shifts WER 0.2-0.3 pts only. 43/1000 refs contain digits. planned distrust -> measured -> negligible. keeping 3 variants in table as evidence
- training targets = raw CV text (casing+punct), normalization lives in eval only

## training
- DataParallel surprise: kaggle gives T4 x2, trainer used both silently. effective batch 32 not 16, epochs ~4.8 not 2.4. caught via scalar-gather warning. kept (same for both models, eval unaffected), documented
- tiny 1500 steps: dev WER 28.4 / 27.7 / 28.0 -> best at 1000, guard kept that checkpoint. training longer didn't help
- tiny_ft test 33.7. dev said 27.7 -> checked: tiny BASE on dev = 34.8 vs 46.9 on test, so dev is apparently just an easier speaker sample. my forecast was wrong, not the measurement
- bonus from that check: trainer's internal dev eval and my harness = 27.7 exactly, two independent code paths agree
- laptop died overnight at step 1473/1500 of small run 1. lost the checkpoint, kept the info: dev flat 500->1000 (10.98->11.09, within noise for n=1000) + val loss rising 0.229->0.239. NOT calling it overfitting (delta too small), calling it "no value past 500"
- rerun small with max_steps=500 (+eval@250 as guard): dev 11.26@250, 10.79@500. lr schedule finishes cooler with shorter horizon, hence slightly better than run1@500
- lesson: interactive kaggle sessions die with the client. push to hub immediately, download jsonls immediately
- processor push said "skipping empty commit" -> checked repo loadable via roundtrip (reload from hub, tokenize test string, ids identical). fine, tokenizer.json is the vocab in fast format
- small_ft test 13.1

## analysis
- paired bootstrap (same resampled clips for both systems, delta CI):
  - tiny: -13.18 [-25.61, -5.05]
  - small: -2.26 [-2.97, -1.60]
- tiny CI 15x wider than small + skewed -> didn't trust it. per-clip errors: 2 loop clips in base (440 errs vs 10 ref words "der der der...", 177 vs 13). ~5 of the 13.2 pts come from these two clips
- loop analysis (flag errs > 2x ref len, union of both systems, excluded from both sides):
  - tiny: 3 flagged, base 40.3 ft 32.0, delta -8.30 [-9.51, -7.01]. CI now normal width -> outliers explained it
  - small: 0 flagged, unchanged -> criterion catches loops not hard audio
- 3rd flagged clip = tiny_FT looping ("bit bit bit...", 147 errs vs 11 words) on a clip base handled ok. so ft cured both old loops, created 1 new milder one. reduced not eliminated. the scary failure mode for clinical use, wrote it up
- ft models nearly stopped emitting digits (997/996 digit-free vs 957/984 base) -> learned CV's spelled-out style, that's why num2words stops mattering after ft

## didn't help / dead ends
- num2words machinery (see above, negligible)
- training past dev optimum (both models)
- duration filter (vacuous on official splits)
- my own test-WER forecast for tiny_ft (26-30, actual 33.7) - dev/test difficulty gap, investigated, forecast wrong not measurement
