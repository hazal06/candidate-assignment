# German ASR fine-tuning

Fine-tuning whisper-tiny and whisper-small on German Common Voice and measuring WER before and after, with the same held-out set, same decoding, and same normalization for every measurement. Write-up in `Write-up.pdf`, additional information in `DECISIONS.md` if needed.

## Results

| Model | Params | WER before | WER after | Delta | 95% CI of delta |
|---|---|---|---|---|---|
| whisper-tiny | 39M | 46.9% | 33.7% | -13.2 pts | [-25.6, -5.1] |
| whisper-small | 244M | 15.3% | 13.1% | -2.2 pts | [-3.0, -1.6] |

Test set: 1000 clips from the official Common Voice 26.0 German test split, never trained on, touched exactly twice per model. Clip list: `data/test_ids.txt`.

CI: paired bootstrap over utterances, 1000 resamples. The same clip indices are resampled for both systems and the WER difference is recomputed each time, so clip difficulty cancels out and the interval is about the delta itself.

Footnote on tiny's wide CI: it is driven by two baseline clips where whisper-tiny got stuck in repetition loops (440 and 177 errors against 10 and 13 reference words). Excluding loop clips from both sides, tiny's improvement is 8.3 pts with CI [-9.5, -7.0]. Small has no loop clips and its numbers do not change. Details in `Write-up.pdf`; the analysis is `scoring/loop_analysis.py`.

Normalization (identical for every number above, applied to reference and hypothesis through one shared function):
1. Whisper's BasicTextNormalizer: lowercase, strip punctuation. Not the English variant, which is wrong for German.
2. Digit sequences spelled out as German words with num2words. Reported with and without this rule; it moves WER by at most 0.3 points.
3. ss and umlauts left as written.

## Reproduce the evaluation (one command, CPU only, no data download)

```
pip install -r requirements-scoring.txt
python run_eval.py
```

`run_eval.py` recomputes every number above from the raw model outputs in `predictions/`: the four WER scores in three normalization variants, both bootstrap CIs, and the loop analysis. It needs no audio, no GPU, and no PyTorch, and runs in seconds.

This works because evaluation is split in two stages. The GPU stage (the Kaggle notebook) decodes audio once per model and writes raw transcriptions to JSONL. The scoring stage reads those files and applies normalization and WER offline. Shipping the JSONLs makes the whole scoring stage reproducible without redistributing audio, which the dataset terms forbid (see Data).

## Repository layout

```
run_eval.py                  reproduces all reported numbers from predictions/
predictions/                 raw model outputs (JSONL): 4 test runs + 2 dev runs
scoring/                     score.py, bootstrap.py, loop_analysis.py, worst_clips.py
local/                       data preparation scripts that ran on my PC
data/test_ids.txt            the exact 1000 test clips
notebook/nixi_asr.ipynb      Kaggle notebook: data checks, frozen decode harness, training
requirements-scoring.txt     loose pins for the scoring path
requirements-scoring-frozen.txt  exact versions from my machine
DECISIONS.md                 decision log/notes
Write-up.pdf                   the half-page write-up
```

## Full pipeline (optional, needs data + GPU)

Retraining or re-decoding needs the audio, which I cannot redistribute. Steps: create a Mozilla Data Collective account, download Common Voice Scripted Speech 26.0 German (34.77 GB), run `local/extract_tsvs.py`, `local/check_and_sample.py` and `local/extract_clips.py` to rebuild the exact subsets (seed 42, verified against `data/test_ids.txt`), upload as a private Kaggle dataset, and run the notebook top to bottom on a T4. Fine-tuned checkpoints are on the Hub: `hazal-karakus/whisper-tiny-de-cv26` and `hazal-karakus/whisper-small-de-cv26`, so decoding can also be verified without retraining.

## Data and licensing

Common Voice 26.0 German (released June 2026), downloaded from Mozilla Data Collective, which is the official distribution since Common Voice left Hugging Face in late 2025. The data is CC0, but the download terms explicitly forbid re-sharing the dataset and identifying speakers, so this repository contains no audio and no speaker metadata beyond what the scoring needs (reference sentences and clip filenames). Official train/dev/test splits used unmodified; speaker and sentence disjointness across splits verified programmatically (all overlaps zero, see `local/check_and_sample.py`).

## Environment notes

Training ran on Kaggle (free tier). The only package added to Kaggle's stock image is jiwer. Kaggle allocates two T4s and the HF Trainer used both via DataParallel, which I detected mid-run from a warning and kept identical for both models; everything fits on a single T4 with the same effective batch size (use per-device batch x accumulation accordingly), and all evaluation is single-GPU. Kaggle-side package versions are printed in the notebook's last cell. The scoring path needs only the packages in `requirements-scoring.txt` and runs without PyTorch.
