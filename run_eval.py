"""
Reproduces every number in the README results section from the raw
prediction files in predictions/. CPU only, no audio, runs in seconds.

Stages: WER table (3 normalization variants) -> paired bootstrap CIs
-> loop analysis. All logic lives in scoring/; this just runs it in order.
"""
import subprocess, sys

P = "predictions"
TEST = [f"{P}/tiny_base.jsonl", f"{P}/tiny_ft.jsonl",
        f"{P}/small_base.jsonl", f"{P}/small_ft.jsonl"]

def run(title, args):
    print(f"\n{'='*60}\n{title}\n{'='*60}")
    r = subprocess.run([sys.executable] + args)
    if r.returncode != 0:
        sys.exit(f"step failed: {' '.join(args)}")

run("WER (all models, all normalization variants)",
    ["scoring/score.py"] + TEST)

run("Paired bootstrap: tiny base -> ft",
    ["scoring/bootstrap.py", f"{P}/tiny_base.jsonl", f"{P}/tiny_ft.jsonl"])

run("Paired bootstrap: small base -> ft",
    ["scoring/bootstrap.py", f"{P}/small_base.jsonl", f"{P}/small_ft.jsonl"])

run("Loop analysis: tiny (expect 3 flagged clips)",
    ["scoring/loop_analysis.py", f"{P}/tiny_base.jsonl", f"{P}/tiny_ft.jsonl"])

run("Loop analysis: small (expect none flagged)",
    ["scoring/loop_analysis.py", f"{P}/small_base.jsonl", f"{P}/small_ft.jsonl"])

print("\ndone. numbers above should match README.md and WRITEUP.md exactly.")
