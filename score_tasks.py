"""Phase 2 scoring: does each ASR model's transcript still produce the RIGHT outcome?

Reads results/per_clip.csv (from evaluate.py), runs the extractor on every hypothesis
and on the reference, and reports per model:
  word accuracy (1 - WER)   vs   intent accuracy, ptp-date accuracy, amount accuracy.
"""
import argparse, pathlib, pandas as pd
from extract import extract

p = argparse.ArgumentParser()
p.add_argument("--per_clip", default="results/per_clip.csv"); p.add_argument("--truth", default="transcripts/ground_truth.csv")
p.add_argument("--method", default="rules", choices=["rules","llm"]); p.add_argument("--out", default="results")
a = p.parse_args()

clips = pd.read_csv(a.per_clip); truth = pd.read_csv(a.truth).set_index("clip_id")
rows = []
for _, r in clips.iterrows():
    g = truth.loc[r.clip_id]; h = extract(str(r.hyp), a.method)
    gi, gd, ga = str(g.intent), (None if pd.isna(g.ptp_date) else str(g.ptp_date).replace(".0","")), (None if pd.isna(g.amount_mentioned) else int(g.amount_mentioned))
    rows.append(dict(model=r.model, clip_id=r.clip_id, condition=r.condition, word_acc=1 - min(1, r.wer),
                     intent_ok=(h["intent"] == gi),
                     date_ok=(str(h["ptp_date"]) == str(gd)) if gd else None,
                     amount_ok=(h["amount_mentioned"] == ga) if ga else None,
                     hyp=r.hyp, got=h))
df = pd.DataFrame(rows); df.to_csv(pathlib.Path(a.out) / "task_per_clip.csv", index=False)
summ = df.groupby("model").agg(word_accuracy=("word_acc","mean"), intent_accuracy=("intent_ok","mean"),
                               ptp_date_accuracy=("date_ok","mean"), amount_accuracy=("amount_ok","mean")).round(3)
md = "## Word accuracy vs outcome accuracy (extractor: %s)\n\n" % a.method + summ.to_markdown() + \
     "\n\nReading this table: the gap between word accuracy and intent accuracy is how much a transcript " \
     "can be wrong while still being actionable; the gap on dates and amounts is where errors cost money.\n"
open(pathlib.Path(a.out) / "task_summary.md", "w").write(md); print(md)
