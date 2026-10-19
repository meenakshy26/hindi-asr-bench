"""Transcribe every clip with each model, score WER overall and per category, write results."""
import argparse, pathlib, time
import pandas as pd, torch, librosa, jiwer
from transformers import pipeline
from normalise import normalise

MODELS = {
    "whisper-tiny":    "openai/whisper-tiny",
    "whisper-small":   "openai/whisper-small",
    "whisper-medium":  "openai/whisper-medium",
    # Indic-specific checkpoint: replace with the current AI4Bharat Hindi Whisper id from the Hub
    "indicwhisper-hi": "ai4bharat/indicwhisper",
}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio", required=True); p.add_argument("--truth", required=True)
    p.add_argument("--out", required=True); p.add_argument("--models", nargs="*", default=list(MODELS))
    a = p.parse_args()
    truth = pd.read_csv(a.truth); out = pathlib.Path(a.out); out.mkdir(exist_ok=True)
    device = 0 if torch.cuda.is_available() else -1
    rows = []
    for name in a.models:
        asr = pipeline("automatic-speech-recognition", model=MODELS[name], device=device)
        for _, r in truth.iterrows():
            path = pathlib.Path(a.audio) / r.filename
            if not path.exists():
                continue
            y, _ = librosa.load(path, sr=16000, mono=True)
            t0 = time.time()
            hyp = asr({"raw": y, "sampling_rate": 16000},
                      generate_kwargs={"language": "hi", "task": "transcribe"})["text"]
            ref_n, hyp_n = normalise(r.transcript), normalise(hyp)
            rows.append(dict(model=name, clip_id=r.clip_id, condition=r.condition, categories=r.categories,
                             ref=r.transcript, hyp=hyp, wer=jiwer.wer(ref_n, hyp_n),
                             seconds=round(time.time() - t0, 2)))
            print(name, r.clip_id, f"WER {rows[-1]['wer']:.2f}")
    df = pd.DataFrame(rows); df.to_csv(out / "per_clip.csv", index=False)
    summary = df.groupby("model").agg(wer=("wer", "mean"), sec_per_clip=("seconds", "mean")).round(3)
    by_cond = df.groupby(["model", "condition"]).wer.mean().unstack().round(3)
    cats = df.assign(cat=df.categories.str.split(",")).explode("cat")
    by_cat = cats.groupby(["model", "cat"]).wer.mean().unstack().round(3)
    with open(out / "summary.md", "w") as f:
        f.write("## Overall\n\n" + summary.to_markdown()
                + "\n\n## By recording condition\n\n" + by_cond.to_markdown()
                + "\n\n## By error category\n\n" + by_cat.to_markdown() + "\n")
    print(open(out / "summary.md").read())

if __name__ == "__main__":
    main()
