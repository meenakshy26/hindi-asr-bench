"""Optional: LoRA fine-tune whisper-small on half the clips, evaluate on the other half.
With ~25 training clips this demonstrates mechanics and measures movement; it is not a model."""
import argparse, pathlib, pandas as pd, torch, librosa, jiwer
from transformers import (WhisperProcessor, WhisperForConditionalGeneration,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments)
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from normalise import normalise

p = argparse.ArgumentParser()
p.add_argument("--audio", required=True); p.add_argument("--truth", required=True)
p.add_argument("--out", default="results/finetune")
a = p.parse_args()

truth = pd.read_csv(a.truth).sample(frac=1, random_state=7)
train, test = truth.iloc[: len(truth) // 2], truth.iloc[len(truth) // 2:]
proc = WhisperProcessor.from_pretrained("openai/whisper-small", language="hi", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05))

def prep(r):
    y, _ = librosa.load(pathlib.Path(a.audio) / r["filename"], sr=16000, mono=True)
    feats = proc.feature_extractor(y, sampling_rate=16000).input_features[0]
    labels = proc.tokenizer(r["transcript"]).input_ids
    return {"input_features": feats, "labels": labels}

ds = Dataset.from_pandas(train).map(prep, remove_columns=list(train.columns))

class Collator:
    def __call__(self, b):
        feats = torch.tensor([x["input_features"] for x in b])
        labels = proc.tokenizer.pad({"input_ids": [x["labels"] for x in b]}, return_tensors="pt").input_ids
        labels[labels == proc.tokenizer.pad_token_id] = -100
        return {"input_features": feats, "labels": labels}

args = Seq2SeqTrainingArguments(output_dir=a.out, per_device_train_batch_size=4, learning_rate=1e-4,
                                num_train_epochs=8, logging_steps=5, save_strategy="no",
                                fp16=torch.cuda.is_available(), report_to=[])
Seq2SeqTrainer(model=model, args=args, train_dataset=ds, data_collator=Collator()).train()

def wer_on(m, rows):
    m.eval(); tot = []
    for _, r in rows.iterrows():
        y, _ = librosa.load(pathlib.Path(a.audio) / r.filename, sr=16000, mono=True)
        x = proc.feature_extractor(y, sampling_rate=16000, return_tensors="pt").input_features.to(m.device)
        with torch.no_grad():
            ids = m.generate(x, language="hi", task="transcribe")
        hyp = proc.batch_decode(ids, skip_special_tokens=True)[0]
        tot.append(jiwer.wer(normalise(r.transcript), normalise(hyp)))
    return sum(tot) / len(tot)

base = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(model.device)
print(f"held-out WER  base: {wer_on(base, test):.3f}   lora: {wer_on(model, test):.3f}   (n={len(test)})")
