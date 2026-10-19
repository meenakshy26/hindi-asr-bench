# Hindi ASR benchmark for collections calls

How well do off-the-shelf speech-to-text models hear a borrower on an EMI reminder call?

This repo benchmarks open ASR models on a small, purpose-built test set of Hindi and Hinglish
utterances that occur on lending collections calls: amounts, dates, promises to pay, disputes,
and the code-switching real borrowers use. It reports word error rate (WER) overall and by
error category, so a product team can choose a model with evidence rather than a demo.

**Why this exists.** A voice agent for BFSI lives or dies on one number: did it hear
"paanch tarikh" or "pandrah tarikh"? Vendor demos use clean English. This set does not.

## Test set

- 50 utterances, recorded on a phone by one speaker (so results are indicative, not
  statistically robust; see Limitations)
- 5 conditions: quiet, street noise, inside a car, speakerphone, fast speech
- Every clip has a hand-written ground-truth transcript in `transcripts/ground_truth.csv`
- Each clip is tagged with the error categories it can expose: `number`, `date`, `hinglish`,
  `name`, `noise`

## Models compared

| Model | Why it is here |
|---|---|
| openai/whisper-tiny | the latency floor |
| openai/whisper-small | the usual on-device compromise |
| openai/whisper-medium | the accuracy reference most teams start from |
| ai4bharat/indicwhisper (Hindi) | trained specifically on Indian speech |

## Phase 1: how well do models hear? (`scripts/evaluate.py`)

Word error rate per model, per recording condition, per error category. Written to `results/summary.md`.

## Phase 2: does the transcript still produce the right outcome? (`scripts/score_tasks.py`)

A transcript is only useful if the system can act on it. Phase 2 runs an extractor over every
model's transcript and pulls out the three fields a collections system needs:

- **intent** — ptp, extension, dispute, waiver_request, distress, unclear
- **ptp_date** — the day the borrower committed to
- **amount_mentioned** — any rupee amount they said

and scores them against what was actually said (`ground_truth.csv` carries the labels).
The result is one table per model: word accuracy next to intent accuracy, date accuracy and
amount accuracy. The gap between the columns is the finding: how wrong a transcript can be and
still be actionable, and where errors (numbers, dates) cost money.

Two extractors share one schema: `rules` (deterministic, auditable, no API) and `llm`
(any OpenAI-compatible chat API: set `LLM_API_KEY`, and optionally `LLM_BASE_URL` and `LLM_MODEL`). Comparing them answers a second product question:
how much does an LLM understanding layer recover from ASR mistakes?

## Results

Filled in after running; see `results/summary.md` and `results/task_summary.md`.

## Running it

```bash
pip install -r requirements.txt
python scripts/evaluate.py --audio audio --truth transcripts/ground_truth.csv --out results
python scripts/score_tasks.py --method rules
LLM_API_KEY=... python scripts/score_tasks.py --method llm   # optional; any OpenAI-compatible provider
```

Or open `notebooks/benchmark.ipynb` in Google Colab (free T4 is enough; medium takes ~10 min for 50 clips).

## Optional: fine-tuning experiment

`scripts/finetune_lora.py` adapts whisper-small with LoRA on 25 clips and evaluates on the
other 25. With this little data the point is not a better model; it is to show the mechanics
and how much (or how little) WER moves. Results are reported with that caveat.

## Limitations

- One speaker, one phone, 50 clips. Enough to rank models on this domain, not enough to
  claim a production WER.
- WER on Hindi is sensitive to normalisation (chandrabindu vs anusvara, Devanagari vs
  romanised numerals). `scripts/normalise.py` documents the rules used; all models are
  scored with the same rules.

## Author

Meenakshy Bijoy, product manager (fintech & AI). Built to understand what a voice agent for
collections actually needs from its speech layer.
