# Recording guide

Record 50 clips, 2 to 8 seconds each. Any phone voice-memo app is fine; `scripts/convert.py`
turns m4a/mp3 into 16 kHz mono WAV. Name them 001 ... 050 and add one row per clip to
ground_truth.csv with the exact words spoken in Devanagari (keep English words in Latin
script: "EMI", "pay", "ok").

Content mix (about 10 each):
- Promise to pay with a date or day: "कल कर दूँगा", "शुक्रवार शाम तक हो जाएगा"
- Numbers and amounts: "चार हज़ार आठ सौ पचास", "पंद्रह तारीख", "पाँच तारीख"
- Dispute / already paid: "पहले ही कर दिया", "ये amount ग़लत है"
- Need time / hardship: "थोड़ा समय चाहिए", "नौकरी चली गई"
- Hinglish code-switching: "payment link भेज दो", "ok ok शाम तक pay कर दूँगा"

Conditions (about 10 each): quiet room, near traffic, inside a car, on speakerphone,
speaking fast. Tag each clip's condition in the CSV.

Phase 2 labels (three more columns per row):
- intent: ptp | extension | dispute | waiver_request | distress | unclear
- ptp_date: today, tomorrow, day_after, a weekday in English, or the day-of-month number; blank if none
- amount_mentioned: the rupee amount as digits (4850); blank if none
