"""Phase 2: turn a transcript into the structured fields a collections system acts on.

Two extractors, same output schema:
  rules  — deterministic keyword/number rules (no API, auditable)
  llm    — Claude via the Anthropic API (set ANTHROPIC_API_KEY)

Output per transcript: {"intent": ptp|extension|dispute|waiver_request|distress|unclear,
                        "ptp_date": e.g. "tomorrow" | "5" | "friday" | null,
                        "amount_mentioned": integer rupees | null}
"""
import json, os, re

HI_NUM = {"एक":1,"दो":2,"तीन":3,"चार":4,"पाँच":5,"पांच":5,"छह":6,"छः":6,"सात":7,"आठ":8,"नौ":9,"दस":10,"ग्यारह":11,"बारह":12,"तेरह":13,"चौदह":14,"सोलह":16,"सत्रह":17,"अठारह":18,"उन्नीस":19,
          "पंद्रह":15,"बीस":20,"पच्चीस":25,"तीस":30,"चालीस":40,"पचास":50,"साठ":60,"सत्तर":70,"अस्सी":80,"नब्बे":90,"सौ":100,"हज़ार":1000,"हजार":1000,"लाख":100000}
# ordered: specific (weekday, day word) before vague (evening)
DAYS = [("सोमवार","monday"),("monday","monday"),("मंगलवार","tuesday"),("tuesday","tuesday"),("बुधवार","wednesday"),("wednesday","wednesday"),
        ("गुरुवार","thursday"),("thursday","thursday"),("शुक्रवार","friday"),("friday","friday"),("शनिवार","saturday"),("saturday","saturday"),
        ("रविवार","sunday"),("sunday","sunday"),("परसों","day_after"),("parso","day_after"),("कल","tomorrow"),("kal","tomorrow"),
        ("tomorrow","tomorrow"),("आज","today"),("aaj","today"),("today","today"),("शाम","today"),("evening","today")]

RX = {
  "distress": re.compile(r"नौकरी|अस्पताल|हॉस्पिटल|बीमार|गुज़र|गुजर|मृत्यु|naukri|job|hospital|bimar|passed away|death", re.I),
  "dispute":  re.compile(r"पहले ही|कर दिया|कर चुका|ग़लत|गलत|ज़्यादा|ज्यादा|already paid|galat|zyada|wrong amount|dispute", re.I),
  "waiver":   re.compile(r"माफ़|माफ|waive|maaf|छोड़ दो|हटा दो", re.I),
  "extension":re.compile(r"समय चाहिए|थोड़ा समय|अगले हफ्ते|अगले महीने|बाद में|अभी नहीं|मुश्किल|time chahiye|next week|next month|later|abhi nahi|mushkil", re.I),
  "ptp":      re.compile(r"कर दूँगा|कर दूंगा|कर दूँगी|कर दूंगी|करूँगा|करूंगा|हो जाएगा|kar dunga|kar doonga|karunga|will pay|pay kar|i'll pay|ok", re.I),
}

def hindi_number(text):
    """Very small Hindi number parser: 'चार हज़ार आठ सौ पचास' -> 4850."""
    toks = [t for t in re.split(r"\s+", text) if t in HI_NUM]
    if not toks: return None
    total, cur = 0, 0
    for t in toks:
        v = HI_NUM[t]
        if v >= 100: cur = (cur or 1) * v; total += cur; cur = 0
        else: cur += v
    return total + cur or None

def extract_rules(text):
    t = text.lower()
    intent = next((k for k in ("distress","dispute","waiver","extension","ptp") if RX[k].search(t)), "unclear")
    if intent == "waiver": intent = "waiver_request"
    ptp_date = None
    if intent == "ptp":
        for k, v in DAYS:
            if k in t: ptp_date = v; break
        m = re.search(r"(\d{1,2})\s*(तारीख|tarikh|tareekh|th)", t)
        if m: ptp_date = m.group(1)
        if not ptp_date:
            for k in ("पाँच","पांच","दस","पंद्रह","बीस","पच्चीस","तीस"):
                if k in t and "तारीख" in t: ptp_date = str(HI_NUM[k]); break
    digits = re.search(r"(\d[\d,]{2,})", t)
    amount = int(digits.group(1).replace(",", "")) if digits else hindi_number(text)
    if amount and amount < 100: amount = None     # a date, not an amount
    return {"intent": intent, "ptp_date": ptp_date, "amount_mentioned": amount}

def extract_llm(text):
    import anthropic
    client = anthropic.Anthropic()
    prompt = f"""A borrower said this on an EMI reminder call (Hindi/Hinglish, possibly mis-transcribed):
"{text}"
Return ONLY JSON: {{"intent": one of ptp|extension|dispute|waiver_request|distress|unclear,
"ptp_date": a day word (today, tomorrow, day_after, monday..sunday) or a day-of-month number as a string, or null,
"amount_mentioned": integer rupees or null}}"""
    r = client.messages.create(model="claude-sonnet-4-6", max_tokens=200, messages=[{"role":"user","content":prompt}])
    txt = r.content[0].text.strip().strip("`")
    txt = txt[txt.find("{"): txt.rfind("}")+1]
    return json.loads(txt)

def extract(text, method="rules"):
    return extract_llm(text) if method == "llm" and os.getenv("ANTHROPIC_API_KEY") else extract_rules(text)

if __name__ == "__main__":
    import sys
    print(extract(" ".join(sys.argv[1:]) or "पाँच तारीख को सैलरी आती है तब कर दूँगा"))
