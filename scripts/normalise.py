"""Text normalisation applied identically to references and hypotheses before WER."""
import re, unicodedata

DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

def normalise(text: str) -> str:
    t = unicodedata.normalize("NFC", text)
    t = t.translate(DEVANAGARI_DIGITS)
    t = t.replace("ँ", "ं")                      # chandrabindu -> anusvara
    t = re.sub(r"[।\.,!?\"'\-–—()]", " ", t)      # punctuation
    t = t.lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t
