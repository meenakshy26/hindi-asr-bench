"""Convert any audio in --src to 16 kHz mono WAV in --dst."""
import argparse, pathlib, librosa, soundfile as sf
p = argparse.ArgumentParser(); p.add_argument("--src", required=True); p.add_argument("--dst", required=True)
a = p.parse_args(); dst = pathlib.Path(a.dst); dst.mkdir(parents=True, exist_ok=True)
for f in sorted(pathlib.Path(a.src).iterdir()):
    if f.suffix.lower() in {".wav", ".m4a", ".mp3", ".ogg", ".flac"}:
        y, _ = librosa.load(f, sr=16000, mono=True)
        sf.write(dst / (f.stem + ".wav"), y, 16000)
        print("converted", f.name)
