#!/usr/bin/env python3
"""Audio -> phoneme timeline for the lip-sync animation.

Pipeline:
  1. faster-whisper transcribes the WAV with word timestamps
     (VAD on, so eating/crowd noise between words is dropped).
  2. Each word is broken into phonemes:
     - English words: CMU pronouncing dictionary (ARPAbet).
     - Chinese words: pinyin (per character), initial+final mapped to
       ARPAbet-like phonemes.
     Phonemes share the word's duration evenly.
  3. Each phoneme maps to one of the 12 mouth shapes in the
     Lip Sync Animation Theory chart (mouth_library_v2).
  4. Gaps > 0.25s between words become CLOSED (silence).
Output: JSON with words, phonemes and merged shape segments.
"""
import argparse, json, re
from pathlib import Path

import pronouncing
from pypinyin import pinyin, Style

# ARPAbet phoneme -> chart shape (from the Lip Sync chart labels)
PHONEME_SHAPE = {
    "M": "M_B_P", "B": "M_B_P", "P": "M_B_P",
    "F": "F_V", "V": "F_V",
    "S": "S_C_Z_SH_CH", "Z": "S_C_Z_SH_CH", "SH": "S_C_Z_SH_CH",
    "CH": "S_C_Z_SH_CH", "JH": "S_C_Z_SH_CH", "D": "S_C_Z_SH_CH",
    "T": "S_C_Z_SH_CH", "K": "S_C_Z_SH_CH", "G": "S_C_Z_SH_CH",
    "N": "S_C_Z_SH_CH", "NG": "S_C_Z_SH_CH", "HH": "S_C_Z_SH_CH",
    "AA": "AH_I_2", "AH": "AH_I_2", "AE": "AH_I_2", "AY": "AH_I_2",
    "IH": "AH_I_1", "Y": "AH_I_1",
    "IY": "EE_UH_EN_1", "UH": "EE_UH_EN_2", "EH": "EE_UH_EN_2",
    "EY": "EE_UH_EN_2", "ER": "EE_UH_EN_2",
    "OW": "OH", "AO": "OH", "AW": "OH",
    "UW": "OO_U_W_R_1", "W": "OO_U_W_R_2", "R": "OO_U_W_R_2",
    "L": "L_TH", "TH": "L_TH", "DH": "L_TH",
}
DEFAULT_SHAPE = "AH_I_1"  # neutral open, for any unmapped phoneme

PINYIN_INITIALS = ["zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n",
                   "l", "g", "k", "h", "j", "q", "x", "r", "z", "c",
                   "s", "w", "y"]
PINYIN_FINAL_TO_ARPABET = {
    "a": ["AA"], "ai": ["AY"], "an": ["AE", "N"], "ang": ["AA", "NG"],
    "ao": ["AW"], "e": ["EH"], "ei": ["EY"], "en": ["EH", "N"],
    "eng": ["EH", "NG"], "er": ["ER"], "i": ["IY"], "ia": ["Y", "AA"],
    "ian": ["Y", "AE", "N"], "iang": ["Y", "AA", "NG"], "iao": ["Y", "AW"],
    "ie": ["Y", "EH"], "in": ["IY", "N"], "ing": ["IY", "NG"],
    "iong": ["Y", "OW", "NG"], "iu": ["Y", "UW"], "o": ["OW"],
    "ong": ["OW", "NG"], "ou": ["OW"], "u": ["UW"], "ua": ["W", "AA"],
    "uai": ["W", "AY"], "uan": ["W", "AE", "N"], "uang": ["W", "AA", "NG"],
    "ue": ["W", "EH"], "ui": ["W", "EY"], "un": ["UW", "N"],
    "uo": ["W", "OW"], "v": ["UW"], "ve": ["UW", "EH"],
    "van": ["UW", "AE", "N"], "vn": ["UW", "N"],
}
# pinyin initial -> ARPAbet ('' means the syllable is a bare final)
PINYIN_INITIAL_TO_ARPABET = {
    "b": "B", "p": "P", "m": "M", "f": "F", "d": "D", "t": "T",
    "n": "N", "l": "L", "g": "G", "k": "K", "h": "HH", "j": "JH",
    "q": "CH", "x": "SH", "zh": "JH", "ch": "CH", "sh": "SH",
    "r": "R", "z": "Z", "c": "S", "s": "S", "w": "W", "y": "Y",
}
CJK = re.compile(r"[\u4e00-\u9fff]")

def pinyin_syllable_to_phonemes(syl):
    """One toneless pinyin syllable, e.g. 'hao' -> ['HH', 'AW']."""
    syl = re.sub(r"\d", "", syl).lower()
    initial = ""
    for cand in PINYIN_INITIALS:
        if syl.startswith(cand) and len(syl) > len(cand):
            initial, final = cand, syl[len(cand):]
            break
    else:
        final = syl
    phones = []
    if initial:
        phones.append(PINYIN_INITIAL_TO_ARPABET[initial])
    phones.extend(PINYIN_FINAL_TO_ARPABET.get(final, ["AH"]))
    return phones

def word_to_phonemes(word):
    """Word (one Whisper word) -> list of ARPAbet phonemes."""
    if CJK.search(word):
        phones = []
        for syl_list in pinyin(word, style=Style.TONE3, heteronym=False):
            phones.extend(pinyin_syllable_to_phonemes(syl_list[0]))
        return phones
    clean = re.sub(r"[^a-zA-Z']", "", word).lower()
    if not clean:
        return []
    phones = pronouncing.phones_for_word(clean)
    if not phones:
        return []
    return [re.sub(r"\d", "", p) for p in phones[0].split()]

MIN_HOLD = 0.0  # seconds: no merging — do not skip any phoneme.
# Every phoneme keeps its shape, however brief. (A 0.1s minimum hold
# was tried and reverted: it cut 429 segments to 109 and whole sounds
# went missing. Even a one-frame hold deleted 4.) At 12 fps a frame
# lasts 0.083s, close to the median phoneme (0.073s), so roughly one
# mouth shape lands on each frame — the classic animation rate.

def merge_short_segments(segments, min_hold=MIN_HOLD):
    """Merge every segment shorter than `min_hold` into its longer neighbour.

    Repeats until no segment is short (or only one segment is left).
    Merging only ever lengthens neighbours, so this always terminates.
    """
    segs = [dict(s) for s in segments]
    while len(segs) > 1:
        short = [i for i, s in enumerate(segs)
                 if s["end"] - s["start"] < min_hold]
        if not short:
            break
        i = short[0]
        if i == 0:
            j = 1  # first segment: merge into the next one
        elif i == len(segs) - 1:
            j = i - 1  # last segment: merge into the previous one
        else:
            # merge into whichever neighbour is longer
            prev_len = segs[i - 1]["end"] - segs[i - 1]["start"]
            next_len = segs[i + 1]["end"] - segs[i + 1]["start"]
            j = i - 1 if prev_len >= next_len else i + 1
        lo_i, hi_i = min(i, j), max(i, j)
        segs[lo_i] = {"shape": segs[j]["shape"],
                      "start": segs[lo_i]["start"],
                      "end": segs[hi_i]["end"]}
        del segs[hi_i]
    return segs

def build_timeline(words):
    """Words [{word, start, end}] -> phonemes + merged shape segments."""
    phonemes = []
    for w in words:
        phones = word_to_phonemes(w["word"])
        if not phones:
            continue
        dur = (w["end"] - w["start"]) / len(phones)
        for i, ph in enumerate(phones):
            s = w["start"] + i * dur
            phonemes.append({
                "phoneme": ph,
                "shape": PHONEME_SHAPE.get(ph, DEFAULT_SHAPE),
                "start": round(s, 3), "end": round(s + dur, 3),
                "word": w["word"],
            })
    # shape segments: merge consecutive phonemes with the same shape,
    # and insert CLOSED for gaps > 0.25s
    segments = []
    prev_end = 0.0
    for ph in phonemes:
        if ph["start"] - prev_end > 0.25:
            segments.append({"shape": "CLOSED",
                             "start": round(prev_end, 3),
                             "end": round(ph["start"], 3)})
        if segments and segments[-1]["shape"] == ph["shape"] \
                and abs(segments[-1]["end"] - ph["start"]) < 0.05:
            segments[-1]["end"] = ph["end"]
        else:
            segments.append({"shape": ph["shape"],
                             "start": ph["start"], "end": ph["end"]})
        prev_end = ph["end"]
    segments = merge_short_segments(segments)
    return phonemes, segments

def load_wav_16k(wav_path):
    """WAV -> float32 mono 16kHz numpy array (avoids faster-whisper's
    PyAV decode path, which breaks on the installed av version)."""
    import numpy as np
    from scipy.io import wavfile
    rate, data = wavfile.read(str(wav_path))
    assert rate == 16000, f"expected 16kHz wav, got {rate}"
    if data.ndim > 1:
        data = data.mean(axis=1)
    return (data.astype(np.float32) / 32768.0)

def transcribe(wav_path, model="base"):
    from faster_whisper import WhisperModel
    print(f"Loading Whisper model '{model}' (CPU, int8)...")
    m = WhisperModel(model, device="cpu", compute_type="int8")
    audio = load_wav_16k(wav_path)
    segments, info = m.transcribe(audio, word_timestamps=True,
                                  vad_filter=True, beam_size=5)
    print(f"Detected language: {info.language} "
          f"(probability {info.language_probability:.2f})")
    words, text = [], []
    for seg in segments:
        text.append(seg.text)
        for w in seg.words:
            words.append({"word": w.word.strip(),
                          "start": round(w.start, 3),
                          "end": round(w.end, 3)})
    return " ".join(text).strip(), info.language, words

def timeline_for_video(video_path, model="base"):
    """Full pipeline for one video: extract its audio, transcribe it,
    build the phoneme timeline. Returns the timeline dict."""
    import subprocess, tempfile
    video_path = Path(video_path)
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "audio.wav"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(video_path),
                        "-vn", "-ac", "1", "-ar", "16000", str(wav)],
                       check=True)
        text, language, words = transcribe(wav, model)
    phonemes, segments = build_timeline(words)
    return {"transcript": text, "language": language, "words": words,
            "phonemes": phonemes, "segments": segments}

def timeline_for_audio(audio_path, model="base"):
    """Full pipeline for one audio file: convert to 16kHz mono WAV,
    transcribe it, build the phoneme timeline. Returns the timeline dict."""
    import subprocess, tempfile
    audio_path = Path(audio_path)
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "audio.wav"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(audio_path),
                        "-vn", "-ac", "1", "-ar", "16000", str(wav)],
                       check=True)
        text, language, words = transcribe(wav, model)
    phonemes, segments = build_timeline(words)
    return {"transcript": text, "language": language, "words": words,
            "phonemes": phonemes, "segments": segments}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="WAV file (16kHz mono)")
    ap.add_argument("--output", required=True, help="timeline JSON to write")
    ap.add_argument("--model", default="base")
    args = ap.parse_args()
    text, language, words = transcribe(args.input, args.model)
    print(f"Transcript ({len(words)} words): {text[:500]}")
    phonemes, segments = build_timeline(words)
    from collections import Counter
    print(f"Phonemes: {len(phonemes)}, shape segments: {len(segments)}")
    print("Shape usage:", dict(Counter(s["shape"] for s in segments)))
    out = {"transcript": text, "language": language, "words": words,
           "phonemes": phonemes, "segments": segments}
    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"Wrote {args.output}")

if __name__ == "__main__":
    main()
