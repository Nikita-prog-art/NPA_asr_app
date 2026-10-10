#!/usr/bin/env python3
"""Character/word error rate of ASR output against reference sentences.

Used for the quantization study in README.md ("Качество квантизаций"): the same
audio is transcribed by every quantization level and the resulting tables are
the score of this script.

Input format
------------
Reference file:  one "<name>\\t<text>" per line.
Hypothesis file: the output of `asr_test batch <model.gguf> <list.txt>`, i.e.
                 "<path/to/name.wav>\\t<recognised text>" per line; the file stem
                 is matched against the reference name.

Both sides are normalised before scoring: lower case, `ё` -> `е`, every
character that is not a letter, a digit or a space dropped, runs of spaces
collapsed.  Numbers stay as digits, so a reference "пятнадцатое марта" against
"15 марта" counts as an error for every model equally - the comparison between
quantizations is what the script is for, not an absolute benchmark number.

Usage:
  tools/score_asr.py --ref ref.tsv --tag f16 out/f16.txt [--tag q4_0 out/q4_0.txt ...]
  tools/score_asr.py --ref ref.tsv --self                      # sanity: score nothing
"""
from __future__ import annotations

import argparse
from pathlib import Path


def normalise(text: str) -> str:
    text = text.lower().replace("ё", "е")
    out = []
    for ch in text:
        if ch.isalnum() or ch == " ":
            out.append(ch)
        else:
            out.append(" ")
    return " ".join("".join(out).split())


def edit_distance(a: list[str], b: list[str]) -> int:
    """Levenshtein distance; O(len(a)) memory."""
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
        previous = current
    return previous[-1]


def read_tsv(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        name, _, text = line.partition("\t")
        rows[Path(name.strip()).stem] = text.strip()
    return rows


def score(reference: dict[str, str], hypothesis: dict[str, str]) -> dict[str, float]:
    per_sentence = per_sentence_errors(reference, hypothesis)
    char_errors = sum(s[0] for s in per_sentence)
    char_total = sum(s[1] for s in per_sentence)
    word_errors = sum(s[2] for s in per_sentence)
    word_total = sum(s[3] for s in per_sentence)
    exact = sum(1 for s in per_sentence if s[4])
    missing = sum(1 for s in per_sentence if s[5])
    return {
        "sentences": len(reference),
        "missing": missing,
        "cer": 100.0 * char_errors / max(char_total, 1),
        "wer": 100.0 * word_errors / max(word_total, 1),
        "exact": 100.0 * exact / max(len(reference), 1),
    }


def per_sentence_errors(reference: dict[str, str],
                        hypothesis: dict[str, str]) -> list[tuple[int, int, int, int, bool, bool]]:
    """(char errors, chars, word errors, words, exact, missing) per sentence."""
    rows = []
    for name, ref_text in reference.items():
        hyp_text = hypothesis.get(name)
        missing = hyp_text is None
        if missing:
            hyp_text = ""
        ref = normalise(ref_text)
        hyp = normalise(hyp_text)
        ref_words = ref.split()
        rows.append((
            edit_distance(list(ref), list(hyp)), len(ref),
            edit_distance(ref_words, hyp.split()), len(ref_words),
            ref == hyp, missing,
        ))
    return rows


def bootstrap_wer_diff(base: list[tuple[int, int, int, int, bool, bool]],
                       other: list[tuple[int, int, int, int, bool, bool]],
                       rounds: int = 2000, seed: int = 1) -> tuple[float, float, float]:
    """Paired bootstrap over sentences: WER(other) - WER(base) with a 95 % interval.

    Sentences are resampled with replacement (the same draw for both models), so
    the interval reflects how much of the difference is due to the particular
    sentences in the set rather than to the quantization.
    """
    import random

    rng = random.Random(seed)
    n = len(base)

    def wer(rows: list[tuple[int, int, int, int, bool, bool]]) -> float:
        return 100.0 * sum(r[2] for r in rows) / max(sum(r[3] for r in rows), 1)

    point = wer(other) - wer(base)
    diffs = []
    for _ in range(rounds):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(wer([other[i] for i in idx]) - wer([base[i] for i in idx]))
    diffs.sort()
    return point, diffs[int(0.025 * rounds)], diffs[int(0.975 * rounds) - 1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--tag", action="append", default=[])
    ap.add_argument("--hyp", action="append", default=[],
                    help="hypothesis file, in the same order as --tag")
    ap.add_argument("--markdown", action="store_true", help="print a markdown table")
    ap.add_argument("--ci", action="store_true",
                    help="paired bootstrap over sentences: 95 %% interval of the WER "
                         "difference against the first model")
    args = ap.parse_args()
    if len(args.tag) != len(args.hyp):
        ap.error("--tag and --hyp must be given the same number of times")

    reference = read_tsv(args.ref)
    print(f"reference: {len(reference)} sentences from {args.ref}")
    rows = []
    for tag, hyp_path in zip(args.tag, args.hyp):
        hyp = read_tsv(Path(hyp_path))
        rows.append((tag, score(reference, hyp), per_sentence_errors(reference, hyp)))

    if args.markdown:
        print("\n| модель | CER | WER | предложений без ошибок |")
        print("|---|---|---|---|")
        for tag, s, _ in rows:
            print(f"| `{tag}` | {s['cer']:.2f} % | {s['wer']:.2f} % | {s['exact']:.1f} % |")
    else:
        for tag, s, _ in rows:
            print(f"{tag:>10}: CER {s['cer']:.2f} %  WER {s['wer']:.2f} %  "
                  f"exact {s['exact']:.1f} %  ({s['sentences']} sentences, "
                  f"{s['missing']} missing)")

    if args.ci and len(rows) >= 2:
        base_tag, _, base_rows = rows[0]
        for tag, _, other_rows in rows[1:]:
            diff, lo, hi = bootstrap_wer_diff(base_rows, other_rows)
            print(f"WER {tag} - {base_tag}: {diff:+.2f} п.п., 95 % интервал [{lo:+.2f}; {hi:+.2f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
