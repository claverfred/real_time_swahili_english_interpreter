# Human evaluation protocol

Automatic metrics (WER, BLEU, chrF) compare the system's output with a single human reference.
They cannot tell whether the English *means* what the speaker said, whether it reads naturally,
or whether a mistake is harmless or dangerous. This protocol collects those judgements from
people.

## Who rates

- **Adequacy, names, numbers, terminology, error severity:** raters who are fluent in both
  Swahili and English. Ideally at least **two raters** per item, so agreement can be measured.
- **Fluency and naturalness:** fluent English speakers. Bilingual raters can do this too.
- Raters should not have built or tuned the system, and they are not shown which model produced
  the output.

## What raters receive

Create a sheet from a speech evaluation run:

```powershell
python -m src.cli evaluate speech --profile cpu                      # creates results/eval-speech-...
python -m src.cli evaluate human-sheet --from results/eval-speech-... --n 60 --audio
```

This writes `human_eval/` inside the run directory:

| File | Who gets it | Contents |
|---|---|---|
| `rating_sheet.csv` | raters | item code, the Swahili that was said, the system's English, audio file name, empty rating columns |
| `audio/H001.wav` … | raters (with `--audio`) | the system's English speech for naturalness ratings |
| `key.csv` | analyst only | English human reference, ASR text, ASR WER and sentence chrF for each item |

Items are a seeded random sample (`--seed`, default 8), shuffled, and coded `H001`, `H002`, ….
The English reference is **not** on the rating sheet, so it can't anchor the raters'
judgement. Raters compare the English output with the Swahili that was said.

## Rating scales

| Column | Scale | Question |
|---|---|---|
| `adequacy_1_5` | 1–5 | How much of the meaning of the Swahili is in the English? 5 all, 4 most, 3 much, 2 little, 1 none or wrong. |
| `fluency_1_5` | 1–5 | Ignoring the Swahili, is the English well-formed? 5 flawless, 4 good, 3 non-native, 2 disfluent, 1 incomprehensible. |
| `names_ok_y_n_na` | y / n / na | Are all people, places and organisations correct? `na` if there are none. |
| `numbers_ok_y_n_na` | y / n / na | Are all numbers, dates, amounts and units correct? |
| `terminology_ok_y_n_na` | y / n / na | Are technical or domain terms translated correctly? |
| `error_severity_none_minor_major_critical` | category | The worst error: **none**; **minor** (meaning intact); **major** (meaning partly lost or misleading); **critical** (opposite meaning, wrong number or name that matters, or offensive). |
| `naturalness_1_5` | 1–5 | (Audio) How natural does the English voice sound? 5 like a person, 1 very robotic or hard to follow. |
| `comment` | free text | What went wrong, for anything rated below 4. |

**Guidance for raters:**
- Rate each item on its own, and don't change earlier ratings after seeing later items.
- Judge the meaning a listener would take away, not word-for-word correspondence.
- If the Swahili itself is unclear, rate against the most plausible meaning and say so in
  `comment`.

## Scoring

Each rater returns their own copy of the sheet (for example `rater_a.csv`, `rater_b.csv`). Then:

```powershell
python -m src.cli evaluate human-score rater_a.csv rater_b.csv --key results/eval-speech-.../human_eval/key.csv
```

The scorer reports:
- per-scale distributions, medians, means and the share rated 4 or 5
- the share of names, numbers and terms correct
- the error-severity mix
- agreement between each pair of raters on adequacy: exact agreement, agreement within one
  point, and mean absolute difference
- adequacy grouped by how well the sentence was recognised (ASR exact, WER ≤ 0.2, WER > 0.2),
  using the key

Report the distributions, not only the means, and report the number of items and raters.
Agreement within one point below about 0.8 means the scale or instructions need clarifying
before the numbers are trusted.

## Status

The kit is built and tested. **No human ratings have been collected yet.** All quality numbers
in the README are automatic metrics against human references.
