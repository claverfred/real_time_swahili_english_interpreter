"""Markdown reports for evaluation runs. Every number is shown with its conditions and method."""

from __future__ import annotations

from typing import Any


def _f(x: Any, nd: int = 3) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _ms(s: dict | None) -> str:
    if not s or s.get("median") is None:
        return "-"
    return f"{s['median'] / 1000:.2f} / {s['p95'] / 1000:.2f} / {s['p99'] / 1000:.2f} s"


def speech_report(results: dict[str, Any]) -> str:
    meta, conds = results["meta"], results["conditions"]
    lines = [f"# Speech evaluation: {meta['dataset']['name']} ({meta['dataset']['items']} recordings)", "",
             f"- Profile `{meta['profile']}`; ASR `{meta['models']['asr']}` (beam {meta['asr_beam']}), "
             f"MT `{meta['models']['mt']}`; device {meta['device']}",
             f"- Dataset: {meta['dataset'].get('subset')}, license {meta['dataset'].get('license')}, "
             f"revision {meta['dataset'].get('revision') or '-'}",
             f"- English references: {meta['dataset'].get('english_references') or 'none'}",
             f"- ASR normalization: {meta['normalization']}",
             "- Pipeline: deployed path (VAD, ASR, correction list, stabilization, sentence aggregation, MT), "
             "one recording at a time, context reset between recordings, batch speed.", "",
             "## Swahili speech recognition", "",
             "| Condition | WER (95% CI) | WER, number words as digits | WER without correction list | CER | Exact sentences | Numbers | Names | Empty | Decode p50/p95/p99 | RTF |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, c in conds.items():
        a = c["asr"]
        ci = a["wer_ci95"]
        nn = a.get("number_words_normalized", {})
        lines.append(f"| {name} | **{a['wer']:.3f}** ({ci['low']:.3f}-{ci['high']:.3f}) | {_f(nn.get('wer'))} | "
                     f"{_f(a.get('wer_uncorrected'))} | {a['cer']:.3f} | "
                     f"{a['sentences_exact']:.0%} | {_f(a['number_recall']['recall'], 2)} | "
                     f"{_f(a['name_recall']['recall'], 2)} | {a['empty_rate']:.1%} | {_ms(a['decode_ms'])} | {a['rtf']} |")
    s2t_rows = [(n, c) for n, c in conds.items() if "speech_to_english" in c]
    if s2t_rows:
        lines += ["", "## Swahili speech to English text (human references)", "",
                  "| Condition | BLEU (95% CI) | chrF (95% CI) | chrF++ | Oracle chrF (perfect ASR) | ASR-induced chrF loss | Numbers | Names | Omission | Addition |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for name, c in s2t_rows:
            s, o = c["speech_to_english"], c.get("oracle_mt", {})
            ci = s.get("ci95", {})
            lines.append(f"| {name} | **{s['bleu']}** ({'-'.join(map(str, ci.get('bleu', [])))}) | **{s['chrf']}** "
                         f"({'-'.join(map(str, ci.get('chrf', [])))}) | {s['chrf_pp']} | {o.get('chrf', '-')} | "
                         f"{c.get('asr_induced_chrf_loss', '-')} | {_f(s['number_recall']['recall'], 2)} | "
                         f"{_f(s['name_recall']['recall'], 2)} | {s['omission_rate']:.1%} | {s['addition_rate']:.1%} |")
        clean = conds.get("clean") or s2t_rows[0][1]
        if "chrf_by_asr_wer" in clean.get("speech_to_english", {}):
            lines += ["", "**Translation quality by how well the sentence was recognised** (first condition):", "",
                      "| Sentence ASR WER | Sentences | Mean sentence chrF |", "|---|---|---|"]
            for b, v in clean["speech_to_english"]["chrf_by_asr_wer"].items():
                lines.append(f"| {b} | {v['n']} | {v['chrf_mean']} |")
        sig = s2t_rows[0][1]["speech_to_english"]["signatures"]
        lines += ["", f"sacrebleu: `{sig['bleu']}`, `{sig['chrf']}`, `{sig['chrf_pp']}`"]
    first = next(iter(conds.values()))
    if first["asr"].get("by_gender"):
        lines += ["", "## By speaker gender (first condition)", "", "| Gender | Recordings | WER |", "|---|---|---|"]
        for g, v in first["asr"]["by_gender"].items():
            lines.append(f"| {g} | {v['n']} | {v['wer']:.3f} |")
    lines += ["", "## Worst recordings (first condition)", ""]
    for f in first["failure_examples"][:8]:
        lines += [f"- **{f['item_id']}**, WER {f['wer']}", f"  - said: {f['sw_ref']}", f"  - heard: {f['sw_hyp']}"]
        if f.get("en_hyp") is not None:
            lines += [f"  - English: {f['en_hyp']}", f"  - reference: {f['en_ref']}"]
    lines += ["", "## Method notes", "",
              "- WER/CER: jiwer on normalized text; CIs are percentile bootstraps over recordings.",
              "- FLEURS references write numbers as digits, the ASR writes words. 'WER, number words as digits' "
              "converts Swahili cardinal phrases to digits on both sides; ASR number recall uses the same conversion.",
              "- Name recall: capitalized non-sentence-initial words of the reference found case-insensitively "
              "in the hypothesis (a proxy).",
              "- Omission/addition: sentence word-length ratio below 0.6 / above 1.6 versus the reference (a proxy).",
              "- Perturbed conditions are simulated from clean recordings (see src/eval/perturb.py)."]
    return "\n".join(lines) + "\n"


def latency_report(r: dict[str, Any], meta: dict[str, Any]) -> str:
    lat = r["latency_from_speech_end_ms"]
    lines = [f"# Real-time latency: {r['n_recordings']} real Swahili recordings", "",
             f"- Profile `{meta['profile']}`, ASR `{meta['models']['asr']}` (beam {meta['asr_beam']}), "
             f"MT `{meta['models']['mt']}`, TTS `{meta['models']['tts']}`",
             f"- {r['audio_s']} s of audio streamed at {r['pace']}, {r['gap_s']} s pause between recordings; "
             f"{r['units']} translation units", "",
             "Latency from the speaker's last speech frame (median / p95 / p99):", "",
             "| Stage | Latency |", "|---|---|"]
    for k, label in (("asr_final", "Swahili text"), ("mt_final", "English text"),
                     ("first_audio", "First English audio"), ("full_audio", "English audio complete")):
        lines.append(f"| {label} | {_ms(lat.get(k))} |")
    lines += ["", f"- Peak RAM {r['rss_peak_mb']} MB; push blocks more than 100 ms late: {r['push_blocks_late']}; "
              f"errors: {len(r['errors'] or [])}"]
    return "\n".join(lines) + "\n"


def mt_report(r: dict[str, Any], meta: dict[str, Any]) -> str:
    ci = r.get("ci95", {})
    return "\n".join([
        f"# Text translation: FLORES-200 {meta['dataset']['split']} ({r['n']} sentences)", "",
        f"- Model `{meta['models']['mt']}`, beam {meta['mt_beam']}, batch 1; human references",
        "", "| BLEU | chrF | chrF++ | Numbers kept | Names (vs ref) | Omission | Addition | Latency p50/p95/p99 |",
        "|---|---|---|---|---|---|---|---|",
        f"| {r['bleu']} ({'-'.join(map(str, ci.get('bleu', [])))}) | {r['chrf']} ({'-'.join(map(str, ci.get('chrf', [])))}) "
        f"| {r['chrf_pp']} | {_f(r['numbers_kept_from_source']['recall'], 2)} | "
        f"{_f(r['name_recall_vs_reference']['recall'], 2)} | {r['omission_rate']:.1%} | {r['addition_rate']:.1%} | "
        f"{_ms(r['latency_ms'])} |", "",
        f"sacrebleu: `{r['signatures']['bleu']}`, `{r['signatures']['chrf']}`"]) + "\n"
