"""Validasi NER oleh LLM (Gemini lewat google.colab.ai), per batch, bisa resume lewat log per doc_id.

`generate(prompt) -> str` selalu dioper dari notebook, jadi modul ini tidak meng-import google.colab.
"""
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from ner.formats import entities_dict_to_spans, spans_to_names
from ner.routing import interleave
from ner.schema import append_jsonl, make_record, read_jsonl

ANSWERED = ("ok", "missing", "conflict")     # LLM menjawab
FINAL = ANSWERED + ("failed",)              # tidak dikirim ulang saat resume
MAX_BATCH_FAILS = 2                         # setelah ini doc dikirim sendirian (batch 1)
MAX_FAILS = 3                               # gagal lagi saat sendirian -> failed


def log_path(cfg, project):
    return os.path.join(cfg["paths"]["checkpoint_dir"], f"llm_log_{project}.jsonl")


def is_quota_error(e):
    msg = str(e).lower()
    return "429" in msg or "resource_exhausted" in msg or "quota" in msg


def parse_json_text(text):
    """JSON pertama di jawaban model (Colab AI tidak punya mode JSON)."""
    starts = [i for i in (text.find("["), text.find("{")) if i >= 0]
    if not starts:
        raise ValueError("tidak ada JSON di jawaban model")
    return json.JSONDecoder().raw_decode(text[min(starts):])[0]


def order_by_index(parsed, n):
    """List objek ber-field 'index' (1..n) -> list urut. Error kalau ada index yang hilang."""
    if not isinstance(parsed, list):
        raise ValueError(f"Jawaban harus JSON array, dapat {type(parsed).__name__}")
    by_idx = {int(it["index"]): it for it in parsed if isinstance(it, dict) and "index" in it}
    if set(by_idx) != set(range(1, n + 1)):
        raise ValueError(f"Index hasil tidak lengkap: dapat {sorted(by_idx)}, harus 1..{n}")
    return [by_idx[i] for i in range(1, n + 1)]


def ask_json(generate, prompt, parse=lambda x: x, retry=2, sleep=time.sleep):
    """LLM -> JSON -> parse(). Kuota habis langsung di-raise; None kalau semua retry gagal."""
    for attempt in range(retry):
        try:
            return parse(parse_json_text(generate(prompt)))
        except Exception as e:
            if is_quota_error(e):
                raise
            wait = (20 if "503" in str(e) or "UNAVAILABLE" in str(e) else 5) * (attempt + 1)
            print(f"[WARN] LLM gagal ({attempt + 1}/{retry}): {str(e)[:200]} -> tunggu {wait}s")
            sleep(wait)
    return None


def build_prompt(items, cfg):
    """items: list (text, {label: [nama prediksi model]})."""
    label_block = "\n".join(
        f"- {lab}: {d['deskripsi']}\n"
        f"  Contoh: {', '.join(d['contoh_positif'])}\n"
        f"  Bukan: {', '.join(d['contoh_negatif'])}\n"
        f"  Aturan: {' '.join(d['aturan'])}"
        for lab, d in ((lab, cfg["label_descriptions"][lab]) for lab in cfg["labels"]))
    joined = "\n".join(f'{i}. Teks: "{text}"\n   Prediksi model: {json.dumps(ents, ensure_ascii=False)}'
                       for i, (text, ents) in enumerate(items, 1))
    empty = json.dumps({lab: [] for lab in cfg["labels"]}, ensure_ascii=False)
    return f"""Kamu validator NER untuk cuitan/postingan bahasa Indonesia tentang politik dan ekonomi.

Label:
{label_block}

Untuk setiap item, periksa prediksi model: pindahkan entitas yang salah label, tambahkan yang
terlewat, hapus yang bukan entitas.
- Tulis entitas PERSIS seperti di teks (ejaan, singkatan, huruf besar-kecil). Jangan dinormalisasi,
  dilengkapi, atau diterjemahkan.
- Satu nama hanya boleh punya satu label.
- Hanya pakai label di atas. Label tanpa entitas diisi list kosong.

Item:
{joined}

Jawab HANYA JSON array berisi tepat {len(items)} objek, tanpa markdown, dengan format:
{{"index": <nomor item>, "entities": {empty}}}"""


def clean_entities(entities, labels):
    """Jawaban LLM -> {label: [nama]} untuk label yang dikenal. Return (entities, jumlah nama label asing)."""
    if not isinstance(entities, dict):
        raise ValueError(f"entities harus objek, dapat {type(entities).__name__}")
    out = {lab: [n for n in (entities.get(lab) or []) if isinstance(n, str) and n.strip()] for lab in labels}
    n_unknown = sum(len(v) for k, v in entities.items() if k not in labels and isinstance(v, list))
    return out, n_unknown


def _name_sets(names):
    return {lab: sorted({n.strip().lower() for n in v}) for lab, v in names.items()}


def review_all(docs, preds, routes, cfg, project, generate, master_path, log_path,
               sleep=time.sleep, sleep_between=4):
    """Kirim doc di `routes` ke LLM, tulis yang lolos ke master, catat semua ke log.

    docs {doc_id: text}, preds {doc_id: spans model}, routes {doc_id: route_reason}.
    Log per doc: status ok | missing | conflict | failed (final) atau parse_fail (dicoba lagi).
    Return True kalau semua doc sudah final, False kalau berhenti karena kuota.
    """
    labels = cfg["labels"]
    final, fails = set(), Counter()
    for r in read_jsonl(log_path):
        if r["status"] in FINAL:
            final.add(r["doc_id"])
        elif r["status"] == "parse_fail":
            fails[r["doc_id"]] += 1

    def ask_batch(batch):
        items = [(docs[d], spans_to_names(docs[d], preds[d], labels)) for d in batch]
        parse = lambda p: [clean_entities(it.get("entities"), labels) for it in order_by_index(p, len(batch))]
        return ask_json(generate, build_prompt(items, cfg), parse=parse, sleep=sleep)

    def handle(batch, result, master_rows, log_rows):
        if result is None:
            for d in batch:
                fails[d] += 1
                status = "failed" if fails[d] >= MAX_FAILS else "parse_fail"
                log_rows.append({"doc_id": d, "status": status, "route_reason": routes[d], "attempts": fails[d]})
                if status == "failed":
                    final.add(d)
            print(f"[WARN] Batch {len(batch)} doc gagal di-parse, dicoba lagi "
                  f"{'sendiri-sendiri' if len(batch) > 1 else '(atau ditandai failed)'}")
            return
        for d, (ents, n_unknown) in zip(batch, result):
            if n_unknown:
                print(f"[WARN] {d}: {n_unknown} nama dengan label di luar config dibuang")
            text = docs[d]
            spans, missing = entities_dict_to_spans(text, ents)
            labels_of = {}
            for lab, names in ents.items():
                for n in names:
                    labels_of.setdefault(n.strip().lower(), set()).add(lab)
            conflict = any(len(labs) > 1 for labs in labels_of.values())
            status = "conflict" if conflict else "missing" if missing else "ok"
            if status == "ok":
                master_rows.append(make_record(
                    text, spans, labels=labels, project=project, label_source="gemini",
                    route_reason=routes[d], model_version=cfg["active_model"],
                    schema_version=cfg["schema_version"]))
            log_rows.append({"doc_id": d, "status": status, "route_reason": routes[d],
                             "attempts": fails[d] + 1, "entities": ents, "missing": missing,
                             "corrected": _name_sets(ents) != _name_sets(spans_to_names(text, preds[d], labels))})
            final.add(d)

    order = interleave(routes)
    todo = [d for d in order if d not in final]
    print(f"[INFO] LLM: {len(final & set(routes))} doc sudah selesai sebelumnya, {len(todo)} doc tersisa")
    workers = cfg["ai_workers"]
    with ThreadPoolExecutor(workers) as pool:
        while True:
            pending = [d for d in order if d not in final]
            if not pending:
                return True
            normal = [d for d in pending if fails[d] < MAX_BATCH_FAILS]
            batches = ([normal[i:i + cfg["batch_size"]] for i in range(0, len(normal), cfg["batch_size"])]
                       + [[d] for d in pending if fails[d] >= MAX_BATCH_FAILS])
            for g in range(0, len(batches), workers):
                group = batches[g:g + workers]
                futures = [pool.submit(ask_batch, b) for b in group]
                master_rows, log_rows, quota_hit = [], [], False
                for batch, fut in zip(group, futures):
                    try:
                        result = fut.result()
                    except Exception as e:
                        if not is_quota_error(e):
                            raise
                        quota_hit = True   # batch ini tidak dicatat -> dikirim ulang saat resume
                        continue
                    handle(batch, result, master_rows, log_rows)
                # master dulu, baru log: kalau mati di antaranya, batch diulang dan master tidak dobel
                append_jsonl(master_path, master_rows, allowed_labels=labels)
                append_jsonl(log_path, log_rows, key=None, validate=False)
                done = len(final & set(routes))
                print(f"[INFO] {done}/{len(routes)} doc selesai")
                if quota_hit:
                    print("[STOP] Kuota habis / rate limit. Semua hasil sebelumnya sudah tersimpan. "
                          "Tunggu, (turunkan ai_workers), lalu jalankan ulang cell ini.")
                    return False
                sleep(sleep_between)


def latest_by_doc(log_rows):
    """Baris log terakhir per doc_id."""
    return {r["doc_id"]: r for r in log_rows}


def summarize_log(log_rows, routes=None):
    """Ringkasan per route_reason: jumlah per status + correction rate (dari doc yang dijawab LLM)."""
    rows = list(latest_by_doc(log_rows).values())
    if routes is not None:
        rows = [r for r in rows if r["doc_id"] in routes]
    summary = {}
    for reason in sorted({r["route_reason"] for r in rows}):
        sub = [r for r in rows if r["route_reason"] == reason]
        answered = [r for r in sub if r["status"] in ANSWERED]
        n_corr = sum(r["corrected"] for r in answered)
        summary[reason] = {**Counter(r["status"] for r in sub), "answered": len(answered),
                           "corrected": n_corr,
                           "correction_rate": n_corr / len(answered) if answered else None}
    return summary


def print_summary(summary):
    print(f"{'route_reason':<18} {'dijawab':>8} {'ok':>6} {'missing':>8} {'conflict':>9} {'failed':>7} {'koreksi':>8}")
    for reason, s in summary.items():
        rate = "-" if s["correction_rate"] is None else f"{s['correction_rate']:.1%}"
        print(f"{reason:<18} {s['answered']:>8} {s.get('ok', 0):>6} {s.get('missing', 0):>8} "
              f"{s.get('conflict', 0):>9} {s.get('failed', 0):>7} {rate:>8}")
    answered = sum(s["answered"] for s in summary.values())
    corrected = sum(s["corrected"] for s in summary.values())
    if answered:
        print(f"[INFO] Total correction rate: {corrected / answered:.1%} dari {answered} doc")


def llm_entities(log_rows):
    """{doc_id: entitas dari LLM} untuk doc yang dijawab LLM (termasuk missing/conflict, untuk export)."""
    return {d: r["entities"] for d, r in latest_by_doc(log_rows).items() if r["status"] in ANSWERED}
