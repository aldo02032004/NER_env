"""Validasi NER oleh LLM (Gemini), per batch, bisa resume lewat log per doc_id.

`review_all` menerima `generate(prompt) -> str`; buat dengan `make_generate(cfg, api_key)`.
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
QUOTA_WAIT = 60                             # detik tunggu saat kena limit per menit
MAX_QUOTA_WAITS = 3                         # kena limit beruntun sebanyak ini -> berhenti


def log_path(cfg, project):
    return os.path.join(cfg["paths"]["checkpoint_dir"], f"llm_log_{project}.jsonl")


def make_generate(cfg, api_key=None):
    """Fungsi generate(prompt) -> str sesuai cfg['llm_provider']. Import dilakukan di sini (bukan di atas)
    supaya test tidak butuh google.colab / google-genai."""
    if cfg["llm_provider"] == "colab":
        from google.colab import ai
        model = f"google/{cfg['llm_model']}"
        return lambda prompt: ai.generate_text(prompt, model_name=model)

    if not api_key:
        raise ValueError("[ERROR] llm_provider = gemini_api butuh API key (Colab Secrets: GOOGLE_API_KEY)")
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(response_mime_type="application/json", temperature=0)

    def generate(prompt):
        return client.models.generate_content(model=cfg["llm_model"], contents=prompt, config=config).text
    return generate


def is_quota_error(e):
    msg = str(e).lower()
    return "429" in msg or "resource_exhausted" in msg or "quota" in msg


def is_daily_quota(e):
    """Kuota harian habis: menunggu semenit tidak ada gunanya."""
    return "perday" in str(e).lower().replace("_", "")


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


def _entity_set(names):
    return {(lab, n.strip().lower()) for lab, v in names.items() for n in v}


def compare_entities(model_names, llm_names):
    """Entitas model vs LLM (per label, huruf kecil): dibenarkan, dibuang, ditambah LLM."""
    m, g = _entity_set(model_names), _entity_set(llm_names)
    return {"kept": len(m & g), "dropped": len(m - g), "added": len(g - m)}


def review_all(docs, preds, routes, cfg, project, generate, master_path, log_path,
               sleep=time.sleep, sleep_between=4):
    """Kirim doc di `routes` ke LLM, tulis yang lolos ke master, catat semua ke log.

    docs {doc_id: text}, preds {doc_id: spans model}, routes {doc_id: route_reason}.
    Urutan: selang-seling per route_reason, low_score dari skor terendah; dibatasi cfg['llm_budget_per_run'].
    Log per doc: status ok | missing | conflict | failed (final) atau parse_fail (dicoba lagi).
    Return True kalau semua doc di routes sudah final.
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
                             **compare_entities(spans_to_names(text, preds[d], labels), ents)})
            final.add(d)

    todo = [d for d in interleave(routes, preds) if d not in final]
    budget = cfg.get("llm_budget_per_run")
    print(f"[INFO] LLM ({cfg['llm_provider']}, {cfg['llm_model']}): {len(final & set(routes))} doc sudah selesai "
          f"sebelumnya, {len(todo)} doc tersisa" + (f", run ini maksimal {budget} doc" if budget else ""))
    todo = todo[:budget] if budget else todo
    workers, quota_waits = cfg["ai_workers"], 0
    with ThreadPoolExecutor(workers) as pool:
        while True:
            pending = [d for d in todo if d not in final]
            if not pending:
                done_all = all(d in final for d in routes)
                if not done_all:
                    print(f"[INFO] Batas llm_budget_per_run ({budget}) tercapai. Jalankan ulang untuk lanjut.")
                return done_all
            normal = [d for d in pending if fails[d] < MAX_BATCH_FAILS]
            batches = ([normal[i:i + cfg["batch_size"]] for i in range(0, len(normal), cfg["batch_size"])]
                       + [[d] for d in pending if fails[d] >= MAX_BATCH_FAILS])
            for g in range(0, len(batches), workers):
                group = batches[g:g + workers]
                futures = [pool.submit(ask_batch, b) for b in group]
                master_rows, log_rows, quota_err = [], [], None
                for batch, fut in zip(group, futures):
                    try:
                        result = fut.result()
                    except Exception as e:
                        if not is_quota_error(e):
                            raise
                        quota_err = e   # batch ini tidak dicatat -> dikirim ulang
                        continue
                    handle(batch, result, master_rows, log_rows)
                # master dulu, baru log: kalau mati di antaranya, batch diulang dan master tidak dobel
                append_jsonl(master_path, master_rows, allowed_labels=labels)
                append_jsonl(log_path, log_rows, key=None, validate=False)
                print(f"[INFO] {len(final & set(routes))}/{len(routes)} doc selesai")
                if quota_err is not None:
                    if is_daily_quota(quota_err) or quota_waits >= MAX_QUOTA_WAITS:
                        print(f"[STOP] Kuota habis ({str(quota_err)[:160]}). Semua hasil sebelumnya sudah "
                              "tersimpan. Jalankan ulang cell ini setelah kuota pulih.")
                        return False
                    quota_waits += 1
                    print(f"[WARN] Kena limit per menit, tunggu {QUOTA_WAIT}s ({quota_waits}/{MAX_QUOTA_WAITS})")
                    sleep(QUOTA_WAIT)
                    break   # susun ulang batch, yang kena limit dikirim lagi
                quota_waits = 0
                sleep(sleep_between)


def latest_by_doc(log_rows):
    """Baris log terakhir per doc_id."""
    return {r["doc_id"]: r for r in log_rows}


def summarize_log(log_rows, routes=None):
    """Ringkasan per route_reason: jumlah per status + perbandingan entitas model vs LLM.
    presisi = porsi entitas model yang dibenarkan LLM; recall = porsi entitas LLM yang sudah ditemukan model."""
    rows = list(latest_by_doc(log_rows).values())
    if routes is not None:
        rows = [r for r in rows if r["doc_id"] in routes]
    summary = {}
    for reason in sorted({r["route_reason"] for r in rows}):
        sub = [r for r in rows if r["route_reason"] == reason]
        answered = [r for r in sub if r["status"] in ANSWERED]
        kept, dropped, added = (sum(r.get(k, 0) for r in answered) for k in ("kept", "dropped", "added"))
        summary[reason] = {**Counter(r["status"] for r in sub), "answered": len(answered),
                           "kept": kept, "dropped": dropped, "added": added,
                           "precision": kept / (kept + dropped) if kept + dropped else None,
                           "recall": kept / (kept + added) if kept + added else None}
    return summary


def print_summary(summary):
    pct = lambda x: "-" if x is None else f"{x:.0%}"
    print(f"{'route_reason':<18} {'dijawab':>7} {'ok':>5} {'missing':>7} {'conflict':>8} {'failed':>6} "
          f"{'dibenarkan':>10} {'dibuang':>7} {'ditambah':>8} {'presisi':>7} {'recall':>6}")
    for reason, s in summary.items():
        print(f"{reason:<18} {s['answered']:>7} {s.get('ok', 0):>5} {s.get('missing', 0):>7} "
              f"{s.get('conflict', 0):>8} {s.get('failed', 0):>6} {s['kept']:>10} {s['dropped']:>7} "
              f"{s['added']:>8} {pct(s['precision']):>7} {pct(s['recall']):>6}")
    print("[INFO] presisi = entitas model yang dibenarkan LLM; recall = entitas versi LLM yang sudah ditemukan model. "
          "Angka paling jujur ada di random_sample (tidak dipilih karena ragu).")


def llm_entities(log_rows):
    """{doc_id: entitas dari LLM} untuk doc yang dijawab LLM (termasuk missing/conflict, untuk export)."""
    return {d: r["entities"] for d, r in latest_by_doc(log_rows).items() if r["status"] in ANSWERED}
