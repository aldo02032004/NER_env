"""Validasi NER oleh LLM lokal (GGUF via llama-cpp), per batch, bisa resume lewat log per doc_id.

`review_all` menerima `generate(prompt, n) -> str` (n = jumlah doc di prompt); buat dengan `make_generate(cfg)`.
"""
import json
import os
from collections import Counter

from ner.formats import entities_dict_to_spans, spans_to_names
from ner.routing import interleave
from ner.schema import append_jsonl, make_record, read_jsonl

ANSWERED = ("ok", "missing", "conflict")     # LLM menjawab
FINAL = ANSWERED + ("failed",)              # tidak dikirim ulang saat resume
MAX_BATCH_FAILS = 1                         # setelah ini doc dikirim sendirian (batch 1)
MAX_FAILS = 2                               # gagal lagi saat sendirian -> failed
TOKENS_PER_DOC = 150                        # pengaman max_tokens per doc; panjang jawaban sudah dibatasi schema
MAX_NAMES_PER_LABEL = 15                    # batas di schema: tanpa batas, model bisa menulis terus sampai terpotong
MAX_NAME_CHARS = 80


def log_path(cfg, project):
    return os.path.join(cfg["paths"]["checkpoint_dir"], f"llm_log_{project}.jsonl")


def answer_schema(labels, n):
    """JSON schema jawaban: tepat n objek {index, entities: {label: [nama]}}. Dipakai llama-cpp sebagai grammar."""
    names = {"type": "array", "items": {"type": "string", "maxLength": MAX_NAME_CHARS}, "maxItems": MAX_NAMES_PER_LABEL}
    entities = {"type": "object", "properties": {lab: names for lab in labels}, "required": list(labels)}
    item = {"type": "object", "properties": {"index": {"type": "integer"}, "entities": entities},
            "required": ["index", "entities"]}
    return {"type": "array", "items": item, "minItems": n, "maxItems": n}


def make_generate(cfg):
    """Muat model GGUF dari Hugging Face, return generate(prompt, n) -> str.
    Import di sini supaya test tidak butuh llama_cpp."""
    from llama_cpp import Llama
    llm = Llama.from_pretrained(repo_id=cfg["llm_model_repo"], filename=cfg["llm_model_file"],
                                n_ctx=cfg["llm_n_ctx"], n_gpu_layers=-1, verbose=False)   # -1 = semua layer di GPU

    def generate(prompt, n):
        fmt = {"type": "json_object", "schema": answer_schema(cfg["labels"], n)}
        out = llm.create_chat_completion(messages=[{"role": "user", "content": prompt}], response_format=fmt,
                                         temperature=0, max_tokens=TOKENS_PER_DOC * n + 200)
        return out["choices"][0]["message"]["content"]
    return generate


def parse_json_text(text):
    """JSON pertama di jawaban model (jaga-jaga kalau ada teks di sekitarnya)."""
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


def ask_json(generate, prompt, n, parse=lambda x: x):
    """LLM -> JSON -> parse(). None kalau gagal (temperature 0: dicoba ulang pun jawabannya sama)."""
    try:
        return parse(parse_json_text(generate(prompt, n)))
    except Exception as e:
        print(f"[WARN] Jawaban LLM tidak valid: {str(e)[:200]}")
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


def review_all(docs, preds, routes, cfg, project, generate, master_path, log_path):
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
        return ask_json(generate, build_prompt(items, cfg), len(batch), parse=parse)

    def handle(batch, result):
        master_rows, log_rows = [], []
        if result is None:
            for d in batch:
                fails[d] += 1
                status = "failed" if fails[d] >= MAX_FAILS else "parse_fail"
                log_rows.append({"doc_id": d, "status": status, "route_reason": routes[d], "attempts": fails[d]})
                if status == "failed":
                    final.add(d)
            if len(batch) > 1:
                print(f"[WARN] Batch {len(batch)} doc gagal di-parse, dicoba lagi satu per satu")
            return master_rows, log_rows
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
                    text, spans, labels=labels, project=project, label_source="llm",
                    route_reason=routes[d], model_version=cfg["active_model"],
                    schema_version=cfg["schema_version"]))
            log_rows.append({"doc_id": d, "status": status, "route_reason": routes[d],
                             "attempts": fails[d] + 1, "entities": ents, "missing": missing,
                             **compare_entities(spans_to_names(text, preds[d], labels), ents)})
            final.add(d)
        return master_rows, log_rows

    todo = [d for d in interleave(routes, preds) if d not in final]
    budget = cfg.get("llm_budget_per_run")
    print(f"[INFO] LLM ({cfg['llm_model_file']}): {len(final & set(routes))} doc sudah selesai "
          f"sebelumnya, {len(todo)} doc tersisa" + (f", run ini maksimal {budget} doc" if budget else ""))
    todo = todo[:budget] if budget else todo
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
        for batch in batches:
            master_rows, log_rows = handle(batch, ask_batch(batch))
            # master dulu, baru log: kalau mati di antaranya, batch diulang dan master tidak dobel
            append_jsonl(master_path, master_rows, allowed_labels=labels)
            append_jsonl(log_path, log_rows, key=None, validate=False)
            print(f"[INFO] {len(final & set(routes))}/{len(routes)} doc selesai")


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
