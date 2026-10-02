"""Prediksi GLiNER: potong teks panjang, batch inference, cache per doc_id supaya bisa resume."""
import hashlib
import json
import os
import re

from ner.formats import RE_TOKEN
from ner.schema import append_jsonl, read_jsonl

CHUNK_OVERLAP = 20   # token yang diulang di potongan berikutnya supaya entitas di batas tidak terbelah


def gliner_chunks(text, max_tokens, overlap=CHUNK_OVERLAP):
    """Teks -> [(offset karakter, potongan)]. Tiap potongan maksimal max_tokens token GLiNER."""
    toks = list(RE_TOKEN.finditer(text))
    chunks, i = [], 0
    while i < len(toks):
        j = min(i + max_tokens, len(toks))
        start = toks[i].start()
        chunks.append((start, text[start:toks[j - 1].end()]))
        if j == len(toks):
            break
        i = max(j - overlap, i + 1)
    return chunks


def resolve_overlaps(spans):
    """Span bertabrakan (dari potongan tumpang tindih / label beda) -> skor tertinggi menang."""
    kept = []
    for span in sorted(spans, key=lambda x: (-x[3], x[0], x[1])):
        if not any(span[0] < e and s < span[1] for s, e, *_ in kept):
            kept.append(span)
    return sorted(kept)


def predict_texts(model, texts, cfg, max_tokens, batch_size=8):
    """Batch inference GLiNER untuk banyak teks. Return per teks: [[start, end, label, score], ...]."""
    prompts = cfg["gliner_label_prompts"]
    label_of = {p: lab for lab, p in prompts.items()}
    pieces = [(i, off, chunk) for i, t in enumerate(texts) for off, chunk in gliner_chunks(t, max_tokens)]
    results = model.inference([c for _, _, c in pieces], list(prompts.values()),
                              threshold=cfg["predict_threshold"], batch_size=batch_size) if pieces else []
    spans = [[] for _ in texts]
    for (i, off, _), ents in zip(pieces, results):
        for e in ents:
            spans[i].append([off + e["start"], off + e["end"], label_of[e["label"]], round(float(e["score"]), 4)])
    return [resolve_overlaps(s) for s in spans]


def preds_cache_path(cfg, project):
    """Cache terpisah per model, threshold, dan prompt label: ganti salah satu = prediksi ulang."""
    model_slug = re.sub(r"[^\w.-]+", "_", cfg["active_model"])
    prompts = json.dumps(cfg["gliner_label_prompts"], sort_keys=True, ensure_ascii=False)
    prompt_hash = hashlib.sha256(prompts.encode("utf-8")).hexdigest()[:8]
    return os.path.join(cfg["paths"]["checkpoint_dir"],
                        f"preds_{project}_{model_slug}_t{cfg['predict_threshold']}_p{prompt_hash}.jsonl")


def predict_all(model, docs, cfg, cache_path, max_tokens=None, batch_size=8, save_every=256):
    """docs: {doc_id: text}. Hasil disimpan tiap save_every doc; doc yang sudah ada di cache dilewati.
    Return {doc_id: spans} untuk semua docs."""
    max_tokens = max_tokens or model.config.max_len
    cached = {r["doc_id"]: r["spans"] for r in read_jsonl(cache_path)}
    todo = [d for d in docs if d not in cached]
    print(f"[INFO] GLiNER: {len(docs) - len(todo)} doc dari cache, {len(todo)} doc baru")
    try:
        from tqdm.auto import tqdm
        bar = tqdm(total=len(todo), desc="GLiNER")
    except ImportError:
        bar = None
    for i in range(0, len(todo), save_every):
        part = todo[i:i + save_every]
        spans = predict_texts(model, [docs[d] for d in part], cfg, max_tokens, batch_size)
        rows = [{"doc_id": d, "spans": s} for d, s in zip(part, spans)]
        append_jsonl(cache_path, rows, validate=False)
        cached.update((r["doc_id"], r["spans"]) for r in rows)
        if bar:
            bar.update(len(part))
    if bar:
        bar.close()
    return {d: cached[d] for d in docs}
