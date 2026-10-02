"""Pilih doc yang dikirim ke LLM. Keputusan deterministik per doc_id supaya resume konsisten."""
import hashlib
import math
from collections import Counter
from itertools import zip_longest

REASONS = ("low_score", "no_entity_capital", "no_entity_sample", "random_sample")
OPENING = "\"'([{“‘#*-"
CLOSING = "\"')]}”’"


def doc_score(spans):
    """Skor terendah entitas di doc, None kalau tidak ada entitas."""
    return min((s[3] for s in spans), default=None)


def has_mid_capital(text):
    """Ada kata berawalan huruf kapital yang BUKAN kata pertama kalimat?"""
    sentence_start = True
    for tok in text.split():
        word = tok.lstrip(OPENING)
        if word and word[0].isupper() and not sentence_start:
            return True
        sentence_start = tok.rstrip(CLOSING).endswith((".", "!", "?", "…"))
    return False


def unit_hash(doc_id, salt):
    """doc_id -> angka [0, 1) yang selalu sama untuk doc_id + salt yang sama."""
    return int(hashlib.sha256(f"{salt}:{doc_id}".encode()).hexdigest(), 16) / 16 ** 64


def route(doc_id, text, spans, threshold, sample_rate, no_entity_sample_rate):
    """Return alasan kirim ke LLM, atau None kalau hasil model dipercaya."""
    if not spans:
        if has_mid_capital(text):
            return "no_entity_capital"
        return "no_entity_sample" if unit_hash(doc_id, "no_entity") < no_entity_sample_rate else None
    if doc_score(spans) < threshold:
        return "low_score"
    return "random_sample" if unit_hash(doc_id, "sample") < sample_rate else None


def interleave(routes, preds=None):
    """Urutan doc_id selang-seling per alasan, supaya tiap putaran kuota mencakup semua route_reason.
    Dengan preds: low_score diurutkan dari skor terendah (paling ragu = paling berguna untuk latihan)."""
    groups = [[d for d, r in routes.items() if r == reason] for reason in REASONS]
    if preds is not None:
        groups[REASONS.index("low_score")].sort(key=lambda d: doc_score(preds[d]))
    return [d for row in zip_longest(*groups) for d in row if d is not None]


def route_all(docs, preds, cfg):
    """docs {doc_id: text}, preds {doc_id: spans} -> {doc_id: alasan} (hanya yang dikirim)."""
    threshold = cfg["ner_review_threshold"][cfg["active_model"]]
    routes = {}
    for d, text in docs.items():
        reason = route(d, text, preds[d], threshold, cfg["review_sample_rate"], cfg["no_entity_sample_rate"])
        if reason:
            routes[d] = reason
    counts = Counter(routes.values())
    print(f"[INFO] Routing (threshold {threshold}): {len(routes)} dari {len(docs)} doc dikirim ke LLM")
    for reason in REASONS:
        print(f"       {reason:<18}: {counts[reason]}")
    print(f"       {'tidak dikirim':<18}: {len(docs) - len(routes)}")
    print(f"[INFO] Perkiraan {math.ceil(len(routes) / cfg['batch_size'])} panggilan LLM "
          f"(batch {cfg['batch_size']}, {cfg['ai_workers']} paralel)")
    return routes
