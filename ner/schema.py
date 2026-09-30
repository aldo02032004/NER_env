"""Format data master (lihat CLAUDE.md): pembuat record, validator, baca/tulis JSONL."""
import hashlib
import json
import os
import re
from datetime import date

from ner.cleaning import dedup_key

LABEL_SOURCES = ("gemini", "manual")
ROUTE_REASONS = ("low_score", "random_sample", "no_entity")
RE_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
FIELDS = {"doc_id": str, "project": str, "text": str, "entities": list, "labels": list,
          "schema_version": int, "label_source": str, "route_reason": str,
          "model_version": str, "run_date": str}


def make_doc_id(text):
    """Hash dari dedup_key(text): beda tanda baca/huruf besar-kecil = dokumen yang sama."""
    return hashlib.sha256(dedup_key(text).encode("utf-8")).hexdigest()[:16]


def make_record(text, entities, labels, project, label_source, route_reason,
                model_version, schema_version, run_date=None):
    return {"doc_id": make_doc_id(text), "project": project, "text": text,
            "entities": [list(s) for s in sorted(entities)], "labels": list(labels),
            "schema_version": schema_version, "label_source": label_source,
            "route_reason": route_reason, "model_version": model_version,
            "run_date": run_date or str(date.today())}


def validate_record(rec, allowed_labels=None):
    """Return list error (kosong = valid)."""
    if not isinstance(rec, dict):
        return [f"record harus dict, dapat {type(rec).__name__}"]
    errors = []
    for key, typ in FIELDS.items():
        if key not in rec:
            errors.append(f"field hilang: {key}")
        elif not isinstance(rec[key], typ) or (typ is int and isinstance(rec[key], bool)):
            errors.append(f"{key} harus {typ.__name__}, dapat {type(rec[key]).__name__}")
    extra = set(rec) - set(FIELDS)
    if extra:
        errors.append(f"field tidak dikenal: {sorted(extra)}")
    if errors:
        return errors

    text = rec["text"]
    if not text.strip():
        errors.append("text kosong")
    elif rec["doc_id"] != make_doc_id(text):
        errors.append(f"doc_id {rec['doc_id']} tidak cocok dengan text (harusnya {make_doc_id(text)})")
    if rec["label_source"] not in LABEL_SOURCES:
        errors.append(f"label_source harus salah satu {LABEL_SOURCES}")
    if rec["route_reason"] not in ROUTE_REASONS:
        errors.append(f"route_reason harus salah satu {ROUTE_REASONS}")
    if not RE_DATE.match(rec["run_date"]):
        errors.append(f"run_date harus YYYY-MM-DD, dapat {rec['run_date']!r}")
    if allowed_labels is not None:
        unknown = set(rec["labels"]) - set(allowed_labels)
        if unknown:
            errors.append(f"labels di luar config: {sorted(unknown)}")

    valid = []
    for span in rec["entities"]:
        if (isinstance(span, list) and len(span) == 3 and all(type(x) is int for x in span[:2])
                and isinstance(span[2], str)):
            valid.append(span)
        else:
            errors.append(f"span harus [start:int, end:int, label:str], dapat {span!r}")
    prev_end = -1
    for span in sorted(valid):
        s, e, label = span
        if not 0 <= s < e <= len(text):
            errors.append(f"span {span} di luar teks (panjang {len(text)})")
        if label not in rec["labels"]:
            errors.append(f"label span {label!r} tidak ada di labels")
        if s < prev_end:
            errors.append(f"span {span} tumpang tindih dengan span sebelumnya")
        prev_end = max(prev_end, e)
    return errors


def read_jsonl(path):
    """Baca JSONL. Baris terakhir terpotong (runtime mati saat menulis) dilewati dengan WARN."""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    lines = raw.split("\n")
    rows = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1:   # tidak diakhiri newline = tulisan terakhir belum selesai
                print(f"[WARN] Baris terakhir {path} terpotong, dilewati")
                continue
            raise ValueError(f"[ERROR] {path} baris {i + 1} bukan JSON valid") from None
    return rows


def _drop_partial_tail(path):
    """Potong sisa baris yang tidak selesai di akhir file, supaya append tidak menyambung ke sana."""
    with open(path, "rb+") as f:
        data = f.read()
        if not data or data.endswith(b"\n"):
            return
        cut = data.rfind(b"\n") + 1
        print(f"[WARN] {path}: {len(data) - cut} byte baris terpotong di akhir dibuang sebelum append")
        f.truncate(cut)


def append_jsonl(path, records, key="doc_id", allowed_labels=None):
    """Append record valid yang key-nya belum ada. Aman diulang saat resume. Return jumlah ditulis."""
    for rec in records:
        errors = validate_record(rec, allowed_labels)
        if errors:
            raise ValueError(f"[ERROR] Record tidak valid ({rec.get('doc_id') if isinstance(rec, dict) else rec}): "
                             + "; ".join(errors))
    seen = {r.get(key) for r in read_jsonl(path)}
    new = []
    for rec in records:
        if rec[key] not in seen:
            seen.add(rec[key])
            new.append(rec)
    if not new:
        return 0
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.exists(path):
        _drop_partial_tail(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in new))
        f.flush()
        os.fsync(f.fileno())
    return len(new)
