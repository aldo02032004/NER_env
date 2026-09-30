"""Konversi entitas: dict LLM -> span karakter -> BIO / format GLiNER, dan kebalikannya."""
import re

RE_TOKEN = re.compile(r"\w+(?:[-_]\w+)*|\S")   # sama dengan WhitespaceTokenSplitter bawaan GLiNER


def entities_dict_to_spans(text, entities):
    """{"lokasi": [nama,...], ...} -> ([[start, end, label], ...], [nama tidak ketemu / konflik]).
    Semua kemunculan, case-insensitive, kata utuh. Nama panjang duluan, span tidak tumpang tindih.
    Nama yang sama dengan dua label = konflik -> masuk missing (baris jangan dipakai training)."""
    pairs = {(n.strip(), lab) for lab, names in entities.items() for n in (names or [])
             if isinstance(n, str) and n.strip()}
    labels_by_name = {}
    for name, lab in pairs:
        labels_by_name.setdefault(name.lower(), set()).add(lab)
    conflicts = {n for n, labs in labels_by_name.items() if len(labs) > 1}

    spans = []
    missing = sorted({name for name, _ in pairs if name.lower() in conflicts})
    for name, label in sorted(pairs, key=lambda p: (-len(p[0]), p[0].lower(), p[1])):
        if name.lower() in conflicts:
            continue
        matches = list(re.finditer(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text, re.IGNORECASE))
        if not matches:
            missing.append(name)
        for m in matches:
            if not any(m.start() < e and s < m.end() for s, e, _ in spans):
                spans.append([m.start(), m.end(), label])
    return sorted(spans), missing


def tokenize(text):
    """Teks -> list (start, end) karakter per token."""
    return [m.span() for m in RE_TOKEN.finditer(text)]


def _span_to_token_range(offsets, start, end):
    """Span karakter -> (token pertama, token terakhir inklusif, sejajar?). None kalau tidak kena token."""
    idx = [i for i, (s, e) in enumerate(offsets) if s < end and start < e]
    if not idx:
        return None
    first, last = idx[0], idx[-1]
    return first, last, offsets[first][0] == start and offsets[last][1] == end


def _spans_to_token_ranges(text, spans):
    offsets = tokenize(text)
    ranges, n_misaligned = [], 0
    for s, e, label in sorted(spans):
        r = _span_to_token_range(offsets, s, e)
        if r is None or not r[2]:
            n_misaligned += 1
        if r is not None:
            ranges.append((r[0], r[1], label))
    if n_misaligned:
        print(f"[WARN] {n_misaligned} span tidak sejajar batas token (melebar ke batas token atau hilang)")
    return offsets, ranges, n_misaligned


def spans_to_bio(text, spans):
    """Return (tokens, tags, n_misaligned). Span yang batasnya tidak sejajar token melebar ke token penuh."""
    offsets, ranges, n_misaligned = _spans_to_token_ranges(text, spans)
    tags = ["O"] * len(offsets)
    for first, last, label in ranges:
        if any(t != "O" for t in tags[first:last + 1]):
            continue   # dua span di token yang sama: yang pertama menang
        tags[first] = f"B-{label}"
        tags[first + 1:last + 1] = [f"I-{label}"] * (last - first)
    return [text[s:e] for s, e in offsets], tags, n_misaligned


def bio_to_spans(text, tags):
    """Tag BIO per token (hasil tokenize(text)) -> span karakter. I- tanpa B- = awal entitas (seperti seqeval)."""
    offsets = tokenize(text)
    if len(tags) != len(offsets):
        raise ValueError(f"Jumlah tag {len(tags)} != jumlah token {len(offsets)}")
    spans, cur = [], None
    for (s, e), tag in zip(offsets, tags):
        prefix, _, label = tag.partition("-")
        if prefix == "I" and cur and cur[2] == label:
            cur[1] = e
            continue
        if cur:
            spans.append(cur)
        cur = [s, e, label] if prefix in ("B", "I") else None
    if cur:
        spans.append(cur)
    return spans


def spans_to_gliner(text, spans):
    """Return ({"tokenized_text": [...], "ner": [[tok_start, tok_end_inklusif, label], ...]}, n_misaligned)."""
    offsets, ranges, n_misaligned = _spans_to_token_ranges(text, spans)
    item = {"tokenized_text": [text[s:e] for s, e in offsets], "ner": [list(r) for r in ranges]}
    return item, n_misaligned


def gliner_to_spans(text, item):
    offsets = tokenize(text)
    if item["tokenized_text"] != [text[s:e] for s, e in offsets]:
        raise ValueError("tokenized_text tidak cocok dengan tokenize(text)")
    return sorted([offsets[ts][0], offsets[te][1], label] for ts, te, label in item["ner"])
