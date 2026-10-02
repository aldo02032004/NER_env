"""Load spreadsheet, siapkan kolom untuk NER, dan susun tabel export per proyek."""
import re
from collections import Counter

import pandas as pd

from ner.cleaning import build_raw_text, clean_for_bert, is_media_account, is_retweet, text_for_ner
from ner.formats import spans_to_names
from ner.schema import make_doc_id

INPUT_COLS = {"headline": "Headline", "mentions": "Mentions", "author": "Author", "media": "Media"}
HELPER_COLS = ["text_raw", "is_retweet", "text", "doc_id", "is_media", "too_short", "is_duplicate", "siap_ner"]


def xlsx_url(link):
    """Link Google Sheets/Drive -> url download .xlsx. Selain itu (path lokal) dipakai apa adanya."""
    match = re.search(r"/d/([\w-]+)", link)
    if not match:
        return link
    if "drive.google.com" in link:
        return f"https://drive.google.com/uc?export=download&id={match.group(1)}"
    return f"https://docs.google.com/spreadsheets/d/{match.group(1)}/export?format=xlsx"


def split_links(links):
    return [s for s in re.split(r"[,\s]+", links) if s] if isinstance(links, str) else list(links)


def load_sheet(links, header=1):
    """Satu/lebih link (dipisah koma) -> satu DataFrame, dengan kolom source_file."""
    frames = []
    for i, link in enumerate(split_links(links), 1):
        try:
            df = pd.read_excel(xlsx_url(link), header=header)
        except Exception as e:
            raise ValueError(f"[ERROR] Gagal baca {link}: {e}\n"
                             "Cek link-nya, dan pastikan Share -> General access = 'Anyone with the link'.") from None
        df.columns = df.columns.astype(str).str.strip()
        frames.append(df.assign(source_file=f"file_{i}"))
    if not frames:
        raise ValueError("[ERROR] Link spreadsheet masih kosong")
    return pd.concat(frames, ignore_index=True)


def prepare_frame(df, cols=INPUT_COLS, min_words=3):
    """Tambah kolom bantu NER. Tidak ada baris yang dibuang: yang tidak layak hanya ditandai.
    siap_ner = bukan akun media, tidak terlalu pendek, bukan duplikat (post asli didahulukan)."""
    need = [c for c in cols.values() if c not in df.columns]
    if need:
        raise KeyError(f"[ERROR] Kolom tidak ada di spreadsheet: {need}. Kolom yang ada: {list(df.columns)}")
    out = df.copy()
    out["text_raw"] = [build_raw_text(h, m) for h, m in zip(out[cols["headline"]], out[cols["mentions"]])]
    out["is_retweet"] = out["text_raw"].map(is_retweet)
    out["text"] = out["text_raw"].map(text_for_ner)
    out["doc_id"] = [make_doc_id(t) if t.strip() else None for t in out["text"]]
    out["is_media"] = [is_media_account(a, p) for a, p in zip(out[cols["author"]], out[cols["media"]])]
    out["too_short"] = [len(clean_for_bert(t).split()) < min_words for t in out["text"]]
    # duplikat dihitung di antara baris layak saja: retweet dari post akun media tetap diproses
    eligible = ~out["is_media"] & ~out["too_short"] & out["doc_id"].notna()
    order = out[eligible].sort_values("is_retweet", kind="stable").index
    out["is_duplicate"] = out.loc[order, "doc_id"].duplicated().reindex(out.index, fill_value=False)
    out["siap_ner"] = eligible & ~out["is_duplicate"]
    return out


def canonical(names, aliases):
    """Variasi nama -> nama baku (aliases huruf kecil), duplikat setelah dinormalisasi dibuang."""
    out = []
    for n in names:
        n = aliases.get(n.strip().lower(), n.strip())
        if n not in out:
            out.append(n)
    return out


def case_map(name_lists):
    """{nama huruf kecil: bentuk yang paling sering muncul} dari semua list nama."""
    counts = Counter(n for names in name_lists for n in names)
    best = {}
    for name, c in counts.items():
        key = name.lower()
        if key not in best or (c, name) > (counts[best[key]], best[key]):
            best[key] = name
    return best


def build_export(df, docs, preds, llm_ents, labels, aliases=None, min_score=0.0):
    """Tabel output kompatibel pipeline lama: kolom asli + doc_id, text, entities_<label>, ner_source.

    df: hasil prepare_frame. docs {doc_id: text yang diprediksi}. llm_ents {doc_id: {label: [nama]}}.
    Entitas model yang belum dicek LLM hanya dipakai kalau skornya >= min_score.
    Baris duplikat ikut entitas doc aslinya (doc_id sama). Alias & penyatuan huruf hanya di sini, bukan di master.
    """
    aliases = {k.lower(): v for k, v in (aliases or {}).items()}
    original = [c for c in df.columns if c not in HELPER_COLS]
    per_doc = {}
    for d in docs:
        if d in llm_ents:
            per_doc[d] = ({lab: llm_ents[d].get(lab, []) for lab in labels}, "gemini")
        elif d in preds:
            trusted = [s for s in preds[d] if s[3] >= min_score]
            per_doc[d] = (spans_to_names(docs[d], trusted, labels), "gliner")

    out = df[original + ["doc_id", "text"]].copy()
    found = [per_doc.get(d) for d in out["doc_id"]]
    names = {lab: [canonical(f[0][lab], aliases) if f else [] for f in found] for lab in labels}
    best = case_map(lst for col in names.values() for lst in col)   # "prabowo" -> "Prabowo"
    for lab in labels:
        out[f"entities_{lab}"] = [canonical([best[n.lower()] for n in lst], {}) for lst in names[lab]]
    out["ner_source"] = [f[1] if f else None for f in found]
    return out
