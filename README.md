# NER_env

NER politik-ekonomi Indonesia: GLiNER + validasi Gemini, distilasi ke IndoBERT.
Aturan proyek dan format data master ada di [CLAUDE.md](CLAUDE.md).

## Syarat

- **Python >= 3.11** (syarat paket `great`). Colab saat ini sudah memenuhi. Cek dengan `python --version`.
- Install: `pip install -r requirements.txt`

## Menjalankan test

```bash
python -m pytest -q
```

## Modul `ner/`

### `ner.config`: config JSON di Drive

```python
from ner.config import load_config, save_config
cfg = load_config("/content/drive/MyDrive/ner/config.json")   # belum ada -> default ditulis dulu
cfg["batch_size"] = 20
save_config(cfg, "/content/drive/MyDrive/ner/config.json")   # divalidasi dulu sebelum ditulis
```
Config yang salah melempar `ValueError` yang berisi **semua** masalahnya sekaligus, termasuk key yang salah ketik.

### `ner.cleaning`: dari teks mentah ke teks master

```python
from ner.cleaning import build_raw_text, is_retweet, text_for_ner, clean_for_bert, is_media_account, dedup_key
raw = build_raw_text(row["Headline"], row["Mentions"])
text = text_for_ner(raw)          # = clean_for_ner(strip_noise(raw)), satu-satunya jalur ke `text` master
```
`clean_for_ner`, `clean_for_bert`, dan `is_media_account` di-re-export dari paket `great`, bukan disalin.

### `ner.schema`: record data master + JSONL

```python
from ner.schema import make_record, validate_record, append_jsonl, read_jsonl
rec = make_record(text, spans, labels=cfg["labels"], project="prabowo_vladivostok",
                  label_source="gemini", route_reason="low_score",
                  model_version=cfg["active_model"], schema_version=cfg["schema_version"])
n = append_jsonl(cfg["paths"]["master_data"], [rec], allowed_labels=cfg["labels"])
```
- `doc_id` = hash dari `dedup_key(text)`. Teks yang hanya beda tanda baca atau huruf besar-kecil dianggap dokumen yang sama.
- `append_jsonl` aman diulang saat resume: `doc_id` yang sudah ada dilewati. Record yang tidak valid ditolak dengan `ValueError`. Baris terakhir yang terpotong karena runtime mati dibuang dengan `[WARN]`.

### `ner.formats`: konversi entitas

```python
from ner.formats import entities_dict_to_spans, spans_to_bio, bio_to_spans, spans_to_gliner, gliner_to_spans
spans, missing = entities_dict_to_spans(text, {"tokoh": ["Prabowo"], "lokasi": ["Rusia"]})
if missing:
    ...   # ada entitas LLM yang tidak ketemu di teks, atau satu nama punya dua label -> JANGAN masuk data latih
tokens, tags, n_misaligned = spans_to_bio(text, spans)   # BIO per token (tokenizer = GLiNER)
item, n_misaligned = spans_to_gliner(text, spans)       # {"tokenized_text", "ner": [[tok_start, tok_end_inklusif, label]]}
```
- Pencocokan entitas tidak peduli huruf besar-kecil dan harus kata utuh. Nama yang lebih panjang menang saat tumpang tindih.
- `n_misaligned` > 0 berarti ada span yang batasnya tidak sejajar dengan token (misalnya "Jakarta" di dalam token "Jakarta-Bandung"). Span itu melebar ke batas token. Ini dilaporkan dengan `[WARN]`.
- Evaluasi: ubah span gold dan span prediksi dengan `spans_to_bio` pada teks yang sama, lalu hitung dengan seqeval.
