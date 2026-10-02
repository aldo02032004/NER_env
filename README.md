# NER_env

NER politik-ekonomi Indonesia: GLiNER + validasi Gemini, distilasi ke IndoBERT.
Aturan proyek dan format data master ada di [CLAUDE.md](CLAUDE.md).

## Syarat

- **Python >= 3.11** (syarat paket `great`). Colab saat ini sudah memenuhi. Cek dengan `python --version`.
- Install: `pip install -r requirements.txt`

## Melabeli data di Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aldo02032004/NER_env/blob/main/notebooks/01_label.ipynb)

Buka [notebooks/01_label.ipynb](notebooks/01_label.ipynb) di Colab (Runtime T4 GPU), lalu ikuti petunjuk di cell pertama.
Alurnya:

```
spreadsheet -> cleaning (siap_ner) -> GLiNER -> routing -> Gemini -> master JSONL (data latih)
                                                                  -> output/<project>_entities.pkl/.csv
```

File yang ditulis ke Drive (lokasinya diatur di `paths` pada config):

| File | Isi |
|---|---|
| `master_ner.jsonl` | data latih (format di CLAUDE.md), dipakai bersama semua proyek |
| `checkpoints/preds_<project>_<model>_t<threshold>.jsonl` | cache prediksi GLiNER per doc |
| `checkpoints/llm_log_<project>.jsonl` | log Gemini per doc (status, entitas, koreksi), dipakai untuk resume |
| `output/<project>_entities.pkl` / `.csv` | kolom asli + `doc_id`, `text`, `entities_<label>`, `ner_source` |

Kalau kuota Gemini habis, jalankan ulang Cell ⑥. Doc yang sudah selesai tidak dikirim lagi.

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
Kalau config lama belum punya key baru, key itu diisi nilai default dan disimpan. Nilai yang sudah Anda ubah tetap dipertahankan.

Key yang perlu diedit:

| Key | Isi |
|---|---|
| `labels` | daftar label (draft) |
| `label_descriptions` | per label: `deskripsi`, `contoh_positif`, `contoh_negatif`, `aturan`. Semuanya masuk ke prompt Gemini. |
| `gliner_label_prompts` | per label: teks yang dikirim ke GLiNER. **Pakai bahasa Inggris** (default: "person", "political party", ...), karena GLiNER v2.1 tidak paham label bahasa Indonesia. Prompt default lama yang belum diedit otomatis diganti saat config di-load. Mengganti prompt = GLiNER memprediksi ulang (cache terpisah). |
| `predict_threshold` | skor minimum entitas GLiNER yang disimpan |
| `ner_review_threshold` | per model: doc dengan entitas di bawah skor ini dikirim ke Gemini (`low_score`) |
| `review_sample_rate` | sampel doc berskor tinggi yang tetap dicek (`random_sample`) |
| `no_entity_sample_rate` | sampel doc tanpa entitas dan tanpa huruf kapital di tengah kalimat (`no_entity_sample`) |
| `aliases` | `{project: {variasi: nama baku}}`, hanya dipakai untuk kolom export, tidak mengubah master |

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

### `ner.io`: spreadsheet dan export

- `load_sheet(links)`: satu atau lebih link Google Sheets/Drive (dipisah koma) digabung jadi satu DataFrame.
- `prepare_frame(df)`: menambah kolom `text`, `doc_id`, `is_media`, `is_retweet`, `too_short`, `is_duplicate`, dan `siap_ner`. Tidak ada baris yang dibuang. Duplikat hanya dihitung di antara baris bukan media dan tidak terlalu pendek, jadi retweet dari post akun media tetap diproses.
- `build_export(df, docs, preds, llm_ents, labels, aliases, min_score)`: menyusun tabel output. Baris duplikat (misalnya retweet) ikut entitas dari post aslinya. Entitas GLiNER yang belum dicek Gemini hanya dipakai kalau skornya >= `min_score` (notebook memakai threshold review). Variasi huruf besar-kecil ("prabowo" / "Prabowo") disatukan ke bentuk yang paling sering muncul.

### `ner.predict`: GLiNER

`predict_all(model, docs, cfg, cache_path)` memakai `model.inference` (batch). Teks panjang dipotong sesuai `max_len` model, dengan potongan yang saling tumpang tindih. Kalau dua span bertabrakan, yang skornya lebih tinggi dipakai. Hasil disimpan per doc, jadi bisa di-resume.

### `ner.routing`: doc mana yang dicek LLM

| route_reason | Syarat |
|---|---|
| `low_score` | ada entitas dengan skor < `ner_review_threshold` |
| `no_entity_capital` | tidak ada entitas, tapi ada kata berhuruf kapital yang bukan kata pertama kalimat |
| `no_entity_sample` | tidak ada entitas, masuk sampel `no_entity_sample_rate` |
| `random_sample` | skor tinggi, masuk sampel `review_sample_rate` |

Sampel ditentukan dari hash `doc_id`, jadi keputusannya selalu sama setiap kali dijalankan.

### `ner.llm`: validasi Gemini

`review_all(docs, preds, routes, cfg, project, generate, master_path, log_path)`. Fungsi `generate(prompt) -> str` dioper dari notebook.
- Status per doc di log: `ok` (masuk master), `missing` (entitas tidak ketemu di teks), `conflict` (satu nama punya dua label), dan `failed`.
- Kalau jawaban batch gagal di-parse 2 kali, doc di batch itu dikirim sendiri-sendiri. Kalau masih gagal, statusnya `failed` dan tidak diulang lagi.
- Kalau kuota habis, proses berhenti. Semua hasil sebelumnya sudah tersimpan, dan saat dijalankan ulang hanya doc yang belum final yang dikirim.
- `summarize_log` / `print_summary`: jumlah per status dan correction rate untuk setiap route_reason.
