\# Proyek: NER politik-ekonomi Indonesia (distilasi LLM -> model NER)



\## Tujuan

Mengekstrak entitas dari cuitan/postingan media sosial berbahasa Indonesia tentang

politik dan ekonomi. Awalnya memakai GLiNER + validasi Gemini. Hasil validasi Gemini

dikumpulkan sebagai data latih untuk fine-tune IndoBERT, sehingga pemanggilan LLM

berkurang bertahap. Tujuan akhirnya kecepatan dan biaya rendah.



\## Lingkungan

\- Dijalankan di Google Colab (GPU T4). Notebook meng-clone repo ini lalu mengimpor paket `ner/`.

\- LLM: `google.colab.ai` dengan model "google/gemini-2.5-flash" (tidak punya mode JSON,

&#x20; jadi JSON di-parse dari teks jawaban).

\- Penyimpanan di Google Drive: config, data latih (JSONL), output per proyek, model.



\## Cleaning

Pakai `clean\_for\_ner`, `clean\_for\_bert`, dan `is\_media\_account` dari paket `great`

(azmkto/GREAT-Tools, publik), di-install sebagai dependensi dan di-pin ke commit tertentu.

Jangan tulis ulang atau salin fungsi-fungsi ini.



\## Struktur

\- `ner/` berisi semua logika (config, cleaning, formats, predict, routing, llm, train, evaluate)

\- `notebooks/` berisi notebook tipis: form Colab + panggil fungsi dari `ner/`

\- `tests/` berisi pytest untuk fungsi murni (tanpa GPU/LLM)

\- `reference/` berisi notebook lama sebagai acuan gaya dan logika. Jangan diubah.



\## Label (DRAFT, schema\_version 1, boleh berubah)

tokoh, instansi\_pemerintah, partai, perusahaan, kebijakan, indikator\_ekonomi, nilai\_uang, lokasi



\## Format data master (JSONL, satu dokumen per baris)

{"doc\_id": str (hash teks), "project": str, "text": str (hasil clean\_for\_ner),

&#x20;"entities": \[\[start\_char, end\_char, label], ...],

&#x20;"labels": \[label yang ditanyakan saat anotasi], "schema\_version": int,

&#x20;"label\_source": "gemini" | "manual", "route\_reason": "low\_score" | "random\_sample" | "no\_entity",

&#x20;"model\_version": str, "run\_date": "YYYY-MM-DD"}

Span memakai posisi karakter pada `text`. Format lain (BIO, format GLiNER) selalu

diturunkan dari format ini, tidak pernah disimpan sebagai sumber utama.



\## Aturan penting

\- Cleaning untuk NER mempertahankan huruf kapital dan tanda baca.

\- Baris yang entitas LLM-nya tidak ditemukan persis di teks TIDAK masuk data latih.

\- Data emas (gold) tidak pernah dipakai untuk training.

\- Split train/val per proyek atau per waktu, tidak acak per baris.

\- Metrik utama: F1 per entitas (seqeval), dilaporkan per label.

\- Semua pemanggilan LLM harus bisa di-resume (checkpoint) saat kuota habis.



\## Gaya kode

\- Komentar dan pesan log berbahasa Indonesia, singkat, format \[INFO]/\[WARN]/\[ERROR] seperti notebook lama.

\- Fungsi kecil dan bisa dites. Hindari variabel global di dalam `ner/`.

\- Sebelum menulis kode untuk tugas besar, tampilkan rencana dulu dan tunggu persetujuan.

