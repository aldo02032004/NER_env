"""Config proyek: dibaca/disimpan sebagai JSON di Drive, divalidasi sebelum dipakai."""
import copy
import json
import os
import re

DRIVE_DIR = "/content/drive/MyDrive/ner"

LABEL_DESCRIPTIONS = {
    "tokoh": {
        "deskripsi": "Nama orang: politisi, pejabat, pengusaha, tokoh publik, termasuk julukan dan username yang jelas merujuk ke orang tertentu.",
        "contoh_positif": ["Prabowo", "Sri Mulyani", "Jokowi", "Mulyono", "Gibran"],
        "contoh_negatif": ["presiden", "menteri keuangan", "rakyat", "netizen"],
        "aturan": ["Jabatan tanpa nama bukan tokoh ('menkeu' saja bukan tokoh).",
                   "Gelar/jabatan yang menempel pada nama tidak ikut ditulis kecuali memang tertulis bersama di teks ('Pak Jokowi' -> 'Jokowi')."],
    },
    "instansi_pemerintah": {
        "deskripsi": "Lembaga negara, kementerian, badan, pemerintah daerah, BUMN regulator, lembaga internasional antarnegara.",
        "contoh_positif": ["Kemenkeu", "Bank Indonesia", "DPR", "KPK", "OJK", "Pemprov DKI", "IMF"],
        "contoh_negatif": ["pemerintah", "aparat", "Gerindra", "Pertamina"],
        "aturan": ["Kata umum 'pemerintah' atau 'negara' tanpa nama lembaga bukan entitas.",
                   "BUMN yang berjualan (Pertamina, PLN) masuk perusahaan, bukan instansi_pemerintah."],
    },
    "partai": {
        "deskripsi": "Nama partai politik, termasuk singkatan.",
        "contoh_positif": ["Gerindra", "PDIP", "PDI Perjuangan", "Golkar", "PKS", "NasDem"],
        "contoh_negatif": ["koalisi", "oposisi", "KIM Plus"],
        "aturan": ["Koalisi atau kubu politik bukan partai."],
    },
    "perusahaan": {
        "deskripsi": "Nama perusahaan swasta, BUMN, bank komersial, startup, merek korporasi.",
        "contoh_positif": ["Pertamina", "PLN", "Bank Mandiri", "GoTo", "Freeport"],
        "contoh_negatif": ["investor", "pengusaha", "industri tekstil"],
        "aturan": ["Sektor atau industri bukan perusahaan."],
    },
    "kebijakan": {
        "deskripsi": "Nama kebijakan, program pemerintah, undang-undang, peraturan, atau skema bantuan yang punya nama spesifik.",
        "contoh_positif": ["Makan Bergizi Gratis", "UU Cipta Kerja", "tax amnesty", "hilirisasi nikel", "PPN 12%"],
        "contoh_negatif": ["kebijakan pemerintah", "aturan baru", "program kerja"],
        "aturan": ["Harus menyebut kebijakan tertentu, bukan istilah umum 'kebijakan'."],
    },
    "indikator_ekonomi": {
        "deskripsi": "Nama indikator atau besaran ekonomi makro/pasar.",
        "contoh_positif": ["inflasi", "suku bunga", "BI Rate", "IHSG", "pertumbuhan ekonomi", "nilai tukar rupiah"],
        "contoh_negatif": ["ekonomi", "harga", "krisis"],
        "aturan": ["Tulis nama indikatornya saja; angka di sebelahnya bukan bagian entitas kecuali berupa nilai uang."],
    },
    "nilai_uang": {
        "deskripsi": "Jumlah uang beserta mata uangnya.",
        "contoh_positif": ["Rp 50 ribu", "Rp71 triliun", "USD 1 miliar", "10 juta rupiah"],
        "contoh_negatif": ["mahal", "5%", "banyak uang"],
        "aturan": ["Persentase bukan nilai_uang.", "Tulis satuan mata uang dan angkanya persis seperti di teks."],
    },
    "lokasi": {
        "deskripsi": "Nama negara, provinsi, kota, daerah, atau tempat geografis tertentu.",
        "contoh_positif": ["Indonesia", "Rusia", "Jakarta", "IKN", "Vladivostok", "Papua"],
        "contoh_negatif": ["daerah", "luar negeri", "desa"],
        "aturan": ["Lokasi yang menjadi bagian nama instansi ('Pemprov DKI') ikut instansi, bukan lokasi."],
    },
}

# GLiNER v2.1 dilatih terutama dengan bahasa Inggris: prompt label Indonesia membuat tebakannya asal.
GLINER_PROMPTS_EN = {
    "tokoh": "person",
    "instansi_pemerintah": "government agency",
    "partai": "political party",
    "perusahaan": "company",
    "kebijakan": "government policy or program",
    "indikator_ekonomi": "economic indicator",
    "nilai_uang": "money amount",
    "lokasi": "location",
}

DEFAULTS = {
    "labels": list(LABEL_DESCRIPTIONS),
    "label_descriptions": LABEL_DESCRIPTIONS,
    "gliner_label_prompts": GLINER_PROMPTS_EN,
    "schema_version": 1,
    "active_model": "urchade/gliner_medium-v2.1",
    "predict_threshold": 0.3,     # skor minimum entitas GLiNER yang disimpan
    "ner_review_threshold": {"urchade/gliner_medium-v2.1": 0.6},   # skor di bawah ini dikirim ke LLM
    "review_sample_rate": 0.05,       # sampel acak skor tinggi yang tetap dicek LLM
    "no_entity_sample_rate": 0.1,     # sampel doc tanpa entitas & tanpa huruf kapital di tengah kalimat
    "batch_size": 40,
    "ai_workers": 2,
    "aliases": {},                    # {project: {variasi huruf kecil: nama baku}}, hanya untuk export
    "paths": {
        "master_data": f"{DRIVE_DIR}/master_ner.jsonl",
        "output_dir": f"{DRIVE_DIR}/output",
        "model_dir": f"{DRIVE_DIR}/models",
        "checkpoint_dir": f"{DRIVE_DIR}/checkpoints",
    },
}
DESCRIPTION_KEYS = {"deskripsi", "contoh_positif", "contoh_negatif", "aturan"}

RE_LABEL = re.compile(r"^[a-z][a-z0-9_]*$")


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def _is_rate(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and 0 <= x <= 1


def _is_str_list(x):
    return isinstance(x, list) and all(isinstance(s, str) and s.strip() for s in x)


def _check_per_label(problems, cfg, key, labels, check_value, what):
    """Objek {label: nilai} harus punya tepat semua label, dan tiap nilai lolos check_value."""
    value = cfg[key]
    if not isinstance(value, dict):
        problems.append(f"{key} harus objek {{label: {what}}}")
        return
    if isinstance(labels, list) and set(value) != set(labels):
        missing, extra = set(labels) - set(value), set(value) - set(labels)
        problems.append(f"{key} harus berisi tepat semua labels (kurang: {sorted(missing)}, lebih: {sorted(extra)})")
    bad = [lab for lab, v in value.items() if not check_value(v)]
    if bad:
        problems.append(f"{key} tidak valid untuk label {bad}: harus {what}")


def _valid_description(d):
    return (isinstance(d, dict) and set(d) == DESCRIPTION_KEYS
            and isinstance(d["deskripsi"], str) and d["deskripsi"].strip()
            and all(_is_str_list(d[k]) for k in ("contoh_positif", "contoh_negatif", "aturan")))


def validate_config(cfg):
    """Raise ValueError berisi SEMUA masalah config sekaligus."""
    if not isinstance(cfg, dict):
        raise ValueError(f"Config harus objek JSON, dapat {type(cfg).__name__}")
    problems = []
    unknown = set(cfg) - set(DEFAULTS)
    missing = set(DEFAULTS) - set(cfg)
    if unknown:
        problems.append(f"key tidak dikenal (salah ketik?): {sorted(unknown)}")
    if missing:
        problems.append(f"key wajib hilang: {sorted(missing)}")

    labels = cfg.get("labels")
    if "labels" in cfg:
        if not isinstance(labels, list) or not labels:
            problems.append("labels harus list tidak kosong")
        else:
            bad = [lab for lab in labels if not isinstance(lab, str) or not RE_LABEL.match(lab)]
            if bad:
                problems.append(f"labels harus snake_case huruf kecil: {bad}")
            if len(set(map(str, labels))) != len(labels):
                problems.append("labels ada yang dobel")

    if "label_descriptions" in cfg:
        _check_per_label(problems, cfg, "label_descriptions", labels, _valid_description,
                         "objek {deskripsi: str, contoh_positif: [str], contoh_negatif: [str], aturan: [str]}")
    if "gliner_label_prompts" in cfg:
        _check_per_label(problems, cfg, "gliner_label_prompts", labels,
                         lambda v: isinstance(v, str) and bool(v.strip()), "string tidak kosong")
        prompts = cfg["gliner_label_prompts"]
        if isinstance(prompts, dict) and len(set(prompts.values())) != len(prompts):
            problems.append("gliner_label_prompts ada yang dobel (dua label dengan prompt sama)")

    if "schema_version" in cfg and not (_is_int(cfg["schema_version"]) and cfg["schema_version"] >= 1):
        problems.append("schema_version harus int >= 1")

    thresholds = cfg.get("ner_review_threshold")
    if "ner_review_threshold" in cfg:
        if not isinstance(thresholds, dict) or not thresholds:
            problems.append("ner_review_threshold harus objek {nama_model: angka 0-1}")
        else:
            bad = {m: t for m, t in thresholds.items() if not _is_rate(t)}
            if bad:
                problems.append(f"ner_review_threshold di luar 0-1: {bad}")
    if "active_model" in cfg:
        model = cfg["active_model"]
        if not isinstance(model, str) or not model:
            problems.append("active_model harus string tidak kosong")
        elif isinstance(thresholds, dict) and model not in thresholds:
            problems.append(f"active_model '{model}' belum punya ner_review_threshold")

    for key in ("predict_threshold", "review_sample_rate", "no_entity_sample_rate"):
        if key in cfg and not _is_rate(cfg[key]):
            problems.append(f"{key} harus angka 0-1, dapat {cfg[key]!r}")
    for key in ("batch_size", "ai_workers"):
        if key in cfg and not (_is_int(cfg[key]) and cfg[key] > 0):
            problems.append(f"{key} harus int > 0, dapat {cfg[key]!r}")

    if "aliases" in cfg:
        aliases = cfg["aliases"]
        ok = isinstance(aliases, dict) and all(
            isinstance(m, dict) and all(isinstance(k, str) and isinstance(v, str) and v.strip() for k, v in m.items())
            for m in aliases.values())
        if not ok:
            problems.append("aliases harus objek {project: {variasi: nama baku}}")

    if "paths" in cfg:
        paths = cfg["paths"]
        if not isinstance(paths, dict):
            problems.append("paths harus objek")
        else:
            need = set(DEFAULTS["paths"])
            if set(paths) != need:
                problems.append(f"paths harus berisi tepat {sorted(need)}, dapat {sorted(paths)}")
            bad = [k for k, v in paths.items() if not isinstance(v, str) or not v]
            if bad:
                problems.append(f"paths harus string tidak kosong: {bad}")

    if problems:
        raise ValueError("Config tidak valid:\n  - " + "\n  - ".join(problems))


def save_config(cfg, path):
    validate_config(cfg)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def load_config(path):
    """Baca + validasi. File belum ada -> DEFAULTS ditulis. Key baru yang belum ada -> diisi default."""
    if not os.path.exists(path):
        save_config(DEFAULTS, path)
        print(f"[INFO] Config belum ada, default ditulis ke {path}. Edit file itu kalau perlu.")
        return copy.deepcopy(DEFAULTS)
    with open(path, encoding="utf-8") as f:
        try:
            cfg = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Config {path} bukan JSON valid: {e}") from None
    added = sorted(set(DEFAULTS) - set(cfg)) if isinstance(cfg, dict) else []
    for key in added:
        cfg[key] = copy.deepcopy(DEFAULTS[key])
    upgraded = upgrade_old_prompts(cfg)
    validate_config(cfg)
    if added or upgraded:
        save_config(cfg, path)
    if added:
        print(f"[INFO] Key baru diisi nilai default dan disimpan ke {path}: {added}")
    if upgraded:
        print("[INFO] gliner_label_prompts masih default lama (bahasa Indonesia) -> diganti ke bahasa Inggris")
    return cfg


def upgrade_old_prompts(cfg):
    """Prompt default lama ('instansi pemerintah', ...) yang belum diedit -> versi Inggris. Return True kalau diganti."""
    prompts, labels = cfg.get("gliner_label_prompts"), cfg.get("labels")
    if not isinstance(labels, list) or not set(labels) <= set(GLINER_PROMPTS_EN):
        return False
    if prompts != {lab: lab.replace("_", " ") for lab in labels}:
        return False   # sudah diedit sendiri, jangan disentuh
    cfg["gliner_label_prompts"] = {lab: GLINER_PROMPTS_EN[lab] for lab in labels}
    return True
