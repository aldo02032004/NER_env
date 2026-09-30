"""Config proyek: dibaca/disimpan sebagai JSON di Drive, divalidasi sebelum dipakai."""
import copy
import json
import os
import re

DRIVE_DIR = "/content/drive/MyDrive/ner"

DEFAULTS = {
    "labels": ["tokoh", "instansi_pemerintah", "partai", "perusahaan", "kebijakan",
               "indikator_ekonomi", "nilai_uang", "lokasi"],
    "schema_version": 1,
    "active_model": "urchade/gliner_medium-v2.1",
    "ner_review_threshold": {"urchade/gliner_medium-v2.1": 0.5},   # skor di bawah ini dikirim ke LLM
    "review_sample_rate": 0.05,   # sampel acak skor tinggi yang tetap dicek LLM
    "batch_size": 40,
    "ai_workers": 2,
    "paths": {
        "master_data": f"{DRIVE_DIR}/master_ner.jsonl",
        "output_dir": f"{DRIVE_DIR}/output",
        "model_dir": f"{DRIVE_DIR}/models",
        "checkpoint_dir": f"{DRIVE_DIR}/checkpoints",
    },
}

RE_LABEL = re.compile(r"^[a-z][a-z0-9_]*$")


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def _is_rate(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and 0 <= x <= 1


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

    if "review_sample_rate" in cfg and not _is_rate(cfg["review_sample_rate"]):
        problems.append(f"review_sample_rate harus angka 0-1, dapat {cfg['review_sample_rate']!r}")
    for key in ("batch_size", "ai_workers"):
        if key in cfg and not (_is_int(cfg[key]) and cfg[key] > 0):
            problems.append(f"{key} harus int > 0, dapat {cfg[key]!r}")

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
    """Baca + validasi. File belum ada -> DEFAULTS ditulis dulu ke path itu."""
    if not os.path.exists(path):
        save_config(DEFAULTS, path)
        print(f"[INFO] Config belum ada, default ditulis ke {path}. Edit file itu kalau perlu.")
        return copy.deepcopy(DEFAULTS)
    with open(path, encoding="utf-8") as f:
        try:
            cfg = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Config {path} bukan JSON valid: {e}") from None
    validate_config(cfg)
    return cfg
