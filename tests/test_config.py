import copy
import json

import pytest

from ner.config import DEFAULTS, load_config, save_config, validate_config


def test_defaults_valid():
    validate_config(DEFAULTS)


def test_all_problems_reported_at_once():
    cfg = copy.deepcopy(DEFAULTS)
    cfg.update(labels=["Tokoh", "tokoh", "tokoh"], active_model="lain", review_sample_rate=1.5,
               batch_size=0, bacth_size=40)
    with pytest.raises(ValueError) as err:
        validate_config(cfg)
    msg = str(err.value)
    for part in ("bacth_size", "snake_case", "dobel", "active_model 'lain'", "review_sample_rate", "batch_size"):
        assert part in msg


def test_load_creates_default_then_roundtrip(tmp_path):
    path = str(tmp_path / "cfg" / "config.json")
    assert load_config(path) == DEFAULTS
    cfg = load_config(path)
    cfg["batch_size"] = 10
    save_config(cfg, path)
    assert load_config(path)["batch_size"] == 10


def test_old_config_gets_new_keys_without_losing_values(tmp_path):
    path = tmp_path / "config.json"
    old = {k: v for k, v in DEFAULTS.items() if k not in ("label_descriptions", "no_entity_sample_rate")}
    path.write_text(json.dumps({**old, "batch_size": 7}), encoding="utf-8")
    cfg = load_config(str(path))
    assert cfg["batch_size"] == 7 and cfg["no_entity_sample_rate"] == 0.1
    assert json.loads(path.read_text(encoding="utf-8"))["label_descriptions"] == DEFAULTS["label_descriptions"]


def test_label_descriptions_must_cover_labels():
    cfg = copy.deepcopy(DEFAULTS)
    cfg["labels"].append("sektor")
    cfg["label_descriptions"]["tokoh"]["aturan"] = "bukan list"
    with pytest.raises(ValueError) as err:
        validate_config(cfg)
    msg = str(err.value)
    assert "label_descriptions harus berisi tepat semua labels (kurang: ['sektor']" in msg
    assert "label_descriptions tidak valid untuk label ['tokoh']" in msg
    assert "gliner_label_prompts harus berisi tepat semua labels" in msg


def test_load_invalid_json(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{bukan json", encoding="utf-8")
    with pytest.raises(ValueError, match="bukan JSON"):
        load_config(str(path))


def test_load_invalid_values(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({**DEFAULTS, "ai_workers": "2"}), encoding="utf-8")
    with pytest.raises(ValueError, match="ai_workers"):
        load_config(str(path))


def test_old_default_prompts_upgraded_but_edited_kept(tmp_path):
    path = tmp_path / "config.json"
    old_prompts = {lab: lab.replace("_", " ") for lab in DEFAULTS["labels"]}
    path.write_text(json.dumps({**DEFAULTS, "gliner_label_prompts": old_prompts}), encoding="utf-8")
    assert load_config(str(path))["gliner_label_prompts"]["partai"] == "political party"
    assert json.loads(path.read_text(encoding="utf-8"))["gliner_label_prompts"]["partai"] == "political party"

    edited = {**old_prompts, "partai": "parpol"}
    path.write_text(json.dumps({**DEFAULTS, "gliner_label_prompts": edited}), encoding="utf-8")
    assert load_config(str(path))["gliner_label_prompts"] == edited
