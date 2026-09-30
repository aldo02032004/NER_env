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
