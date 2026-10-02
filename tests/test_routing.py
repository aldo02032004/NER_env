import pytest

from ner.config import DEFAULTS
from ner.routing import has_mid_capital, route, route_all, unit_hash


@pytest.mark.parametrize("text,expect", [
    ("harga cabai naik lagi", False),
    ("Harga cabai naik lagi", False),                       # kata pertama kalimat
    ("harga naik. Menkeu diam saja", False),                # awal kalimat kedua
    ("kata menkeu Sri Mulyani harga naik", True),
    ("harga naik! \"Kenapa\" tanya warga", False),
    ("harga naik, (Jokowi) diam", True),
    ("inflasi 2,5% menurut BPS", True),
    ("", False),
])
def test_has_mid_capital(text, expect):
    assert has_mid_capital(text) is expect


def test_route_reasons():
    hi, lo = [[0, 1, "tokoh", 0.9]], [[0, 1, "tokoh", 0.9], [2, 3, "lokasi", 0.4]]
    assert route("d", "x", lo, 0.6, 0, 0) == "low_score"
    assert route("d", "x", hi, 0.6, 0, 0) is None
    assert route("d", "x", hi, 0.6, 1, 0) == "random_sample"
    assert route("d", "kata Jokowi", [], 0.6, 0, 0) == "no_entity_capital"
    assert route("d", "kata jokowi", [], 0.6, 0, 1) == "no_entity_sample"
    assert route("d", "kata jokowi", [], 0.6, 1, 0) is None


def test_sampling_deterministic_and_close_to_rate():
    ids = [f"doc{i}" for i in range(5000)]
    picked = [d for d in ids if route(d, "x", [], 0.6, 0, 0.1)]
    assert picked == [d for d in ids if route(d, "x", [], 0.6, 0, 0.1)]
    assert 0.08 < len(picked) / len(ids) < 0.12
    assert unit_hash("a", "sample") != unit_hash("a", "no_entity")


def test_route_all(capsys):
    cfg = {**DEFAULTS, "review_sample_rate": 0, "no_entity_sample_rate": 0}
    docs = {"a": "kata Jokowi", "b": "biasa saja", "c": "x"}
    preds = {"a": [], "b": [], "c": [[0, 1, "tokoh", 0.1]]}
    assert route_all(docs, preds, cfg) == {"a": "no_entity_capital", "c": "low_score"}
    assert "2 dari 3 doc" in capsys.readouterr().out


def test_interleave():
    from ner.routing import interleave
    routes = {"a": "low_score", "b": "low_score", "c": "low_score", "d": "random_sample", "e": "no_entity_capital"}
    assert interleave(routes) == ["a", "e", "d", "b", "c"]
