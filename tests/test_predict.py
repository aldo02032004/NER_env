from types import SimpleNamespace

from ner.config import DEFAULTS
from ner.predict import gliner_chunks, predict_all, predict_texts, resolve_overlaps
from ner.schema import read_jsonl

CFG = {**DEFAULTS, "gliner_label_prompts": {**DEFAULTS["gliner_label_prompts"], "tokoh": "nama orang"}}


class FakeGliner:
    """Cari kata dari `known` di tiap potongan, seperti GLiNER.inference (start/end relatif potongan)."""
    def __init__(self, known, max_len=384):
        self.known, self.calls = known, []
        self.config = SimpleNamespace(max_len=max_len)

    def inference(self, texts, labels, threshold=0.5, batch_size=8):
        self.calls.append(list(texts))
        out = []
        for t in texts:
            ents = []
            for word, (label, score) in self.known.items():
                i = t.find(word)
                if i >= 0 and label in labels and score >= threshold:
                    ents.append({"start": i, "end": i + len(word), "text": word, "label": label, "score": score})
            out.append(ents)
        return out


def test_chunks_cover_text_with_overlap():
    text = " ".join(f"w{i}" for i in range(10))
    chunks = gliner_chunks(text, max_tokens=4, overlap=1)
    assert all(text[off:off + len(c)] == c for off, c in chunks)
    assert chunks[0][1] == "w0 w1 w2 w3" and chunks[1][1].startswith("w3")
    assert chunks[-1][1].endswith("w9")
    assert gliner_chunks("", 4) == [] and gliner_chunks("halo", 4) == [(0, "halo")]


def test_resolve_overlaps_highest_score_wins():
    spans = [[0, 14, "instansi_pemerintah", 0.6], [5, 14, "lokasi", 0.9], [20, 25, "tokoh", 0.8],
             [20, 25, "tokoh", 0.7]]
    assert resolve_overlaps(spans) == [[5, 14, "lokasi", 0.9], [20, 25, "tokoh", 0.8]]


def test_predict_texts_maps_offsets_and_labels():
    model = FakeGliner({"Putin": ("nama orang", 0.9), "Rusia": ("location", 0.8), "abc": ("location", 0.1)})
    text = " ".join(["kata"] * 30) + " Putin ke Rusia"
    [spans] = predict_texts(model, [text], CFG, max_tokens=10)
    assert len(model.calls[0]) > 1                        # teks dipotong, dikirim sekali sebagai batch
    assert [(text[s:e], lab) for s, e, lab, _ in spans] == [("Putin", "tokoh"), ("Rusia", "lokasi")]


def test_predict_all_uses_cache(tmp_path):
    cfg = {**CFG, "paths": {**CFG["paths"], "checkpoint_dir": str(tmp_path)}}
    model = FakeGliner({"Putin": ("nama orang", 0.9)})
    cache = tmp_path / "preds.jsonl"
    docs = {"a": "Putin datang", "b": "tidak ada", "c": "lagi Putin"}
    first = predict_all(model, {"a": docs["a"]}, cfg, cache)
    assert first == {"a": [[0, 5, "tokoh", 0.9]]}
    model.calls.clear()
    res = predict_all(model, docs, cfg, cache, save_every=1)
    assert sum(len(c) for c in model.calls) == 2           # "a" dari cache
    assert res["b"] == [] and res["c"] == [[5, 10, "tokoh", 0.9]]
    assert [r["doc_id"] for r in read_jsonl(cache)] == ["a", "b", "c"]


def test_cache_path_changes_with_prompts():
    from ner.predict import preds_cache_path
    other = {**CFG, "gliner_label_prompts": {**CFG["gliner_label_prompts"], "tokoh": "person"}}
    assert preds_cache_path(CFG, "p") != preds_cache_path(other, "p")
    assert preds_cache_path(CFG, "p") == preds_cache_path(dict(CFG), "p")
