import json
import re
import sys
import types

import pytest

from ner.config import DEFAULTS
from ner.llm import (build_prompt, llm_entities, order_by_index, parse_json_text, review_all,
                     summarize_log)
from ner.schema import read_jsonl, validate_record

CFG = {**DEFAULTS, "batch_size": 2}


def test_parse_json_text():
    assert parse_json_text('Berikut hasilnya:\n```json\n[{"index": 1}]\n``` selesai') == [{"index": 1}]
    with pytest.raises(ValueError):
        parse_json_text("tidak ada json")


def test_order_by_index():
    assert order_by_index([{"index": 2, "x": "b"}, {"index": "1", "x": "a"}], 2) == [
        {"index": "1", "x": "a"}, {"index": 2, "x": "b"}]
    with pytest.raises(ValueError, match="tidak lengkap"):
        order_by_index([{"index": 1}], 2)


def test_prompt_contains_label_guidance():
    prompt = build_prompt([("Prabowo ke Rusia", {"tokoh": ["Prabowo"]})], CFG)
    for part in ("Jabatan tanpa nama bukan tokoh", "Bukan: presiden", "Contoh: Prabowo", '1. Teks: "Prabowo ke Rusia"',
                 "tepat 1 objek"):
        assert part in prompt


class FakeLLM:
    """Jawab dari `answers` {text: entities}. `broken` = teks yang bikin jawaban batch rusak."""
    def __init__(self, answers, broken=()):
        self.answers, self.broken, self.calls = answers, set(broken), []

    def __call__(self, prompt):
        texts = re.findall(r'^\d+\. Teks: "(.*)"$', prompt, re.M)
        self.calls.append(texts)
        if self.broken & set(texts):
            return "maaf, saya tidak bisa"
        return json.dumps([{"index": i, "entities": self.answers[t]} for i, t in enumerate(texts, 1)])


DOCS = {"a": "Prabowo bertemu Putin", "b": "harga cabai naik", "c": "Jokowi di Solo", "d": "kata Luhut"}
PREDS = {"a": [[0, 7, "tokoh", 0.5]], "b": [], "c": [[0, 6, "tokoh", 0.9]], "d": []}
ROUTES = {"a": "low_score", "b": "no_entity_sample", "c": "random_sample", "d": "no_entity_capital"}
ANSWERS = {
    "Prabowo bertemu Putin": {"tokoh": ["Prabowo", "Putin"]},       # koreksi: Putin ditambah
    "harga cabai naik": {},                                         # tidak ada entitas
    "Jokowi di Solo": {"tokoh": ["Jokowi"], "lokasi": ["Solo"], "hewan": ["x"]},   # label asing dibuang
    "kata Luhut": {"tokoh": ["Luhut Binsar"]},                      # tidak ada di teks -> missing
}


def run(tmp_path, llm, cfg=CFG, routes=ROUTES):
    master, log = tmp_path / "master.jsonl", tmp_path / "log.jsonl"
    done = review_all(DOCS, PREDS, routes, cfg, "proj", llm, master, log)
    return done, read_jsonl(master), read_jsonl(log)


def test_review_all_statuses_and_master_fields(tmp_path):
    done, master, log = run(tmp_path, FakeLLM(ANSWERS))
    assert done
    status = {r["doc_id"]: r["status"] for r in log}
    assert status == {"a": "ok", "b": "ok", "c": "ok", "d": "missing"}
    assert {r["text"] for r in master} == {DOCS["a"], DOCS["b"], DOCS["c"]}
    for r in master:
        assert validate_record(r, CFG["labels"]) == []
        assert (r["labels"], r["schema_version"], r["model_version"], r["label_source"]) == (
            CFG["labels"], CFG["schema_version"], CFG["active_model"], "llm")
    by_text = {r["text"]: r for r in master}
    assert by_text[DOCS["a"]]["route_reason"] == "low_score"
    assert by_text[DOCS["a"]]["entities"] == [[0, 7, "tokoh"], [16, 21, "tokoh"]]
    assert by_text[DOCS["b"]]["entities"] == []


def test_entity_metrics_per_route_reason(tmp_path):
    _, _, log = run(tmp_path, FakeLLM(ANSWERS))
    s = summarize_log(log)
    low = s["low_score"]                                      # Prabowo dibenarkan, Putin ditambah
    assert (low["kept"], low["dropped"], low["added"], low["precision"], low["recall"]) == (1, 0, 1, 1.0, 0.5)
    assert s["no_entity_sample"]["precision"] is None and s["no_entity_sample"]["recall"] is None
    assert s["random_sample"]["recall"] == 0.5                # Solo ditambah
    assert s["no_entity_capital"]["missing"] == 1 and s["no_entity_capital"]["added"] == 1
    assert llm_entities(log)["d"] == {**{lab: [] for lab in CFG["labels"]}, "tokoh": ["Luhut Binsar"]}


def test_conflict_not_in_master(tmp_path):
    answers = {**ANSWERS, "Jokowi di Solo": {"tokoh": ["Jokowi"], "lokasi": ["jokowi"]}}
    _, master, log = run(tmp_path, FakeLLM(answers), routes={"c": "random_sample"})
    assert master == [] and log[0]["status"] == "conflict"


def test_budget_stop_then_resume_without_duplicates(tmp_path):
    llm = FakeLLM(ANSWERS)
    done, master, log = run(tmp_path, llm, {**CFG, "llm_budget_per_run": 2})
    assert not done and len(log) == 2 and len(llm.calls) == 1   # satu batch, lalu berhenti
    llm2 = FakeLLM(ANSWERS)
    done, master, log = run(tmp_path, llm2)
    assert done and len(llm2.calls) == 1                  # hanya batch yang belum selesai dikirim
    assert sorted(r["doc_id"] for r in log) == ["a", "b", "c", "d"]
    assert len(master) == len({r["doc_id"] for r in master}) == 3
    llm3 = FakeLLM(ANSWERS)
    assert run(tmp_path, llm3)[0] and llm3.calls == []    # semua final, tidak ada panggilan


def test_parse_fail_splits_then_marks_failed(tmp_path):
    cfg = {**CFG, "batch_size": 4}
    llm = FakeLLM(ANSWERS, broken=[DOCS["d"]])
    done, master, log = run(tmp_path, llm, cfg)
    assert done
    # gagal sebagai batch 4 doc, lalu sendiri-sendiri: a,b,c lolos, d gagal lagi -> failed
    assert [len(c) for c in llm.calls] == [4, 1, 1, 1, 1]
    last = {r["doc_id"]: r for r in log}
    assert last["d"]["status"] == "failed" and last["d"]["attempts"] == 2
    assert {last[d]["status"] for d in "abc"} == {"ok"}
    assert len(master) == 3
    llm2 = FakeLLM(ANSWERS)
    assert run(tmp_path, llm2, cfg)[0] and llm2.calls == []   # failed tidak diulang


def test_compare_entities_drop():
    from ner.llm import compare_entities
    assert compare_entities({"tokoh": ["Prabowo", "Menkeu"]}, {"tokoh": ["prabowo"], "lokasi": []}) == {
        "kept": 1, "dropped": 1, "added": 0}


def test_budget_and_low_score_priority(tmp_path):
    docs = {f"d{i}": f"Prabowo bertemu Putin {i}" for i in range(4)}
    preds = {f"d{i}": [[0, 7, "tokoh", s]] for i, s in enumerate([0.5, 0.2, 0.4, 0.3])}
    routes = {d: "low_score" for d in docs}
    answers = {t: {"tokoh": ["Prabowo"]} for t in docs.values()}
    llm = FakeLLM(answers)
    cfg = {**CFG, "batch_size": 1, "llm_budget_per_run": 2}
    done = review_all(docs, preds, routes, cfg, "p", llm, tmp_path / "m.jsonl", tmp_path / "l.jsonl")
    assert not done
    assert [c[0] for c in llm.calls] == [docs["d1"], docs["d3"]]   # skor terendah dulu, maksimal 2 doc


def test_answer_schema_requires_every_label():
    from ner.llm import answer_schema
    item = answer_schema(["tokoh", "lokasi"])["items"]
    assert item["required"] == ["index", "entities"]
    assert item["properties"]["entities"]["required"] == ["tokoh", "lokasi"]


def test_make_generate_loads_gguf_and_forces_json(monkeypatch):
    from ner.llm import make_generate
    seen = {}

    class FakeLlama:
        @classmethod
        def from_pretrained(cls, **kw):
            seen["load"] = kw
            return cls()

        def create_chat_completion(self, messages, response_format, temperature, max_tokens):
            seen.update(prompt=messages[0]["content"], fmt=response_format["type"], temperature=temperature,
                        max_tokens=max_tokens, labels=response_format["schema"]["items"]["properties"]["entities"]["required"])
            return {"choices": [{"message": {"content": "[]"}}]}

    monkeypatch.setitem(sys.modules, "llama_cpp", types.SimpleNamespace(Llama=FakeLlama))
    assert make_generate(CFG)("halo") == "[]"
    assert seen["load"] == {"repo_id": CFG["llm_model_repo"], "filename": CFG["llm_model_file"],
                            "n_ctx": CFG["llm_n_ctx"], "n_gpu_layers": -1, "verbose": False}
    assert (seen["prompt"], seen["fmt"], seen["temperature"], seen["labels"]) == ("halo", "json_object", 0, CFG["labels"])
    assert seen["max_tokens"] == 150 * CFG["batch_size"] + 200
