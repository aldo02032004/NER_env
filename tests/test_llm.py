import json
import re

import pytest

from ner.config import DEFAULTS
from ner.llm import (build_prompt, llm_entities, order_by_index, parse_json_text, review_all,
                     summarize_log)
from ner.schema import read_jsonl, validate_record

CFG = {**DEFAULTS, "batch_size": 2, "ai_workers": 2}
NO_SLEEP = dict(sleep=lambda s: None, sleep_between=0)


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
    """Jawab dari `answers` {text: entities}. `broken` = teks yang bikin jawaban batch rusak. `quota_after` n panggilan."""
    def __init__(self, answers, broken=(), quota_after=None):
        self.answers, self.broken, self.quota_after, self.calls = answers, set(broken), quota_after, []

    def __call__(self, prompt):
        texts = re.findall(r'^\d+\. Teks: "(.*)"$', prompt, re.M)
        self.calls.append(texts)
        if self.quota_after is not None and len(self.calls) > self.quota_after:
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
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
    done = review_all(DOCS, PREDS, routes, cfg, "proj", llm, master, log, **NO_SLEEP)
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
            CFG["labels"], CFG["schema_version"], CFG["active_model"], "gemini")
    by_text = {r["text"]: r for r in master}
    assert by_text[DOCS["a"]]["route_reason"] == "low_score"
    assert by_text[DOCS["a"]]["entities"] == [[0, 7, "tokoh"], [16, 21, "tokoh"]]
    assert by_text[DOCS["b"]]["entities"] == []


def test_correction_rate_per_route_reason(tmp_path):
    _, _, log = run(tmp_path, FakeLLM(ANSWERS))
    s = summarize_log(log)
    assert s["low_score"]["correction_rate"] == 1.0          # Putin ditambah
    assert s["no_entity_sample"]["correction_rate"] == 0.0
    assert s["random_sample"]["correction_rate"] == 1.0      # Solo ditambah
    assert s["no_entity_capital"]["missing"] == 1
    assert llm_entities(log)["d"] == {**{lab: [] for lab in CFG["labels"]}, "tokoh": ["Luhut Binsar"]}


def test_conflict_not_in_master(tmp_path):
    answers = {**ANSWERS, "Jokowi di Solo": {"tokoh": ["Jokowi"], "lokasi": ["jokowi"]}}
    _, master, log = run(tmp_path, FakeLLM(answers), routes={"c": "random_sample"})
    assert master == [] and log[0]["status"] == "conflict"


def test_quota_stop_then_resume_without_duplicates(tmp_path):
    cfg = {**CFG, "ai_workers": 1}
    llm = FakeLLM(ANSWERS, quota_after=1)
    done, master, log = run(tmp_path, llm, cfg)
    assert not done and len(log) == 2                     # batch pertama tersimpan, batch kedua kena kuota
    llm2 = FakeLLM(ANSWERS)
    done, master, log = run(tmp_path, llm2, cfg)
    assert done and len(llm2.calls) == 1                  # hanya batch yang belum selesai dikirim
    assert sorted(r["doc_id"] for r in log) == ["a", "b", "c", "d"]
    assert len(master) == len({r["doc_id"] for r in master}) == 3
    done, _, _ = run(tmp_path, FakeLLM(ANSWERS, quota_after=0), cfg)
    assert done                                           # semua final, tidak ada panggilan


def test_parse_fail_splits_then_marks_failed(tmp_path):
    cfg = {**CFG, "batch_size": 4, "ai_workers": 1}
    llm = FakeLLM(ANSWERS, broken=[DOCS["d"]])
    done, master, log = run(tmp_path, llm, cfg)
    assert done
    # 2x gagal sebagai batch 4 doc, lalu sendiri-sendiri: a,b,c lolos, d gagal ke-3 -> failed
    # 2 putaran x 2 retry sebagai batch 4 doc, lalu a,b,c sekali, d 2 retry
    assert [len(c) for c in llm.calls] == [4, 4, 4, 4, 1, 1, 1, 1, 1]
    last = {r["doc_id"]: r for r in log}
    assert last["d"]["status"] == "failed" and last["d"]["attempts"] == 3
    assert {last[d]["status"] for d in "abc"} == {"ok"}
    assert len(master) == 3
    llm2 = FakeLLM(ANSWERS)
    assert run(tmp_path, llm2, cfg)[0] and llm2.calls == []   # failed tidak diulang
