import json

import pytest

from ner.schema import append_jsonl, make_doc_id, make_record, read_jsonl, validate_record

LABELS = ["tokoh", "lokasi"]


def rec(text="Prabowo ke Rusia", entities=([0, 7, "tokoh"], [11, 16, "lokasi"]), **kw):
    args = dict(labels=LABELS, project="p", label_source="gemini", route_reason="low_score",
                model_version="m", schema_version=1, run_date="2026-09-30")
    return make_record(text, list(entities), **{**args, **kw})


def test_doc_id_stable_and_ignores_case_punct():
    assert make_doc_id("Prabowo ke Rusia!") == make_doc_id("prabowo  ke rusia")
    assert make_doc_id("Prabowo ke Rusia") != make_doc_id("Prabowo ke China")
    assert len(make_doc_id("x")) == 16


def test_valid_record():
    assert validate_record(rec(), allowed_labels=LABELS) == []


@pytest.mark.parametrize("patch,expect", [
    ({"doc_id": "salah"}, "doc_id"),
    ({"label_source": "gliner"}, "label_source"),
    ({"route_reason": "x"}, "route_reason"),
    ({"run_date": "30-09-2026"}, "run_date"),
    ({"entities": [[0, 99, "tokoh"]]}, "di luar teks"),
    ({"entities": [[0, 7, "partai"]]}, "tidak ada di labels"),
    ({"entities": [[0, 7, "tokoh"], [3, 10, "lokasi"]]}, "tumpang tindih"),
    ({"entities": [[0, 7, "tokoh"], [0, "7", "tokoh"], "x"]}, "span harus"),
    ({"schema_version": True}, "schema_version"),
    ({"extra": 1}, "tidak dikenal"),
])
def test_invalid_record(patch, expect):
    errors = validate_record({**rec(), **patch})
    assert any(expect in e for e in errors), errors


def test_missing_field():
    r = rec()
    del r["project"]
    assert validate_record(r) == ["field hilang: project"]


def test_labels_outside_config():
    assert validate_record(rec(), allowed_labels=["tokoh"])


def test_append_no_duplicate_on_resume(tmp_path):
    path = tmp_path / "master.jsonl"
    a, b = rec(), rec("Jokowi di Solo", [[0, 6, "tokoh"], [10, 14, "lokasi"]])
    assert append_jsonl(path, [a, a]) == 1           # dobel di dalam batch
    assert append_jsonl(path, [a, b]) == 1           # resume: a sudah ada
    assert append_jsonl(path, [rec("PRABOWO ke Rusia!!", [])]) == 0   # varian tanda baca = dokumen sama
    assert [r["doc_id"] for r in read_jsonl(path)] == [a["doc_id"], b["doc_id"]]


def test_append_rejects_invalid(tmp_path):
    with pytest.raises(ValueError, match="tidak valid"):
        append_jsonl(tmp_path / "m.jsonl", [{**rec(), "label_source": "gliner"}])
    assert not (tmp_path / "m.jsonl").exists()


def test_truncated_last_line(tmp_path, capsys):
    path = tmp_path / "master.jsonl"
    a = rec()
    path.write_text(json.dumps(a) + "\n" + '{"doc_id": "pot', encoding="utf-8")
    assert read_jsonl(path) == [a]
    assert "[WARN]" in capsys.readouterr().out

    b = rec("Jokowi di Solo", [])
    assert append_jsonl(path, [b]) == 1
    assert read_jsonl(path) == [a, b]
    assert path.read_text(encoding="utf-8").count("\n") == 2


def test_corrupt_middle_line_raises(tmp_path):
    path = tmp_path / "master.jsonl"
    path.write_text("{rusak\n" + json.dumps(rec()) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="baris 1"):
        read_jsonl(path)
