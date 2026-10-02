import pandas as pd
import pytest

from ner.io import build_export, canonical, prepare_frame, split_links, xlsx_url

LABELS = ["tokoh", "lokasi"]


def test_xlsx_url():
    assert xlsx_url("https://docs.google.com/spreadsheets/d/AbC-1_x/edit?usp=sharing") == \
        "https://docs.google.com/spreadsheets/d/AbC-1_x/export?format=xlsx"
    assert xlsx_url("https://drive.google.com/file/d/XyZ/view") == \
        "https://drive.google.com/uc?export=download&id=XyZ"
    assert xlsx_url("/content/data.xlsx") == "/content/data.xlsx"
    assert split_links("a, b  c") == ["a", "b", "c"]


def frame():
    return pd.DataFrame({
        "Headline": [None, None, None, None, "Tempo"],
        "Mentions": ["Prabowo bertemu Putin di Rusia", "RT Prabowo bertemu Putin di Rusia!",
                     "ok", "Wowo ke Rusia lagi ya", "Berita Prabowo hari ini sekali"],
        "Author": ["a1", "a2", "a3", "a4", "tempodotco"],
        "Media": ["Twitter", "Twitter", "Twitter", "Twitter", "News"],
        "Followers": [10, 20, 30, 40, 50],
    })


def test_prepare_frame_marks_without_dropping():
    df = prepare_frame(frame())
    assert len(df) == 5
    assert df["is_retweet"].tolist() == [False, True, False, False, False]
    assert df["is_duplicate"].tolist() == [False, True, False, False, False]
    assert df["doc_id"][0] == df["doc_id"][1]
    assert df["too_short"].tolist() == [False, False, True, False, False]
    assert df["is_media"].tolist() == [False, False, False, False, True]
    assert df["siap_ner"].tolist() == [True, False, False, True, False]
    assert df["text"][1] == "Prabowo bertemu Putin di Rusia!"


def test_prepare_frame_missing_column():
    with pytest.raises(KeyError, match="Media"):
        prepare_frame(frame().drop(columns="Media"))


def test_canonical():
    assert canonical(["Wowo", "Prabowo Subianto", " Putin "], {"wowo": "Prabowo Subianto"}) == [
        "Prabowo Subianto", "Putin"]


def test_build_export():
    df = prepare_frame(frame())
    ready = df[df["siap_ner"]]
    docs = dict(zip(ready["doc_id"], ready["text"]))
    d0, d3 = ready["doc_id"]
    preds = {d0: [[0, 7, "tokoh", 0.9], [25, 30, "lokasi", 0.8]], d3: [[0, 4, "tokoh", 0.9]]}
    llm = {d3: {"tokoh": ["Wowo"], "lokasi": ["Rusia"]}}
    out = build_export(df, docs, preds, llm, LABELS, aliases={"Wowo": "Prabowo Subianto"})
    assert list(out.columns) == ["Headline", "Mentions", "Author", "Media", "Followers", "doc_id", "text",
                                 "entities_tokoh", "entities_lokasi", "ner_source"]
    assert out["ner_source"].fillna("-").tolist() == ["gliner", "gliner", "-", "gemini", "-"]   # retweet ikut post asli
    assert out["entities_tokoh"][1] == ["Prabowo"] and out["entities_lokasi"][0] == ["Rusia"]
    assert out["entities_tokoh"][3] == ["Prabowo Subianto"]
    assert out["entities_tokoh"][2] == []


def test_retweet_of_media_post_is_processed():
    df = prepare_frame(pd.DataFrame({
        "Headline": [None, None], "Mentions": ["Harga BBM naik mulai besok pagi", "RT Harga BBM naik mulai besok pagi"],
        "Author": ["kompascom", "warga1"], "Media": ["News", "Twitter"]}))
    assert df["is_media"].tolist() == [True, False]
    assert df["is_duplicate"].tolist() == [False, False]
    assert df["siap_ner"].tolist() == [False, True]


def test_export_min_score_and_case_unify():
    df = prepare_frame(pd.DataFrame({
        "Headline": [None] * 3, "Mentions": ["Prabowo ke Rusia hari ini", "prabowo ke Solo besok pagi", "Prabowo pidato di Jakarta"],
        "Author": ["a", "b", "c"], "Media": ["Twitter"] * 3}))
    docs = dict(zip(df["doc_id"], df["text"]))
    d0, d1, d2 = df["doc_id"]
    preds = {d0: [[0, 7, "tokoh", 0.9], [11, 16, "lokasi", 0.4]], d1: [[0, 7, "tokoh", 0.8]], d2: [[0, 7, "tokoh", 0.9]]}
    out = build_export(df, docs, preds, {}, LABELS, min_score=0.6)
    assert out["entities_lokasi"][0] == []                       # skor 0.4 < 0.6, belum dicek LLM
    assert out["entities_tokoh"].tolist() == [["Prabowo"]] * 3   # "prabowo" disatukan ke bentuk tersering
