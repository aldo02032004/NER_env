import pytest

from ner.formats import (bio_to_spans, entities_dict_to_spans, gliner_to_spans, spans_to_bio,
                         spans_to_gliner, tokenize)


def surfaces(text, spans):
    return [(text[s:e], lab) for s, e, lab in spans]


# --- entities_dict_to_spans ---

def test_overlap_longest_wins_but_other_occurrences_kept():
    text = "Bank Indonesia menaikkan suku bunga, rupiah Indonesia menguat"
    spans, missing = entities_dict_to_spans(text, {"instansi_pemerintah": ["Bank Indonesia"],
                                                   "lokasi": ["Indonesia"]})
    assert missing == []
    assert surfaces(text, spans) == [("Bank Indonesia", "instansi_pemerintah"), ("Indonesia", "lokasi")]


def test_entity_at_start_and_end():
    text = "Prabowo bertemu Putin"
    spans, missing = entities_dict_to_spans(text, {"tokoh": ["Prabowo", "Putin"]})
    assert missing == []
    assert spans == [[0, 7, "tokoh"], [16, 21, "tokoh"]]


def test_case_insensitive_keeps_text_casing():
    text = "jokowi dan JOKOWI"
    spans, missing = entities_dict_to_spans(text, {"tokoh": ["Jokowi"]})
    assert missing == []
    assert surfaces(text, spans) == [("jokowi", "tokoh"), ("JOKOWI", "tokoh")]


def test_not_found_and_whole_word():
    text = "Putinisme makin populer"
    spans, missing = entities_dict_to_spans(text, {"tokoh": ["Putin", "  "], "lokasi": []})
    assert spans == []
    assert missing == ["Putin"]


def test_same_name_two_labels_is_conflict():
    text = "Gerindra mendukung kebijakan itu"
    spans, missing = entities_dict_to_spans(text, {"partai": ["Gerindra"], "instansi_pemerintah": ["gerindra"]})
    assert spans == []
    assert sorted(missing) == ["Gerindra", "gerindra"]


def test_equal_length_ties_are_deterministic():
    text = "Aceh Bali"
    ents = {"lokasi": ["Bali", "Aceh"]}
    assert entities_dict_to_spans(text, ents) == entities_dict_to_spans(text, {"lokasi": ["Aceh", "Bali"]})


# --- BIO ---

def test_spans_to_bio_basic():
    text = "Sri Mulyani: inflasi 2,5%."
    spans = [[0, 11, "tokoh"], [13, 20, "indikator_ekonomi"]]
    tokens, tags, n_mis = spans_to_bio(text, spans)
    assert n_mis == 0
    assert tokens == ["Sri", "Mulyani", ":", "inflasi", "2", ",", "5", "%", "."]
    assert tags == ["B-tokoh", "I-tokoh", "O", "B-indikator_ekonomi", "O", "O", "O", "O", "O"]


@pytest.mark.parametrize("text,ents", [
    ("Prabowo bertemu Putin", {"tokoh": ["Prabowo", "Putin"]}),
    ("Bank Indonesia dan Kementerian Keuangan RI.", {"instansi_pemerintah": ["Bank Indonesia",
                                                                           "Kementerian Keuangan RI"]}),
    ("Harga cabai di Jakarta naik Rp 50 ribu", {"lokasi": ["Jakarta"], "nilai_uang": ["Rp 50 ribu"]}),
])
def test_bio_and_gliner_roundtrip_exact(text, ents):
    spans, missing = entities_dict_to_spans(text, ents)
    assert missing == [] and spans
    _, tags, n_mis = spans_to_bio(text, spans)
    assert n_mis == 0                       # round-trip hanya sah kalau tidak ada span yang melebar
    assert bio_to_spans(text, tags) == spans
    item, n_mis = spans_to_gliner(text, spans)
    assert n_mis == 0
    assert gliner_to_spans(text, item) == spans


def test_misaligned_span_is_counted_and_widens(capsys):
    # "Jakarta" kata utuh menurut regex, tapi "Jakarta-Bandung" satu token
    text = "Kereta cepat Jakarta-Bandung molor"
    spans, missing = entities_dict_to_spans(text, {"lokasi": ["Jakarta"]})
    assert missing == [] and surfaces(text, spans) == [("Jakarta", "lokasi")]

    _, tags, n_mis = spans_to_bio(text, spans)
    assert n_mis == 1
    assert "[WARN]" in capsys.readouterr().out
    back = bio_to_spans(text, tags)
    assert back != spans and surfaces(text, back) == [("Jakarta-Bandung", "lokasi")]

    item, n_mis = spans_to_gliner(text, spans)
    assert n_mis == 1


def test_span_mid_token_counted():
    _, _, n_mis = spans_to_bio("Rp50ribu", [[0, 2, "nilai_uang"]])
    assert n_mis == 1


def test_bio_to_spans_orphan_i_and_label_switch():
    text = "a b c d"
    assert bio_to_spans(text, ["I-x", "I-x", "I-y", "O"]) == [[0, 3, "x"], [4, 5, "y"]]
    assert bio_to_spans(text, ["B-x", "B-x", "O", "B-y"]) == [[0, 1, "x"], [2, 3, "x"], [6, 7, "y"]]


def test_bio_to_spans_length_mismatch():
    with pytest.raises(ValueError):
        bio_to_spans("a b", ["O"])


def test_gliner_format_inclusive_end():
    text = "Menteri Sri Mulyani"
    item, _ = spans_to_gliner(text, [[8, 19, "tokoh"]])
    assert item == {"tokenized_text": ["Menteri", "Sri", "Mulyani"], "ner": [[1, 2, "tokoh"]]}


def test_tokenize_offsets():
    assert tokenize("PT. Pertamina-Persero") == [(0, 2), (2, 3), (4, 21)]
