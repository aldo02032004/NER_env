from ner.cleaning import (build_raw_text, clean_for_ner, dedup_key, is_media_account, is_retweet,
                          strip_noise, text_for_ner)


def test_strip_noise():
    assert strip_noise("RT @abc: halo [RE def]").strip() == "halo"
    assert strip_noise("RT halo").strip() == "halo"
    assert strip_noise("mantaaaap") == "mantaap"


def test_build_raw_text():
    assert build_raw_text("Judul", "isi") == "Judul. isi"
    assert build_raw_text(float("nan"), "isi") == "isi"
    assert build_raw_text(None, None) == ""
    assert build_raw_text("ISI", "isi") == "isi"
    assert build_raw_text("Judul", "nan") == "Judul"


def test_is_retweet():
    assert is_retweet("RT @abc: halo")
    assert is_retweet("RT halo")
    assert not is_retweet("ART bagus")
    assert not is_retweet("halo RT")


def test_dedup_key():
    assert dedup_key("Halo,  DUNIA!!") == dedup_key("halo dunia") == "halo dunia"


def test_clean_for_ner_keeps_case_and_punctuation():
    out = clean_for_ner("Prabowo ke Rusia, cek https://x.co/a #PrabowoDiRusia @KemluRI")
    assert "Prabowo ke Rusia," in out
    assert "https" not in out
    assert "Prabowo Di Rusia" in out and "KemluRI" in out


def test_text_for_ner_pipeline():
    assert text_for_ner("RT @abc: Jokowi ke IKN [RE def]") == "Jokowi ke IKN"


def test_is_media_account_reexport():
    assert is_media_account("siapa", "News")
    assert is_media_account("akunbaru", extra_accounts=["akunbaru"])
