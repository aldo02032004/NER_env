"""Cleaning teks. Cleaner utama dari paket `great` (jangan disalin ke sini)."""
import re
import unicodedata

import emoji
from great import clean_for_bert, clean_for_ner, is_media_account  # noqa: F401 (re-export)

# Export menulis retweet sebagai "RT <teks>" (sering TANPA @user) dan balasan sebagai "... [RE username]".
RE_RT_PREFIX = re.compile(r"^\s*RT\b\s*(?:@\w+\s*:\s*)?")      # ponytail: "RT" huruf besar di awal = retweet
RE_REPLY = re.compile(r"\[RE\s+@?(\w+)\]", re.IGNORECASE)
RE_ELONGATION = re.compile(r"(.)\1{2,}")
EMOJI_LANG = "id" if "id" in emoji.LANGUAGES else "en"


def _blank(x):
    return x is None or x != x or str(x).strip().lower() in ("", "nan")   # x != x -> NaN


def _emoji_to_words(chars, _data):
    return " " + emoji.demojize(chars, language=EMOJI_LANG, delimiters=("", "")).replace("_", " ") + " "


def strip_noise(text):
    """Huruf hias -> biasa (NFKC), buang prefix RT & penanda [RE user], emoji -> kata, 'mantaaap' -> 'mantaap'.
    Hanya karakter emoji yang diganti, jadi link (https://...) dan @user_name tetap utuh."""
    text = unicodedata.normalize("NFKC", text)
    text = RE_REPLY.sub(" ", RE_RT_PREFIX.sub("", text))
    text = emoji.replace_emoji(text, replace=_emoji_to_words)
    return RE_ELONGATION.sub(r"\1\1", text)


def build_raw_text(headline, mentions):
    """Gabung kolom Headline + Mentions. Headline kosong/sama dengan mentions -> mentions saja."""
    headline = "" if _blank(headline) else str(headline).strip()
    mentions = "" if _blank(mentions) else str(mentions).strip()
    if headline.lower() in ("", mentions.lower()):
        return mentions or headline
    return f"{headline}. {mentions}" if mentions else headline


def is_retweet(text):
    return bool(RE_RT_PREFIX.match(text))


def dedup_key(text):
    """Kunci pembanding duplikat: huruf kecil, tanpa tanda baca, spasi dirapikan."""
    return " ".join(re.sub(r"[^\w\s]", "", text.lower()).split())


def text_for_ner(raw):
    """Teks mentah -> `text` di data master. Satu-satunya jalur, supaya span konsisten."""
    return clean_for_ner(strip_noise(raw))
