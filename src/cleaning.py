"""Text cleaning utilities for customer support tweets."""
import html
import re

import contractions
import emoji

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase; strip URLs, @mentions, emojis, HTML entities, special chars; expand contractions."""
    text = html.unescape(text)
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = emoji.replace_emoji(text, replace=" ")
    text = contractions.fix(text)
    text = NON_ALNUM_RE.sub(" ", text.lower())
    return WS_RE.sub(" ", text).strip()

SIGNATURE_RE = re.compile(r"\s*[\^\-–~/*]\s*[A-Za-z]{1,4}\d?\s*$")  # agent sign-offs: ^JD, -AB, /KM
PUNCT_RE = re.compile(r"([.,!?])")
REPEAT_PUNCT_RE = re.compile(r"([.,!?])(?:\s+\1)+")


def clean_for_seq2seq(text: str) -> str:
    """Seq2Seq cleaning: like clean_text, but keeps . , ! ? as tokens and drops agent sign-offs."""
    text = html.unescape(text)
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text).rstrip()
    text = SIGNATURE_RE.sub(" ", text)
    text = emoji.replace_emoji(text, replace=" ")
    text = contractions.fix(text).lower()
    text = re.sub(r"[^a-z0-9\s.,!?]", " ", text)
    text = PUNCT_RE.sub(r" \1 ", text)
    text = REPEAT_PUNCT_RE.sub(r"\1", text)
    return WS_RE.sub(" ", text).strip()

GREETING_NAME_RE = re.compile(
    r"^(hi|hey|hello|hiya|dear)\s*,?\s+(?!there\b|again\b|team\b|all\b|everyone\b|folks\b)([a-z]+)(?=\s*[,.!])"
)


def mask_greeting_name(text: str) -> str:
    """Replace a customer's name in the opening greeting ('hi emma ,') with a placeholder ('hi xname ,')."""
    return GREETING_NAME_RE.sub(r"\1 xname", text)