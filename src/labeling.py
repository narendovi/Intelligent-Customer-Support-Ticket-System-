"""Rule-based labeling functions (weak supervision) for ticket intent."""
import numpy as np
import pandas as pd

# Dict order = tie-break priority (specific issues beat generic ones).
INTENT_LFS = {
    "Billing": [
        r"\b(?:refund\w*|reimburs\w*|money back)\b",
        r"\b(?:charged?|charges|charging|overcharged|billed|billing|bill)\b",
        r"\b(?:payment|paid|invoice|receipt|credit card|debit card|transaction|declined)\b",
        r"\b(?:price|pricing|fee|fees|subscription|renewal|discount|promo|coupon)\b",
    ],
    "Account": [
        r"\b(?:log ?in|logged out|sign ?in|log on|logging in)\b",
        r"\b(?:password|passcode|username|verification code|two factor|2fa|verify)\b",
        r"\b(?:account|acct|profile|membership)\b",
        r"\b(?:locked|hacked|suspended|deactivat\w*|disabled|unauthori[sz]ed)\b",
        r"\b(?:email address|phone number|my address|my details|personal (?:info|details))\b",
    ],
    "Technical":[
        r"\bnot (?:working|work|loading|load|connecting|connect|responding|syncing|updating|opening|playing)\b",
        r"\b(?:error|bug|glitch\w*|crash\w*|freez\w*|frozen|outage|offline|buffering)\b",
        r"\b(?:no service|no signal|no internet|wifi|wi fi|broadband|connection|connectivity|network)\b",
        r"\b(?:reset|restart|reboot|reinstall\w*|troubleshoot\w*|update|ios|software)\b",
        r"\b(?:no longer|stopped|keeps?) (?:work\w*|load\w*|crash\w*|freez\w*|drop\w*|disconnect\w*)\b",
    ],
    "Complaint": [
        r"\b(?:worst|terrible|horrible|awful|pathetic|ridiculous|unacceptable|disgusting|useless)\b",
        r"\b(?:bad|poor|rude|lousy) (?:service|customer service|experience|support|staff)\b",
        r"\b(?:disappoint\w*|frustrat\w*|annoyed|angry|furious|fed up|sick of|tired of)\b",
        r"\b(?:never again|still waiting|no one|nobody|switching to|done with)\b",
        r"\b(?:delayed|delay|lost (?:my )?(?:bag|luggage|package)|never (?:arrived|received|came))\b",
    ],
    "General": [
        r"^(?:how|what|when|where|why|which|who|is|are|can|do|does|will|could|would)\b",
        r"\b(?:thank\w*|thx|appreciate|love|great|awesome|amazing|kudos)\b",
        r"\b(?:question|wondering|info|information|hours|available|availability|policy)\b",
    ],
}
INTENTS = list(INTENT_LFS)


def intent_lf_matrix(text: pd.Series) -> pd.DataFrame:
    """Boolean matrix (rows x labeling functions): True = rule fired, False = abstained."""
    return pd.DataFrame(
        {f"{lab}_{i}": text.str.contains(p, regex=True)
         for lab, pats in INTENT_LFS.items() for i, p in enumerate(pats, 1)},
        index=text.index,
    )


def rule_intent(text: pd.Series):
    """Majority vote over LFs. Returns (labels_df, lf_matrix).

    labels_df columns: rule_label (NaN if every rule abstained),
    rule_votes (total votes), rule_agreement (share of votes for the winner).
    """
    lf = intent_lf_matrix(text)
    votes = pd.DataFrame({lab: lf.filter(like=f"{lab}_").sum(axis=1) for lab in INTENTS})
    has_specific = votes.drop(columns="General").sum(axis=1) > 0
    votes.loc[has_specific, "General"] = 0   # General only counts when no specific rule fired
    total = votes.sum(axis=1)
    winner = np.array(INTENTS)[votes.to_numpy().argmax(axis=1)]  # argmax = first max → tie-break
    labels = pd.DataFrame({
        "rule_label": pd.Series(winner, index=text.index).where(total > 0),
        "rule_votes": total,
        "rule_agreement": (votes.max(axis=1) / total.replace(0, np.nan)).round(2),
    })
    return labels, lf

import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

ESCALATION = re.compile(r"\b(?:urgent\w*|immediately|asap|emergency|right now|cancel\w*|worst|lawsuit|sue|never again)\b", re.I)
RISK = re.compile(r"\b(?:hack\w*|fraud\w*|unauthori[sz]ed|stolen|scam\w*|double charged|charged twice|stranded|outage|no service|no internet|missed (?:my )?flight)\b", re.I)
CALM = re.compile(r"\b(?:thank\w*|thx|appreciate|love|great|awesome|kudos|just wondering|quick question)\b", re.I)
_vader = SentimentIntensityAnalyzer()


def rule_priority(raw: pd.Series, intent: pd.Series) -> pd.DataFrame:
    """Keyword + sentiment priority heuristic on raw text (caps and '!' preserved).

    Returns the signal columns (reused later as PS-2 features) plus rule_priority.
    """
    text = raw.astype(str)
    words = text.str.findall(r"[A-Za-z]{3,}")
    sig = pd.DataFrame({
        "vader": text.map(lambda t: _vader.polarity_scores(t)["compound"]),
        "n_exclaim": text.str.count("!"),
        "caps_ratio": words.map(lambda ws: sum(w.isupper() for w in ws) / len(ws) if ws else 0.0),
        "escalation_kw": text.str.contains(ESCALATION).astype(bool),
        "risk_kw": text.str.contains(RISK).astype(bool),
        "calm_kw": text.str.contains(CALM).astype(bool),
    }, index=raw.index)

    urgency = ((sig["vader"] <= -0.6).astype(int) + sig["escalation_kw"].astype(int)
               + (sig["n_exclaim"] >= 2).astype(int) + (sig["caps_ratio"] >= 0.3).astype(int))
    high = sig["risk_kw"] | (urgency >= 2)
    low = ~high & ((sig["vader"] >= 0.4)
                   | (sig["calm_kw"] & (sig["vader"] >= 0))
                   | ((intent == "General") & (sig["vader"] >= -0.2)))
    sig["rule_priority"] = np.select([high, low], ["High", "Low"], default="Medium")
    return sig

def rule_priority_v2(d: pd.DataFrame, intent: pd.Series, neg: float = -0.2) -> np.ndarray:
    """Priority rules v2, calibrated on gold-dev: ANY single urgency signal → High (v1 required two).

    Expects the signal columns produced by rule_priority (vader, n_exclaim, caps_ratio, *_kw).
    """
    risk = d["risk_kw"].astype(bool)
    esc = d["escalation_kw"].astype(bool)
    calm = d["calm_kw"].astype(bool)
    high = risk | esc | (d["vader"] <= neg) | (d["caps_ratio"] >= 0.3) | (d["n_exclaim"] >= 2)
    positive = d["vader"] >= 0.4
    calm_ok = calm & (d["vader"] >= 0)
    general_ok = (intent == "General") & (d["vader"] >= -0.2)
    low = ~high & (positive | calm_ok | general_ok)
    return np.select([high, low], ["High", "Low"], default="Medium")