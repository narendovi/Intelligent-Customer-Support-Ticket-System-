"""Intelligent Customer Support Ticket System: Streamlit demo.

Type a customer tweet → predicted intent (PS-1), priority (PS-2),
suggested first reply (PS-3) and estimated response time + SLA risk (PS-4).
Run from the project root:  streamlit run app/app.py
"""
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cleaning import MENTION_RE, URL_RE, clean_for_seq2seq, clean_text  # noqa: E402
from src.labeling import rule_priority  # noqa: E402
from src.seq2seq import greedy_decode  # noqa: E402

MODELS = ROOT / "models"
SIG = ["vader", "n_exclaim", "caps_ratio", "escalation_kw", "risk_kw", "calm_kw"]
SRC_LEN, TGT_LEN = 45, 40
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
EXAMPLES = {
    "💳 Billing": "I was charged twice for my order and still no refund. This is ridiculous, sort it out NOW!",
    "🛠️ Technical": "My iPhone keeps freezing after the latest iOS update and the camera app crashes every time",
    "❓ General": "Hi, what time does your store close on Sundays? Thanks!",
}

st.set_page_config(page_title="Support Ticket Assistant", page_icon="🎫", layout="wide")


@st.cache_resource(show_spinner="Loading models (first run takes ~30 s)…")
def load_models():
    """Load every trained artifact once per server session."""
    from tensorflow import keras
    from tensorflow.keras import layers

    src_vocab = json.loads((MODELS / "ps3_src_vocab.json").read_text())
    src_vec = layers.TextVectorization(max_tokens=None, output_sequence_length=SRC_LEN,
                                       standardize=None, vocabulary=src_vocab[2:])
    ps4 = joblib.load(MODELS / "ps4_xgb_pipeline.joblib")
    cat_enc = ps4["pipeline"].named_steps["pre"].named_transformers_["cat"]
    companies = sorted(c for c in cat_enc.categories_[0] if c != "Other")
    return {
        "intent": joblib.load(MODELS / "ps1_tfidf_lr.joblib"),
        "priority": joblib.load(MODELS / "ps2_lr_v2.joblib")["model"],
        "priority_v1": joblib.load(MODELS / "ps2_best_classical.joblib"),  # PS-4 was trained on v1 predictions
        "seq2seq": keras.models.load_model(MODELS / "ps3_seq2seq_attn.keras"),
        "src_vec": src_vec,
        "tgt_vocab": json.loads((MODELS / "ps3_tgt_vocab.json").read_text()),
        "ps4": ps4,
        "companies": companies,
    }


def prettify(reply: str, name: str) -> str:
    """Re-attach punctuation, fill the name placeholder, capitalise sentences."""
    reply = re.sub(r"\s+([.,!?])", r"\1", reply)
    reply = reply.replace("xname", name.strip() or "there")
    reply = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), reply)
    return re.sub(r"\bi\b", "I", reply)


def fmt_minutes(m: float) -> str:
    if m < 60:
        return f"{m:.0f} min"
    h, mm = divmod(int(round(m)), 60)
    return f"{h} h {mm:02d} min" if h < 24 else f"{h / 24:.1f} days"


def analyse(text, company, name, hour, dow, replies_60, requests_60, M):
    """Run the full PS-1 → PS-4 pipeline on one ticket."""
    clean = clean_text(text)
    base = pd.DataFrame({"text": [text], "text_clean": [clean]})

    # PS-1: intent
    clf = M["intent"]
    intent_p = pd.Series(clf.predict_proba(base["text_clean"])[0], index=clf.classes_).sort_values(ascending=False)
    intent = intent_p.index[0]

    # Signals (VADER, caps, '!', keywords) on raw text minus URLs/mentions
    raw = MENTION_RE.sub(" ", URL_RE.sub(" ", text)).strip()
    sig = rule_priority(pd.Series([raw]), pd.Series([intent]))[SIG].astype(float)
    sig["n_exclaim"] = sig["n_exclaim"].clip(upper=5)
    feats = pd.concat([base, sig], axis=1)

    # PS-2: priority (final v2 model)
    pclf = M["priority"]
    prio_p = pd.Series(pclf.predict_proba(feats)[0], index=pclf.classes_).reindex(["High", "Medium", "Low"])
    priority = prio_p.idxmax()

    # PS-3: suggested reply (attention Seq2Seq, no-repeat-trigram greedy decoding)
    src = f"co_{company.lower()} {clean_for_seq2seq(text)}"
    src_ids = M["src_vec"](np.array([src], dtype=object)).numpy()
    reply = greedy_decode(M["seq2seq"], src_ids, M["tgt_vocab"], max_len=TGT_LEN, no_repeat_ngram=3)[0]

    # PS-4: response time + SLA flag
    ps4 = M["ps4"]
    x = feats.assign(
        company_grp=company if company in M["companies"] else "Other",
        pred_intent=intent, pred_priority=M["priority_v1"].predict(feats)[0],
        n_words=len(clean.split()), n_chars=len(text), n_question=text.count("?"),
        has_url=int("http" in text), hour=hour, dayofweek=dow, is_weekend=int(dow >= 5),
        replies_prev_60m=replies_60, requests_prev_60m=requests_60,
    )
    minutes = float(np.expm1(ps4["pipeline"].predict(x[ps4["cat"] + ps4["num"]])[0]))
    sla_cut = float(np.expm1(ps4["sla_threshold_log"]))
    return {"intent": intent, "intent_p": intent_p, "priority": priority, "prio_p": prio_p,
            "reply": prettify(reply, name), "minutes": minutes, "sla_risk": minutes > sla_cut,
            "sla_min": ps4["sla_min"], "signals": sig.iloc[0]}


M = load_models()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Ticket context")
    default_co = M["companies"].index("AmazonHelp") if "AmazonHelp" in M["companies"] else 0
    company = st.selectbox("Company handle", M["companies"], index=default_co)
    name = st.text_input("Customer first name (optional)", "")
    st.subheader("Support-team workload")
    now = datetime.now(timezone.utc)
    hour = st.slider("Hour of day (UTC)", 0, 23, now.hour)
    dow = DAYS.index(st.selectbox("Day of week", DAYS, index=now.weekday()))
    replies_60 = st.number_input("Company replies in last hour", 0, 500, 23,
                                 help="Proxy for active agents (dataset median: 23)")
    requests_60 = st.number_input("Customer requests in last hour", 0, 400, 21,
                                  help="Proxy for incoming queue (dataset median: 21)")
    with st.expander("📋 Model card (gold-test, 240 hand-labelled tweets)"):
        st.markdown(
            "- **PS-1 Intent:** TF-IDF + LR · macro-F1 **0.544**\n"
            "- **PS-2 Priority:** LR v2 · macro-F1 **0.422**, High recall **0.558**\n"
            "- **PS-3 Reply:** Seq2Seq LSTM + attention · BLEU 0.102, ROUGE-L 0.244\n"
            "- **PS-4 Response time:** tuned XGBoost · R² (log) 0.411, MedAE 21 min, SLA recall 78%\n\n"
            "Trained on weakly labelled Twitter support data (twcs.csv)."
        )

# ---------------- Main ----------------
st.title("🎫 Intelligent Customer Support Ticket Assistant")
st.caption("NLP triage for incoming customer tweets: intent · priority · draft reply · response-time risk")

if "ticket" not in st.session_state:
    st.session_state["ticket"] = ""
cols = st.columns(len(EXAMPLES))
for col, (label, example) in zip(cols, EXAMPLES.items()):
    col.button(label, use_container_width=True,
               on_click=lambda e=example: st.session_state.update(ticket=e))

text = st.text_area("Customer message", key="ticket", height=110,
                    placeholder="Paste or type a customer tweet…")

if st.button("Analyse ticket", type="primary", disabled=not text.strip()):
    with st.spinner("Analysing…"):
        r = analyse(text, company, name, hour, dow, replies_60, requests_60, M)

    badge = {"High": "🔴 High", "Medium": "🟠 Medium", "Low": "🟢 Low"}[r["priority"]]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Intent (PS-1)", r["intent"], f"{r['intent_p'].iloc[0]:.0%} confidence", delta_color="off")
    c2.metric("Priority (PS-2)", badge, f"{r['prio_p'][r['priority']]:.0%} confidence", delta_color="off")
    c3.metric("Est. first response (PS-4)", fmt_minutes(r["minutes"]))
    c4.metric(f"SLA risk (> {r['sla_min']} min)", "⚠️ At risk" if r["sla_risk"] else "✅ On track")

    st.subheader("Suggested first reply (PS-3)")
    st.info(r["reply"])
    st.caption("Draft for agent review: edit before sending.")

    route = f"{r['priority']}-priority {r['intent']}"
    st.markdown(f"**Routing:** `{route}` → {r['intent']} team queue"
                + (" · escalate to senior agent" if r["priority"] == "High" else ""))

    e1, e2 = st.columns(2)
    with e1.expander("Intent probabilities"):
        st.bar_chart(r["intent_p"])
    with e2.expander("Priority probabilities"):
        st.bar_chart(r["prio_p"])
    with st.expander("Urgency signals extracted from the text"):
        s = r["signals"]
        st.dataframe(pd.DataFrame({
            "signal": ["Sentiment (VADER)", "Exclamation marks", "ALL-CAPS share",
                       "Escalation words", "Risk words", "Calm words"],
            "value": [f"{s['vader']:+.2f}", int(s["n_exclaim"]), f"{s['caps_ratio']:.0%}",
                      bool(s["escalation_kw"]), bool(s["risk_kw"]), bool(s["calm_kw"])],
        }), hide_index=True, use_container_width=True)