# Intelligent Customer Support Ticket System

**NLP-based classification, prioritisation and response automation for customer support tweets**
GUVI × HCL Data Science capstone · Narendran Panneerselva

Given a customer's tweet to a company, the system predicts the ticket's **intent** and **priority**, drafts a **first reply**, and estimates the company's **response time** with an SLA-risk flag. It is built on the public *Customer Support on Twitter* dataset (`twcs.csv`, ~2.8M tweets) and evaluated against **400 hand-labelled tweets**.

---

## Results at a glance

All PS-1 and PS-2 numbers are on the **240-tweet gold-test split**: hand-labelled tweets that influenced no modelling decision.

| Problem statement | Final model | Key result |
|---|---|---|
| **PS-1** Intent classification (5 classes) | TF-IDF (1–2 grams) + Logistic Regression | Macro-F1 **0.544** |
| **PS-2** Priority detection (Low / Medium / High) | Logistic Regression on rules-v2 labels | Macro-F1 **0.422**, High-priority recall **0.558** |
| **PS-3** Automated response generation | Seq2Seq LSTM + attention, no-repeat-trigram decoding | BLEU **0.102**, ROUGE-L **0.244**, perplexity **9.6** |
| **PS-4** Response-time prediction | Tuned XGBoost + workload features | R² (log) **0.411**, median error **21 min**, SLA-breach recall **78%** |

**Three findings**

1. **Silver scores overstate quality.** Intent models scored 0.97 macro-F1 on rule-derived (silver) labels but 0.54 on human labels. The gold set exposed this circularity.
2. **Urgency is situational, not stylistic.** Keyword and sentiment rules caught 10% of human-judged urgent tickets. Recalibrating the rules on a gold-dev split raised this to 56%.
3. **Response time is operational.** Company and the past hour's workload explain most of the predictable variation; the tweet's wording barely matters.

📄 The full project report (method, results, error analysis, limitations) accompanies this repo.

---

## Pipeline

```
Customer tweet + company handle          Team workload (last hour)
              │                                     │
        Pre-processing                              │
  (clean text · urgency signals · tokens)           │
              │                                     │
   ┌──────────┼──────────────┬──────────────┐       │
   ▼          ▼              ▼              ▼       ▼
 PS-1       PS-2           PS-3           PS-4 ◄────┘
 Intent     Priority       Reply          Response time
   └──────────┴──────────────┴──────────────┘
                         ▼
   Routing (priority × intent) · draft reply · SLA flag
```

---

## Data

| Step | Rows |
|---|---|
| Raw `twcs.csv` | 2,811,774 |
| First-contact tickets (inbound, start a thread) | 787,346 |
| Answered by the company, 2017 | 713,907 |
| English, ≥ 3 words after cleaning (from a 130k pool) | 117,807 |
| **Modelling sample** | **100,000** |

Cleaning (`src/cleaning.py`): HTML unescape, URL / @mention / emoji removal, contraction expansion, lower-casing. Raw text is kept for PS-2's caps and `!` signals. Language detection uses Lingua.

---

## Labelling strategy (weak supervision, Option D)

`twcs.csv` has no intent or priority labels, so they were engineered:

1. **Rules:** 3–5 regex labelling functions per class vote; majority wins; ties go to the more specific class; General counts only when no specific rule fires.
2. **Zero-shot:** `MoritzLaurer/deberta-v3-base-zeroshot-v2.0`, chosen over BART-large-MNLI (3× faster, 59% vs 51% rule agreement, realistic Billing rate).
3. **Silver labels:** rows where both signals agree. Intent: 32,168 rows. Priority v1: 55,480 rows.
4. **Gold set:** 400 random tweets hand-labelled from a written guide (`reports/labeling_guide.md`), shown text only, removed from all training. Split into **gold-dev (160)** for tuning and **gold-test (240)** for final reporting.

PS-2's v1 labels failed on gold (High recall 0.08): silver called 5% of tickets High, while human judgement called 40% High. **Rules v2**, designed on gold-dev, call a ticket High when *any single* urgency signal fires, which lifted High recall to 0.56 on gold-test.

---

## Per-task summary

### PS-1 Intent

| Model | Silver test macro-F1 | Gold macro-F1 (400) |
|---|---|---|
| **TF-IDF + LR** | 0.971 | **0.538** |
| BiLSTM (Word2Vec-initialised) | 0.976 | 0.532 |
| GloVe-Twitter + LR | 0.783 | 0.484 |
| Word2Vec (self-trained) + LR | 0.824 | 0.474 |

Main error: calm or sarcastic complaints classified as General (49 of 123). Errors: `reports/ps1_gold_errors.csv`.

### PS-2 Priority (gold-test, 240)

| Approach | Macro-F1 | High recall |
|---|---|---|
| Rules v1 | 0.370 | 0.095 |
| BiLSTM v1 + tuned threshold | 0.411 | 0.200 |
| Rules v2 (no model) | 0.435 | 0.558 |
| **LR v2 (final)** | **0.422** | **0.558** |

### PS-3 Response generation (1,000 test pairs, greedy decoding)

| Model | Perplexity | BLEU-4 | ROUGE-L | Distinct replies |
|---|---|---|---|---|
| LSTM encoder-decoder | 10.2 | 0.085 | 0.229 | 71.7% |
| + attention | 9.6 | 0.102 | 0.243 | 78.6% |
| **+ attention, no-repeat-trigram** | – | **0.102** | **0.244** | **78.7%** |

Inputs carry a company token (`co_amazonhelp …`) for on-brand replies; customer names are masked as `xname`. Samples: `reports/ps3_samples.csv`.

### PS-4 Response time (test, 9,984 tickets)

Target: **first company reply time** (Option B: earliest of all linked replies), modelled as `log(1 + minutes)`; replies after 7+ days excluded (0.17%).

| Model (validation) | Median AE | R² (log) |
|---|---|---|
| Naive company median | 29.2 min | 0.220 |
| Linear Regression | 34.9 min | 0.253 |
| Random Forest | 21.1 min | 0.413 |
| **XGBoost, 5-fold CV tuned** (test) | **21.0 min** | **0.411** |

Workload features (company replies and customer requests in the previous 60 minutes, past-only) lifted R² from 0.32 to 0.41. SLA flag (> 60 min): recall 0.777, precision 0.625. Feature importance: `reports/ps4_feature_importance.csv`.

---

## Project structure

```
├── app/
│   └── app.py                    # Streamlit demo (all four models)
├── data/
│   ├── raw/                      # twcs.csv (not tracked)
│   ├── processed/                # samples, silver labels, splits (not tracked)
│   └── gold/                     # gold_to_label.csv, gold_labeled.csv, gold_split.csv
├── models/                       # trained artifacts (.joblib, .keras, vocab .json, w2v)
├── notebooks/
│   ├── 01_data_prep.ipynb
│   ├── 02_labeling.ipynb
│   ├── 03_ps1_intent.ipynb
│   ├── 04_ps2_priority.ipynb
│   ├── 05_ps3_seq2seq.ipynb
│   ├── 06_ps4_response_time.ipynb
│   ├── 07_gold_labeling.ipynb
│   └── 08_gold_evaluation.ipynb
├── reports/
│   ├── experiments.csv           # every run: model, params, metrics, eval set
│   ├── labeling_guide.md
│   ├── intent_lf_coverage.csv
│   ├── ps1_gold_errors.csv
│   ├── ps2_gold_errors.csv
│   ├── ps3_samples.csv
│   └── ps4_feature_importance.csv
├── src/
│   ├── cleaning.py               # text cleaning, Seq2Seq cleaning, name masking
│   ├── labeling.py               # intent rules, priority rules v1 and v2
│   ├── seq2seq.py                # greedy decoding, BLEU / ROUGE
│   └── experiments.py            # experiment logger
├── requirements.txt
└── README.md
```

---

## Setup and usage

**Requirements:** Python 3.11 (TensorFlow does not support 3.14), about 8 GB free RAM.

```bash
git clone https://github.com/narendovi/Intelligent-Customer-Support-Ticket-System-.git
cd Intelligent-Customer-Support-Ticket-System-
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

1. Download `twcs.csv` (Customer Support on Twitter) into `data/raw/`.
2. Run the notebooks **01 → 08** in order. The slow steps are zero-shot labelling (~75 min on Apple-Silicon GPU) and Seq2Seq training (~3 h on CPU).
3. Launch the demo from the project root:

```bash
streamlit run app/app.py
```

The app takes a tweet and a company handle and shows intent, priority, a suggested reply, estimated response time and SLA risk. Sidebar inputs set the team's current workload.

**Notes**
- On Apple Silicon, TensorFlow runs on CPU; `tensorflow-metal` was removed after a version conflict. Zero-shot labelling uses PyTorch MPS.
- All random seeds are fixed (42 unless noted). Every experiment is logged to `reports/experiments.csv`.

---

## Limitations and future work

- Gold set labelled by one annotator; next step is a second annotator for an agreement score (Cohen's kappa).
- Intent classes overlap (a billing complaint is both Billing and Complaint); multi-label intent would fit better.
- Priority from tone and keywords has a ceiling (~0.43 macro-F1); fine-tuning a transformer on human urgency labels is the next step.
- LSTM Seq2Seq replies lean on templates and occasionally hallucinate details; retrieval or a fine-tuned pretrained generator would help.
- Timestamps are UTC; company-local time zones would sharpen PS-4.

---

## Tech stack

Python · pandas · scikit-learn · XGBoost · TensorFlow / Keras · PyTorch · Hugging Face Transformers · gensim · NLTK · VADER · Lingua · rouge-score · Streamlit

**Dataset:** [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) (ThoughtVector, Kaggle)

---

**Author:** Narendran Panneerselva · [LinkedIn](https://www.linkedin.com/in/narendran-panneerselva) · [GitHub](https://github.com/narendovi)
