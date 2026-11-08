"""Greedy decoding and evaluation helpers for the Seq2Seq response generator."""
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from rouge_score import rouge_scorer


def greedy_decode(model, src_ids, tgt_vocab, max_len=40, chunk=128, no_repeat_ngram=0):
    """Greedy decoding; if no_repeat_ngram=n > 0, an n-gram may not appear twice in one reply."""
    start, end = tgt_vocab.index("startseq"), tgt_vocab.index("endseq")
    n = no_repeat_ngram
    results = []
    for i in range(0, len(src_ids), chunk):
        src = src_ids[i:i + chunk]
        dec = np.zeros((len(src), max_len), dtype="int64")
        dec[:, 0] = start
        done = np.zeros(len(src), dtype=bool)
        for t in range(max_len - 1):
            probs = model({"src": src, "tgt_in": dec}, training=False)[:, t, :].numpy()
            probs[:, :2] = 0                                   # never emit padding or [UNK]
            if n and t >= n - 1:
                for r in np.where(~done)[0]:
                    seq = dec[r, 1:t + 1].tolist()             # words generated so far
                    prefix = tuple(seq[-(n - 1):])
                    banned = [seq[k + n - 1] for k in range(len(seq) - n + 1)
                              if tuple(seq[k:k + n - 1]) == prefix]
                    probs[r, banned] = 0
            nxt = probs.argmax(axis=-1)
            nxt[done] = 0
            dec[:, t + 1] = nxt
            done |= nxt == end
            if done.all():
                break
        for row in dec[:, 1:]:
            words = []
            for idx in row:
                if idx in (0, end):
                    break
                words.append(tgt_vocab[idx])
            results.append(" ".join(words))
    return results


def evaluate_generation(refs, hyps):
    """Corpus BLEU-4, mean ROUGE-1/ROUGE-L F1, and share of distinct generated replies."""
    bleu = corpus_bleu([[r.split()] for r in refs], [h.split() for h in hyps],
                       smoothing_function=SmoothingFunction().method1)
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"])
    scores = [scorer.score(r, h) for r, h in zip(refs, hyps)]
    return {
        "bleu": bleu,
        "rouge1_f": float(np.mean([s["rouge1"].fmeasure for s in scores])),
        "rougeL_f": float(np.mean([s["rougeL"].fmeasure for s in scores])),
        "distinct_ratio": len(set(hyps)) / len(hyps),
    }