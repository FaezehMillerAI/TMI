import re

import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .retrieval import extract_label


def _tokens(text):
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def rouge_l_f1(pred, ref):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(ref, pred)["rougeL"].fmeasure


def bleu1(pred, ref):
    smooth = SmoothingFunction().method1
    return sentence_bleu([_tokens(ref)], _tokens(pred), weights=(1, 0, 0, 0), smoothing_function=smooth)


def bleu2(pred, ref):
    smooth = SmoothingFunction().method1
    return sentence_bleu([_tokens(ref)], _tokens(pred), weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)


def bleu3(pred, ref):
    smooth = SmoothingFunction().method1
    return sentence_bleu(
        [_tokens(ref)],
        _tokens(pred),
        weights=(1 / 3, 1 / 3, 1 / 3, 0),
        smoothing_function=smooth,
    )


def bleu4(pred, ref):
    smooth = SmoothingFunction().method1
    return sentence_bleu(
        [_tokens(ref)],
        _tokens(pred),
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smooth,
    )


def cider_score(pred, ref):
    """
    CIDEr if possible:
    1) official pycocoevalcap implementation if installed
    2) fallback: TF-IDF cosine sentence score as CIDEr proxy
    """
    try:
        from pycocoevalcap.cider.cider import Cider

        scorer = Cider()
        gts = {0: [ref]}
        res = {0: [pred]}
        score, _ = scorer.compute_score(gts, res)
        return float(score)
    except Exception:
        vec = TfidfVectorizer(ngram_range=(1, 4), min_df=1)
        mat = vec.fit_transform([ref, pred])
        return float(cosine_similarity(mat[0], mat[1])[0, 0])


def dice_x_scores(pred, ref, retrieval_result, cfg):
    pred_toks = set(_tokens(pred))
    factual_toks = set(_tokens(retrieval_result.factual_report))

    pred_abn = {k for k in cfg.abnormality_keywords if k in pred_toks}
    support = sum(1 for k in pred_abn if k in factual_toks)
    fidelity = support / max(1, len(pred_abn))

    plausible = bleu1(pred, ref)

    qlabel = extract_label(ref, cfg.abnormality_keywords)
    f_label = extract_label(retrieval_result.factual_report, cfg.abnormality_keywords)
    a_label = extract_label(retrieval_result.anatomy_report, cfg.abnormality_keywords)
    consistency = 0.5 * float(f_label == qlabel) + 0.5 * float(a_label == qlabel)

    usefulness = forte_proxy(pred, ref, cfg)

    return {
        "consistency": float(consistency),
        "plausibility": float(plausible),
        "fidelity": float(fidelity),
        "usefulness": float(usefulness),
    }


def _f1_set(pred_set, ref_set):
    if not pred_set and not ref_set:
        return 1.0
    if not pred_set or not ref_set:
        return 0.0
    tp = len(pred_set & ref_set)
    p = tp / len(pred_set)
    r = tp / len(ref_set)
    if (p + r) == 0:
        return 0.0
    return 2 * p * r / (p + r)


def forte_proxy(pred, ref, cfg):
    degree_words = {"mild", "moderate", "severe", "small", "large"}
    landmark_words = set(cfg.anatomy_keywords)
    feature_words = set(cfg.abnormality_keywords)
    impression_words = {"no", "evidence", "suggest", "compatible", "likely", "normal"}

    pred_t = set(_tokens(pred))
    ref_t = set(_tokens(ref))

    f_degree = _f1_set(pred_t & degree_words, ref_t & degree_words)
    f_landmark = _f1_set(pred_t & landmark_words, ref_t & landmark_words)
    f_feature = _f1_set(pred_t & feature_words, ref_t & feature_words)
    f_impression = _f1_set(pred_t & impression_words, ref_t & impression_words)

    return float(np.mean([f_degree, f_landmark, f_feature, f_impression]))


def summarize_metrics(items):
    if not items:
        return {}
    out = {}
    keys = items[0].keys()
    for k in keys:
        out[k] = float(np.mean([x[k] for x in items]))
    return out
