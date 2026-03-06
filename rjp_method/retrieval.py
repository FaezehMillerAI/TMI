from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def extract_label(report: str, abnormality_keywords):
    text = report.lower()
    found = [k for k in abnormality_keywords if k in text]
    if not found:
        return "normal"
    return found[0]


def anatomy_overlap_score(a: str, b: str, anatomy_keywords):
    sa = {k for k in anatomy_keywords if k in a}
    sb = {k for k in anatomy_keywords if k in b}
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


@dataclass
class RetrievalResult:
    factual_idx: int
    counterfactual_idx: int
    anatomy_idx: int
    factual_report: str
    counterfactual_report: str
    anatomy_report: str


class RetrievalEngine:
    """
    RJP-style retrieval loop:
    1) factual retrieval (most similar)
    2) counterfactual retrieval (similar but different pseudo-label)
    3) anatomy-aware retrieval (best anatomy keyword overlap)
    """

    def __init__(
        self,
        train_records: List[Dict],
        top_k: int,
        abnormality_keywords,
        anatomy_keywords,
        use_counterfactual: bool = True,
        use_anatomy: bool = True,
    ):
        self.train_records = train_records
        self.top_k = top_k
        self.abnormality_keywords = abnormality_keywords
        self.anatomy_keywords = anatomy_keywords
        self.use_counterfactual = use_counterfactual
        self.use_anatomy = use_anatomy

        self.reports = [r["report"] for r in train_records]
        self.labels = [extract_label(t, abnormality_keywords) for t in self.reports]

        self.vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.mat = self.vec.fit_transform(self.reports)

    def retrieve(self, query_report: str):
        q = self.vec.transform([query_report])
        sim = cosine_similarity(q, self.mat)[0]
        ranking = np.argsort(-sim)

        factual_idx = int(ranking[0])

        counterfactual_idx = factual_idx
        if self.use_counterfactual:
            qlabel = extract_label(query_report, self.abnormality_keywords)
            for idx in ranking:
                if self.labels[idx] != qlabel:
                    counterfactual_idx = int(idx)
                    break

        top = ranking[: self.top_k]
        if self.use_anatomy:
            best_anat = top[0]
            best_score = -1.0
            for idx in top:
                score = anatomy_overlap_score(query_report, self.reports[idx], self.anatomy_keywords)
                if score > best_score:
                    best_score = score
                    best_anat = idx
            anatomy_idx = int(best_anat)
        else:
            anatomy_idx = factual_idx

        return RetrievalResult(
            factual_idx=factual_idx,
            counterfactual_idx=counterfactual_idx,
            anatomy_idx=anatomy_idx,
            factual_report=self.reports[factual_idx],
            counterfactual_report=self.reports[counterfactual_idx],
            anatomy_report=self.reports[anatomy_idx],
        )

    def retrieve_batch(self, query_reports: List[str]):
        return [self.retrieve(q) for q in query_reports]
