import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def split_claims(report_text):
    report_text = (report_text or "").strip()
    if not report_text:
        return []
    claims = [c.strip() for c in re.split(r"[.;]\s+", report_text) if c.strip()]
    return claims


def claim_scores_against_evidence(claim, factual, counterfactual, anatomy):
    corpus = [claim, factual or "", counterfactual or "", anatomy or ""]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    mat = vec.fit_transform(corpus)

    def sim(i):
        return float(cosine_similarity(mat[0], mat[i])[0, 0])

    sf = sim(1)
    sc = sim(2)
    sa = sim(3)

    # Differential reasoning margin: support minus counter-support.
    differential_margin = sf - sc

    return {
        "support_factual": sf,
        "support_counterfactual": sc,
        "support_anatomy": sa,
        "differential_margin": differential_margin,
    }


def build_claim_evidence_graph(sample):
    claims = split_claims(sample.get("generated_report", ""))
    nodes = []
    edges = []

    nodes.extend(
        [
            {"id": "factual", "label": "Factual Evidence", "type": "evidence"},
            {"id": "counterfactual", "label": "Counterfactual Evidence", "type": "evidence"},
            {"id": "anatomy", "label": "Anatomy Evidence", "type": "evidence"},
        ]
    )

    claim_rows = []
    for i, c in enumerate(claims):
        cid = f"claim_{i}"
        nodes.append({"id": cid, "label": c, "type": "claim"})
        scores = claim_scores_against_evidence(
            c,
            sample.get("factual_evidence", ""),
            sample.get("counterfactual_evidence", ""),
            sample.get("anatomy_evidence", ""),
        )
        claim_rows.append({"claim": c, **scores})

        edges.extend(
            [
                {"source": cid, "target": "factual", "weight": scores["support_factual"], "relation": "supports"},
                {
                    "source": cid,
                    "target": "counterfactual",
                    "weight": scores["support_counterfactual"],
                    "relation": "contrasts",
                },
                {"source": cid, "target": "anatomy", "weight": scores["support_anatomy"], "relation": "localizes"},
            ]
        )

    return {
        "uid": sample.get("uid"),
        "nodes": nodes,
        "edges": edges,
        "claims": claim_rows,
    }


def _plot_claim_radar(graph, out_html):
    import plotly.graph_objects as go

    fig = go.Figure()
    for c in graph["claims"]:
        fig.add_trace(
            go.Scatterpolar(
                r=[c["support_factual"], c["support_counterfactual"], c["support_anatomy"], c["support_factual"]],
                theta=["factual", "counterfactual", "anatomy", "factual"],
                fill="toself",
                name=(c["claim"][:48] + "...") if len(c["claim"]) > 48 else c["claim"],
            )
        )

    fig.update_layout(
        title=f"Claim-Evidence Profile (UID={graph.get('uid')})",
        polar={"radialaxis": {"visible": True, "range": [0, 1]}},
        showlegend=True,
    )
    fig.write_html(str(out_html), include_plotlyjs="cdn")


def _plot_differential_bars(graph, out_html):
    import plotly.express as px

    data = []
    for c in graph["claims"]:
        data.append({"claim": c["claim"][:70], "differential_margin": c["differential_margin"]})

    if not data:
        data = [{"claim": "no_claims", "differential_margin": 0.0}]

    fig = px.bar(
        data,
        x="claim",
        y="differential_margin",
        title=f"Differential Reasoning Margin by Claim (UID={graph.get('uid')})",
    )
    fig.write_html(str(out_html), include_plotlyjs="cdn")


def generate_qualitative_bundle(samples, out_dir: Path, top_k=15):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sort by fidelity-like proxy if present.
    with_scores = [s for s in samples if "sample_scores" in s]
    if not with_scores:
        with_scores = samples

    def _score(s):
        return float(s.get("sample_scores", {}).get("fidelity", 0.0))

    good = sorted(with_scores, key=_score, reverse=True)[:top_k]
    bad = sorted(with_scores, key=_score)[:top_k]

    differential_cases = []
    graph_dir = out_dir / "explainability_graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)

    for subset_name, subset in [("good", good), ("bad", bad)]:
        for i, s in enumerate(subset):
            g = build_claim_evidence_graph(s)
            differential_cases.append({"subset": subset_name, "uid": s.get("uid"), "graph": g})
            radar_path = graph_dir / f"{subset_name}_{i:02d}_uid_{s.get('uid')}_radar.html"
            bar_path = graph_dir / f"{subset_name}_{i:02d}_uid_{s.get('uid')}_differential.html"
            _plot_claim_radar(g, radar_path)
            _plot_differential_bars(g, bar_path)

    out = {
        "good_cases": good,
        "bad_cases": bad,
        "differential_reasoning_graphs": differential_cases,
    }

    with open(out_dir / "qualitative_bundle.json", "w") as f:
        json.dump(out, f, indent=2)

    # Also write quick markdown report for paper drafting.
    lines = ["# Qualitative Case Studies", "", "## Good Cases", ""]
    for s in good[: min(5, len(good))]:
        lines.extend(
            [
                f"### UID {s.get('uid')}",
                f"- Generated: {s.get('generated_report', '')}",
                f"- Reference: {s.get('reference_report', '')}",
                f"- Factual evidence: {s.get('factual_evidence', '')}",
                f"- Counterfactual evidence: {s.get('counterfactual_evidence', '')}",
                f"- Anatomy evidence: {s.get('anatomy_evidence', '')}",
                f"- Scores: {s.get('sample_scores', {})}",
                "",
            ]
        )

    lines.extend(["## Bad Cases", ""])
    for s in bad[: min(5, len(bad))]:
        lines.extend(
            [
                f"### UID {s.get('uid')}",
                f"- Generated: {s.get('generated_report', '')}",
                f"- Reference: {s.get('reference_report', '')}",
                f"- Factual evidence: {s.get('factual_evidence', '')}",
                f"- Counterfactual evidence: {s.get('counterfactual_evidence', '')}",
                f"- Anatomy evidence: {s.get('anatomy_evidence', '')}",
                f"- Scores: {s.get('sample_scores', {})}",
                "",
            ]
        )

    (out_dir / "qualitative_report.md").write_text("\n".join(lines))

    return out
