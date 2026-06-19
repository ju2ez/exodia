"""Consensus / majority-vote theme analysis over the knowledge base.

Deterministic and dependency-light: TF-IDF over each entry's text (title +
abstract + any cached video transcript / PDF full text, via :mod:`corpus`) yields
the field's dominant keyphrases; KMeans (k chosen by a silhouette sweep, fixed seed)
groups the corpus into theme clusters whose *sizes* are the "majority vote" over
what the field works on; a per-year keyphrase tally shows what is rising. If
scikit-learn is unavailable it degrades to a plain keyword count.

An optional LLM "consensus statement" can be layered on later (analysis.use_llm);
it is left as None here and clearly AI-labeled on the site when enabled.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

from .config import Settings
from .corpus import corpus_text
from .logging_setup import get_logger
from .models import Entry, ThemeReport
from .util import now_utc_iso, write_json

log = get_logger(__name__)

# Single-token domain stopwords (the corpus is saturated with these).
EXTRA_STOPWORDS = {
    "open", "ended", "openendedness", "endedness", "using", "via", "toward",
    "towards", "learning", "model", "models", "based", "approach", "method",
    "methods", "novel", "paper", "study", "results", "propose", "proposed", "new",
}
# Paper-boilerplate stopwords — only relevant once we mine full PDF text, where
# inline citations, figure/table refs and bibliography cruft would otherwise
# dominate. (sklearn drops these tokens before building n-grams, so removing
# "et"/"al" also removes the "et al" bigram.)
_BOILERPLATE_STOPWORDS = {
    "et", "al", "fig", "figs", "figure", "figures", "table", "tables", "eq", "eqs",
    "equation", "equations", "section", "sec", "appendix", "appendices", "arxiv",
    "preprint", "preprints", "doi", "https", "http", "www", "pp", "vol", "eg", "ie",
    "cf", "etc", "ref", "refs", "supplementary", "proceedings", "conference",
    "journal", "press", "university", "abstract", "introduction", "conclusion",
}
EXTRA_STOPWORDS |= _BOILERPLATE_STOPWORDS
_BASIC_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "to", "in", "for", "on", "with", "we",
    "our", "is", "are", "by", "as", "that", "this", "from", "at", "be", "can",
}

_TOKEN_RE = re.compile(r"[a-z][a-z+]+")
# Word endings that look plural but aren't (so we don't mangle them).
_PLURAL_GUARD = ("ss", "is", "us", "ics", "ous", "ness", "sis", "ias")


def _singularize(word: str) -> str:
    """Conservative plural -> singular fold so "agents" and "agent" share a term.

    Handles regular English plurals (-s, -es, -ies) only, with guards for common
    words that merely end in 's' (analysis, bias, status, ...). Irregular plurals
    are left alone — good enough to cluster inflected forms in the TF-IDF.
    """
    if len(word) <= 3 or word.endswith(_PLURAL_GUARD):
        return word
    if word.endswith("ies"):
        return word[:-3] + "y"  # studies -> study
    if word.endswith(("ches", "shes", "sses", "xes", "zes")):
        return word[:-2]  # searches -> search, classes -> class
    if word.endswith("s"):
        return word[:-1]  # agents -> agent
    return word


def _stem_analyzer(stop_words: set[str]):
    """A TF-IDF analyzer that singularizes tokens, drops stop words, then 1–3-grams.

    Replaces scikit-learn's default analyzer so morphological variants collapse to
    one feature; stop words are matched in singularized form too.
    """
    stop = {_singularize(s) for s in stop_words} | set(stop_words)

    def analyze(doc: str) -> list[str]:
        toks = [_singularize(t) for t in _TOKEN_RE.findall(doc.lower())]
        toks = [t for t in toks if len(t) > 1 and t not in stop]
        grams = list(toks)
        for n in (2, 3):
            grams.extend(" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1))
        return grams

    return analyze


def _entry_link(e: Entry) -> str:
    """Best outbound link for a cluster member (paper page, else abstract/any)."""
    for k in ("paper", "website", "blog", "code", "video"):
        if k in e.links:
            return e.links[k]
    return e.abstract_url or (next(iter(e.links.values()), "") if e.links else "")


def _corpus(entries: list[Entry], settings: Settings) -> tuple[list[str], list[Entry]]:
    texts, metas = [], []
    for e in entries:
        texts.append(corpus_text(e, settings))
        metas.append(e)
    return texts, metas


def analyze(entries: list[Entry], settings: Settings) -> ThemeReport:
    texts, metas = _corpus(entries, settings)
    n = len(texts)
    if n < 2:
        return ThemeReport(generated_utc=now_utc_iso(), method="insufficient_data", n_docs=n)

    try:
        import numpy as np
        from sklearn.cluster import KMeans
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
        from sklearn.metrics import silhouette_score
        from sklearn.metrics.pairwise import cosine_similarity
    except Exception as ex:  # pragma: no cover - exercised only without sklearn
        log.warning("scikit-learn unavailable (%s); using keyword counts", ex)
        return _fallback(texts, n)

    analyzer = _stem_analyzer(set(ENGLISH_STOP_WORDS) | EXTRA_STOPWORDS)

    def _fit(max_df: float):
        v = TfidfVectorizer(analyzer=analyzer, min_df=1, max_df=max_df, sublinear_tf=True)
        return v, v.fit_transform(texts)

    try:
        # Drop terms shared by >90% of docs (ubiquitous = uninformative).
        vec, X = _fit(0.9)
    except ValueError:
        # Tiny/uniform corpus: every term is ubiquitous, so keep them all.
        vec, X = _fit(1.0)
    features = vec.get_feature_names_out()

    sums = X.sum(axis=0).A1
    order = sums.argsort()[::-1]
    top_keyphrases = [
        {"phrase": features[i], "score": round(float(sums[i]), 4)}
        for i in order[:25]
        if sums[i] > 0
    ]

    clusters = _cluster(X, features, metas, settings, np, KMeans, silhouette_score, cosine_similarity)
    themes_by_year = _themes_by_year(metas, [k["phrase"] for k in top_keyphrases[:12]], settings)

    log.info("Analysis: %d docs, %d keyphrases, %d clusters", n, len(top_keyphrases), len(clusters))
    return ThemeReport(
        generated_utc=now_utc_iso(),
        method="tfidf_kmeans",
        n_docs=n,
        top_keyphrases=top_keyphrases,
        clusters=clusters,
        themes_by_year=themes_by_year,
    )


def _cluster(X, features, metas, settings, np, KMeans, silhouette_score, cosine_similarity):
    n = X.shape[0]
    kmin, kmax = settings.analysis_k_range
    kmax = min(kmax, n - 1)
    kmin = max(2, min(kmin, kmax))
    if n < 4 or kmax < kmin:
        return []

    best = None
    for k in range(kmin, kmax + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        try:
            score = silhouette_score(X, labels, metric="cosine")
        except Exception:
            continue
        if best is None or score > best[0]:
            best = (score, model)

    model = best[1] if best else KMeans(n_clusters=kmin, random_state=42, n_init=10).fit(X)
    labels = model.labels_
    centers = model.cluster_centers_

    clusters = []
    for cid in range(model.n_clusters):
        idxs = [i for i, lab in enumerate(labels) if lab == cid]
        if not idxs:
            continue
        center = centers[cid]
        top_terms = [features[i] for i in center.argsort()[::-1][:6] if center[i] > 0]
        sims = cosine_similarity(X[idxs], center.reshape(1, -1)).ravel()
        order = sims.argsort()[::-1]  # most representative first
        # Every paper in the cluster, made explicit with a link back to the source.
        members = [
            {"title": metas[idxs[o]].title, "url": _entry_link(metas[idxs[o]])}
            for o in order
        ]
        clusters.append({
            "id": int(cid),
            "label": ", ".join(top_terms[:3]) if top_terms else f"Cluster {cid + 1}",
            "size": len(idxs),
            "top_terms": top_terms,
            "representative_titles": [m["title"] for m in members[:3]],
            "members": members,
        })
    clusters.sort(key=lambda c: c["size"], reverse=True)
    return clusters


def _themes_by_year(
    metas: list[Entry], phrases: list[str], settings: Settings
) -> dict[str, dict[str, int]]:
    by_year: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    plower = [p.lower() for p in phrases]
    for e in metas:
        if not e.year:
            continue
        text = corpus_text(e, settings).lower()
        for p in plower:
            if p in text:
                by_year[str(e.year)][p] += 1
    return {y: dict(d) for y, d in sorted(by_year.items())}


def _fallback(texts: list[str], n: int) -> ThemeReport:
    stop = EXTRA_STOPWORDS | _BASIC_STOPWORDS
    words: Counter[str] = Counter()
    for t in texts:
        for w in re.findall(r"[a-zA-Z][a-zA-Z-]{2,}", t.lower()):
            if w not in stop:
                words[w] += 1
    top = [{"phrase": w, "score": c} for w, c in words.most_common(25)]
    return ThemeReport(
        generated_utc=now_utc_iso(), method="keyword_counts", n_docs=n, top_keyphrases=top
    )


def save_themes(settings: Settings, report: ThemeReport) -> None:
    write_json(settings.themes_path, report.to_dict())
