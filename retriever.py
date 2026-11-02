"""Hybrid retriever: BM25 + dense embeddings + cross-encoder reranking.

Architecture
------------
  1. BM25        — exact term matching (existing, fast, no deps)
  2. Dense       — semantic similarity via sentence-transformers (optional)
  3. RRF fusion  — reciprocal rank fusion of BM25 + dense
  4. Reranker    — cross-encoder rerank of top-K (optional)

If dense/reranker deps are absent, gracefully degrades to BM25 only — zero
config required, progressive enhancement when libraries are installed.

Usage:
  from retriever import HybridRetriever, build_hybrid_index

  retriever = HybridRetriever(segments)
  hits = retriever.search("what is weak entity", limit=5)
                  -> list[(Segment, float)]

  # For LLM RAG: get formatted context
  ctx = retriever.context_for("what is weak entity", max_chars=6000)
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Re-use existing generator utilities where possible
from generator import (
    Segment,
    TermStats,
    _collect_sentences,
    _find_definition,
    _strip_column_bleed,
    _is_junk,
    _tokenize,
    _stem_token,
    _focus_terms,
    _ARTICLE_RE,
)

# ---------------------------------------------------------------------------
# Lazy optional imports — no hard dependency
# ---------------------------------------------------------------------------

def _try_import_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        return SentenceTransformer
    except ImportError:
        return None

def _try_import_cross_encoder():
    try:
        from sentence_transformers import CrossEncoder  # type: ignore
        return CrossEncoder
    except ImportError:
        return None


_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # 384 dim, 80 MB, fast
_RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_embed_model = None
_rerank_model = None

def _get_embed_model():
    global _embed_model
    if _embed_model is not None:
        return _embed_model
    ST = _try_import_sentence_transformers()
    if ST is None:
        return None
    try:
        _embed_model = ST(_EMBED_MODEL_NAME)
        logger.info("Loaded embedding model %s", _EMBED_MODEL_NAME)
    except Exception as exc:
        logger.info("Could not load embedding model: %s", exc)
        _embed_model = False  # sentinel = tried and failed
        return None
    return _embed_model

def _get_rerank_model():
    global _rerank_model
    if _rerank_model is not None:
        return _rerank_model if _rerank_model is not False else None
    CE = _try_import_cross_encoder()
    if CE is None:
        return None
    try:
        _rerank_model = CE(_RERANK_MODEL_NAME)
        logger.info("Loaded reranker model %s", _RERANK_MODEL_NAME)
    except Exception as exc:
        logger.info("Could not load reranker: %s", exc)
        _rerank_model = False
        return None
    return _rerank_model


# ---------------------------------------------------------------------------
# BM25 sub-retriever (wraps generator.Retriever logic)
# ---------------------------------------------------------------------------

class BM25Index:
    K1 = 1.5
    B = 0.75

    def __init__(self, sentences: list[Segment]):
        self.sentences = sentences
        self.docs = [_tokenize(s.text) for s in sentences]
        self.lengths = [len(d) for d in self.docs]
        self.avg_len = sum(self.lengths) / len(self.lengths) if self.docs else 0.0
        self.freqs: list[Counter] = [Counter(d) for d in self.docs]
        df: Counter = Counter()
        for freq in self.freqs:
            df.update(freq.keys())
        n = len(self.docs)
        self.idf: dict[str, float] = {
            term: math.log(1.0 + (n - c + 0.5) / (c + 0.5))
            for term, c in df.items()
        }
        # definitions map for exact-match boost
        self.definitions: dict[str, list[Segment]] = {}
        for sent in sentences:
            d = _find_definition(sent.text)
            if d is None:
                continue
            term = _ARTICLE_RE.sub("", d.term).strip().lower()
            if term:
                self.definitions.setdefault(term, []).append(sent)

    def score(self, idx: int, query_terms: list[str]) -> float:
        total = 0.0
        length = self.lengths[idx]
        for term in set(query_terms):
            tf = self.freqs[idx].get(term, 0)
            if not tf:
                continue
            norm = tf * (self.K1 + 1) / (tf + self.K1 * (1 - self.B + self.B * length / (self.avg_len or 1)))
            total += self.idf.get(term, 0.0) * norm
        return total

    def search(self, query: str, limit: int = 20) -> list[tuple[Segment, float]]:
        terms = [_stem_token(w) for w in _focus_terms(query)]
        if not terms or not self.sentences:
            return []
        scored = [(self.score(i, terms), s) for i, s in enumerate(self.sentences)]
        hits = [(s, sc) for sc, s in scored if sc > 0]
        hits.sort(key=lambda p: p[1], reverse=True)
        return hits[:limit]

    def define(self, word: str) -> Segment | None:
        word = word.strip().lower()
        if not word:
            return None
        if word in self.definitions:
            return self.definitions[word][0]
        for length in (3, 2):
            parts = word.split()
            if len(parts) < length:
                continue
            for start in range(len(parts) - length + 1):
                phrase = " ".join(parts[start:start + length])
                if phrase in self.definitions:
                    return self.definitions[phrase][0]
        return None


# ---------------------------------------------------------------------------
# Dense sub-retriever
# ---------------------------------------------------------------------------

class DenseIndex:
    def __init__(self, sentences: list[Segment]):
        self.sentences = sentences
        self.embeddings = None
        model = _get_embed_model()
        if model is None or not sentences:
            return
        try:
            import numpy as np
            texts = [s.text for s in sentences]
            self.embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        except Exception as exc:
            logger.info("Dense indexing failed: %s", exc)
            self.embeddings = None

    def search(self, query: str, limit: int = 20) -> list[tuple[Segment, float]]:
        if self.embeddings is None:
            return []
        model = _get_embed_model()
        if model is None:
            return []
        try:
            import numpy as np
            q_emb = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
            scores = (self.embeddings @ q_emb).tolist()  # cosine (normalized)
            ranked = sorted(zip(self.sentences, scores), key=lambda p: p[1], reverse=True)
            return [(s, float(sc)) for s, sc in ranked[:limit] if sc > 0.15]
        except Exception as exc:
            logger.info("Dense search failed: %s", exc)
            return []


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------

def _rrf_fuse(
    bm25_hits: list[tuple[Segment, float]],
    dense_hits: list[tuple[Segment, float]],
    k: int = 60,
) -> list[tuple[Segment, float]]:
    """Reciprocal Rank Fusion of two ranked lists."""
    if not dense_hits:
        return bm25_hits
    if not bm25_hits:
        return dense_hits
    # map text -> rank (1-indexed)
    bm25_rank = {id(s): r for r, (s, _) in enumerate(bm25_hits, 1)}
    dense_rank = {id(s): r for r, (s, _) in enumerate(dense_hits, 1)}
    # union of all segments
    all_segs: dict[int, Segment] = {}
    for s, _ in bm25_hits:
        all_segs[id(s)] = s
    for s, _ in dense_hits:
        all_segs[id(s)] = s
    fused: list[tuple[Segment, float]] = []
    for sid, seg in all_segs.items():
        r1 = bm25_rank.get(sid)
        r2 = dense_rank.get(sid)
        score = 0.0
        if r1 is not None:
            score += 1.0 / (k + r1)
        if r2 is not None:
            score += 1.0 / (k + r2)
        # boost if in both lists
        if r1 is not None and r2 is not None:
            score *= 1.5
        fused.append((seg, score))
    fused.sort(key=lambda p: p[1], reverse=True)
    return fused


# ---------------------------------------------------------------------------
# Cross-encoder reranking
# ---------------------------------------------------------------------------

def _rerank(query: str, candidates: list[tuple[Segment, float]], top_k: int = 5) -> list[tuple[Segment, float]]:
    """Rerank candidates with a cross-encoder. Falls back to input order."""
    model = _get_rerank_model()
    if model is None or not candidates:
        return candidates[:top_k]
    try:
        pairs = [(query, s.text) for s, _ in candidates]
        scores = model.predict(pairs, show_progress_bar=False).tolist()
        reranked = sorted(zip([s for s, _ in candidates], scores), key=lambda p: p[1], reverse=True)
        return [(s, float(sc)) for s, sc in reranked[:top_k]]
    except Exception as exc:
        logger.info("Reranking failed: %s", exc)
        return candidates[:top_k]


# ---------------------------------------------------------------------------
# Public: HybridRetriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """Hybrid BM25 + dense + reranking retriever.

    Gracefully degrades: BM25 always works, dense/rerank enhance when available.
    """

    def __init__(self, segments: list[Segment], enable_dense: bool = True, enable_rerank: bool = True):
        # Clean sentences once
        tagged = _collect_sentences(segments)
        stats = TermStats.build(tagged) if tagged else None
        clean: list[Segment] = []
        for s in tagged:
            text = _strip_column_bleed(s.text, stats) if stats else s.text
            if text and not _is_junk(text):
                clean.append(Segment(text, s.course, s.filename, s.page))
        self.sentences = clean
        self.bm25 = BM25Index(clean)

        self.dense: DenseIndex | None = None
        if enable_dense:
            self.dense = DenseIndex(clean)

        self.enable_rerank = enable_rerank
        # definition map (from BM25)
        self.definitions = self.bm25.definitions

    def __bool__(self) -> bool:
        return bool(self.sentences)

    def search(self, query: str, limit: int = 5, rerank: bool | None = None) -> list[tuple[Segment, float]]:
        """Search and return top `limit` segments with scores."""
        if not self.sentences:
            return []
        if rerank is None:
            rerank = self.enable_rerank

        # Stage 1: retrieve broadly
        bm25_hits = self.bm25.search(query, limit=20)
        dense_hits: list[tuple[Segment, float]] = []
        if self.dense is not None:
            dense_hits = self.dense.search(query, limit=20)

        fused = _rrf_fuse(bm25_hits, dense_hits)
        if not fused:
            return []

        # Stage 2: rerank top candidates
        if rerank and len(fused) > limit:
            fused = _rerank(query, fused, top_k=limit)
        else:
            fused = fused[:limit]
        return fused

    def define(self, word: str) -> Segment | None:
        return self.bm25.define(word)

    def vocabulary(self) -> set[str]:
        return set(self.bm25.idf)

    def context_for(self, query: str, max_chars: int = 6000, limit: int = 8) -> str:
        """Formatted context string for LLM prompts (with citations)."""
        hits = self.search(query, limit=limit)
        if not hits:
            return ""
        parts: list[str] = []
        total = 0
        for seg, _ in hits:
            citation = ""
            if seg.course or seg.filename:
                bits = [p for p in (seg.course, seg.filename) if p]
                citation = " — " + " · ".join(bits)
                if seg.page is not None:
                    citation += f" · p.{seg.page}"
            chunk = f"[{citation.strip(' —')}]\n{seg.text}" if citation else seg.text
            if total + len(chunk) > max_chars:
                break
            parts.append(chunk)
            total += len(chunk)
        return "\n\n".join(parts)

    def context_segments(self, query: str, limit: int = 8) -> list[Segment]:
        return [s for s, _ in self.search(query, limit=limit)]


# ---------------------------------------------------------------------------
# Legacy adapter — drop-in for generator.Retriever
# ---------------------------------------------------------------------------

def build_hybrid_index(segments: list[Segment]) -> HybridRetriever:
    return HybridRetriever(segments)
