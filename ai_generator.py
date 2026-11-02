"""Next-level generation: LLM-powered quiz / answer / note with rule-based fallback.

All three features share the same philosophy:
  * Try the LLM (via llm_client) using RAG context from the PDFs.
  * If no LLM is configured or it fails/returns invalid JSON, fall back to the
    proven rule-based engine in generator.py — zero-config, no breakage.
  * Every result is validated against Telegram limits and provenance rules.

This is the "better than predefined" option: a real model generates fluent,
Bloom-taxonomy-aware questions, grounded answers, and synthesized notes — but
the bot never depends on it.

Public API (all synchronous for generator.py compatibility, async variants too):
  ai_generate_questions(segments, count, seed) -> list[Question]
  ai_generate_note(segments, seed)           -> Note
  ai_answer_question(query, segments)        -> Answer | None
"""
from __future__ import annotations

import json
import logging
import random
import re
from html import escape

from generator import (
    MAX_EXPLANATION_LEN,
    MAX_OPTION_LEN,
    MAX_QUESTION_LEN,
    MIN_WORDS,
    Answer,
    Note,
    Question,
    Segment,
    InsufficientTextError,
    _collect_sentences,
    _compose_explanation,
    _find_definition,
    _is_junk,
    _quizworthy,
    _strip_column_bleed,
    TermStats,
    generate_questions_from_segments as _rb_questions,
    generate_note as _rb_note,
    answer_question as _rb_answer,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers — context selection
# ---------------------------------------------------------------------------

def _top_sentences_for_quiz(segments: list[Segment], max_chars: int = 8000) -> str:
    """Pick diverse, high-value sentences for quiz generation context."""
    tagged = _collect_sentences(segments)
    if not tagged:
        return ""
    stats = TermStats.build(tagged)
    # strip column bleed and re-filter
    cleaned: list[Segment] = []
    for s in tagged:
        t = _strip_column_bleed(s.text, stats)
        if t and not _is_junk(t):
            cleaned.append(Segment(t, s.course, s.filename, s.page))
    # rank by quizworthy + value, then diversify by course/page
    scored = [(_quizworthy(s, stats) + stats.value_of_text(s.text), s) for s in cleaned]
    scored.sort(key=lambda p: p[0], reverse=True)
    # take top pool and sample to keep diversity
    pool = [s for _, s in scored if _ > 0] or [s for _, s in scored[:20]]
    # ensure every course appears at least once
    seen_courses: set[str] = set()
    picked: list[Segment] = []
    for s in pool:
        if s.course not in seen_courses and len(picked) < 5:
            picked.append(s)
            seen_courses.add(s.course or "")
    for s in pool:
        if s not in picked:
            picked.append(s)
        if len(picked) >= 30:
            break
    # build context string with citations
    parts: list[str] = []
    total = 0
    for s in picked:
        citation = ""
        if s.course or s.filename:
            bits = [p for p in (s.course, s.filename) if p]
            citation = " · ".join(bits)
            if s.page is not None:
                citation += f" · p.{s.page}"
            citation = f"[{citation}] "
        chunk = f"{citation}{s.text}"
        if total + len(chunk) > max_chars:
            break
        parts.append(chunk)
        total += len(chunk) + 2
    return "\n".join(parts)


def _top_sentences_for_note(segments: list[Segment], max_chars: int = 6000) -> str:
    tagged = _collect_sentences(segments)
    if not tagged:
        return ""
    stats = TermStats.build(tagged)
    cleaned = []
    for s in tagged:
        t = _strip_column_bleed(s.text, stats)
        if t and not _is_junk(t):
            cleaned.append(Segment(t, s.course, s.filename, s.page))
    scored = [(_quizworthy(s, stats), s) for s in cleaned]
    # prefer definitions
    scored.sort(key=lambda p: p[0], reverse=True)
    top = [s for _, s in scored[:15] if _ > 0] or [s for _, s in scored[:10]]
    parts = []
    total = 0
    for s in top:
        cit = ""
        if s.course or s.filename:
            bits = [p for p in (s.course, s.filename) if p]
            cit = " · ".join(bits)
            if s.page is not None:
                cit += f" · p.{s.page}"
            cit = f"[{cit}] "
        chunk = f"{cit}{s.text}"
        if total + len(chunk) > max_chars:
            break
        parts.append(chunk)
        total += len(chunk) + 2
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_question(raw: dict, source_segments: list[Segment]) -> Question | None:
    """Validate a single LLM-produced question dict."""
    try:
        text = str(raw.get("text") or raw.get("question") or "").strip()
        options = raw.get("options") or []
        correct = raw.get("correct_index")
        if correct is None:
            correct = raw.get("answer_index")
        kind = str(raw.get("kind") or "mcq").strip().lower()
        if kind not in ("mcq", "tf"):
            kind = "mcq"
        explanation = raw.get("explanation")
        subtype = str(raw.get("subtype") or "llm").strip()

        if not text or len(text) > MAX_QUESTION_LEN or len(text) < 5:
            return None
        if not isinstance(options, list) or len(options) < 2 or len(options) > 10:
            return None
        options = [str(o).strip() for o in options if str(o).strip()]
        if len(options) < 2:
            return None
        for o in options:
            if len(o) > MAX_OPTION_LEN or len(o) < 1:
                return None
        if len({o.lower() for o in options}) != len(options):
            return None
        if not isinstance(correct, int) or not (0 <= correct < len(options)):
            return None
        # tf must be True/False
        if kind == "tf":
            if set(options) != {"True", "False"}:
                # normalize
                return None
        if explanation and len(explanation) > MAX_EXPLANATION_LEN:
            explanation = explanation[: MAX_EXPLANATION_LEN - 1].rsplit(" ", 1)[0] + "…"

        # attach provenance from the most relevant segment (heuristic: longest overlap)
        course = filename = None
        page = None
        if source_segments:
            # find segment with most word overlap
            q_words = set(re.findall(r"[A-Za-z0-9]+", text.lower()))
            best = None
            best_overlap = -1
            for seg in source_segments:
                overlap = len(q_words & set(re.findall(r"[A-Za-z0-9]+", seg.text.lower())))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best = seg
            if best is not None:
                course, filename, page = best.course, best.filename, best.page
                if not explanation:
                    explanation = _compose_explanation(best.text, best)
                elif "📄" not in explanation:
                    ref = _compose_explanation(None, best)
                    if ref:
                        # append ref if room
                        if len(explanation) + 2 + len(ref) <= MAX_EXPLANATION_LEN:
                            explanation = f"{explanation}\n\n{ref}"
        return Question(
            text=text, options=options, correct_index=correct,
            kind=kind, subtype=subtype, explanation=explanation,
            course=course, filename=filename, page=page,
        )
    except Exception as exc:
        logger.info("question validation failed: %s | raw=%.200s", exc, str(raw))
        return None


# ---------------------------------------------------------------------------
# Quiz — LLM
# ---------------------------------------------------------------------------

_QUIZ_SYSTEM = (
    "You are an expert university exam creator. You generate high-quality quiz "
    "questions STRICTLY from the provided study material. Rules:\n"
    "- Every question must be answerable ONLY from the material — never use outside knowledge.\n"
    "- Vary cognitive levels: Recall, Comprehension, Application (scenario), Analysis.\n"
    "- Include these types across the set: definition, fill-in-the-blank, numeric, true/false, "
    "scenario/application, comparison, cause-effect.\n"
    "- Distractors must be plausible and from the same topic — never random.\n"
    "- Telegram limits: question 5-300 chars, each option 1-100 chars, 2-4 options.\n"
    "- For true/false, options must be exactly [\"True\",\"False\"].\n"
    "- Reply with JSON only."
)

def _quiz_prompt(material: str, count: int) -> str:
    return (
        f"Study material (with [Course · File · p.N] citations):\n"
        f"---\n{material}\n---\n\n"
        f"Task: Create {count} quiz questions from ONLY this material.\n"
        f"Return a JSON array of objects with keys:\n"
        f'  "text": string (the question),\n'
        f'  "options": string[] (2-4 options),\n'
        f'  "correct_index": number (0-based),\n'
        f'  "kind": "mcq" or "tf",\n'
        f'  "subtype": one of "def2term","term2def","number","tf","scenario","comparison","cause_effect",\n'
        f'  "explanation": string (1 sentence from material + why it is correct, max 120 chars)\n'
        f"Example:\n"
        f'[{{\n'
        f'  "text": "Which term is described as: \\"a collection of related data stored together\\"",\n'
        f'  "options": ["Database","Relation","Attribute","Tuple"],\n'
        f'  "correct_index": 0, "kind": "mcq", "subtype": "def2term",\n'
        f'  "explanation": "A database is a collection of related data stored together."\n'
        f"}}]\n"
        f"Return ONLY the JSON array."
    )


def _try_llm_quiz(segments: list[Segment], count: int, seed: int | None) -> list[Question] | None:
    try:
        from llm_client import generate_json, is_llm_available
    except ImportError:
        return None
    if not is_llm_available():
        return None
    material = _top_sentences_for_quiz(segments)
    if not material.strip():
        return None
    prompt = _quiz_prompt(material, count)
    # add seed hint for determinism
    if seed is not None:
        prompt += f"\nRandom seed: {seed} — use it to vary the questions."
    data = generate_json(prompt, _QUIZ_SYSTEM, timeout=30)
    if not isinstance(data, list) or not data:
        return None
    questions: list[Question] = []
    seen: set[str] = set()
    for raw in data:
        if not isinstance(raw, dict):
            continue
        q = _validate_question(raw, segments)
        if q is None:
            continue
        key = re.sub(r"\W+", "", q.text.lower())[:60]
        if key in seen:
            continue
        seen.add(key)
        questions.append(q)
        if len(questions) >= count:
            break
    if not questions:
        return None
    logger.info("LLM quiz: %d/%d valid questions", len(questions), len(data))
    return questions


# ---------------------------------------------------------------------------
# Answer — LLM (RAG)
# ---------------------------------------------------------------------------

_ANSWER_SYSTEM = (
    "You are a study assistant. Answer the student's question STRICTLY from the "
    "provided material. Rules:\n"
    "- If the material contains the answer, quote/synthesize 1-2 sentences and cite the source.\n"
    "- If the material does NOT contain the answer, reply exactly: NOT_FOUND\n"
    "- Never use outside knowledge. Never hallucinate.\n"
    "- Reply with JSON: {\"answer\": string, \"found\": boolean, \"citation\": string}\n"
)

def _answer_prompt(query: str, context: str) -> str:
    return (
        f"Material:\n---\n{context}\n---\n\n"
        f"Student question: {query}\n\n"
        f"Answer strictly from the material above. If not found, set found=false and answer=\"NOT_FOUND\".\n"
        f'Return JSON: {{"answer": "...", "found": true/false, "citation": "Course · File · p.N"}}'
    )


def _try_llm_answer(query: str, segments: list[Segment]) -> Answer | None:
    try:
        from llm_client import generate_json, is_llm_available
    except ImportError:
        return None
    if not is_llm_available():
        return None
    # build context via hybrid retriever if available, else simple
    context = ""
    provenance: Segment | None = None
    try:
        from retriever import HybridRetriever
        hr = HybridRetriever(segments)
        context = hr.context_for(query, max_chars=6000, limit=8)
        hits = hr.search(query, limit=1)
        if hits:
            provenance = hits[0][0]
    except Exception:
        # fallback: top quizworthy sentences containing query words
        material = _top_sentences_for_quiz(segments, max_chars=6000)
        context = material
    if not context.strip():
        return None
    prompt = _answer_prompt(query, context)
    data = generate_json(prompt, _ANSWER_SYSTEM, timeout=20)
    if not isinstance(data, dict):
        return None
    if not data.get("found"):
        return None
    text = str(data.get("answer") or "").strip()
    if not text or text == "NOT_FOUND" or len(text) < 5:
        return None
    # hallucination guard: answer must overlap with context (at least 3 content words)
    ctx_words = set(re.findall(r"[A-Za-z]{3,}", context.lower()))
    ans_words = set(re.findall(r"[A-Za-z]{3,}", text.lower()))
    if len(ctx_words & ans_words) < 3:
        logger.info("LLM answer rejected by overlap guard: %.80s", text)
        return None
    # resolve provenance
    course = filename = None
    page = None
    if provenance:
        course, filename, page = provenance.course, provenance.filename, provenance.page
    # try to parse citation
    citation = str(data.get("citation") or "")
    also: list[str] = []
    return Answer(query=query, text=text, course=course, filename=filename, page=page, also=also, is_definition=False)


# ---------------------------------------------------------------------------
# Note — LLM (synthesis)
# ---------------------------------------------------------------------------

_NOTE_SYSTEM = (
    "You are a study-note synthesizer. From the provided material, create ONE "
    "concise, well-structured study note. Rules:\n"
    "- Synthesize — don't just copy one sentence.\n"
    "- Include: a clear title (the key term), 2-3 bullet points, and an example if present.\n"
    "- Stay STRICTLY within the material — no outside facts.\n"
    "- Keep note text under 500 chars.\n"
    "- Reply with JSON: {\"term\": string, \"text\": string, \"detail\": string or null, \"source\": string}\n"
)

def _note_prompt(material: str) -> str:
    return (
        f"Material:\n---\n{material}\n---\n\n"
        f"Create ONE study note from this material (pick the most central concept).\n"
        f'Return JSON: {{"term": "Photosynthesis", "text": "Photosynthesis is ...", "detail": "Light reactions occur ...", "source": "Course · File · p.N"}}'
    )


def _try_llm_note(segments: list[Segment], seed: int | None) -> Note | None:
    try:
        from llm_client import generate_json, is_llm_available
    except ImportError:
        return None
    if not is_llm_available():
        return None
    material = _top_sentences_for_note(segments)
    if not material.strip():
        return None
    prompt = _note_prompt(material)
    if seed is not None:
        prompt += f"\nSeed: {seed}"
    data = generate_json(prompt, _NOTE_SYSTEM, timeout=20)
    if not isinstance(data, dict):
        return None
    term = str(data.get("term") or "").strip() or None
    text = str(data.get("text") or "").strip()
    detail = str(data.get("detail") or "").strip() or None
    if not text or len(text) < 10:
        return None
    if len(text) > 600:
        text = text[:597] + "…"
    # provenance: best overlapping segment
    best: Segment | None = None
    if term:
        low = term.lower()
        for seg in segments:
            if low in seg.text.lower():
                best = seg
                break
    if best is None:
        tagged = _collect_sentences(segments)
        if tagged:
            best = tagged[0]
    course = best.course if best else None
    filename = best.filename if best else None
    page = best.page if best else None
    # hallucination guard
    ctx_words = set(re.findall(r"[A-Za-z]{3,}", material.lower()))
    note_words = set(re.findall(r"[A-Za-z]{3,}", (text + " " + (detail or "")).lower()))
    if len(ctx_words & note_words) < 4:
        logger.info("LLM note rejected by overlap guard")
        return None
    return Note(text=text, course=course, filename=filename, page=page, term=term, detail=detail)


# ---------------------------------------------------------------------------
# Public hybrid entry points (LLM first, rule-based fallback)
# ---------------------------------------------------------------------------

def ai_generate_questions(
    segments: list[Segment],
    count: int = 15,
    seed: int | None = None,
) -> list[Question]:
    """Hybrid quiz generation — LLM if available, else rule-based."""
    if not segments:
        raise InsufficientTextError("There is no material to build questions from.")
    # try LLM for 70% of quizzes when available (keeps variety, respects rate limits)
    # but if LLM returns partial, fill remainder with rule-based
    llm_qs = _try_llm_quiz(segments, count, seed)
    if llm_qs is not None and len(llm_qs) >= max(2, count // 2):
        if len(llm_qs) >= count:
            return llm_qs[:count]
        # fill remainder with rule-based, deduplicated
        try:
            rb_qs = _rb_questions(segments, count - len(llm_qs), seed=(seed + 1) if seed is not None else None)
            seen = {re.sub(r"\W+", "", q.text.lower())[:60] for q in llm_qs}
            for q in rb_qs:
                key = re.sub(r"\W+", "", q.text.lower())[:60]
                if key not in seen:
                    llm_qs.append(q)
                    seen.add(key)
                if len(llm_qs) >= count:
                    break
        except InsufficientTextError:
            pass
        return llm_qs
    # fallback entirely
    return _rb_questions(segments, count, seed)


def ai_generate_note(segments: list[Segment], seed: int | None = None) -> Note:
    """Hybrid note — LLM synthesis if available, else rule-based."""
    note = _try_llm_note(segments, seed)
    if note is not None:
        return note
    return _rb_note(segments, seed)


def ai_answer_question(query: str, segments: list[Segment], limit: int = 5) -> Answer | None:
    """Hybrid answering — LLM RAG if available, else BM25. LLM answer is verified."""
    llm_ans = _try_llm_answer(query, segments)
    if llm_ans is not None:
        return llm_ans
    return _rb_answer(query, segments, limit)


# Async variants for bot.py

async def ai_generate_questions_async(segments: list[Segment], count: int = 15, seed: int | None = None) -> list[Question]:
    try:
        from llm_client import generate_json_async, is_llm_available
        if is_llm_available():
            material = _top_sentences_for_quiz(segments)
            if material.strip():
                prompt = _quiz_prompt(material, count)
                if seed is not None:
                    prompt += f"\nRandom seed: {seed}"
                data = await generate_json_async(prompt, _QUIZ_SYSTEM, timeout=30)
                if isinstance(data, list) and data:
                    questions: list[Question] = []
                    seen: set[str] = set()
                    for raw in data:
                        if not isinstance(raw, dict):
                            continue
                        q = _validate_question(raw, segments)
                        if q is None:
                            continue
                        key = re.sub(r"\W+", "", q.text.lower())[:60]
                        if key in seen:
                            continue
                        seen.add(key)
                        questions.append(q)
                        if len(questions) >= count:
                            break
                    if questions and len(questions) >= max(2, count // 2):
                        return questions
    except Exception as exc:
        logger.info("async LLM quiz failed, falling back: %s", exc)
    # fallback
    import asyncio
    return await asyncio.to_thread(_rb_questions, segments, count, seed)


async def ai_answer_question_async(query: str, segments: list[Segment], limit: int = 5) -> Answer | None:
    try:
        from llm_client import generate_json_async, is_llm_available
        if is_llm_available():
            context = ""
            provenance = None
            try:
                from retriever import HybridRetriever
                hr = HybridRetriever(segments)
                context = hr.context_for(query, max_chars=6000, limit=8)
                hits = hr.search(query, limit=1)
                if hits:
                    provenance = hits[0][0]
            except Exception:
                context = _top_sentences_for_quiz(segments, max_chars=6000)
            if context.strip():
                prompt = _answer_prompt(query, context)
                data = await generate_json_async(prompt, _ANSWER_SYSTEM, timeout=20)
                if isinstance(data, dict) and data.get("found"):
                    text = str(data.get("answer") or "").strip()
                    if text and text != "NOT_FOUND" and len(text) >= 5:
                        ctx_words = set(re.findall(r"[A-Za-z]{3,}", context.lower()))
                        ans_words = set(re.findall(r"[A-Za-z]{3,}", text.lower()))
                        if len(ctx_words & ans_words) >= 3:
                            course = provenance.course if provenance else None
                            filename = provenance.filename if provenance else None
                            page = provenance.page if provenance else None
                            return Answer(query=query, text=text, course=course, filename=filename, page=page, also=[], is_definition=False)
    except Exception as exc:
        logger.info("async LLM answer failed, falling back: %s", exc)
    import asyncio
    return await asyncio.to_thread(_rb_answer, query, segments, limit)


async def ai_generate_note_async(segments: list[Segment], seed: int | None = None) -> Note:
    try:
        from llm_client import generate_json_async, is_llm_available
        if is_llm_available():
            material = _top_sentences_for_note(segments)
            if material.strip():
                prompt = _note_prompt(material)
                if seed is not None:
                    prompt += f"\nSeed: {seed}"
                data = await generate_json_async(prompt, _NOTE_SYSTEM, timeout=20)
                if isinstance(data, dict):
                    term = str(data.get("term") or "").strip() or None
                    text = str(data.get("text") or "").strip()
                    detail = str(data.get("detail") or "").strip() or None
                    if text and len(text) >= 10:
                        if len(text) > 600:
                            text = text[:597] + "…"
                        best = None
                        if term:
                            low = term.lower()
                            for seg in segments:
                                if low in seg.text.lower():
                                    best = seg
                                    break
                        if best is None:
                            tagged = _collect_sentences(segments)
                            if tagged:
                                best = tagged[0]
                        course = best.course if best else None
                        filename = best.filename if best else None
                        page = best.page if best else None
                        ctx_words = set(re.findall(r"[A-Za-z]{3,}", material.lower()))
                        note_words = set(re.findall(r"[A-Za-z]{3,}", (text + " " + (detail or "")).lower()))
                        if len(ctx_words & note_words) >= 4:
                            return Note(text=text, course=course, filename=filename, page=page, term=term, detail=detail)
    except Exception as exc:
        logger.info("async LLM note failed, falling back: %s", exc)
    import asyncio
    return await asyncio.to_thread(_rb_note, segments, seed)
