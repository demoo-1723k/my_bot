"""Tests for the PDF → quiz generator.

Builds a tiny valid PDF on the fly (no external files needed), then checks
extraction and that every generated question respects Telegram poll limits.

Run from the project root:
    python -m unittest -v
"""

from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path

from generator import (
    MAX_OPTION_LEN,
    MAX_QUESTION_LEN,
    InsufficientTextError,
    Note,
    Segment,
    TermStats,
    _collect_sentences,
    _find_definition,
    _stem,
    _swap_word,
    extract_pages_from_pdf,
    extract_text_from_pdf,
    generate_note,
    generate_questions,
    generate_questions_from_segments,
)

SAMPLE_LINES = [
    "Photosynthesis is the process by which plants convert light energy into chemical energy.",
    "The mitochondria is the powerhouse of the cell and produces ATP.",
    "Water molecules are made of two hydrogen atoms and one oxygen atom.",
    "The human heart has four chambers that pump blood through the body.",
    "Osmosis is the movement of water across a semipermeable membrane.",
    "The periodic table lists 118 known chemical elements.",
    "Enzymes are biological catalysts that speed up chemical reactions.",
    "During respiration cells break down glucose to release energy.",
    "The pancreas produces insulin which regulates blood sugar levels.",
    "Ribosomes are the cell structures where proteins are synthesized.",
    # a real course repeats its terms; the generator only asks about central
    # ones, so the fixture has to repeat them too
    "The cell membrane controls which substances enter and leave the cell.",
    "Chloroplasts capture light energy and store it as chemical energy.",
    "Proteins are built by ribosomes and folded inside the cell.",
    "Osmosis moves water across the cell membrane until both sides balance.",
]


def _stream_for(lines: list[str]) -> str:
    def esc(s: str) -> str:
        return s.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")

    content = ["BT", "/F1 12 Tf", "50 760 Td"]
    for i, line in enumerate(lines):
        if i:
            content.append("0 -18 Td")
        content.append(f"({esc(line)}) Tj")
    content.append("ET")
    return "\n".join(content)


def _write_objects(objects: list[str]) -> bytes:
    pdf = b"%PDF-1.4\n"
    offsets: list[int] = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf += f"{i} 0 obj\n{body}\nendobj\n".encode("latin-1")

    xref_pos = len(pdf)
    pdf += f"xref\n0 {len(objects) + 1}\n".encode()
    pdf += b"0000000000 65535 f \n"
    for off in offsets:
        pdf += f"{off:010d} 00000 n \n".encode()
    pdf += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF"
    ).encode()
    return pdf


def _build_pdf_pages(pages: list[list[str]]) -> bytes:
    """Write a minimal multi-page PDF — one text page per entry."""
    n = len(pages)
    font_obj = 3 + 2 * n
    objects = ["<< /Type /Catalog /Pages 2 0 R >>"]
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(n))
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {n} >>")
    for i, lines in enumerate(pages):
        stream = _stream_for(lines)
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {4 + 2 * i} 0 R "
            f"/Resources << /Font << /F1 {font_obj} 0 R >> >> >>"
        )
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
    objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    return _write_objects(objects)


def _build_pdf(lines: list[str]) -> bytes:
    """Write a minimal single-page PDF containing `lines` of text."""
    return _build_pdf_pages([lines])


class GeneratorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.pdf_path = Path(self.tmp.name) / "notes.pdf"
        self.pdf_path.write_bytes(_build_pdf(SAMPLE_LINES))
        self.text = extract_text_from_pdf(self.pdf_path)

    def test_extract_text_from_pdf(self) -> None:
        self.assertIn("Photosynthesis", self.text)
        self.assertIn("Ribosomes", self.text)
        # every sample line survived extraction
        for line in SAMPLE_LINES:
            self.assertIn(line.split()[0], self.text)

    def test_generates_requested_count(self) -> None:
        questions = generate_questions(self.text, count=6, seed=42)
        self.assertGreaterEqual(len(questions), 4)
        self.assertLessEqual(len(questions), 6)

    def test_questions_fit_telegram_poll_limits(self) -> None:
        questions = generate_questions(self.text, count=6, seed=7)
        self.assertTrue(questions)
        for q in questions:
            with self.subTest(question=q.text[:40]):
                self.assertGreaterEqual(len(q.text), 1)
                self.assertLessEqual(len(q.text), MAX_QUESTION_LEN)
                self.assertGreaterEqual(len(q.options), 2)
                self.assertLessEqual(len(q.options), 10)
                self.assertIn(q.correct_index, range(len(q.options)))
                self.assertEqual(
                    len(q.options), len({o.lower() for o in q.options}),
                    "options must be unique",
                )
                for opt in q.options:
                    self.assertLessEqual(len(opt), MAX_OPTION_LEN)
                self.assertIn(q.kind, {"mcq", "tf"})
                if q.explanation:
                    self.assertLessEqual(len(q.explanation), 190)

    def test_true_false_shape(self) -> None:
        questions = generate_questions(self.text, count=8, seed=3)
        tf = [q for q in questions if q.kind == "tf"]
        self.assertTrue(tf, "expected at least one true/false question")
        for q in tf:
            self.assertEqual(set(q.options), {"True", "False"})
            self.assertIn(q.correct_index, (0, 1))

    def test_mcq_correct_answer_comes_from_source(self) -> None:
        questions = generate_questions(self.text, count=8, seed=11)
        mcq = [q for q in questions if q.kind == "mcq"]
        self.assertTrue(mcq, "expected at least one MCQ question")
        for q in mcq:
            self.assertIsNotNone(q.explanation)
            correct = q.options[q.correct_index]
            # strip punctuation/symbols so "12%" matches "12"
            norm = lambda s: "".join(c for c in s.lower() if c.isalnum())
            self.assertIn(norm(correct), norm(q.explanation or ""))

    def test_insufficient_text_raises(self) -> None:
        with self.assertRaises(InsufficientTextError):
            generate_questions("", count=3)
        with self.assertRaises(InsufficientTextError):
            generate_questions("too short", count=3)


class MixedGenerationTest(unittest.TestCase):
    """generate_mixed_questions pulls from every course, round-robin."""

    COURSE_A = "\n".join(SAMPLE_LINES)
    COURSE_B = "\n".join(
        [
            "Zeolites are microporous aluminosilicate minerals used as catalysts.",
            "Xenon is a noble gas with atomic number 54 on the periodic table.",
            "The Haber process combines nitrogen and hydrogen to make ammonia.",
            "Benzene is an aromatic hydrocarbon with the molecular formula C6H6.",
            "Polyethylene is a plastic made from repeating ethylene monomers.",
            "Titration is a laboratory technique to determine concentration.",
            "Gypsum is a soft sulfate mineral used to make plaster of Paris.",
        ]
    )
    B_TOKENS = ("zeolite", "xenon", "haber", "benzene", "polyethylene", "titration", "gypsum")
    A_TOKENS = ("photosynthesis", "osmosis", "enzyme", "ribosome", "membrane", "chloroplast")

    def test_mixed_quiz_contains_questions_from_both_courses(self) -> None:
        from generator import generate_mixed_questions

        questions, skipped = generate_mixed_questions(
            [("A", self.COURSE_A), ("B", self.COURSE_B)], count=6, seed=5
        )
        self.assertEqual(len(questions), 6)
        self.assertEqual(skipped, [])

        haystack = " ".join(
            q.text + " " + " ".join(q.options) + " " + (q.explanation or "")
            for q in questions
        ).lower()
        self.assertTrue(any(t in haystack for t in self.A_TOKENS),
                        "expected at least one question from course A")
        self.assertTrue(any(t in haystack for t in self.B_TOKENS),
                        "expected at least one question from course B")

    def test_mixed_quiz_respects_limits(self) -> None:
        from generator import generate_mixed_questions

        questions, _ = generate_mixed_questions(
            [("A", self.COURSE_A), ("B", self.COURSE_B)], count=15, seed=9
        )
        # count is an upper bound: this fixture is only ~14 sentences, and the
        # engine only builds questions from sentences that carry a fact.
        self.assertLessEqual(len(questions), 15)
        self.assertGreaterEqual(len(questions), 5)
        for q in questions:
            self.assertLessEqual(len(q.text), MAX_QUESTION_LEN)
            self.assertGreaterEqual(len(q.options), 2)
            self.assertIn(q.correct_index, range(len(q.options)))

    def test_mixed_quiz_skips_unreadable_source(self) -> None:
        from generator import generate_mixed_questions

        questions, skipped = generate_mixed_questions(
            [("good", self.COURSE_A), ("broken", "tiny")], count=5, seed=1
        )
        self.assertTrue(questions)
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0][0], "broken")

    def test_mixed_quiz_empty_sources_raises(self) -> None:
        from generator import generate_mixed_questions

        with self.assertRaises(InsufficientTextError):
            generate_mixed_questions([], count=5)
        with self.assertRaises(InsufficientTextError):
            generate_mixed_questions([("bad", "tiny")], count=5)


class ProvenanceTest(unittest.TestCase):
    """Questions must point back to course · file · page."""

    PAGES = {
        1: [
            "Photosynthesis is the process by which plants convert light energy into chemical energy.",
            "The light dependent reactions take place in the thylakoid membranes of chloroplasts.",
            "Chloroplasts capture light energy and store it as chemical energy.",
            "Water molecules are used in photosynthesis to release oxygen.",
        ],
        2: [
            "Osmosis is the movement of water across a semipermeable membrane.",
            "The periodic table lists 118 known chemical elements in order of atomic number.",
            "Enzymes are biological catalysts that speed up chemical reactions in living cells.",
            "Water molecules are made of two hydrogen atoms and one oxygen atom.",
            "The cell membrane controls which substances enter and leave the cell.",
            "Chemical energy from food is released by respiration inside every cell.",
        ],
    }

    def _segments(self) -> list[Segment]:
        return [
            Segment(
                text="\n".join(lines),
                course="Biology",
                filename="ch1.pdf",
                page=page,
            )
            for page, lines in self.PAGES.items()
        ]

    def test_extract_pages_preserves_page_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / "two.pdf"
            pdf.write_bytes(_build_pdf_pages(list(self.PAGES.values())))
            pages = extract_pages_from_pdf(pdf)
        self.assertEqual(len(pages), 2)
        self.assertIn("Photosynthesis", pages[0])
        self.assertIn("Osmosis", pages[1])

    def test_questions_carry_provenance(self) -> None:
        questions = generate_questions_from_segments(self._segments(), count=6, seed=3)
        self.assertEqual(len(questions), 6)
        for q in questions:
            self.assertEqual(q.course, "Biology")
            self.assertEqual(q.filename, "ch1.pdf")
            self.assertIn(q.page, (1, 2))

    def test_explanation_references_material(self) -> None:
        questions = generate_questions_from_segments(self._segments(), count=6, seed=3)
        mcq = [q for q in questions if q.kind == "mcq"]
        self.assertTrue(mcq)
        for q in mcq:
            self.assertIn("📄 Biology · ch1.pdf · p.", q.explanation or "")
            self.assertLessEqual(len(q.explanation or ""), 190)
            # the source sentence is still there, before the reference
            self.assertNotIn("📄", q.explanation.split("\n\n")[0])
        # true/false also cites the material — students who pick wrong need it
        tf = [q for q in questions if q.kind == "tf"]
        self.assertTrue(tf)
        for q in tf:
            self.assertIn("📄", q.explanation or "")

    def test_plain_text_has_no_material_reference(self) -> None:
        questions = generate_questions("\n".join(SAMPLE_LINES), count=4, seed=2)
        for q in questions:
            self.assertNotIn("📄", q.explanation or "")

    def test_page_boundary_fragments_are_dropped(self) -> None:
        from generator import _collect_sentences

        segments = [
            Segment(
                text="Photosynthesis is the process by which plants convert light energy",
                course="B", filename="f.pdf", page=1,
            ),
            Segment(
                text=(
                    "ical energy happens inside the chloroplasts of green plant cells. "
                    "Osmosis is the movement of water across a membrane."
                ),
                course="B", filename="f.pdf", page=2,
            ),
        ]
        texts = [s.text for s in _collect_sentences(segments)]
        self.assertFalse(any(t.endswith("light energy") for t in texts),
                         "page-1 tail fragment should be dropped")
        self.assertFalse(any(t.startswith("ical energy") for t in texts),
                         "page-2 continuation fragment should be dropped")
        self.assertTrue(any(t.startswith("Osmosis") for t in texts),
                        "complete sentences must survive")


class NoteTest(unittest.TestCase):
    """A note is one good sentence plus the place it came from."""

    PAGES = {
        1: [
            "Photosynthesis is the process by which plants convert light energy into chemical energy.",
            "The light dependent reactions take place in the thylakoid membranes of chloroplasts.",
        ],
        2: [
            "Osmosis is the movement of water across a semipermeable membrane.",
            "The periodic table lists 118 known chemical elements in order of atomic number.",
            "Enzymes are biological catalysts that speed up chemical reactions in living cells.",
            "Water molecules are made of two hydrogen atoms and one oxygen atom.",
        ],
    }

    def _segments(self) -> list[Segment]:
        return [
            Segment(
                text="\n".join(lines),
                course="Biology",
                filename="ch1.pdf",
                page=page,
            )
            for page, lines in self.PAGES.items()
        ]

    def test_note_comes_from_the_material(self) -> None:
        note = generate_note(self._segments(), seed=3)
        haystack = " ".join(
            " ".join(lines) for lines in self.PAGES.values()
        )
        self.assertIn(note.text.rstrip("."), haystack)
        self.assertTrue(note.text.endswith((".", "!", "?")), "notes end as sentences")

    def test_note_carries_provenance(self) -> None:
        note = generate_note(self._segments(), seed=3)
        self.assertEqual(note.course, "Biology")
        self.assertEqual(note.filename, "ch1.pdf")
        self.assertIn(note.page, (1, 2))

    def test_render_shows_course_file_and_page(self) -> None:
        note = generate_note(self._segments(), seed=3)
        out = note.render()
        self.assertIn("Course: <b>Biology</b>", out)
        self.assertIn("File: ch1.pdf", out)
        self.assertIn(f"Page: <b>{note.page}</b>", out)
        # the note itself comes first, and its term is bolded rather than
        # repeated, so compare with the tags taken out
        plain = re.sub(r"</?b>", "", out)
        self.assertIn(note.text, plain)
        if note.term:
            self.assertIn(f"<b>{note.term}</b>", out)
        self.assertIn("Daily Note", out)

    def test_render_uses_the_given_title(self) -> None:
        out = generate_note(self._segments(), seed=3).render("Note")
        self.assertIn("Note", out)
        self.assertNotIn("Daily Note", out)

    def test_render_escapes_html(self) -> None:
        """Course names and PDF text come from users — never trusted as HTML."""
        out = Note(text="A <b>cell</b> & its membrane", course="Bio & Chem").render()
        self.assertIn("A &lt;b&gt;cell&lt;/b&gt; &amp; its membrane", out)
        self.assertIn("Course: <b>Bio &amp; Chem</b>", out)
        self.assertNotIn("<b>cell</b>", out)

    def test_render_omits_an_unknown_source(self) -> None:
        out = Note(text="Water boils at 100 degrees celsius at sea level.").render()
        self.assertIn("Water boils", out)
        self.assertNotIn("Page:", out)
        self.assertNotIn("File:", out)

    def test_note_without_material_raises(self) -> None:
        with self.assertRaises(InsufficientTextError):
            generate_note([])
        with self.assertRaises(InsufficientTextError):
            generate_note([Segment(text="hi", course="B", filename="f.pdf", page=1)])

    def test_seeds_pick_from_the_best_sentences(self) -> None:
        """Different seeds vary the note, but never a low-value line."""
        picks = {generate_note(self._segments(), seed=s).text for s in range(25)}
        self.assertGreater(len(picks), 1, "notes should not be identical every day")
        good = {
            line
            for lines in self.PAGES.values()
            for line in lines
        }
        for text in picks:
            self.assertIn(text, good)


class CleanUpTest(unittest.TestCase):
    """Real course PDFs extract as layout debris; none of it may survive."""

    def _kept(self, *texts: str) -> list[str]:
        segs = [Segment(text=t, course="C", filename="f.pdf", page=1) for t in texts]
        return [s.text for s in _collect_sentences(segs)]

    def test_wingdings_bullets_split_into_separate_sentences(self) -> None:
        # a Symbol-font bullet extracts as a private-use codepoint
        kept = self._kept(
            "Ternary relationship \uf076Ternary (degree 3) \uf0a7 "
            "An association among three different entity types \uf0a7 E.g."
        )
        self.assertNotIn("�", " ".join(kept))
        self.assertTrue(
            any(t.startswith("An association among") for t in kept),
            f"the bullet should end the previous item, got {kept}",
        )

    def test_truncated_sentence_is_dropped(self) -> None:
        kept = self._kept("Definition: a table is in 1NF If \uf076There are no duplicated rows.")
        self.assertFalse(any(t.endswith("1NF If") for t in kept))

    def test_placeholder_and_code_lines_are_dropped(self) -> None:
        kept = self._kept(
            "The general format is something along the lines of: \uf076 "
            "CREATE TABLE <table-name> ( . . . . . .); \uf076The ..... is where columns go."
        )
        for text in kept:
            self.assertNotIn(".....", text)
            self.assertNotIn("<table-name>", text)

    def test_assignment_instructions_are_dropped(self) -> None:
        kept = self._kept(
            "Indiviual Assignment(5%) #Do all the questions below accordingly 1)."
        )
        self.assertEqual(kept, [])

    def test_purpose_clause_is_not_a_definition(self) -> None:
        """'The goal is to get exactly one liter' states a purpose, not a meaning."""
        self.assertIsNone(_find_definition("The goal is to get exactly one liter of water."))
        self.assertIsNotNone(
            _find_definition("A database is a collection of related data stored together.")
        )

    def test_demonstrative_subject_is_not_a_term(self) -> None:
        self.assertIsNone(_find_definition("This schema is in its 1NF since it has no repeating groups."))

    def test_glued_running_head_is_never_a_term(self) -> None:
        stats = TermStats.build([
            Segment(text="SearchStrategies Exercises SearchAlgorithms are central topics in AI."),
            Segment(text="Search strategies are central topics in AI search."),
            Segment(text="A search tree holds a state space for AI search."),
        ])
        pool = stats.candidates()
        self.assertTrue(pool)
        for term in pool:
            self.assertIsNone(re.search(r"[a-z][A-Z]", term), term)

    def test_stem_dedupes_morphological_variants(self) -> None:
        self.assertEqual(_stem("Algorithms"), _stem("Algorithm"))
        self.assertEqual(_stem("entity types"), _stem("entity type"))
        self.assertEqual(_stem("entities"), _stem("entity"))
        self.assertNotEqual(_stem("tree"), _stem("search"))

    def test_swap_word_does_not_corrupt_other_words(self) -> None:
        """"search" -> "goal" must not turn "searching" into "goaling"."""
        out = _swap_word("The searching process is like building the search tree.", "search", "goal")
        self.assertEqual(out, "The searching process is like building the goal tree.")
        self.assertIsNone(_swap_word("The searching process.", "search", "goal"))

    def test_adverbs_are_never_offered_as_terms(self) -> None:
        stats = TermStats.build([
            Segment(text="A function can be applied functionally to every input value."),
            Segment(text="The function maps each input to exactly one output value."),
            Segment(text="Every function has a domain and a range of input values."),
        ])
        for term in stats.candidates():
            self.assertNotEqual(term.lower(), "functionally")


class QuizShapeTest(unittest.TestCase):
    """The mix of question kinds a quiz is made of."""

    TEXT = "\n".join([
        "A search tree is a representation in which nodes denote paths and branches.",
        "The state space is a representation of every possible search problem state.",
        "A goal state is the state a search algorithm tries to reach.",
        "A solution quality is measured by the path cost of the search tree.",
        "Search algorithms are judged on completeness and optimality.",
        "Heuristic functions estimate the cost of reaching the goal state.",
        "The path cost of a search tree grows with the depth of the search.",
        "An informed search uses knowledge to order the state space.",
        "The periodic table lists 118 known chemical elements.",
        "Water molecules are made of two hydrogen atoms and one oxygen atom.",
    ])

    def test_true_false_stays_a_minority(self) -> None:
        """T/F is the easiest question to build, so it must not swamp a quiz."""
        questions = generate_questions(self.TEXT, count=9, seed=4)
        tf = [q for q in questions if q.kind == "tf"]
        self.assertTrue(questions)
        self.assertLessEqual(len(tf), (9 + 2) // 3)

    def test_every_question_keeps_its_source_sentence(self) -> None:
        """A blanked question is its source sentence with one span hidden."""
        def norm(text: str) -> str:
            text = " ".join(text.split()).rstrip(".").lower()
            return re.sub(r"^(the|a|an) ", "", text)

        for seed in range(6):
            for q in generate_questions(self.TEXT, count=8, seed=seed):
                self.assertTrue(q.explanation)
                if q.subtype not in {"term2def", "number"}:
                    continue
                source = q.explanation.split("\n\n")[0]
                answer = q.options[q.correct_index]
                rebuilt = q.text.replace("______", answer)
                self.assertEqual(
                    norm(rebuilt), norm(source),
                    f"dropping in {answer!r} must restore {source!r}, got {rebuilt!r}",
                )


if __name__ == "__main__":
    unittest.main()
