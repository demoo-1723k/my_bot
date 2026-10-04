"""Tests for old exam papers, forum topics and the "baymax" wake word.

Run from the project root:
    python -m unittest -v
"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from telegram.error import BadRequest

import bot
import storage
from generator import (
    InsufficientTextError,
    Segment,
    _normalize_exam_text,
    exam_pairs,
    questions_from_exams,
)
from tests.test_generator import _build_pdf

MC_PAPER = """Database Systems Final Exam
1) Which normal form removes transitive dependency of non key attributes?
a) 1NF
b) 2NF
c) 3NF
Ans: c
2) Which command is used to remove a table from a database schema?
a) DELETE
b) DROP
c) TRUNCATE
d) ALTER
Ans: b
3) What is a candidate key in a relation schema?
Ans: A minimal superkey that uniquely identifies a tuple
"""

LONG_PAPER = """AI Chapter 3 Review
1) Which search strategy expands the node with the lowest estimated cost?
a) Depth first search
b) Breadth first search
c) Informed search
d) Uniform cost search
Ans: d
2) Which data structure stores the nodes discovered but not yet expanded?
a) The frontier
b) The goal state
c) The action space
d) The heuristic table
Ans: a
3) Which structure holds every state reachable from the initial state?
a) The frontier
b) The goal state
c) The state space
d) The problem graph
Ans: c
"""


def _segments(text: str, name: str = "Exam") -> list[Segment]:
    return [Segment(text=text, course=name, filename="paper.pdf", page=1)]


# The layout of the real "Computer Science Exit Exam 2018" paper: the answer
# marker has no separator ("Ans."), the option letter is repeated on its own
# line ("A." then "a. the text"), a bare question number starts each item, and
# the export leaves "Question 4Answer" and "2 | Page" artefacts behind.
REAL_PAPER = """4 Which automaton is powerful enough to recognize the language
L = {aⁿbⁿcⁿ | n ≥ 1}?
Question 4Answer
A. a.
DFA
B. b.
Linear Bounded Automaton (LBA)
C. c.
NFA
D. d.
Pushdown Automaton (PDA)
Ans. b. Linear Bounded Automaton (LBA)
5 In two-phase locking, during which phase may a transaction acquire locks but
not release any?
A. a.
Growing phase
B. b.
Commit phase
C. c.
Validation phase
D. d.
Shrinking phase
ans. a. Growing phase
6 To extend the connectivity of processor bus we use:
A. a.
SCSI
B. b.
PCI
C. c.
Controllers
D. d.
Multi bus
Ans. d. Multi bus
2 | Page
"""


class ExamParsingTest(unittest.TestCase):
    """Reading a past paper's questions and its own answer key."""

    def setUp(self) -> None:
        self.pairs = exam_pairs(_segments(MC_PAPER))
        self.by_question = {p.question: p for p in self.pairs}

    def test_reads_every_question_with_an_answer(self) -> None:
        self.assertEqual(len(self.pairs), 3)
        self.assertIn(
            "Which normal form removes transitive dependency of non key attributes?",
            self.by_question,
        )

    def test_resolves_a_multiple_choice_letter_to_the_option_text(self) -> None:
        """"Ans. c" with a) 1NF b) 2NF c) 3NF must answer "3NF", not "c"."""
        pair = self.by_question[
            "Which normal form removes transitive dependency of non key attributes?"
        ]
        self.assertEqual(pair.answer, "3NF")
        self.assertEqual(pair.options, ["1NF", "2NF", "3NF"])

    def test_resolves_the_second_letter_answer_too(self) -> None:
        pair = self.by_question[
            "Which command is used to remove a table from a database schema?"
        ]
        self.assertEqual(pair.answer, "DROP")
        self.assertEqual(pair.options, ["DELETE", "DROP", "TRUNCATE", "ALTER"])

    def test_keeps_a_written_answer_as_is(self) -> None:
        pair = self.by_question["What is a candidate key in a relation schema?"]
        self.assertEqual(
            pair.answer, "A minimal superkey that uniquely identifies a tuple"
        )

    def test_options_are_not_left_in_the_question(self) -> None:
        for pair in self.pairs:
            self.assertNotIn("a)", pair.question)
            self.assertNotIn("Ans", pair.question)

    def test_pairs_keep_their_page(self) -> None:
        for pair in self.pairs:
            self.assertEqual(pair.page, 1)
            self.assertEqual(pair.filename, "paper.pdf")

    def test_survives_a_paper_extracted_onto_one_line(self) -> None:
        """Some PDF extractors glue a whole page together; layout is rebuilt."""
        glued = MC_PAPER.replace("\n", " ")
        pairs = exam_pairs(_segments(glued))
        self.assertEqual(len(pairs), 3)
        self.assertIn(
            "Which normal form removes transitive dependency of non key attributes?",
            {p.question for p in pairs},
        )

    def test_strips_broken_font_markers(self) -> None:
        noisy = MC_PAPER.replace("\n", "(cid:12)")
        normalized = _normalize_exam_text(noisy)
        self.assertNotIn("cid:", normalized)

    def test_a_paper_without_an_answer_key_yields_nothing(self) -> None:
        """Better no exam question than one with an invented answer."""
        paper = (
            "1) Which normal form removes transitive dependency of attributes?\n"
            "a) 1NF\nb) 2NF\nc) 3NF\n"
            "2) Define a candidate key in a relational schema.\n"
        )
        self.assertEqual(exam_pairs(_segments(paper)), [])


class RealPaperLayoutTest(unittest.TestCase):
    """The "Ans." / "A. a." layout of the actual uploaded exit exam."""

    def setUp(self) -> None:
        self.pairs = exam_pairs(_segments(REAL_PAPER, "Exit Exam 2018"))
        self.by_question = {p.question: p for p in self.pairs}

    def test_finds_every_question(self) -> None:
        self.assertEqual(len(self.pairs), 3)

    def test_answer_marker_without_a_separator_is_read(self) -> None:
        """"Ans. b. Linear Bounded Automaton" — no colon after "Ans"."""
        pair = self.pairs[0]
        self.assertEqual(pair.answer, "Linear Bounded Automaton (LBA)")

    def test_the_repeated_option_label_is_not_left_in_the_option(self) -> None:
        pair = self.pairs[0]
        self.assertEqual(
            pair.options,
            ["DFA", "Linear Bounded Automaton (LBA)", "NFA", "Pushdown Automaton (PDA)"],
        )

    def test_lowercase_ans_is_read_too(self) -> None:
        self.assertEqual(self.pairs[1].answer, "Growing phase")

    def test_a_trailing_colon_is_turned_into_a_question(self) -> None:
        self.assertEqual(
            self.pairs[2].question,
            "To extend the connectivity of processor bus we use?",
        )
        self.assertEqual(self.pairs[2].answer, "Multi bus")

    def test_export_artefacts_are_dropped(self) -> None:
        for pair in self.pairs:
            self.assertNotIn("Question 4Answer", pair.question)
            self.assertNotIn("| Page", pair.question)
            self.assertFalse(any("|" in o for o in pair.options))

    def test_set_notation_in_a_question_survives(self) -> None:
        """The code-block stripper must not eat "{aⁿbⁿcⁿ | n ≥ 1}"."""
        self.assertIn("{aⁿbⁿcⁿ | n ≥ 1}", self.pairs[0].question)

    def test_these_become_poll_questions(self) -> None:
        import random

        questions = questions_from_exams(
            [("Exit Exam 2018", _segments(REAL_PAPER, "Exit Exam 2018"))],
            count=3,
            rng=random.Random(5),
        )
        self.assertEqual(len(questions), 3)
        for q in questions:
            self.assertEqual(q.kind, "mcq")
            self.assertEqual(q.subtype, "exam")
            self.assertGreaterEqual(len(q.options), 2)
            self.assertIn(q.correct_index, range(len(q.options)))


class ExamQuestionTest(unittest.TestCase):
    def test_uses_the_papers_own_options(self) -> None:
        import random

        questions = questions_from_exams(
            [("Exam", _segments(MC_PAPER))], count=3, rng=random.Random(1)
        )
        first = next(
            q for q in questions
            if q.text.startswith("Which normal form")
        )
        self.assertEqual(
            sorted(first.options), sorted(["1NF", "2NF", "3NF"])
        )
        self.assertEqual(first.options[first.correct_index], "3NF")
        self.assertEqual(first.subtype, "exam")

    def test_explanation_says_where_it_came_from(self) -> None:
        import random

        questions = questions_from_exams(
            [("Exam", _segments(LONG_PAPER))], count=3, rng=random.Random(2)
        )
        self.assertTrue(questions)
        for q in questions:
            self.assertIn("past exam", q.explanation or "")
            self.assertIn("Exam · paper.pdf · p.1", q.explanation or "")

    def test_options_stay_within_telegram_limits(self) -> None:
        import random

        for q in questions_from_exams(
            [("Exam", _segments(LONG_PAPER))], count=3, rng=random.Random(4)
        ):
            self.assertGreaterEqual(len(q.options), 2)
            self.assertLessEqual(len(q.options), 10)
            self.assertLessEqual(len(q.text), 300)
            self.assertEqual(
                len(q.options), len({o.lower() for o in q.options})
            )

    def test_no_exams_means_no_questions(self) -> None:
        self.assertEqual(questions_from_exams([], count=5), [])


class ExamStorageTest(unittest.TestCase):
    """Uploaded papers are stored apart from courses and can be removed."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        storage.set_data_dir(self.tmp.name)
        self.root = Path(self.tmp.name)
        self.pdf = self.root / "paper.pdf"
        self.pdf.write_bytes(_build_pdf([MC_PAPER]))

    def _upload(self, name: str = "DB Final") -> int:
        exam = storage.create_exam(name)
        storage.add_exam_pdf(exam.id, self.pdf, "paper.pdf")
        storage.finalize_exam(exam.id)
        return exam.id

    def test_create_list_delete(self) -> None:
        exam_id = self._upload()
        self.assertEqual([e.name for e in storage.list_exams()], ["DB Final"])
        storage.delete_exam(exam_id)
        self.assertEqual(storage.list_exams(), [])

    def test_deleting_removes_the_folder(self) -> None:
        exam_id = self._upload()
        folder = self.root / "exams" / str(exam_id)
        self.assertTrue(folder.exists())
        storage.delete_exam(exam_id)
        self.assertFalse(folder.exists())

    def test_segments_carry_page_numbers(self) -> None:
        exam_id = self._upload()
        segments = storage.exam_segments(exam_id)
        self.assertTrue(segments)
        self.assertEqual(segments[0].page, 1)
        self.assertEqual(segments[0].filename, "paper.pdf")

    def test_exams_do_not_appear_as_courses(self) -> None:
        self._upload()
        self.assertEqual(storage.list_courses(), [])

    def test_an_image_only_pdf_is_rejected(self) -> None:
        exam = storage.create_exam("Scanned")
        blank = self.root / "blank.pdf"
        blank.write_bytes(_build_pdf(["   "]))
        storage.add_exam_pdf(exam.id, blank, "blank.pdf")
        with self.assertRaises(InsufficientTextError):
            storage.finalize_exam(exam.id)

    def test_empty_name_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            storage.create_exam("   ")


class TopicTest(unittest.TestCase):
    """The forum topic the bot posts into."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        storage.set_data_dir(self.tmp.name)

    def test_learned_and_forgotten(self) -> None:
        self.assertIsNone(storage.get_topic("study room"))
        storage.save_topic("Study Room", 42)
        self.assertEqual(storage.get_topic("study room"), 42)
        storage.forget_topic("STUDY ROOM")
        self.assertIsNone(storage.get_topic("study room"))

    def test_a_broken_file_reads_as_no_topics(self) -> None:
        (Path(self.tmp.name) / "topics.json").write_text("{not json", "utf-8")
        self.assertEqual(storage.load_topics(), {})

    def test_thread_kwargs_adds_the_thread_only_when_known(self) -> None:
        with patch.object(bot, "QUIZ_TOPIC_ID", ""):
            storage.forget_topic(bot.QUIZ_TOPIC_NAME)
            self.assertEqual(bot._thread_kwargs(text="hi"), {"text": "hi"})
            storage.save_topic(bot.QUIZ_TOPIC_NAME, 77)
            self.assertEqual(bot._thread_kwargs(text="hi")["message_thread_id"], 77)

    def test_pinned_env_id_wins(self) -> None:
        storage.save_topic(bot.QUIZ_TOPIC_NAME, 77)
        with patch.object(bot, "QUIZ_TOPIC_ID", " 12 "):
            self.assertEqual(bot._thread_id(), 12)

    def test_a_broken_env_id_falls_back_to_the_learned_one(self) -> None:
        storage.save_topic(bot.QUIZ_TOPIC_NAME, 77)
        with patch.object(bot, "QUIZ_TOPIC_ID", "not-a-number"):
            self.assertEqual(bot._thread_id(), 77)


class StaleTopicRecoveryTest(unittest.TestCase):
    """A topic id Telegram rejects must not silence the bot forever.

    A wrong or deleted topic made every poll fail with "message thread not
    found". The send now falls back to General and forgets the bad id.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        storage.set_data_dir(self.tmp.name)
        self.saved: list[dict] = []

    def _bot(self, *, fail_thread: bool = True):
        bot_obj = SimpleNamespace()

        async def send_poll(**kwargs):
            self.saved.append(dict(kwargs))
            if fail_thread and "message_thread_id" in kwargs:
                raise BadRequest("Bad Request: message thread not found")
            return SimpleNamespace(message_id=len(self.saved))

        async def send_message(**kwargs):
            self.saved.append(dict(kwargs))
            if fail_thread and "message_thread_id" in kwargs:
                raise BadRequest("Bad Request: Message thread not found")
            return SimpleNamespace(message_id=len(self.saved))

        bot_obj.send_poll = send_poll
        bot_obj.send_message = send_message
        return bot_obj

    def _learn_bad_id(self) -> None:
        storage.save_topic(bot.QUIZ_TOPIC_NAME, 314)

    def _run(self, coro):
        return asyncio.run(coro)

    def test_recognises_the_error(self) -> None:
        self.assertTrue(
            bot._is_missing_thread(BadRequest("Bad Request: message thread not found"))
        )
        self.assertTrue(
            bot._is_missing_thread(BadRequest("message_thread_not_found"))
        )
        self.assertFalse(bot._is_missing_thread(BadRequest("chat not found")))

    def test_a_poll_falls_back_to_general(self) -> None:
        self._learn_bad_id()
        question = SimpleNamespace(
            text="Which term is described as X?", options=["a", "b"],
            correct_index=0, explanation="Because.", kind="mcq",
        )
        sent = self._run(bot._post_questions(self._bot(), [question]))
        self.assertEqual(sent, 1)
        self.assertEqual(len(self.saved), 2)
        self.assertEqual(self.saved[0]["message_thread_id"], 314)
        self.assertNotIn("message_thread_id", self.saved[1])

    def test_a_message_falls_back_to_general(self) -> None:
        self._learn_bad_id()
        ok = self._run(bot._send_message_safe(self._bot(), "hello"))
        self.assertTrue(ok)
        self.assertNotIn("message_thread_id", self.saved[-1])

    def test_the_bad_id_is_forgotten_so_it_cannot_repeat(self) -> None:
        self._learn_bad_id()
        self.assertEqual(bot._thread_id(), 314)
        self._run(bot._send_message_safe(self._bot(), "hello"))
        self.assertIsNone(storage.get_topic(bot.QUIZ_TOPIC_NAME))

    def test_later_sends_go_straight_to_general(self) -> None:
        self._learn_bad_id()
        self._run(bot._send_message_safe(self._bot(), "first"))
        self.saved.clear()
        self.assertTrue(self._run(bot._send_message_safe(self._bot(), "second")))
        self.assertNotIn("message_thread_id", self.saved[0])

    def test_a_good_topic_is_untouched(self) -> None:
        storage.save_topic(bot.QUIZ_TOPIC_NAME, 77)
        self.assertTrue(
            self._run(bot._send_message_safe(self._bot(fail_thread=False), "hi"))
        )
        self.assertEqual(storage.get_topic(bot.QUIZ_TOPIC_NAME), 77)
        self.assertEqual(self.saved[0]["message_thread_id"], 77)

    def test_other_bad_requests_do_not_drop_the_topic(self) -> None:
        """A genuine rejection (bad poll) must not clear a working topic."""
        storage.save_topic(bot.QUIZ_TOPIC_NAME, 77)

        async def send_message(**kwargs):
            raise BadRequest("Bad Request: poll question is too long")

        fake = SimpleNamespace(send_message=send_message, send_poll=None)
        self.assertFalse(self._run(bot._send_message_safe(fake, "hi")))
        self.assertEqual(storage.get_topic(bot.QUIZ_TOPIC_NAME), 77)


class WakeWordTest(unittest.TestCase):
    """"baymax what is 2NF" is a question; ordinary chatter is not."""

    def _update(self, chat_type: str = "group", bot_id: int = 7):
        return SimpleNamespace(
            effective_chat=SimpleNamespace(type=chat_type),
            bot=SimpleNamespace(bot=SimpleNamespace(id=bot_id)),
        )

    def _message(self, text: str, reply_from: int | None = None):
        message = SimpleNamespace(text=text)
        if reply_from is not None:
            message.reply_to_message = SimpleNamespace(
                from_user=SimpleNamespace(id=reply_from)
            )
        return message

    def test_recognises_the_wake_word(self) -> None:
        for word in ["baymax", "BayMax", "jarvis", "jarves", "bay max", "J.A.R.V.I.S."]:
            with self.subTest(word=word):
                query = bot._question_from_message(
                    self._update(), self._message(f"{word} what is a weak entity")
                )
                self.assertEqual(query, "what is a weak entity")

    def test_a_question_after_the_wake_word_keeps_its_question_frame(self) -> None:
        query = bot._question_from_message(
            self._update(), self._message("baymax what is 2nf please")
        )
        self.assertEqual(query, "what is 2nf")

    def test_drops_a_trailing_thanks(self) -> None:
        query = bot._question_from_message(
            self._update(), self._message("baymax explain 3nf thanks bot")
        )
        self.assertEqual(query, "explain 3nf")

    def test_ordinary_group_chatter_is_ignored(self) -> None:
        for text in [
            "anyone got the notes?",
            "see you at 5",
            "",
        ]:
            with self.subTest(text=text):
                self.assertEqual(
                    bot._question_from_message(self._update(), self._message(text)),
                    "",
                )

    def test_a_reply_to_the_bot_is_a_question(self) -> None:
        query = bot._question_from_message(
            self._update(), self._message("and 3NF?", reply_from=7)
        )
        self.assertEqual(query, "and 3NF?")

    def test_a_reply_to_someone_else_is_not(self) -> None:
        query = bot._question_from_message(
            self._update(), self._message("and 3NF?", reply_from=999)
        )
        self.assertEqual(query, "")

    def test_private_chat_always_answers(self) -> None:
        query = bot._question_from_message(
            self._update("private"), self._message("what is a weak entity")
        )
        self.assertEqual(query, "what is a weak entity")


class ExamQuizMixTest(unittest.TestCase):
    """Exams take a share of each quiz, not all of it."""

    COURSE_LINES = [
        "A search tree is a representation in which nodes denote paths.",
        "The frontier holds nodes discovered but not yet expanded.",
        "Uniform-cost search orders the frontier by total path cost.",
        "Heuristic functions estimate the cost of reaching the goal state.",
        "The state space holds every state reachable from the initial state.",
        "A goal state is the state a search algorithm tries to reach.",
        "Search algorithms are judged on completeness and optimality.",
        "The path cost of a search tree grows with its depth.",
        "An informed search uses knowledge to order the state space.",
        "A breadth first search expands the shallowest node first.",
    ]

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        storage.set_data_dir(self.tmp.name)
        bot._drop_ask_index()
        self.addCleanup(bot._drop_ask_index)

    def _add_exam(self) -> None:
        paper = Path(self.tmp.name) / "p.pdf"
        paper.write_bytes(_build_pdf([LONG_PAPER]))
        exam = storage.create_exam("AI Review")
        storage.add_exam_pdf(exam.id, paper, "p.pdf")
        storage.finalize_exam(exam.id)

    def _add_course(self) -> None:
        notes = Path(self.tmp.name) / "notes.pdf"
        notes.write_bytes(_build_pdf(self.COURSE_LINES))
        course = storage.create_course("AI")
        storage.add_pdf(course.id, notes, "notes.pdf")
        storage.finalize_course(course.id)

    def test_exam_questions_appear_in_the_quiz(self) -> None:
        self._add_exam()
        questions = bot._build_quiz(10, seed=1)
        self.assertTrue(any(q.subtype == "exam" for q in questions))

    def test_courses_and_exams_together(self) -> None:
        """The usual case: notes *and* papers, which exercises the full path."""
        self._add_course()
        self._add_exam()
        questions = bot._build_quiz(15, seed=1)
        self.assertTrue(questions)
        kinds = {q.subtype for q in questions}
        self.assertIn("exam", kinds)
        self.assertTrue(kinds - {"exam"}, "notes must still supply questions")

    def test_a_paper_alone_can_still_make_a_quiz(self) -> None:
        self._add_exam()
        self.assertTrue(bot._build_quiz(6, seed=1))

    def test_exams_stay_a_share_of_the_quiz(self) -> None:
        self._add_course()
        self._add_exam()
        questions = bot._build_quiz(20, seed=1)
        exam_qs = [q for q in questions if q.subtype == "exam"]
        self.assertLessEqual(len(exam_qs), round(20 * bot.EXAM_SHARE) + 1)

    def test_quiz_questions_respect_poll_limits(self) -> None:
        self._add_course()
        self._add_exam()
        for q in bot._build_quiz(20, seed=3):
            with self.subTest(question=q.text[:40]):
                self.assertLessEqual(len(q.text), 300)
                self.assertGreaterEqual(len(q.options), 2)
                self.assertIn(q.correct_index, range(len(q.options)))

    def test_no_exams_means_no_exam_questions(self) -> None:
        self._add_course()
        self.assertFalse(
            any(q.subtype == "exam" for q in bot._build_quiz(10, seed=1))
        )

    def test_nothing_at_all_raises(self) -> None:
        with self.assertRaises(InsufficientTextError):
            bot._build_quiz(5, seed=1)


if __name__ == "__main__":
    unittest.main()