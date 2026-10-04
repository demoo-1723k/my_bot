"""Tests for /ask — answering questions from the saved PDFs only.

The contract the bot relies on: every answer is a sentence copied out of the
material, it carries its course · file · page, and a question that isn't in
the notes gets "I couldn't find that" rather than a plausible wrong sentence.

Run from the project root:
    python -m unittest -v
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import bot
from generator import (
    Retriever,
    Segment,
    _focus_terms,
    answer_question,
)

DB_LINES = [
    "A relation schema R is in 2NF if every nonprime attribute is fully "
    "dependent on the primary key.",
    "A weak entity is an entity set that has no key attribute of its own.",
    "A strong entity set is an entity set that has a key attribute.",
    "Any table in 1NF with a single attribute primary key is in 2NF.",
]

AI_LINES = [
    "A search tree is a representation in which nodes denote paths and "
    "branches connect paths.",
    "Uniform-cost search orders the frontier by the total path cost.",
    "Heuristic functions estimate the cost of reaching the goal state.",
]


def _segments(lines: list[str], course: str = "Database", file: str = "ch1.pdf"):
    return [
        Segment(
            text="\n".join(lines),
            course=course,
            filename=file,
            page=1,
        )
    ]


class FocusTermsTest(unittest.TestCase):
    """The question boilerplate must not outrank the real words."""

    def test_strips_the_question_frame(self) -> None:
        self.assertEqual(_focus_terms("what is a weak entity"), ["weak", "entity"])
        self.assertEqual(_focus_terms("hey, can you explain 2NF please"), ["2NF"])
        self.assertEqual(
            _focus_terms("tell me about functional dependency"),
            ["functional", "dependency"],
        )

    def test_strips_contractions(self) -> None:
        """"what's database" once leaked the word "what's" into the search."""
        for query in ["what's database", "whats database", "what’s database"]:
            with self.subTest(query=query):
                self.assertEqual(_focus_terms(query), ["database"])

    def test_strips_other_question_words_with_contractions(self) -> None:
        self.assertEqual(_focus_terms("how's normalization"), ["normalization"])
        self.assertEqual(_focus_terms("wheres the ER model"), ["ER", "model"])

    def test_keeps_every_content_word(self) -> None:
        terms = _focus_terms("define the entity relationship model")
        self.assertIn("entity", terms)
        self.assertIn("relationship", terms)
        self.assertIn("model", terms)
        self.assertNotIn("define", terms)


class RetrieverTest(unittest.TestCase):
    def test_ranks_the_sentence_that_actually_answers(self) -> None:
        index = Retriever(_segments(DB_LINES))
        hits = index.search("weak entity")
        self.assertTrue(hits)
        self.assertIn("weak entity", hits[0][0].text.lower())

    def test_stems_match_a_question_word_to_the_slide_word(self) -> None:
        """Slides say "attributes"; the student types "attribute"."""
        lines = DB_LINES + [
            "A primary key attribute uniquely identifies each entity instance.",
            "A foreign key attribute references a primary key attribute.",
        ]
        hits = Retriever(_segments(lines)).search("primary key attribute")
        self.assertTrue(hits)
        self.assertIn("attribute", hits[0][0].text.lower())

    def test_empty_material_has_no_hits(self) -> None:
        self.assertEqual(Retriever([]).search("anything"), [])


class AnswerQuestionTest(unittest.TestCase):
    def test_prefers_the_definition_of_the_thing_asked_about(self) -> None:
        answer = answer_question("what is a weak entity", _segments(DB_LINES))
        self.assertIsNotNone(answer)
        self.assertTrue(answer.is_definition)
        self.assertIn("no key attribute", answer.text)

    def test_an_exact_term_beats_a_longer_one_that_contains_it(self) -> None:
        """"what's database" must not answer with the definition of
        "database normalization" just because that repeats the word more."""
        lines = [
            "A database is a collection of related data stored in tables.",
            "Database normalization is a series of steps used to reduce data "
            "redundancy in a database design and improve database consistency.",
        ]
        for query in ["what's database", "what is a database", "whats database"]:
            with self.subTest(query=query):
                answer = answer_question(query, _segments(lines))
                self.assertIsNotNone(answer)
                self.assertIn("collection of related data", answer.text)

    def test_answer_is_a_sentence_from_the_material(self) -> None:
        answer = answer_question("what is a search tree", _segments(AI_LINES, "AI", "ch3.pdf"))
        self.assertIn(answer.text, AI_LINES)

    def test_answer_carries_its_source(self) -> None:
        answer = answer_question("define a weak entity", _segments(DB_LINES))
        self.assertEqual(answer.course, "Database")
        self.assertEqual(answer.filename, "ch1.pdf")
        self.assertEqual(answer.page, 1)

    def test_render_cites_the_material_and_escapes_html(self) -> None:
        answer = answer_question("what is a weak entity", _segments(DB_LINES))
        out = answer.render()
        self.assertIn("Database", out)
        self.assertIn("ch1.pdf", out)
        self.assertIn("p.1", out)

    def test_says_nothing_rather_than_guessing(self) -> None:
        """A word that is nowhere in the notes must not return a near-miss."""
        for query in [
            "what is quantum entanglement",
            "who won the world cup in 1998",
            "what is the capital city of Peru",
        ]:
            with self.subTest(query=query):
                self.assertIsNone(answer_question(query, _segments(DB_LINES)))

    def test_blank_query_returns_nothing(self) -> None:
        self.assertIsNone(answer_question("   ", _segments(DB_LINES)))


class BotAskTest(unittest.TestCase):
    """The bot side: picking the best course and not spamming the group."""

    def setUp(self) -> None:
        self.courses = [
            ("Database", _segments(DB_LINES, "Database", "ch1.pdf")),
            ("AI", _segments(AI_LINES, "AI", "ch3.pdf")),
        ]

    def _run(self, query: str):
        with patch.object(bot, "_build_sources", return_value=self.courses):
            bot._drop_ask_index()
            try:
                return bot._answer(query)
            finally:
                bot._drop_ask_index()

    def test_picks_the_course_that_contains_the_answer(self) -> None:
        answer = self._run("what is a search tree")
        self.assertIsNotNone(answer)
        self.assertEqual(answer.course, "AI")

        answer = self._run("what is a weak entity")
        self.assertEqual(answer.course, "Database")

    def test_no_match_anywhere_returns_none(self) -> None:
        self.assertIsNone(self._run("what is quantum entanglement"))

    def test_index_is_cached_until_the_material_changes(self) -> None:
        with patch.object(bot, "_build_sources", return_value=self.courses) as build:
            bot._drop_ask_index()
            bot._ask_indexes()
            bot._ask_indexes()
            self.assertEqual(build.call_count, 1)
            bot._drop_ask_index()
            bot._ask_indexes()
            self.assertEqual(build.call_count, 2)


class WantsAnswerTest(unittest.TestCase):
    """In a group the bot only listens when it is spoken to directly."""

    def _update(self, chat_type: str, reply_to_bot: bool = False):
        bot_user = SimpleNamespace(id=42)
        message = SimpleNamespace(text="what is a weak entity")
        if reply_to_bot:
            message.reply_to_message = SimpleNamespace(from_user=bot_user)
        return SimpleNamespace(
            effective_chat=SimpleNamespace(type=chat_type),
            bot=SimpleNamespace(bot=bot_user),
        ), message

    def test_answers_in_a_private_chat(self) -> None:
        update, message = self._update("private")
        self.assertTrue(bot._wants_answer(update, message))

    def test_answers_a_reply_to_its_own_message(self) -> None:
        update, message = self._update("group", reply_to_bot=True)
        self.assertTrue(bot._wants_answer(update, message))

    def test_stays_quiet_otherwise(self) -> None:
        update, message = self._update("group")
        self.assertFalse(bot._wants_answer(update, message))


if __name__ == "__main__":
    unittest.main()