"""Tests for the daily automations: time input, scheduling and note rotation.

Run from the project root:
    python -m unittest -v
"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import bot
import storage
from bot import _first_run_date, _is_due, _parse_time, _pick_note
from generator import Segment
from tests.test_generator import _build_pdf

PROJECT_DATA = Path(__file__).resolve().parent.parent / "data"

BIO_LINES = [
    "Photosynthesis is the process by which plants convert light energy into chemical energy.",
    "The mitochondria is the powerhouse of the cell and produces ATP for the cell.",
    "Osmosis is the movement of water across a semipermeable membrane in plants.",
]

CHEM_LINES = [
    "Xenon is a noble gas with atomic number 54 on the periodic table.",
    "The Haber process combines nitrogen and hydrogen to make ammonia gas.",
    "Titration is a laboratory technique to determine the concentration of a solution.",
]


class ParseTimeTest(unittest.TestCase):
    """Custom times are only accepted in 24-hour format."""

    def test_accepts_24_hour_format(self) -> None:
        self.assertEqual(_parse_time("08:30"), "08:30")
        self.assertEqual(_parse_time("19:05"), "19:05")
        self.assertEqual(_parse_time("23:59"), "23:59")
        self.assertEqual(_parse_time("0:00"), "00:00")
        self.assertEqual(_parse_time("7:30"), "07:30")  # zero-padded on save
        self.assertEqual(_parse_time("  08:30  "), "08:30")

    def test_rejects_anything_else(self) -> None:
        for bad in [
            "24:00", "25:00", "12:60", "8", "8.30", "0730", "12:5",
            "half past eight", "", "   ", "-1:00", "7:300", "12:00:00",
            "8:30 pm", "noon", "08:3o",
        ]:
            with self.subTest(value=bad):
                self.assertIsNone(_parse_time(bad))


class ScheduleTest(unittest.TestCase):
    """When a saved schedule should fire."""

    def _now(self, hour: int, minute: int = 0) -> datetime:
        return datetime(2026, 10, 3, hour, minute)

    def _quiz(self, **overrides) -> dict:
        return {"enabled": True, "time": "08:00", "last_run": None, **overrides}

    def test_not_due_before_the_time(self) -> None:
        self.assertFalse(_is_due(self._quiz(), self._now(7, 59)))

    def test_due_at_and_after_the_time(self) -> None:
        self.assertTrue(_is_due(self._quiz(), self._now(8, 0)))
        # later than the time: a restart after 08:00 still sends today's quiz
        self.assertTrue(_is_due(self._quiz(), self._now(9, 30)))

    def test_only_once_a_day(self) -> None:
        ran_today = self._quiz(last_run="2026-10-03")
        self.assertFalse(_is_due(ran_today, self._now(8, 5)))
        self.assertFalse(_is_due(ran_today, self._now(23, 59)))

    def test_runs_again_the_next_day(self) -> None:
        ran_yesterday = self._quiz(last_run="2026-10-02")
        self.assertTrue(_is_due(ran_yesterday, self._now(8, 0)))

    def test_disabled_or_untimed_never_fires(self) -> None:
        self.assertFalse(_is_due(self._quiz(enabled=False), self._now(9, 0)))
        self.assertFalse(_is_due(self._quiz(time=None), self._now(9, 0)))

    def test_new_schedule_starts_tomorrow_when_the_time_has_passed(self) -> None:
        # picked at 20:00 for 08:00 — 08:00 today is gone, so it waits a day
        self.assertEqual(_first_run_date("08:00", self._now(20, 0)), "2026-10-03")

    def test_new_schedule_still_runs_today_when_the_time_is_ahead(self) -> None:
        # picked at 07:00 for 08:00 — keep the slot open for today
        self.assertIsNone(_first_run_date("08:00", self._now(7, 0)))


class PickNoteTest(unittest.TestCase):
    """The daily note rotates courses — never the same one two days running."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        storage.set_data_dir(self.tmp.name)
        self.addCleanup(storage.set_data_dir, PROJECT_DATA)

        self.bio = storage.create_course("Biology")
        self.chem = storage.create_course("Chemistry")

        def fake_segments(course_id: int) -> list[Segment]:
            if course_id == self.bio.id:
                return [
                    Segment(
                        text="\n".join(BIO_LINES),
                        course="Biology",
                        filename="ch1.pdf",
                        page=2,
                    )
                ]
            return [
                Segment(
                    text="\n".join(CHEM_LINES),
                    course="Chemistry",
                    filename="ch1.pdf",
                    page=1,
                )
            ]

        patcher = patch.object(storage, "course_segments", side_effect=fake_segments)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_skips_the_course_used_last(self) -> None:
        storage.set_note_last_course(self.bio.id)
        course, note = _pick_note(exclude_course_id=storage.get_note_last_course())
        self.assertEqual(course.id, self.chem.id)
        self.assertEqual(note.course, "Chemistry")
        self.assertEqual(note.page, 1)

    def test_any_course_when_none_was_used(self) -> None:
        course, note = _pick_note()
        self.assertIn(course.id, (self.bio.id, self.chem.id))
        self.assertTrue(note.text.endswith((".", "!", "?")))

    def test_single_course_is_used_even_if_it_repeats(self) -> None:
        storage.delete_course(self.chem.id)
        storage.set_note_last_course(self.bio.id)
        course, note = _pick_note(exclude_course_id=self.bio.id)
        self.assertEqual(course.id, self.bio.id)
        self.assertTrue(note.text)

    def test_no_courses_gives_no_note(self) -> None:
        storage.delete_course(self.bio.id)
        storage.delete_course(self.chem.id)
        self.assertEqual(_pick_note(), (None, None))

    def test_unreadable_course_is_skipped(self) -> None:
        with patch.object(
            storage, "course_segments", side_effect=[RuntimeError("boom")]
        ):
            self.assertEqual(_pick_note(), (None, None))


# --- fakes for driving the menu ------------------------------------------


class FakeBot:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.edits: list[dict] = []

    async def send_message(self, **kwargs):
        self.sent.append(kwargs)

    async def send_poll(self, **kwargs):
        self.sent.append(kwargs)

    async def edit_message_text(self, **kwargs):
        self.edits.append(kwargs)
        return SimpleNamespace(**kwargs)


class FakeMessage:
    def __init__(self, chat_id: int = 1, message_id: int = 2, text: str | None = None):
        self.chat_id = chat_id
        self.message_id = message_id
        self.text = text
        self.replies: list[dict] = []

    async def reply_text(self, text, **kwargs):
        self.replies.append({"text": text, **kwargs})
        return self


class FakeQuery:
    def __init__(self, data: str, message: FakeMessage, fake_bot: FakeBot):
        self.data = data
        self.message = message
        self.answers: list[tuple] = []
        self._bot = fake_bot

    async def answer(self, text=None, show_alert=False):
        self.answers.append((text, show_alert))

    def get_bot(self):
        return self._bot


class FakeContext:
    def __init__(self, fake_bot: FakeBot):
        self.bot = fake_bot
        self.user_data: dict = {}


class AutomationFlowTest(unittest.TestCase):
    """Drive the menu itself: status screens, time picking and turning off."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        storage.set_data_dir(self.tmp.name)
        self.addCleanup(storage.set_data_dir, PROJECT_DATA)

        course = storage.create_course("Biology")
        pdf = Path(self.tmp.name) / "ch1.pdf"
        pdf.write_bytes(_build_pdf(BIO_LINES))
        storage.add_pdf(course.id, pdf, "ch1.pdf")
        storage.finalize_course(course.id)

        self.bot = FakeBot()
        self.context = FakeContext(self.bot)
        self.message = FakeMessage()

    # -- helpers --

    def _press(self, data: str) -> dict:
        query = FakeQuery(data, self.message, self.bot)
        asyncio.run(bot.on_button(SimpleNamespace(callback_query=query), self.context))
        return self.bot.edits[-1] if self.bot.edits else {}

    def _type(self, text: str) -> None:
        message = FakeMessage(text=text)
        asyncio.run(bot.on_text(SimpleNamespace(effective_message=message), self.context))

    @staticmethod
    def _labels(edit: dict) -> list[str]:
        markup = edit.get("reply_markup") or {}
        return [b.text for row in markup.inline_keyboard for b in row]

    @staticmethod
    def _data(edit: dict) -> list[str]:
        markup = edit.get("reply_markup") or {}
        return [b.callback_data for row in markup.inline_keyboard for b in row]

    # -- auto quiz --

    def test_auto_quiz_screen_starts_off(self) -> None:
        edit = self._press("menu:aq")
        self.assertIn("Status: ⭕ OFF", edit["text"])
        self.assertIn("08:00", self._labels(edit))       # 24-hour presets
        self.assertIn("⌨️ Custom time", self._labels(edit))
        self.assertNotIn("❌ Turn off", self._labels(edit))  # nothing to turn off

    def test_auto_quiz_set_up_time_then_count(self) -> None:
        self._press("menu:aq")
        edit = self._press("aq:time:08:00")
        self.assertIn("How many questions", edit["text"])
        # not live yet — the quantity is still missing
        self.assertFalse(storage.get_auto_quiz()["enabled"])

        edit = self._press("aq:count:15")
        self.assertIn("Auto Quiz is ON", edit["text"])
        settings = storage.get_auto_quiz()
        self.assertTrue(settings["enabled"])
        self.assertEqual(settings["time"], "08:00")
        self.assertEqual(settings["count"], 15)

    def test_auto_quiz_screen_shows_the_running_schedule(self) -> None:
        self._press("menu:aq")
        self._press("aq:time:08:00")
        self._press("aq:count:15")

        edit = self._press("menu:aq")
        self.assertIn("Status: ✅ ON — every day at 08:00, 15 questions", edit["text"])
        self.assertIn("❌ Turn off", self._labels(edit))

    def test_auto_quiz_turn_off(self) -> None:
        self._press("menu:aq")
        self._press("aq:time:08:00")
        self._press("aq:count:15")

        edit = self._press("aq:off")
        self.assertIn("turned off", edit["text"])
        self.assertIn("Status: ⭕ OFF", edit["text"])
        self.assertFalse(storage.get_auto_quiz()["enabled"])
        self.assertEqual(storage.get_auto_quiz()["time"], "08:00")  # kept for later

    def test_auto_quiz_custom_time_typed(self) -> None:
        self._press("menu:aq")
        self._press("aq:custom")

        self._type("18:45")  # 24-hour format is accepted
        self.assertEqual(self.context.user_data.get("pending_time"), "18:45")

        self._type("20")      # quantity can be typed too
        settings = storage.get_auto_quiz()
        self.assertTrue(settings["enabled"])
        self.assertEqual(settings["time"], "18:45")
        self.assertEqual(settings["count"], 20)

    def test_custom_time_must_be_24_hour(self) -> None:
        self._press("menu:aq")
        self._press("aq:custom")

        self._type("25:00")
        self.assertIsNone(self.context.user_data.get("pending_time"))
        self.assertFalse(storage.get_auto_quiz()["enabled"])

    def test_custom_time_prompt_keeps_the_flow_open(self) -> None:
        self._press("menu:aq")
        self._press("aq:custom")
        self._type("8 pm")  # rejected — the prompt stays on screen
        edit = self._press("aq:time:08:00")  # a preset still works afterwards
        self.assertIn("How many questions", edit["text"])

    # -- auto note --

    def test_auto_note_set_up_and_turn_off(self) -> None:
        edit = self._press("menu:an")
        self.assertIn("Status: ⭕ OFF", edit["text"])

        edit = self._press("an:time:17:00")
        self.assertIn("Auto Note is ON", edit["text"])
        # no quantity is asked for a note — it goes straight back to the menu
        self.assertIn("menu:add", self._data(edit))
        self.assertTrue(storage.get_auto_note()["enabled"])
        self.assertEqual(storage.get_auto_note()["time"], "17:00")

        self._press("menu:an")
        edit = self._press("an:off")
        self.assertIn("turned off", edit["text"])
        self.assertFalse(storage.get_auto_note()["enabled"])

    def test_auto_note_custom_time_typed(self) -> None:
        self._press("menu:an")
        self._press("an:custom")
        self._type("21:05")
        settings = storage.get_auto_note()
        self.assertTrue(settings["enabled"])
        self.assertEqual(settings["time"], "21:05")

    # -- generate note --

    def test_generate_note_posts_the_note_to_the_group(self) -> None:
        self._press("menu:genote")

        posted = [s for s in self.bot.sent if "text" in s]
        self.assertEqual(len(posted), 1)
        self.assertEqual(posted[0]["chat_id"], bot.QUIZ_CHAT_ID)
        self.assertEqual(posted[0].get("parse_mode"), "HTML")
        self.assertIn("📚 Course: <b>Biology</b>", posted[0]["text"])
        self.assertIn("📄 File: ch1.pdf", posted[0]["text"])
        self.assertIn("📃 Page: <b>1</b>", posted[0]["text"])

        # the button message only confirms the post
        edit = self.bot.edits[-1]
        self.assertIn("Posted a note from Biology", edit["text"])
        self.assertIn("menu:add", self._data(edit))  # back to the menu


if __name__ == "__main__":
    unittest.main()
