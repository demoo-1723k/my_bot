"""Who is allowed to run the bot, and where a quiz/note is posted.

The rules:
  * the group's owner and admins control the bot — courses, exams, schedules
    and posting — in the group and in a private chat alike;
  * everyone else can only talk to the bot in private: ask questions and get
    a quiz of their own, never posted into the group;
  * an admin who generates a quiz or a note in private is asked whether it
    should go to the group or to that chat.

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

from telegram.error import TelegramError

import bot
import storage
from generator import Answer
from tests.test_generator import _build_pdf

PROJECT_DATA = Path(__file__).resolve().parent.parent / "data"


COURSE_LINES = [
    "A search tree is a representation in which nodes denote paths.",
    "The frontier holds nodes discovered but not yet expanded.",
    "Uniform-cost search orders the frontier by total path cost.",
    "Heuristic functions estimate the cost of reaching the goal state.",
    "The state space holds every state reachable from the initial state.",
    "A goal state is the state a search algorithm tries to reach.",
    "Search algorithms are judged by completeness and optimality.",
    "The path cost of a search tree grows with its depth.",
    "An informed search uses knowledge to order the state space.",
    "A breadth first search expands the shallowest node first.",
]

class FakeChat:
    """A group or private chat; `status` is the caller's status in the group."""

    def __init__(self, type_: str = "group", chat_id: int = -1001, status: str = "administrator"):
        self.type = type_
        self.id = chat_id
        self.status = status

    async def get_member(self, user_id: int):
        if self.status == "none":
            raise TelegramError("user not found")
        return SimpleNamespace(status=self.status)


class FakeMessage:
    def __init__(self, chat: FakeChat, message_id: int = 2, text: str | None = None):
        self.chat = chat
        self.chat_id = chat.id
        self.message_id = message_id
        self.text = text
        self.replies: list[dict] = []

    async def reply_text(self, text, **kwargs):
        self.replies.append({"text": text, **kwargs})
        return self


class FakeQuery:
    def __init__(self, data: str, message: FakeMessage, bot_obj: "FakeBot"):
        self.data = data
        self.message = message
        self.bot = bot_obj
        self.answers: list[tuple] = []

    async def answer(self, text=None, show_alert=False):
        self.answers.append((text, show_alert))

    def get_bot(self):
        return self.bot


class FakeBot:
    def __init__(self, status: str = "administrator"):
        self.status = status
        self.sent: list[dict] = []
        self.edits: list[dict] = []

    async def get_chat_member(self, chat_id, user_id: int):
        if self.status == "none":
            raise TelegramError("user not found")
        return SimpleNamespace(status=self.status)

    async def send_message(self, **kwargs):
        self.sent.append({"kind": "message", **kwargs})
        return SimpleNamespace(message_id=len(self.sent))

    async def send_poll(self, **kwargs):
        self.sent.append({"kind": "poll", **kwargs})
        return SimpleNamespace(message_id=len(self.sent))

    async def edit_message_text(self, **kwargs):
        self.edits.append(kwargs)
        return SimpleNamespace(**kwargs)

    def polls(self) -> list[dict]:
        return [s for s in self.sent if s.get("kind") == "poll"]

    def notes(self) -> list[dict]:
        return [s for s in self.sent if s.get("kind") == "message"]


class FakeContext:
    def __init__(self, fake_bot: FakeBot):
        self.bot = fake_bot
        self.user_data: dict = {}
        self.args: list[str] = []


class BaseTest(unittest.TestCase):
    """Shared setup: a temp data dir, one course, a fresh permission cache."""

    STATUS = "administrator"
    CHAT_TYPE = "group"
    CHAT_ID = -1001
    USER_ID = 5

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        storage.set_data_dir(self.tmp.name)
        self.addCleanup(storage.set_data_dir, PROJECT_DATA)
        bot._clear_member_cache()
        self.addCleanup(bot._clear_member_cache)
        bot._drop_ask_index()
        self.addCleanup(bot._drop_ask_index)

        pdf = Path(self.tmp.name) / "notes.pdf"
        pdf.write_bytes(_build_pdf(COURSE_LINES))
        course = storage.create_course("AI")
        storage.add_pdf(course.id, pdf, "notes.pdf")
        storage.finalize_course(course.id)

        self.bot = FakeBot(self.STATUS)
        self.context = FakeContext(self.bot)

    def update(self, *, data: str | None = None, text: str | None = None):
        chat = FakeChat(self.CHAT_TYPE, self.CHAT_ID, self.STATUS)
        message = FakeMessage(chat, text=text)
        return SimpleNamespace(
            effective_chat=chat,
            effective_user=SimpleNamespace(id=self.USER_ID),
            effective_message=message,
            callback_query=FakeQuery(data, message, self.bot) if data else None,
        )

    def press(self, data: str) -> dict:
        update = self.update(data=data)
        asyncio.run(bot.on_button(update, self.context))
        return self.bot.edits[-1] if self.bot.edits else {}

    def type(self, text: str) -> FakeMessage:
        update = self.update(text=text)
        message = update.effective_message
        asyncio.run(bot.on_text(update, self.context))
        return message

    @staticmethod
    def labels(edit: dict) -> list[str]:
        markup = edit.get("reply_markup") or {}
        return [b.text for row in markup.inline_keyboard for b in row]

    @staticmethod
    def data_of(edit: dict) -> list[str]:
        markup = edit.get("reply_markup") or {}
        return [b.callback_data for row in markup.inline_keyboard for b in row]


class MemberStatusTest(BaseTest):
    """The owner and admins control the bot; plain members do not."""

    def _admin(self, status: str) -> bool:
        bot._clear_member_cache()
        update = SimpleNamespace(
            effective_chat=FakeChat("group", -1001, status),
            effective_user=SimpleNamespace(id=9),
        )
        return asyncio.run(bot._is_admin(update, self.context))

    def test_owner_and_admins_are_allowed(self) -> None:
        self.assertTrue(self._admin("creator"))
        self.assertTrue(self._admin("administrator"))

    def test_a_plain_member_is_not(self) -> None:
        self.assertFalse(self._admin("member"))
        self.assertFalse(self._admin("restricted"))

    def test_someone_not_in_the_group_is_not(self) -> None:
        self.assertFalse(self._admin("none"))

    def test_private_chat_checks_the_group_membership(self) -> None:
        for status, expected in [
            ("administrator", True),
            ("creator", True),
            ("member", False),
            ("none", False),
        ]:
            with self.subTest(status=status):
                bot._clear_member_cache()
                self.bot.status = status
                update = SimpleNamespace(
                    effective_chat=FakeChat("private", 4242, "member"),
                    effective_user=SimpleNamespace(id=9),
                )
                self.assertEqual(asyncio.run(bot._is_admin(update, self.context)), expected)

    def test_an_unknown_answer_does_not_lock_anyone_out(self) -> None:
        """No chat attached (or no API) must not be read as 'not allowed'."""
        update = SimpleNamespace()
        self.assertTrue(asyncio.run(bot._is_admin(update, self.context)))

    def test_the_answer_is_cached(self) -> None:
        bot._clear_member_cache()
        update = SimpleNamespace(
            effective_chat=FakeChat("group", -1001, "administrator"),
            effective_user=SimpleNamespace(id=11),
        )
        asyncio.run(bot._is_admin(update, self.context))
        self.assertIn(11, bot._MEMBER_CACHE)
        bot._MEMBER_CACHE[11] = (bot._MEMBER_CACHE[11][0], "member")
        # the cached value wins, so the API is not called again
        self.assertFalse(asyncio.run(bot._is_admin(update, self.context)))


class GroupGateTest(BaseTest):
    """In the group only the owner and admins may use the bot."""

    STATUS = "member"

    def test_a_member_cannot_press_the_menu(self) -> None:
        update = self.update(data="menu:add")
        asyncio.run(bot.on_button(update, self.context))
        self.assertEqual(self.bot.edits, [])                       # nothing opened
        self.assertEqual(update.callback_query.answers[-1][1], True)  # alert
        self.assertIn("owner and admins", update.callback_query.answers[-1][0])

    def test_an_admin_gets_the_menu(self) -> None:
        self.STATUS = "administrator"
        self.bot.status = "administrator"
        edit = self.press("menu:aq")
        self.assertIn("Auto Quiz", edit["text"])

    def test_a_member_is_pointed_at_a_private_chat_on_start(self) -> None:
        update = self.update(text="/start")
        asyncio.run(bot.cmd_start(update, self.context))
        reply = update.effective_message.replies[-1]
        self.assertIn("owner and admins", reply["text"])
        self.assertNotIn("reply_markup", reply)                   # no menu for them

    def test_a_member_is_told_off_when_they_call_the_bot(self) -> None:
        message = self.type("baymax what is a weak entity")
        self.assertTrue(message.replies)
        self.assertIn("owner and admins", message.replies[-1]["text"])

    def test_ordinary_group_chatter_stays_silent(self) -> None:
        message = self.type("anyone got the notes?")
        self.assertEqual(message.replies, [])

    def test_a_member_cannot_ask_with_the_command(self) -> None:
        update = self.update(text="/ask what is 2NF")
        update.effective_message.text = "/ask what is 2NF"
        asyncio.run(bot.cmd_ask(update, self.context))
        self.assertIn("owner and admins", update.effective_message.replies[-1]["text"])

    def test_a_member_cannot_move_the_topic(self) -> None:
        update = self.update(text="/topic")
        asyncio.run(bot.cmd_topic(update, self.context))
        self.assertIn("admins only", update.effective_message.replies[-1]["text"])

    def test_a_member_cannot_upload_material(self) -> None:
        update = self.update(text=None)
        update.effective_message.document = SimpleNamespace(
            mime_type="application/pdf", file_name="x.pdf"
        )
        asyncio.run(bot.on_document(update, self.context))
        self.assertEqual(update.effective_message.replies, [])     # silent in the group


class StudentPrivateTest(BaseTest):
    """A member in a private chat: questions and their own quiz."""

    STATUS = "member"
    CHAT_TYPE = "private"
    CHAT_ID = 4242

    def test_start_offers_the_student_menu(self) -> None:
        update = self.update(text="/start")
        asyncio.run(bot.cmd_start(update, self.context))
        reply = update.effective_message.replies[-1]
        self.assertIn("Ask me anything", reply["text"])
        labels = [b.text for row in reply["reply_markup"].inline_keyboard for b in row]
        self.assertIn("🎯 Send me a quiz", labels)
        self.assertNotIn("Add Course", labels)                    # no admin controls

    def test_admin_in_private_gets_the_full_menu(self) -> None:
        self.STATUS = "administrator"
        self.bot.status = "administrator"
        update = self.update(text="/start")
        asyncio.run(bot.cmd_start(update, self.context))
        reply = update.effective_message.replies[-1]
        labels = [b.text for row in reply["reply_markup"].inline_keyboard for b in row]
        self.assertIn("Add Course", labels)

    def test_a_student_cannot_press_an_admin_button(self) -> None:
        update = self.update(data="menu:add")
        asyncio.run(bot.on_button(update, self.context))
        self.assertEqual(self.bot.edits, [])
        self.assertIn("admins", update.callback_query.answers[-1][0])

    def test_a_student_is_stopped_from_holding_an_admin_state(self) -> None:
        self.context.user_data["state"] = bot.ADD_NAME
        message = self.type("Database")
        self.assertIn("admins only", message.replies[-1]["text"])
        self.assertNotIn("state", self.context.user_data)         # it was cleared

    def test_their_quiz_lands_in_their_own_chat(self) -> None:
        self.press("menu:squiz")
        self.assertIn("How many questions", self.bot.edits[-1]["text"])
        self.type("2")

        polls = self.bot.polls()
        self.assertTrue(polls)
        for poll in polls:
            self.assertEqual(poll["chat_id"], self.CHAT_ID)       # not the group
            self.assertNotIn("message_thread_id", poll)

    def test_the_answer_offers_a_quiz(self) -> None:
        answer = Answer(query="what is a search tree", text="A search tree is x.",
                        course="AI", filename="notes.pdf", page=1)
        update = self.update(text="what is a search tree")
        with patch.object(bot, "_ask_indexes", return_value=[("AI", object())]), \
                patch.object(bot, "_answer", return_value=answer):
            asyncio.run(bot._ask("what is a search tree", update.effective_message))
        reply = update.effective_message.replies[-1]
        labels = [b.text for row in reply["reply_markup"].inline_keyboard for b in row]
        self.assertIn("🎯 Send me a quiz", labels)


class AdminInGroupTest(BaseTest):
    """In the group the destination is obvious, so nothing is asked."""

    def test_an_answer_in_the_group_carries_no_quiz_button(self) -> None:
        answer = Answer(query="q", text="A search tree is x.")
        update = self.update(text="q")
        with patch.object(bot, "_ask_indexes", return_value=[("AI", object())]), \
                patch.object(bot, "_answer", return_value=answer):
            asyncio.run(bot._ask("q", update.effective_message))
        self.assertIsNone(update.effective_message.replies[-1].get("reply_markup"))

    def test_quiz_goes_straight_to_the_count(self) -> None:
        edit = self.press("menu:gen")
        self.assertIn("How many questions", edit["text"])

    def test_note_is_posted_without_asking(self) -> None:
        self.press("menu:genote")
        notes = self.bot.notes()
        self.assertTrue(notes)
        self.assertEqual(notes[-1]["chat_id"], bot.QUIZ_CHAT_ID)
        self.assertIn("Posted a note", self.bot.edits[-1]["text"])


class AdminDestinationTest(BaseTest):
    """An admin in a private chat is asked where the quiz/note should go."""

    CHAT_TYPE = "private"
    CHAT_ID = 4242

    def test_quiz_asks_where_to_post(self) -> None:
        edit = self.press("menu:gen")
        self.assertIn("Where should I post the quiz?", edit["text"])
        self.assertEqual(
            self.data_of(edit), ["dest:group", "dest:private", "menu:back"]
        )

    def test_choosing_the_group_posts_to_the_group(self) -> None:
        self.press("menu:gen")
        self.press("dest:group")
        self.type("2")
        polls = self.bot.polls()
        self.assertTrue(polls)
        for poll in polls:
            self.assertEqual(poll["chat_id"], bot.QUIZ_CHAT_ID)

    def test_choosing_this_chat_posts_here(self) -> None:
        self.press("menu:gen")
        self.press("dest:private")
        self.type("2")
        polls = self.bot.polls()
        self.assertTrue(polls)
        for poll in polls:
            self.assertEqual(poll["chat_id"], self.CHAT_ID)
            self.assertNotIn("message_thread_id", poll)

    def test_note_asks_where_to_post(self) -> None:
        edit = self.press("menu:genote")
        self.assertIn("Where should I post the note?", edit["text"])
        self.assertEqual(self.data_of(edit)[0], "dest:group")

    def test_a_note_for_this_chat_goes_here(self) -> None:
        self.press("menu:genote")
        self.press("dest:private")
        notes = self.bot.notes()
        self.assertTrue(notes)
        self.assertEqual(notes[-1]["chat_id"], self.CHAT_ID)
        self.assertNotIn("message_thread_id", notes[-1])
        self.assertIn("this chat", self.bot.edits[-1]["text"])

    def test_a_note_for_the_group_goes_to_the_group(self) -> None:
        self.press("menu:genote")
        self.press("dest:group")
        notes = self.bot.notes()
        self.assertTrue(notes)
        self.assertEqual(notes[-1]["chat_id"], bot.QUIZ_CHAT_ID)

    def test_the_choice_is_forgotten_next_time(self) -> None:
        self.press("menu:gen")
        self.press("dest:private")
        self.assertEqual(self.context.user_data.get("dest_chat"), self.CHAT_ID)
        edit = self.press("menu:back")
        self.assertNotIn("dest_chat", self.context.user_data)
        self.assertIn("Generate Quiz", self.labels(edit))          # admin menu again

    def test_a_student_is_not_asked_where(self) -> None:
        """They only ever get their own quiz, so there is nothing to decide."""
        self.STATUS = "member"
        self.bot.status = "member"
        edit = self.press("menu:squiz")
        self.assertIn("How many questions", edit["text"])
        self.assertNotIn("dest:group", self.data_of(edit))


if __name__ == "__main__":
    unittest.main()
