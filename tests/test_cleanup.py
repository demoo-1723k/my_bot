"""Everything the bot posts belongs in the Study Room topic — and can be cleaned up.

  * a question or command asked in General (or another topic) is answered
    *inside* the Study Room, never where it was asked;
  * every message the bot sends to the group is written down with the topic it
    landed in, because Telegram offers no way to list a bot's own messages;
  * /cleanup deletes the ones that ended up outside the topic.

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
from generator import Answer
from tests.test_generator import _build_pdf
from tests.test_permissions import COURSE_LINES

PROJECT_DATA = Path(__file__).resolve().parent.parent / "data"
TOPIC = 4832
GROUP = bot.QUIZ_CHAT_ID


class SentBot:
    """Just enough Bot for the send/delete paths under test."""

    def __init__(self):
        self.sent: list[dict] = []
        self.polls: list[dict] = []
        self.edited: list[dict] = []
        self.deleted: list[tuple] = []
        self.deletable: set[int] | None = None   # None = everything works
        self.delete_error: Exception | None = None
        self.can_delete_any = False       # admin right to delete anyone's message

    async def send_message(self, **kwargs):
        self.sent.append(kwargs)
        return SimpleNamespace(
            chat_id=kwargs.get("chat_id"),
            message_id=5000 + len(self.sent),
            message_thread_id=kwargs.get("message_thread_id"),
        )

    async def send_poll(self, **kwargs):
        self.polls.append(kwargs)
        return SimpleNamespace(
            chat_id=kwargs.get("chat_id"),
            message_id=7000 + len(self.polls),
            message_thread_id=kwargs.get("message_thread_id"),
        )

    async def delete_message(self, chat_id, message_id):
        if self.delete_error is not None:
            raise self.delete_error
        if self.deletable is not None and message_id not in self.deletable:
            # what Telegram says for an id that is not the bot's own message
            raise BadRequest("Bad Request: message to delete not found")
        self.deleted.append((chat_id, message_id))
        return True

    async def get_chat(self, chat_id):
        return SimpleNamespace(type="supergroup")

    async def get_me(self):
        return SimpleNamespace(id=42)

    async def get_chat_member(self, chat_id, user_id):
        return SimpleNamespace(
            status="administrator", can_delete_messages=self.can_delete_any
        )

    async def edit_message_text(self, **kwargs):
        self.edited.append(kwargs)
        return SimpleNamespace(**kwargs)

    async def reply_text(self, text, **kwargs):  # pragma: no cover - unused
        return SimpleNamespace(text=text, **kwargs)


class SentMessage:
    """A Message: optionally in the group, optionally in the Study Room."""

    def __init__(self, bot_obj, chat_id=GROUP, thread=None, text="hi", message_id=7):
        self._bot = bot_obj
        self.chat_id = chat_id
        self.message_id = message_id
        self.message_thread_id = thread
        self.text = text
        self.chat = SimpleNamespace(
            type="private" if str(chat_id) != str(GROUP) else "supergroup"
        )
        self.replies: list[dict] = []
        self.edits: list[dict] = []

    def get_bot(self):
        return self._bot

    async def reply_text(self, text, **kwargs):
        self.replies.append({"text": text, **kwargs})
        return SimpleNamespace(chat_id=self.chat_id, message_id=1, message_thread_id=self.message_thread_id)

    async def edit_text(self, text, **kwargs):
        self.edits.append({"text": text, **kwargs})
        return self


class BaseTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        storage.set_data_dir(self.tmp.name)
        self.addCleanup(storage.set_data_dir, PROJECT_DATA)
        bot._clear_member_cache()
        self.addCleanup(bot._clear_member_cache)
        bot._drop_ask_index()
        self.addCleanup(bot._drop_ask_index)
        storage.save_topic(bot.QUIZ_TOPIC_NAME, TOPIC)

        self.bot = SentBot()

    def run_async(self, coro):
        return asyncio.run(coro)


class OutsideTopicTest(BaseTest):
    """Which messages count as "outside the Study Room"."""

    def test_general_is_outside(self) -> None:
        msg = SentMessage(self.bot, thread=None)
        self.assertTrue(bot._is_outside_topic(msg))

    def test_another_topic_is_outside(self) -> None:
        msg = SentMessage(self.bot, thread=999)
        self.assertTrue(bot._is_outside_topic(msg))

    def test_the_study_room_itself_is_not(self) -> None:
        msg = SentMessage(self.bot, thread=TOPIC)
        self.assertFalse(bot._is_outside_topic(msg))

    def test_a_private_chat_is_never_redirected(self) -> None:
        msg = SentMessage(self.bot, chat_id=4242, thread=None)
        self.assertFalse(bot._is_outside_topic(msg))

    def test_nothing_is_redirected_before_the_topic_is_known(self) -> None:
        storage.forget_topic(bot.QUIZ_TOPIC_NAME)
        msg = SentMessage(self.bot, thread=None)
        self.assertFalse(bot._is_outside_topic(msg))


class RedirectTest(BaseTest):
    """Answers to General land in the topic, not in General."""

    def test_a_reply_from_general_goes_into_the_topic(self) -> None:
        msg = SentMessage(self.bot, thread=None)
        self.run_async(bot._reply(msg, "hello"))

        self.assertEqual(msg.replies, [])              # not answered in place
        self.assertEqual(len(self.bot.sent), 1)
        self.assertEqual(self.bot.sent[0]["message_thread_id"], TOPIC)
        self.assertEqual(str(self.bot.sent[0]["chat_id"]), str(GROUP))

    def test_a_reply_inside_the_topic_stays_a_reply(self) -> None:
        msg = SentMessage(self.bot, thread=TOPIC)
        self.run_async(bot._reply(msg, "hello"))

        self.assertEqual(len(msg.replies), 1)
        self.assertEqual(self.bot.sent, [])

    def test_a_reply_in_private_stays_in_private(self) -> None:
        msg = SentMessage(self.bot, chat_id=4242, thread=None)
        self.run_async(bot._reply(msg, "hello"))

        self.assertEqual(len(msg.replies), 1)
        self.assertEqual(self.bot.sent, [])

    def test_the_answer_carries_the_question_with_it(self) -> None:
        """Nobody in the topic would know what is being answered otherwise."""
        msg = SentMessage(self.bot, thread=None)
        answer = Answer(query="what is a weak entity",
                        text="A weak entity depends on another.",
                        course="Database", filename="ch3.pdf", page=4)
        with patch.object(bot, "_ask_indexes", return_value=[("Database", object())]), \
                patch.object(bot, "_answer", return_value=answer):
            self.run_async(bot._ask("what is a weak entity", msg))

        self.assertEqual(len(self.bot.sent), 1)
        text = self.bot.sent[0]["text"]
        self.assertTrue(text.startswith("❓ <i>what is a weak entity</i>"))
        self.assertEqual(self.bot.sent[0]["parse_mode"], "HTML")
        self.assertIn("weak entity depends", text)
        self.assertEqual(self.bot.sent[0]["message_thread_id"], TOPIC)

    def test_an_answer_inside_the_topic_does_not_repeat_the_question(self) -> None:
        msg = SentMessage(self.bot, thread=TOPIC)
        answer = Answer(query="q", text="A weak entity depends on another.")
        with patch.object(bot, "_ask_indexes", return_value=[("Database", object())]), \
                patch.object(bot, "_answer", return_value=answer):
            self.run_async(bot._ask("what is a weak entity", msg))

        self.assertEqual(len(msg.replies), 1)
        self.assertNotIn("❓", msg.replies[0]["text"])

    def test_a_screen_outside_the_topic_moves_in_and_is_retired(self) -> None:
        msg = SentMessage(self.bot, thread=None)
        query = SimpleNamespace(message=msg, get_bot=lambda: self.bot)

        self.run_async(bot._edit(query, "the screen", bot._menu_kb()))

        self.assertEqual(len(self.bot.sent), 1)
        self.assertEqual(self.bot.sent[0]["message_thread_id"], TOPIC)
        self.assertEqual(self.bot.sent[0]["text"], "the screen")
        # the old screen in General is blanked so nobody keeps pressing it
        self.assertEqual(len(msg.edits), 1)
        self.assertIn("Moved to the Study Room topic", msg.edits[0]["text"])
        self.assertIsNone(msg.edits[0].get("reply_markup"))

    def test_send_to_topic_actually_sends(self) -> None:
        """Regression: the message used to be built but never awaited."""
        sent = self.run_async(bot._send_to_topic(self.bot, "hello"))
        self.assertEqual(len(self.bot.sent), 1)
        self.assertIsNotNone(sent)
        self.assertEqual(self.bot.sent[0]["message_thread_id"], TOPIC)


class OutboxTest(BaseTest):
    """Every group message is written down as it goes out."""

    def test_a_group_message_is_recorded_with_its_topic(self) -> None:
        self.run_async(bot._send_to_topic(self.bot, "hello"))
        entries = storage.load_outbox()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["thread"], TOPIC)

    def test_a_message_outside_the_topic_is_recorded_as_such(self) -> None:
        storage.forget_topic(bot.QUIZ_TOPIC_NAME)
        self.run_async(bot._send_to_topic(self.bot, "hello"))
        entries = storage.load_outbox()
        self.assertEqual(len(entries), 1)
        self.assertIsNone(entries[0]["thread"])

    def test_private_chats_are_not_recorded(self) -> None:
        bot._record_sent(SimpleNamespace(chat_id=4242, message_id=3, message_thread_id=None))
        self.assertEqual(storage.load_outbox(), [])

    def test_a_send_helper_records_what_it_sent(self) -> None:
        self.assertTrue(self.run_async(bot._send_message_safe(self.bot, "note")))
        entries = storage.load_outbox()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["thread"], TOPIC)

    def test_forgetting_removes_only_the_named_messages(self) -> None:
        storage.record_sent(1, None)
        storage.record_sent(2, TOPIC)
        storage.forget_sent([1])
        self.assertEqual([e["id"] for e in storage.load_outbox()], [2])

    def test_a_broken_outbox_reads_as_empty(self) -> None:
        (Path(self.tmp.name) / "outbox.json").write_text("{not json", "utf-8")
        self.assertEqual(storage.load_outbox(), [])


class RefInTopicTest(BaseTest):
    """When a remembered screen may still be edited in place."""

    def test_a_screen_left_in_general_is_not_editable(self) -> None:
        self.assertFalse(bot._ref_in_topic((GROUP, 777, None)))

    def test_a_screen_in_the_topic_is(self) -> None:
        self.assertTrue(bot._ref_in_topic((GROUP, 777, TOPIC)))

    def test_a_private_chat_has_no_topic_so_it_always_is(self) -> None:
        self.assertTrue(bot._ref_in_topic((4242, 777, None)))

    def test_a_reference_without_a_thread_keeps_the_simple_behaviour(self) -> None:
        self.assertTrue(bot._ref_in_topic((GROUP, 777)))

    def test_no_reference_at_all_is_fine(self) -> None:
        self.assertTrue(bot._ref_in_topic(None))


class ProgressEditTest(BaseTest):
    """Typed-input progress must not rewrite a screen outside the topic."""

    def setUp(self) -> None:
        super().setUp()
        pdf = Path(self.tmp.name) / "notes.pdf"
        pdf.write_bytes(_build_pdf(COURSE_LINES))
        course = storage.create_course("AI")
        storage.add_pdf(course.id, pdf, "notes.pdf")
        storage.finalize_course(course.id)

    def _type_quiz_count(self, ref):
        async def get_member(user_id: int):
            return SimpleNamespace(status="administrator")

        chat = SimpleNamespace(
            type="supergroup", id=int(GROUP), get_member=get_member
        )
        message = SentMessage(self.bot, thread=None, text="2")
        message.chat = chat
        update = SimpleNamespace(
            effective_chat=chat,
            effective_user=SimpleNamespace(id=5),
            effective_message=message,
            bot=SimpleNamespace(bot=SimpleNamespace(id=42)),
        )
        context = SimpleNamespace(
            user_data={"state": bot.GEN_COUNT, "gen_msg": ref},
            args=[],
            bot=self.bot,
        )
        self.run_async(bot.on_text(update, context))
        return message

    def test_a_screen_in_general_is_left_alone(self) -> None:
        message = self._type_quiz_count((GROUP, 777, None))

        self.assertEqual(self.bot.edited, [])          # the retired screen is safe
        self.assertTrue(self.bot.polls)                # the quiz still went out
        for poll in self.bot.polls:
            self.assertEqual(poll["message_thread_id"], TOPIC)
        # the progress went into the topic instead of editing the old screen
        self.assertEqual(message.replies, [])
        self.assertTrue(
            any(s.get("message_thread_id") == TOPIC for s in self.bot.sent)
        )

    def test_a_screen_in_the_topic_is_still_edited(self) -> None:
        self._type_quiz_count((GROUP, 777, TOPIC))

        self.assertTrue(self.bot.edited)
        self.assertEqual(self.bot.edited[0]["message_id"], 777)
        self.assertTrue(self.bot.polls)


class CleanupTest(BaseTest):
    """/cleanup removes whatever is outside the topic."""

    def _update(self, *, status="administrator", message=None, args=()):
        async def get_member(user_id: int):
            return SimpleNamespace(status=status)

        chat = SimpleNamespace(
            type="group",
            id=int(GROUP) if str(GROUP).lstrip("-").isdigit() else -1001,
            get_member=get_member,
        )
        message = message or SentMessage(self.bot, thread=TOPIC, text="/cleanup")
        message.chat = chat
        return SimpleNamespace(
            effective_chat=chat,
            effective_user=SimpleNamespace(id=5),
            effective_message=message,
            bot=SimpleNamespace(bot=SimpleNamespace(id=42)),
        )

    def _context(self, args=()):
        return SimpleNamespace(user_data={}, args=list(args), bot=self.bot)

    def _run(self, update, args=()):
        return self.run_async(bot.cmd_cleanup(update, self._context(args)))

    def test_it_deletes_only_what_is_outside_the_topic(self) -> None:
        storage.record_sent(11, None)        # went to General
        storage.record_sent(12, 999)         # went to another topic
        storage.record_sent(13, TOPIC)       # correctly placed

        self._run(self._update())

        self.assertEqual(
            self.bot.deleted,
            [(bot._group_chat_id(), 11), (bot._group_chat_id(), 12)],
        )
        self.assertEqual([e["id"] for e in storage.load_outbox()], [13])

    def test_it_reports_how_many_were_deleted(self) -> None:
        storage.record_sent(11, None)
        update = self._update()
        self._run(update)
        reply = update.effective_message.replies[-1]["text"]
        self.assertIn("Deleted 1 message", reply)
        self.assertIn(bot.QUIZ_TOPIC_NAME, reply)

    def test_nothing_to_delete_says_so(self) -> None:
        storage.record_sent(13, TOPIC)
        update = self._update()
        self._run(update)
        self.assertEqual(self.bot.deleted, [])
        self.assertIn("Nothing I remember", update.effective_message.replies[-1]["text"])

    def test_it_needs_the_topic_to_know_what_outside_means(self) -> None:
        storage.forget_topic(bot.QUIZ_TOPIC_NAME)
        storage.record_sent(11, None)
        update = self._update()
        self._run(update)
        self.assertEqual(self.bot.deleted, [])
        self.assertIn("/topic", update.effective_message.replies[-1]["text"])

    def test_a_member_may_not_use_it(self) -> None:
        storage.record_sent(11, None)
        update = self._update(status="member")
        self._run(update)
        self.assertEqual(self.bot.deleted, [])
        self.assertIn("admins", update.effective_message.replies[-1]["text"])

    def test_replying_to_a_message_deletes_that_one(self) -> None:
        storage.record_sent(13, TOPIC)       # correctly placed — still deletable
        message = SentMessage(self.bot, thread=TOPIC, text="/cleanup")
        message.reply_to_message = SimpleNamespace(
            chat_id=GROUP, message_id=13, from_user=SimpleNamespace(id=42)
        )
        self._run(self._update(message=message))

        self.assertEqual(self.bot.deleted, [(GROUP, 13)])
        self.assertEqual(storage.load_outbox(), [])

    def test_it_never_deletes_somebody_elses_message(self) -> None:
        """A student replies to their own message and runs /cleanup."""
        message = SentMessage(self.bot, thread=TOPIC, text="/cleanup")
        message.reply_to_message = SimpleNamespace(
            chat_id=GROUP, message_id=99, from_user=SimpleNamespace(id=999)
        )
        update = self._update(message=message)
        self._run(update)

        self.assertEqual(self.bot.deleted, [])
        self.assertIn("isn't mine", update.effective_message.replies[-1]["text"])

    def test_a_refusal_from_telegram_is_shown_verbatim(self) -> None:
        self.bot.delete_error = BadRequest("Bad Request: message to delete not found")
        message = SentMessage(self.bot, thread=TOPIC, text="/cleanup")
        message.reply_to_message = SimpleNamespace(
            chat_id=GROUP, message_id=7, from_user=None
        )
        update = self._update(message=message)
        self._run(update)

        reply = update.effective_message.replies[-1]["text"]
        self.assertIn("Telegram would not delete", reply)
        self.assertIn("not found", reply.lower())
        self.assertEqual(update.effective_message.replies[-1]["parse_mode"], "HTML")

    def test_a_range_sweep_deletes_only_my_own_messages(self) -> None:
        self.bot.deletable = {11, 12, 14, 21}
        storage.record_sent(20, TOPIC)   # known to be inside the topic -> spared
        storage.record_sent(21, None)    # an old one in General
        update = self._update()
        self._run(update, args=["10-21"])

        deleted = sorted(m[1] for m in self.bot.deleted)
        self.assertEqual(deleted, [11, 12, 14, 21])   # 20 protected, 10/13/15/… not mine
        reply = update.effective_message.replies[-1]["text"]
        self.assertIn("Deleted 4", reply)

    def test_a_sweep_is_refused_if_it_could_hit_other_peoples_messages(self) -> None:
        """"My own messages only" must survive a bot that may delete anything."""
        self.bot.can_delete_any = True
        self.bot.deletable = {11, 12}
        update = self._update()
        self._run(update, args=["10-15"])

        self.assertEqual(self.bot.deleted, [])
        reply = update.effective_message.replies[-1]["text"]
        self.assertIn("other people's messages", reply)
        self.assertIn("Reply to each of my messages", reply)

    def test_the_reply_form_still_works_when_a_sweep_is_refused(self) -> None:
        self.bot.can_delete_any = True
        message = SentMessage(self.bot, thread=TOPIC, text="/cleanup")
        message.reply_to_message = SimpleNamespace(
            chat_id=GROUP, message_id=21, from_user=SimpleNamespace(id=42)
        )
        self._run(self._update(message=message))
        self.assertEqual([m[1] for m in self.bot.deleted], [21])

    def test_a_sweep_accepts_the_long_form(self) -> None:
        self.bot.deletable = {5}
        update = self._update()
        self._run(update, args=["from 5 to 5"])
        self.assertEqual([m[1] for m in self.bot.deleted], [5])

    def test_a_sweep_refuses_an_enormous_range(self) -> None:
        update = self._update()
        self._run(update, args=["1-5000"])
        self.assertEqual(self.bot.deleted, [])
        self.assertIn("at most", update.effective_message.replies[-1]["text"])

    def test_a_sweep_needs_a_range(self) -> None:
        update = self._update()
        self._run(update, args=["everything"])
        self.assertEqual(self.bot.deleted, [])
        self.assertIn("Usage", update.effective_message.replies[-1]["text"])

    def test_nothing_recorded_explains_how_to_reach_older_messages(self) -> None:
        update = self._update()
        self._run(update)
        reply = update.effective_message.replies[-1]["text"]
        self.assertIn("/cleanup 1234-1290", reply)
        self.assertIn("reply to one of them", reply)


if __name__ == "__main__":
    unittest.main()
