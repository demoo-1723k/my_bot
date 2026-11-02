"""Tests for the operator-only /log activity feed.

The log must be a strict secret: only @kena_ng ever sees it, only in a
private chat, covering a 12-hour window. It records commands, button
presses, uploads and the daily jobs — not the group owner, not admins.

Run from the project root:
    python -m unittest -v
"""

from __future__ import annotations

import asyncio
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import bot
import storage


def _update(
    username: str | None,
    chat_type: str = "private",
    chat_id: int = 555,
    title: str | None = None,
    text: str = "/log",
):
    """A barebones Update double that records every reply it is asked to send."""
    chat = SimpleNamespace(id=chat_id, type=chat_type, title=title, username=None)
    replies: list[str] = []

    async def reply_text(msg_text, **kwargs):
        replies.append(msg_text)
        return SimpleNamespace(message_id=1)

    message = SimpleNamespace(
        text=text, chat_id=chat_id, chat=chat, reply_text=reply_text
    )
    update = SimpleNamespace(
        effective_user=SimpleNamespace(username=username, full_name="Someone", id=1),
        effective_chat=chat,
        effective_message=message,
    )
    return update, replies


class _StorageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        storage.set_data_dir(self.tmp.name)
        self.addCleanup(
            storage.set_data_dir, Path(__file__).resolve().parent.parent / "data"
        )


class CmdLogTest(_StorageTest):
    """Only @kena_ng sees the 12h activity feed, and only in private."""

    def _run_cmd(self, update):
        return asyncio.run(bot.cmd_log(update, SimpleNamespace()))

    def test_everyone_else_gets_silence(self) -> None:
        # the group owner, an admin, a student — no reply at all, as if the
        # command did not exist
        for name in ("some_admin", "group_owner", None):
            update, replies = _update(name)
            self._run_cmd(update)
            self.assertEqual(replies, [], f"{name!r} must not see the log")

    def test_operator_gets_the_log_in_private(self) -> None:
        storage.log_activity("@kena_ng", "private", "/start")
        storage.log_activity("@alice", "Study Group", "📝 quiz: 5/5 questions → group")
        update, replies = _update("kena_ng")
        self._run_cmd(update)
        self.assertEqual(len(replies), 1)
        self.assertIn("Activity log", replies[0])
        self.assertIn("@alice", replies[0])
        self.assertIn("quiz: 5/5", replies[0])
        # newest first: alice's quiz is newer than the /start
        self.assertLess(replies[0].find("quiz: 5/5"), replies[0].find("/start"))

    def test_operator_in_a_group_gets_no_log_only_a_hint(self) -> None:
        storage.log_activity("@alice", "private", "secret detail")
        update, replies = _update(
            "kena_ng", chat_type="group", chat_id=777, title="Study Group"
        )
        self._run_cmd(update)
        self.assertEqual(len(replies), 1)
        self.assertIn("private chat", replies[0])
        self.assertNotIn("Activity log", replies[0])
        self.assertNotIn("secret detail", replies[0])

    def test_empty_window_says_so(self) -> None:
        update, replies = _update("kena_ng")
        self._run_cmd(update)
        self.assertEqual(len(replies), 1)
        self.assertIn("nothing happened", replies[0])

    def test_only_the_last_12_hours_are_shown(self) -> None:
        storage.log_activity("@x", "private", "too old", ts=time.time() - 12 * 3600 - 60)
        storage.log_activity("@x", "private", "fresh")
        update, replies = _update("kena_ng")
        self._run_cmd(update)
        self.assertIn("fresh", replies[0])
        self.assertNotIn("too old", replies[0])


class LogEventTest(_StorageTest):
    """_log_event records who did what, where — best-effort, never raises."""

    def test_event_recorded_from_update(self) -> None:
        update, _ = _update("kena_ng", chat_type="group", chat_id=777, title="Study Group")
        bot._log_event(update, '🔘 "📚 Add Course" [menu:add]')
        events = storage.load_activity(12.0)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["who"], "@kena_ng")
        self.assertEqual(events[0]["where"], "Study Group")
        self.assertIn("Add Course", events[0]["what"])

    def test_private_chat_and_usernameless_users(self) -> None:
        update, _ = _update(None)  # no username -> display name
        bot._log_event(update, "/start")
        event = storage.load_activity(12.0)[0]
        self.assertEqual(event["who"], "Someone")
        self.assertEqual(event["where"], "private")

    def test_scheduled_jobs_log_as_scheduler(self) -> None:
        bot._log_event(what="⏰ auto quiz: 10/10 questions → the group")
        self.assertEqual(storage.load_activity(12.0)[0]["who"], "scheduler")

    def test_broken_disk_never_raises(self) -> None:
        storage.DATA_DIR.mkdir(parents=True, exist_ok=True)
        storage._activity_path().write_text("[broken", encoding="utf-8")
        # corrupt file -> reads as empty, write still succeeds
        bot._log_event(what="still works")
        self.assertEqual(len(storage.load_activity(12.0)), 1)


if __name__ == "__main__":
    unittest.main()
