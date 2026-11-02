"""Regression tests for network hardening in bot.py.

These cover the failures seen live on 2026-10-03:
  - TimedOut while answering a callback aborted the whole "save course" flow
  - Stale callback queries ("Query is too old") crashed as unhandled errors
  - Poll sends died on transient timeouts instead of retrying

Run from the project root:
    python -m unittest -v
"""

from __future__ import annotations

import asyncio
import unittest

from telegram.error import BadRequest, RetryAfter, TimedOut

from bot import _safe_answer, _send_poll_with_retry

POLL = {"question": "Q?", "options": ["a", "b"], "chat_id": 1,
        "correct_option_id": 0}


class FakeQuery:
    def __init__(self, error: Exception | None = None):
        self.error = error
        self.answered: tuple | None = None

    async def answer(self, text=None, show_alert=False):
        if self.error:
            raise self.error
        self.answered = (text, show_alert)


class FakeBot:
    """Raises the queued exceptions first, then succeeds."""

    def __init__(self, failures: list):
        self.failures = list(failures)
        self.sent: list[dict] = []

    async def send_poll(self, **kwargs):
        if self.failures:
            exc = self.failures.pop(0)
            if exc is not None:
                raise exc
        self.sent.append(kwargs)


class SafeAnswerTest(unittest.TestCase):
    def test_timeout_does_not_propagate(self) -> None:
        asyncio.run(_safe_answer(FakeQuery(TimedOut()), "Saving…"))

    def test_stale_query_does_not_propagate(self) -> None:
        asyncio.run(
            _safe_answer(
                FakeQuery(BadRequest("Query is too old and response timeout expired")),
                "hi",
                True,
            )
        )

    def test_answer_is_passed_through_when_network_works(self) -> None:
        q = FakeQuery()
        asyncio.run(_safe_answer(q, "Saved!", show_alert=True))
        self.assertEqual(q.answered, ("Saved!", True))


class SendPollRetryTest(unittest.TestCase):
    def test_recovers_after_timeouts(self) -> None:
        bot = FakeBot([TimedOut(), TimedOut(), None])
        ok = asyncio.run(_send_poll_with_retry(bot, POLL, backoff=0))
        self.assertTrue(ok)
        self.assertEqual(len(bot.sent), 1)

    def test_gives_up_after_repeated_failures(self) -> None:
        bot = FakeBot([TimedOut()] * 4)
        ok = asyncio.run(_send_poll_with_retry(bot, POLL, backoff=0))
        self.assertFalse(ok)
        self.assertEqual(bot.sent, [])

    def test_bad_request_is_not_retried(self) -> None:
        bot = FakeBot([BadRequest("poll is too long")])
        ok = asyncio.run(_send_poll_with_retry(bot, POLL, backoff=0))
        self.assertFalse(ok)
        self.assertEqual(bot.sent, [])

    def test_retry_after_honoured(self) -> None:
        bot = FakeBot([RetryAfter(0), None])
        ok = asyncio.run(_send_poll_with_retry(bot, POLL, backoff=0))
        self.assertTrue(ok)
        self.assertEqual(len(bot.sent), 1)


if __name__ == "__main__":
    unittest.main()
