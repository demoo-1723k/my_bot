"""One bot instance at a time.

Telegram delivers a bot's updates to a single getUpdates poller. A second copy
of bot.py does not fail cleanly — it produces

    Conflict: terminated by other getUpdates request; make sure that only one
    bot instance is running

over and over, and whichever console you are watching may be the copy that is
losing updates. A lock file makes the second copy refuse to start with one
clear line instead.

Run from the project root:
    python -m unittest -v
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import bot
import storage

PROJECT_DATA = Path(__file__).resolve().parent.parent / "data"


def _spawn(args: list[str]) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, *args])


def _stop(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.kill()
        proc.wait(timeout=10)


def _finished_pid() -> int:
    """A pid that belonged to a process which has now exited."""
    proc = _spawn(["-c", "pass"])
    proc.wait(timeout=30)
    return proc.pid


class PidAliveTest(unittest.TestCase):
    def test_our_own_process_is_alive(self) -> None:
        self.assertTrue(bot._pid_alive(os.getpid()))

    def test_nonsense_ids_are_not_alive(self) -> None:
        for pid in (0, -1, -7):
            with self.subTest(pid=pid):
                self.assertFalse(bot._pid_alive(pid))

    def test_a_finished_process_is_not_alive(self) -> None:
        self.assertFalse(bot._pid_alive(_finished_pid()))


class SingleInstanceLockTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        storage.set_data_dir(self.tmp.name)
        self.addCleanup(storage.set_data_dir, PROJECT_DATA)
        self.path = Path(self.tmp.name) / bot.LOCK_FILE

    def _live_foreign_pid(self) -> int:
        """A pid belonging to some *other* process that is still running."""
        proc = _spawn(["-c", "import time; time.sleep(60)"])
        self.addCleanup(_stop, proc)
        return proc.pid

    def test_takes_the_lock_when_it_is_free(self) -> None:
        lock = bot._acquire_single_instance_lock()
        self.assertEqual(lock, self.path)
        self.assertEqual(self.path.read_text(encoding="utf-8"), str(os.getpid()))

    def test_a_running_second_copy_is_refused(self) -> None:
        other = self._live_foreign_pid()
        self.path.write_text(str(other), encoding="utf-8")

        self.assertIsNone(bot._acquire_single_instance_lock())
        # the other copy's lock was left untouched
        self.assertEqual(self.path.read_text(encoding="utf-8"), str(other))

    def test_a_lock_left_by_a_crashed_copy_is_taken_over(self) -> None:
        self.path.write_text(str(_finished_pid()), encoding="utf-8")

        lock = bot._acquire_single_instance_lock()
        self.assertEqual(lock, self.path)
        self.assertEqual(self.path.read_text(encoding="utf-8"), str(os.getpid()))

    def test_a_garbage_lock_file_is_taken_over(self) -> None:
        self.path.write_text("{not a pid", encoding="utf-8")
        self.assertIsNotNone(bot._acquire_single_instance_lock())

    def test_no_lock_file_is_taken_over(self) -> None:
        self.assertFalse(self.path.exists())
        self.assertIsNotNone(bot._acquire_single_instance_lock())

    def test_releasing_removes_our_own_lock(self) -> None:
        lock = bot._acquire_single_instance_lock()
        bot._release_single_instance_lock(lock)
        self.assertFalse(self.path.exists())

    def test_releasing_leaves_a_lock_someone_else_took(self) -> None:
        lock = bot._acquire_single_instance_lock()
        other = self._live_foreign_pid()
        self.path.write_text(str(other), encoding="utf-8")

        bot._release_single_instance_lock(lock)
        self.assertTrue(self.path.exists())

    def test_releasing_nothing_is_harmless(self) -> None:
        bot._release_single_instance_lock(None)

    def test_main_refuses_to_start_a_second_copy(self) -> None:
        other = self._live_foreign_pid()
        self.path.write_text(str(other), encoding="utf-8")

        with patch.object(bot, "BOT_TOKEN", "123456789:EXAMPLETOKENVALUE"):
            with self.assertRaises(SystemExit) as caught:
                bot.main()

        message = str(caught.exception)
        self.assertIn("already running", message)
        self.assertIn(bot.LOCK_FILE, message)


if __name__ == "__main__":
    unittest.main()
