#!/usr/bin/env python3
"""Course quiz bot for @practicemakesperfect2.

Flow:
  /start          -> greeting + inline menu
  Add Course      -> ask for name -> accept PDFs until [Done] -> save -> back to menu
  Delete Course   -> list courses -> "Are you sure?" [Yes, Delete][Cancel]
  Generate Quiz   -> asks HOW MANY questions -> mixed quiz from ALL courses,
                     posted to the group; explanations cite course · file · page
  Generate Note   -> one study note from a random course, posted in the
                     group with its source
  Auto Quiz       -> status + 24-hour time picker (or a custom time) -> how many
                     questions -> that mixed quiz is posted in the group daily
  Auto Note       -> status + 24-hour time picker (or a custom time) -> a note
                     from a random course is posted in the group daily
  Both automation screens carry a "Turn off" button that switches the schedule
  back off (the saved time and question count are kept for next time).
  /ask <question>  -> BM25 search over every saved PDF; the reply is a sentence
                     copied from the material plus its course · file · page.
                     Also triggered by a plain message in a private chat or by
                     replying to one of the bot's messages, never by ordinary
                     group chatter.

State lives in context.user_data (per user, per chat) — a single message is
re-used as the "screen" and edited as the flow progresses.
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import os
import random
import re
import tempfile
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from html import escape
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import PollType
from telegram.error import BadRequest, NetworkError, RetryAfter, TelegramError, TimedOut
from telegram.request import HTTPXRequest
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

import storage
from generator import (
    DEFAULT_QUESTION_COUNT,
    MAX_QUESTIONS,
    Answer,
    InsufficientTextError,
    Note,
    Retriever,
    Segment,
    generate_mixed_questions,
    generate_note,
    questions_from_exams,
)
# AI layer — graceful fallback if llm_client not configured
try:
    from ai_generator import (
        ai_answer_question as _ai_answer,
        ai_answer_question_async as _ai_answer_async,
        ai_generate_note as _ai_note,
        ai_generate_note_async as _ai_note_async,
        ai_generate_questions as _ai_questions,
        ai_generate_questions_async as _ai_questions_async,
    )
    _AI_AVAILABLE = True
except ImportError:
    _AI_AVAILABLE = False

try:
    from retriever import HybridRetriever
    _HYBRID_AVAILABLE = True
except ImportError:
    _HYBRID_AVAILABLE = False
    HybridRetriever = Retriever  # type: ignore

load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
QUIZ_CHAT_ID = os.getenv("QUIZ_CHAT_ID", "-1002292353150")
# Forum topic the bot posts into. The topic id can be pinned in .env, or the
# bot learns it from /topic run inside the topic, or from the service message
# Telegram sends when the topic is created.
QUIZ_TOPIC_NAME = os.getenv("QUIZ_TOPIC_NAME", "Study Room")
QUIZ_TOPIC_ID = os.getenv("QUIZ_TOPIC_ID", "")

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

GROUP_LABEL = QUIZ_CHAT_ID if QUIZ_CHAT_ID.startswith("@") else "the group"

# conversation states
ADD_NAME = "add_name"
ADD_FILES = "add_files"
DEL_SELECT = "del_select"
DEL_CONFIRM = "del_confirm"
GEN_COUNT = "gen_count"
AUTO_QUIZ_TIME = "auto_quiz_time"   # waiting for a typed 24-hour time
AUTO_QUIZ_COUNT = "auto_quiz_count" # time chosen, waiting for the quantity
AUTO_NOTE_TIME = "auto_note_time"   # waiting for a typed 24-hour time
EXAM_NAME = "exam_name"             # exams: waiting for the paper's name
EXAM_FILES = "exam_files"           # exams: waiting for the PDF(s)
GEN_DEST = "gen_dest"               # admin in private: group or this chat?

# states only the group owner/admins may hold (a student in a private chat is
# never in one of these — but if one ever is, it is cleared instead of run)
ADMIN_STATES = {
    ADD_NAME,
    ADD_FILES,
    DEL_SELECT,
    DEL_CONFIRM,
    EXAM_NAME,
    EXAM_FILES,
    AUTO_QUIZ_TIME,
    AUTO_QUIZ_COUNT,
    AUTO_NOTE_TIME,
}

# Wake words that turn a normal group message into a question. Students type
# "baymax what is 2NF" instead of using the /ask command.
_WAKE_WORDS = re.compile(
    r"\b(?:bay\s*max|baymax|j\.?a\.?r\.?v\.?i\.?s\.?|jarvis|jarves|jarvas|jarviz)\b",
    re.IGNORECASE,
)

# A share of each quiz that may come from old exam papers, so exams enrich the
# quiz instead of replacing it.
EXAM_SHARE = 0.3

# times offered on the pickers (24-hour format, like the custom input)
TIME_PRESETS = ("07:00", "08:00", "12:00", "13:00", "17:00", "18:00", "20:00", "21:00")

# how often the scheduler checks whether a daily task is due
TICK_SECONDS = 20

GREETING = (
    "👋 Hey! I'm baymax an AI assitance trained from your old matrial you used to read.\n\n"
    "Organize your study PDFs into courses, and I'll turn them into mixed question to test youhow much you remenber from past"
    "quiz polls for the group.\n\n"
    "• Add Course — save PDFs under a course name\n"
    "• Delete Course — remove a course (with confirmation)\n"
    "• Generate Quiz — choose how many questions, and I'll post a mixed "
    "quiz from ALL your courses\n"
    "• Generate Note — one study note from a random course, posted in the "
    "group with its source\n"
    "• Auto Quiz — a mixed quiz in the group every day at the time you pick\n"
    "• Auto Note — a note from a random course every day (never the same "
    "course twice in a row)\n"
    "• Exams — save past papers (with answers); a slice of every quiz then "
    "uses the real questions from them\n\n"
    "🔒 <b>This menu is for the group owner and admins.</b> Everyone else can "
    "still talk to me in a private chat: ask a question or get a quiz for "
    "themselves.\n\n"
    "Ask me anything about your notes: <b>/ask what is a weak entity</b>, or "
    "just say <b>baymax what is 2NF</b> in the group, or reply to one of my "
    "messages. I answer only from your PDFs and show the course · file · "
    "page.\n\n"
    f"Everything is posted in the <b>{QUIZ_TOPIC_NAME}</b> topic. "
    "Every answer shows where the material comes from so you can re-read "
    "that part. /cancel aborts a step, /cleanup removes anything of mine "
    "that ended up outside the topic."
)

HELP = (
    "How it works:\n"
    "1. Add Course — name it, send the PDFs, press Done\n"
    "2. Generate Quiz — pick the quantity (preset or any number); I mix "
    "every course and post quiz polls to @practicemakesperfect2\n"
    "3. Generate Note — one note from a random course, posted in the group, "
    "to check how it looks before turning the daily note on\n"
    "4. Auto Quiz / Auto Note — pick a time in 24-hour format (or type "
    "your own) and the daily post runs by itself; “Turn off” stops it\n"
    "5. Delete Course — remove a course you no longer need\n"
    "6. Exams — upload past papers with their answers; about "
    f"{int(EXAM_SHARE * 100)}% of each quiz then uses the real questions "
    "from them, and you can delete any paper from the same screen\n"
    "7. /ask &lt;question&gt; — I search your PDFs and quote the sentence "
    "that answers it, with its course · file · page\n\n"
    "In the group you can just say <b>baymax what is 2NF</b> (or reply to "
    "one of my messages) instead of using /ask. I only answer when I'm "
    "called by name, so I don't read along with the rest of the chat.\n\n"
    "Everything is posted in the "
    f"<b>{QUIZ_TOPIC_NAME}</b> topic. Quiz questions are fill-in-the-blank "
    "and true/false. After answering, Telegram shows the original sentence "
    "plus its source: course, file and page number.\n\n"
    "🔒 The menu is for the group owner and admins. In a private chat, press "
    "Generate Quiz or Generate Note and I ask whether to post it to the "
    "group or keep it in that chat. Everyone else gets questions and their "
    "own quiz in a private chat.\n\n"
    "Everything I post goes in the topic — if something of mine is sitting in "
    "General or another topic, <b>/cleanup</b> deletes it (admins only, and "
    "Telegram only allows this for 48 hours). Reply to one of my messages "
    "with /cleanup to delete just that one, or send <code>/cleanup 1234-1290</code> "
    "with a range of message ids for older ones. I can only ever delete my "
    "own messages.\n\n"
    "Note: PDFs need selectable text — scanned images can't be read."
)

# Shown to anyone who is not the group owner or an admin: they can study, they
# cannot run the group's bot.
MEMBER_HINT = (
    "🔒 Only the group owner and admins can run me here.\n\n"
    "💬 Open a private chat with me and ask there — I answer your question "
    "and can send you a quiz of your own."
)

ADMIN_ONLY_HINT = "🔒 That's for the group owner and admins only."

STUDENT_GREETING = (
    "👋 Hey! I'm Baymax, the study bot for computer Science 2016 batch students.\n\n"
    "🔎 <b>Ask me anything</b> from the course PDFs — just type your question, "
    "for example <i>what is a weak entity</i>. I answer with the sentence from "
    "the material and show its course · file · page.\n\n"
    "🎯 <b>Send me a quiz</b> — a mixed quiz posted right here in this chat, "
    "with as many questions as you want.\n\n"
    "The group owner and admins manage the courses, exams and schedules; "
    "whatever you ask stays in this chat.feel free"
)

STUDENT_HELP = (
    "How to study with me:\n"
    "• Just type your question here — e.g. what is a weak entity\n"
    "• /ask &lt;question&gt; works the same way\n"
    "• Press 🎯 Send me a quiz for a quiz of your own\n\n"
    "I only answer from the PDFs the admins have saved, and I always show "
    "where the answer came from. If it isn't in the material, I say so "
    "instead of guessing.\n\n"
    "In the group only the owner and admins can use the bot, so this chat is "
    "your place to ask."
)


# --- keyboards ------------------------------------------------------------

def _menu_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("Add Course", callback_data="menu:add"),
                InlineKeyboardButton("Delete Course", callback_data="menu:del"),
            ],
            [
                InlineKeyboardButton("Generate Quiz", callback_data="menu:gen"),
                InlineKeyboardButton("Generate Note", callback_data="menu:genote"),
            ],
            [
                InlineKeyboardButton("⏰ Auto Quiz", callback_data="menu:aq"),
                InlineKeyboardButton("📝 Auto Note", callback_data="menu:an"),
            ],
            [
                InlineKeyboardButton("📚 Exams", callback_data="menu:exams"),
            ],
        ]
    )


def _student_kb() -> InlineKeyboardMarkup:
    """What someone who is not an admin is allowed to do."""
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("🎯 Send me a quiz", callback_data="menu:squiz")],
            [InlineKeyboardButton("🔎 How to ask", callback_data="menu:how")],
        ]
    )


def _menu_screen(is_admin: bool) -> tuple[str, InlineKeyboardMarkup]:
    """The /start screen for whoever pressed it."""
    if is_admin:
        return GREETING, _menu_kb()
    return STUDENT_GREETING, _student_kb()


def _dest_kb() -> InlineKeyboardMarkup:
    """Where should this quiz/note go — the group, or this private chat?"""
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("📌 To the group", callback_data="dest:group")],
            [InlineKeyboardButton("💬 Here in this chat", callback_data="dest:private")],
            [InlineKeyboardButton("⬅️ Back", callback_data="menu:back")],
        ]
    )


def _done_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton("✅ Done", callback_data="add:done")]]
    )


def _course_list_kb(courses: list[storage.Course]) -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton(c.name, callback_data=f"del:{c.id}")]
        for c in courses
    ]
    rows.append([InlineKeyboardButton("⬅️ Back", callback_data="menu:back")])
    return InlineKeyboardMarkup(rows)


def _confirm_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("Yes, Delete", callback_data="del:yes")],
            [InlineKeyboardButton("Cancel", callback_data="del:cancel")],
        ]
    )


def _qty_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton(str(n), callback_data=f"gen:{n}") for n in (5, 10, 15)],
            [InlineKeyboardButton(str(n), callback_data=f"gen:{n}") for n in (25, 50)],
            [InlineKeyboardButton("⬅️ Back", callback_data="menu:back")],
        ]
    )


def _time_kb(prefix: str, *, can_turn_off: bool) -> InlineKeyboardMarkup:
    """Time picker for an automation screen.

    `prefix` is the callback namespace ("aq" for the auto quiz, "an" for the
    auto note). The custom option types the time instead of picking a preset,
    and the off button is only offered while something is actually running.
    """
    rows = [
        [
            InlineKeyboardButton(t, callback_data=f"{prefix}:time:{t}")
            for t in TIME_PRESETS[i : i + 3]
        ]
        for i in range(0, len(TIME_PRESETS), 3)
    ]
    rows.append(
        [InlineKeyboardButton("⌨️ Custom time", callback_data=f"{prefix}:custom")]
    )
    if can_turn_off:
        rows.append([InlineKeyboardButton("❌ Turn off", callback_data=f"{prefix}:off")])
    rows.append([InlineKeyboardButton("⬅️ Back", callback_data="menu:back")])
    return InlineKeyboardMarkup(rows)


def _auto_qty_kb() -> InlineKeyboardMarkup:
    """Quantity picker for the daily quiz (asks after the time is chosen)."""
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(str(n), callback_data=f"aq:count:{n}")
                for n in (5, 10, 15)
            ],
            [
                InlineKeyboardButton(str(n), callback_data=f"aq:count:{n}")
                for n in (25, 50)
            ],
            [InlineKeyboardButton("⬅️ Back", callback_data="menu:aq")],
        ]
    )


def _back_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton("⬅️ Back", callback_data="menu:back")]]
    )


def _exams_kb() -> InlineKeyboardMarkup:
    """Upload a new paper, or delete one already saved."""
    rows = [
        [InlineKeyboardButton("📥 Upload exam PDF", callback_data="ex:upload")]
    ]
    for e in storage.list_exams():
        rows.append(
            [InlineKeyboardButton(f"🗑 {e.name}", callback_data=f"ex:del:{e.id}")]
        )
    rows.append([InlineKeyboardButton("⬅️ Back", callback_data="menu:back")])
    return InlineKeyboardMarkup(rows)


def _exams_screen() -> tuple[str, InlineKeyboardMarkup]:
    exams = storage.list_exams()
    if exams:
        lines = [
            "📚 <b>Old exams</b>\n\n"
            "Saved papers take part in quiz generation: a share of each quiz "
            "uses the real questions and answers from these papers.",
            "",
            "Tap a paper below to delete it, or upload another:",
        ]
        for e in exams:
            pages = len({s.page for s in storage.exam_segments(e.id) if s.page})
            files = len(e.files)
            lines.append(f"• <b>{e.name}</b> — {files} file(s), {pages or '?'} pages")
    else:
        lines = [
            "📚 <b>Old exams</b>\n\n"
            "No papers yet. Upload a past exam <b>with its answers</b> as a "
            "text PDF, and I'll use its real questions in your quizzes.",
        ]
    return "\n".join(lines), _exams_kb()


# --- one bot instance only ------------------------------------------------
# Telegram hands a bot's updates to a single getUpdates poller. A second copy
# of this script does not fail cleanly: it fills the log with
# "Conflict: terminated by other getUpdates request", and the two copies fight
# over the same updates (the console you are watching may be the one that is
# losing). A lock file makes the second copy say so in one line instead.
LOCK_FILE = "bot.lock"


def _pid_alive(pid: int) -> bool:
    """Is a process with this id still running?"""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # it exists, just owned by another user
    except OSError:
        return False
    return True


def _lock_path() -> Path:
    return storage.DATA_DIR / LOCK_FILE


def _acquire_single_instance_lock() -> Path | None:
    """Claim the right to run. None means another copy is already running.

    A lock left behind by a copy that crashed (its process is gone) is taken
    over instead of blocking the next start.
    """
    path = _lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        owner = int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        owner = 0
    if owner and owner != os.getpid() and _pid_alive(owner):
        return None
    path.write_text(str(os.getpid()), encoding="utf-8")
    return path


def _release_single_instance_lock(path: Path | None) -> None:
    if path is None:
        return
    try:
        if path.read_text(encoding="utf-8").strip() == str(os.getpid()):
            path.unlink()
    except OSError:
        pass  # a leftover lock is harmless: it is overwritten next start


# --- helpers --------------------------------------------------------------

# Who may control the bot. Telegram reports the group's owner as "creator" and
# its admins as "administrator"; everyone else is a plain member. The answer
# is cached briefly so a burst of messages does not become an API call each.
_MEMBER_CACHE: dict[int, tuple[float, str]] = {}
_MEMBER_CACHE_TTL = 60.0


def _clear_member_cache() -> None:
    _MEMBER_CACHE.clear()


def _chat_type(update: Update) -> str | None:
    """"group" / "private" / ... — None when the update carries no chat."""
    return getattr(getattr(update, "effective_chat", None), "type", None)


def _is_private(update: Update) -> bool:
    """True only for a chat we can see is private (no chat = not private)."""
    return _chat_type(update) == "private"


def _chat_id(update: Update) -> int | None:
    return getattr(getattr(update, "effective_chat", None), "id", None)


def _message_is_private(message) -> bool:
    """True when a Message object is a private chat with the bot."""
    return getattr(getattr(message, "chat", None), "type", None) == "private"


def _group_chat_id():
    """QUIZ_CHAT_ID as a number when it is one, else as the @username."""
    try:
        return int(str(QUIZ_CHAT_ID).strip())
    except (TypeError, ValueError):
        return QUIZ_CHAT_ID


async def _member_status(update: Update, context) -> str | None:
    """The user's status in the study group, or None when it is unknown.

    "none" means Telegram says they are not in the group at all. None means we
    could not find out — no chat attached, an API/network failure, or a test
    double — and the caller then does not block anyone, so a flaky connection
    can never lock the owner out of their own bot.
    """
    uid = getattr(getattr(update, "effective_user", None), "id", None)
    if uid is None:
        return None
    now = time.monotonic()
    cached = _MEMBER_CACHE.get(uid)
    if cached is not None and now - cached[0] < _MEMBER_CACHE_TTL:
        return cached[1]

    is_private = _is_private(update)
    try:
        if is_private:
            # in a private chat, ask the group directly
            getter = getattr(getattr(context, "bot", None), "get_chat_member", None)
            if getter is None:
                return None
            member = await getter(_group_chat_id(), uid)
        else:
            getter = getattr(getattr(update, "effective_chat", None), "get_member", None)
            if getter is None:
                return None
            member = await getter(uid)
    except NetworkError:
        return None  # transient — do not cache, try again next time
    except TelegramError as exc:
        status = "none"  # not in the group (or the bot cannot see it)
        logger.info(
            "membership check for %s: %s (they get the student view)", uid, exc
        )
    except Exception:
        return None  # not a Telegram object at all (tests)
    else:
        status = getattr(member, "status", None) or "none"
    _MEMBER_CACHE[uid] = (time.monotonic(), status)
    return status


async def _is_admin(update: Update, context) -> bool:
    """May this user run the bot: the group's owner or one of its admins?

    Checked in the group and in a private chat alike, so an admin gets the full
    menu in both. An unknown answer is treated as "yes" — the failure mode of
    failing open is a student seeing a menu, the failure mode of failing closed
    is the owner being locked out.
    """
    status = await _member_status(update, context)
    if status is None:
        return True
    return status in ("creator", "administrator")


def _thread_id() -> int | None:
    """The forum topic id the bot posts into, or None to use General.

    Order: an explicit QUIZ_TOPIC_ID in .env (the dependable way to pin it),
    then the id learned at runtime from /topic or from topic creation.
    """
    if QUIZ_TOPIC_ID.strip():
        try:
            return int(QUIZ_TOPIC_ID.strip())
        except ValueError:
            logger.warning("QUIZ_TOPIC_ID is not a number; ignoring it")
    return storage.get_topic(QUIZ_TOPIC_NAME)


def _thread_kwargs(**extra):
    """Add message_thread_id so a send lands in the study-room topic."""
    tid = _thread_id()
    if tid is None:
        return extra
    extra["message_thread_id"] = tid
    return extra


def _is_missing_thread(exc: BaseException) -> bool:
    """Did Telegram reject this because the topic id no longer exists?

    A topic can be deleted, renamed or the pinned id can be wrong; when that
    happens every send fails with "message thread not found", which would
    otherwise mean no quiz or note could ever be posted again.
    """
    text = str(exc).lower()
    return "message thread not found" in text or "thread_not_found" in text


def _forget_stale_topic() -> None:
    """Drop a topic id Telegram rejected, so posting can continue."""
    storage.forget_topic(QUIZ_TOPIC_NAME)
    if QUIZ_TOPIC_ID.strip():
        logger.warning(
            "QUIZ_TOPIC_ID=%s in .env is not a real topic — fix or clear it, "
            "then send /topic inside '%s' to relearn it",
            QUIZ_TOPIC_ID.strip(),
            QUIZ_TOPIC_NAME,
        )
    logger.warning(
        "Posting to General instead: the '%s' topic was not found.",
        QUIZ_TOPIC_NAME,
    )


def _record_sent(message) -> None:
    """Write down an outgoing group message so /cleanup can remove it later.

    Telegram has no API for "list my own messages", so every send to the group
    is remembered with the topic it landed in. Private chats are not recorded:
    they are already where they belong.
    """
    if message is None:
        return
    if str(getattr(message, "chat_id", "")) != str(QUIZ_CHAT_ID):
        return
    try:
        storage.record_sent(
            int(message.message_id), getattr(message, "message_thread_id", None)
        )
    except Exception:  # never let bookkeeping break a send
        logger.debug("could not record sent message", exc_info=True)


def _is_outside_topic(message) -> bool:
    """True when this message came from the group but not from the topic.

    Everything the bot does belongs in the Study Room, so a command, question
    or button pressed in General (or in another topic) is answered *inside* the
    topic rather than where it came from. Private chats are never redirected.
    """
    if str(getattr(message, "chat_id", "")) != str(QUIZ_CHAT_ID):
        return False
    tid = _thread_id()
    if tid is None:
        return False  # topic unknown — there is nowhere to redirect to
    return getattr(message, "message_thread_id", None) != tid


def _bot_of(message):
    """The Bot bound to a Message, or None (test doubles have no bot)."""
    getter = getattr(message, "get_bot", None)
    if getter is None:
        return None
    try:
        return getter()
    except Exception:
        return None


def _ref_in_topic(ref) -> bool:
    """May a remembered screen still be edited in place?

    A screen outside the Study Room is retired when the flow moves into the
    topic, so editing it afterwards would rewrite the "Moved to the Study Room
    topic" notice instead of showing progress — in that case the update is sent
    as a fresh message (which `_reply` puts in the topic). Private chats have
    no topic and are always fine.
    """
    if not ref or len(ref) < 3:
        return True  # no thread recorded — keep the straightforward behaviour
    chat_id, _message_id, thread = ref
    if str(chat_id) != str(QUIZ_CHAT_ID):
        return True  # a private chat, not a topic
    return thread == _thread_id()


def _screen_ref(message) -> tuple:
    """Remember a screen message together with the topic it lives in."""
    return (
        message.chat_id,
        message.message_id,
        getattr(message, "message_thread_id", None),
    )


async def _send_to_topic(bot, text: str, **kwargs):
    """Post a message into the study-room topic; returns the sent Message.

    A topic id Telegram no longer recognises is forgotten and the send is
    retried in General, so one deleted topic cannot silence the bot. None means
    the message could not be posted at all.
    """
    try:
        sent = await bot.send_message(
            chat_id=QUIZ_CHAT_ID, text=text, **_thread_kwargs(**kwargs)
        )
    except BadRequest as exc:
        if not _is_missing_thread(exc):
            logger.warning("could not post to the topic: %s", exc)
            return None
        _forget_stale_topic()
        try:
            sent = await bot.send_message(chat_id=QUIZ_CHAT_ID, text=text, **kwargs)
        except (BadRequest, NetworkError, TimedOut, TelegramError) as retry_exc:
            logger.warning("could not post to General either: %s", retry_exc)
            return None
    except (NetworkError, TimedOut, TelegramError) as exc:
        logger.warning("could not post to the topic: %s", exc)
        return None
    _record_sent(sent)
    return sent


def _clear_state(context: ContextTypes.DEFAULT_TYPE) -> None:
    for key in (
        "state",
        "course_id",
        "del_id",
        "gen_msg",
        "pending_time",
        "aq_msg",
        "dest_flow",
        "dest_chat",
    ):
        context.user_data.pop(key, None)


# What a non-admin may press — their own quiz and the help screen. Everything
# else changes the group's material or schedules and belongs to the owner and
# admins.
_STUDENT_CALLBACKS = ("menu:squiz", "menu:how", "menu:back", "menu:cancel")


def _student_can(data: str) -> bool:
    return data in _STUDENT_CALLBACKS or data.startswith("gen:")


# --- Schedule helpers -----------------------------------------------------

_TIME_RE = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")


def _parse_time(text: str) -> str | None:
    """'7:30' / '07:30' -> '07:30'. Anything not 24-hour format -> None."""
    match = _TIME_RE.match((text or "").strip())
    if not match:
        return None
    return f"{int(match.group(1)):02d}:{match.group(2)}"


def _first_run_date(hhmm: str, now: datetime | None = None) -> str | None:
    """Start a new schedule today, or tomorrow if its time is already past."""
    now = now or datetime.now()
    return now.date().isoformat() if hhmm < now.strftime("%H:%M") else None


def _is_due(settings: dict, now: datetime) -> bool:
    """True when a schedule should fire right now.

    Enabled, its time has arrived, and it hasn't run today. That last check
    also covers a restart after the scheduled minute: the day's quiz/note is
    sent late instead of being skipped.
    """
    if not settings.get("enabled") or not settings.get("time"):
        return False
    return (
        settings["time"] <= now.strftime("%H:%M")
        and settings.get("last_run") != now.date().isoformat()
    )


def _enable_auto_quiz(hhmm: str, count: int) -> dict:
    return storage.save_auto_quiz(
        enabled=True, time=hhmm, count=count, last_run=_first_run_date(hhmm)
    )


def _enable_auto_note(hhmm: str) -> dict:
    return storage.save_auto_note(
        enabled=True, time=hhmm, last_run=_first_run_date(hhmm)
    )


def _auto_quiz_screen() -> tuple[str, InlineKeyboardMarkup]:
    """Description + current status of the daily quiz, plus the time picker."""
    settings = storage.get_auto_quiz()
    active = bool(settings["enabled"] and settings["time"])
    if active:
        status = f"✅ ON — every day at {settings['time']}, {settings['count']} questions"
    else:
        status = "⭕ OFF"
    text = (
        "⏰ Auto Quiz\n\n"
        f"Status: {status}\n\n"
        "I build a mixed quiz from all your courses and post it to "
        f"{GROUP_LABEL} every day, at the time you pick here.\n\n"
        "Choose a time below (24-hour format). ⌨️ Custom time takes a time "
        "you type yourself."
        + (" Use ❌ Turn off to stop the daily quiz.\n" if active else "\n")
    )
    return text, _time_kb("aq", can_turn_off=active)


def _auto_note_screen() -> tuple[str, InlineKeyboardMarkup]:
    """Description + current status of the daily note, plus the time picker."""
    settings = storage.get_auto_note()
    active = bool(settings["enabled"] and settings["time"])
    status = f"✅ ON — every day at {settings['time']}" if active else "⭕ OFF"
    text = (
        "📝 Auto Note\n\n"
        f"Status: {status}\n\n"
        "Every day I post one short note from a random course to "
        f"{GROUP_LABEL}, with its course, file and page, so you can re-read "
        "that part. The course changes every day — never the same one twice "
        "in a row.\n\n"
        "Choose a time below (24-hour format). ⌨️ Custom time takes a time "
        "you type yourself."
        + (" Use ❌ Turn off to stop the daily note.\n" if active else "\n")
    )
    return text, _time_kb("an", can_turn_off=active)


def _set_state(context: ContextTypes.DEFAULT_TYPE, state: str, **payload) -> None:
    context.user_data["state"] = state
    context.user_data.update(payload)


def _build_httpx() -> HTTPXRequest:
    """Generous timeouts — the connection to Telegram can be flaky.

    PTB's defaults (5s connect, 1s pool) cause TimedOut storms on unstable
    networks; this gives connections room to breathe instead.
    """
    return HTTPXRequest(
        connect_timeout=20.0,
        read_timeout=30.0,
        write_timeout=30.0,
        pool_timeout=10.0,
        media_write_timeout=60.0,
    )


async def _safe_answer(query, text: str = "", show_alert: bool = False) -> None:
    """Acknowledge a button press; a network hiccup must never break the flow.

    If this fails, Telegram shows a spinner on the button — cosmetic only.
    """
    try:
        await query.answer(text, show_alert=show_alert)
    except TelegramError as exc:
        logger.info("callback answer skipped: %s", exc)


async def _reply(message, text: str, **kwargs):
    """Reply to a message — from the group, always inside the Study Room.

    A question asked in General (or another topic) is answered in the topic, so
    everything the bot produces lives in one place; elsewhere the reply simply
    stays where it was asked. One retry so a transient timeout doesn't drop it.
    """
    if _is_outside_topic(message):
        bot = _bot_of(message)
        if bot is not None:
            sent = await _send_to_topic(bot, text, **kwargs)
            if sent is not None:
                return sent
            # the topic refused the message: fall through and answer in place
            # so the flow the user started still has something on screen
    for attempt in (1, 2):
        try:
            return await message.reply_text(text, **kwargs)
        except (TimedOut, NetworkError) as exc:
            if attempt == 1:
                await asyncio.sleep(1.5)
                continue
            logger.warning("reply failed after retry: %s", exc)
    return None


async def _edit_text_safe(
    bot,
    chat_id: int,
    message_id: int,
    text: str,
    kb: InlineKeyboardMarkup | None = None,
    **kwargs,
) -> bool:
    """Edit a specific message, retrying once on timeout. True = shown."""
    for attempt in (1, 2):
        try:
            await bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=text,
                reply_markup=kb,
                **kwargs,
            )
            return True
        except BadRequest as exc:
            if "not modified" in str(exc).lower():
                return True
            logger.warning("edit_message_text failed: %s", exc)
            return False
        except (TimedOut, NetworkError) as exc:
            if attempt == 1:
                logger.info("edit timed out, retrying once: %s", exc)
                await asyncio.sleep(1.5)
                continue
            logger.warning("edit_message_text failed after retry: %s", exc)
    return False


async def _edit(query, text: str, kb: InlineKeyboardMarkup | None = None, **kwargs) -> None:
    """Edit the message a button was pressed on; retry once, then fall back
    to sending a fresh message so the user still sees the outcome.

    A screen that lives outside the Study Room topic is not edited in place —
    the new screen goes into the topic and the old one is retired, so no flow
    keeps running in General.
    """
    msg = query.message
    if msg is None:  # very old message — best effort
        try:
            await query.edit_message_text(text, reply_markup=kb, **kwargs)
        except TelegramError as exc:
            logger.warning("edit_message_text failed: %s", exc)
        return
    if _is_outside_topic(msg):
        bot = _bot_of(msg)
        if bot is None:
            bot = query.get_bot()
        sent = await _send_to_topic(bot, text, reply_markup=kb, **kwargs)
        if sent is not None:
            await _retire(msg)
            return
        # could not reach the topic: edit the old screen so the flow continues
    if await _edit_text_safe(query.get_bot(), msg.chat_id, msg.message_id, text, kb, **kwargs):
        return
    try:
        await msg.reply_text(text, reply_markup=kb, **kwargs)
    except (TimedOut, NetworkError) as exc:
        logger.warning("fallback reply failed: %s", exc)


async def _retire(message) -> None:
    """Blank a screen left outside the topic so nobody keeps pressing it."""
    try:
        await message.edit_text("➡️ Moved to the Study Room topic.", reply_markup=None)
    except TelegramError as exc:
        logger.info("could not retire an old screen: %s", exc)


async def _send_message_safe(
    bot, text: str, attempts: int = 3, chat_id=None, **kwargs
) -> bool:
    """Send one message, retrying transient network errors.

    Used by the daily jobs, which have no chat UI to fall back to. `chat_id`
    defaults to the group topic; pass a private chat id to send it to one
    person instead. A topic id Telegram no longer knows is dropped and the
    send retried in General.
    """
    target = QUIZ_CHAT_ID if chat_id is None else chat_id
    in_group = str(target) == str(QUIZ_CHAT_ID)
    for attempt in range(attempts):
        try:
            sent = await bot.send_message(
                chat_id=target,
                text=text,
                **(_thread_kwargs(**kwargs) if in_group else kwargs),
            )
            _record_sent(sent)
            return True
        except BadRequest as exc:
            if in_group and _is_missing_thread(exc):
                _forget_stale_topic()
                try:
                    sent = await bot.send_message(chat_id=target, text=text, **kwargs)
                    _record_sent(sent)
                    return True
                except (BadRequest, TimedOut, NetworkError) as retry_exc:
                    logger.error("Message rejected even in General: %s", retry_exc)
                    return False
            logger.error("Message rejected by Telegram: %s", exc)
            return False
        except (TimedOut, NetworkError) as exc:
            if attempt == attempts - 1:
                logger.warning("Giving up on a message after network errors: %s", exc)
                return False
            await asyncio.sleep(1.0 * (attempt + 1))
    return False


async def _send_poll_with_retry(bot, kwargs: dict, attempts: int = 4, backoff: float = 1.0) -> bool:
    """Send one poll, retrying timeouts and honouring flood limits (RetryAfter)."""
    for attempt in range(attempts):
        try:
            sent = await bot.send_poll(**kwargs)
            _record_sent(sent)
            return True
        except RetryAfter as exc:
            wait = exc.retry_after
            if not isinstance(wait, int):
                wait = max(1, int(wait.total_seconds()))
            logger.info("Flood control: waiting %ss before the next poll", wait)
            await asyncio.sleep(wait + backoff)
        except BadRequest as exc:
            if _is_missing_thread(exc):
                # the topic is gone; post this quiz in General and move on
                _forget_stale_topic()
                try:
                    sent = await bot.send_poll(
                        **{k: v for k, v in kwargs.items() if k != "message_thread_id"}
                    )
                    _record_sent(sent)
                    return True
                except (BadRequest, TimedOut, NetworkError) as retry_exc:
                    logger.error(
                        "Poll rejected even in General: %s (%s)",
                        retry_exc,
                        kwargs.get("question", "")[:60],
                    )
                    return False
            logger.error("Poll rejected by Telegram: %s (%s)", exc, kwargs.get("question", "")[:60])
            return False
        except (TimedOut, NetworkError) as exc:
            if attempt == attempts - 1:
                logger.warning("Giving up on a poll after network errors: %s", exc)
                return False
            await asyncio.sleep(backoff * (attempt + 1))
    return False


# --- commands -------------------------------------------------------------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _clear_state(context)
    admin = await _is_admin(update, context)
    if not admin and not _is_private(update):
        # the group belongs to the owner and admins; members get pointed at a
        # private chat instead of being handed the menu
        await _reply(update.effective_message, MEMBER_HINT)
        return
    text, kb = _menu_screen(admin)
    await _reply(update.effective_message, text, reply_markup=kb, parse_mode="HTML")


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    admin = await _is_admin(update, context)
    if not admin and not _is_private(update):
        await _reply(update.effective_message, MEMBER_HINT)
        return
    text, kb = (HELP, _menu_kb()) if admin else (STUDENT_HELP, _student_kb())
    await _reply(update.effective_message, text, reply_markup=kb, parse_mode="HTML")


async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    had_state = bool(context.user_data.get("state"))
    _clear_state(context)
    text = "✋ Cancelled." if had_state else "Nothing to cancel."
    admin = await _is_admin(update, context)
    if not admin and not _is_private(update):
        await _reply(update.effective_message, text)
        return
    _, kb = _menu_screen(admin)
    await _reply(update.effective_message, text, reply_markup=kb)


# --- Ask (Q&A over your own notes) ----------------------------------------

# Building a BM25 index over a course takes a moment; the material only
# changes when a course is added or deleted, so the indexes are kept around.
_ASK_INDEX: list[tuple[str, Retriever]] | None = None


def _ask_indexes() -> list[tuple[str, Retriever]]:
    """One BM25 index per course, built on first use and reused after."""
    global _ASK_INDEX
    if _ASK_INDEX is None:
        indexes: list[tuple[str, Retriever]] = []
        for name, segments in _build_sources():
            try:
                index = Retriever(segments)
            except Exception:
                logger.exception("failed indexing course %s for /ask", name)
                continue
            if index:
                indexes.append((name, index))
        _ASK_INDEX = indexes
    return _ASK_INDEX


def _drop_ask_index() -> None:
    """Forget the cached indexes after the material changes."""
    global _ASK_INDEX
    _ASK_INDEX = None


def _all_segments() -> list[Segment]:
    """Every segment from every course — for hybrid LLM answering."""
    segs: list[Segment] = []
    for _, segments in _build_sources():
        segs.extend(segments)
    return segs


def _answer(query: str) -> Answer | None:
    """Hybrid answer: LLM RAG first, then BM25 fallback.

    LLM answers are grounded (overlap guard) and never hallucinated;
    rule-based is the safety net so the bot always responds.
    """
    if _AI_AVAILABLE:
        try:
            all_segs = _all_segments()
            if all_segs:
                llm_ans = _ai_answer(query, all_segs)
                if llm_ans is not None:
                    return llm_ans
        except Exception:
            logger.exception("AI answer failed, falling back to BM25")
    # Fallback: hybrid semantic+BM25 when available, else plain BM25
    if _HYBRID_AVAILABLE:
        try:
            all_segs = _all_segments()
            if all_segs:
                hr = HybridRetriever(all_segs, enable_rerank=False)
                # Try semantic-enhanced search
                from generator import _focus_terms, _GENERIC_QUERY_WORDS, _QWORDS
                hits = hr.search(query, limit=4, rerank=False)
                if hits:
                    # Grounding check: best hit must contain a required query word
                    focus = _focus_terms(query)
                    required = [w for w in focus if w.lower() not in _GENERIC_QUERY_WORDS and len(w) >= 4]
                    if required:
                        best_words = {w.lower() for w in _QWORDS.findall(hits[0][0].text)}
                        if not any(w.lower() in best_words for w in required):
                            return None
                    # Prefer definition if material defines the queried term
                    for w in focus:
                        exact = hr.define(w)
                        if exact is not None:
                            return Answer(query=query, text=exact.text, course=exact.course, filename=exact.filename, page=exact.page, is_definition=True)
                    sent, _ = hits[0]
                    # Find best course name
                    course_name = sent.course or ""
                    return Answer(query=query, text=sent.text, course=course_name, filename=sent.filename, page=sent.page)
        except Exception:
            logger.exception("Hybrid fallback failed, using BM25")
    # Last resort: best BM25 hit across courses (original behaviour)
    best: tuple[float, Answer] | None = None
    for name, index in _ask_indexes():
        hits = index.search(query, limit=4)
        if not hits:
            continue
        sent, score = hits[0]
        candidate = Answer(
            query=query,
            text=sent.text,
            course=sent.course or name,
            filename=sent.filename,
            page=sent.page,
        )
        if best is None or score > best[0]:
            best = (score, candidate)
    if best is None:
        return None
    return best[1]


async def _answer_async(query: str) -> Answer | None:
    """Async hybrid answer for the event loop."""
    if _AI_AVAILABLE:
        try:
            all_segs = _all_segments()
            if all_segs:
                ans = await _ai_answer_async(query, all_segs)
                if ans is not None:
                    return ans
        except Exception:
            logger.exception("AI async answer failed, falling back")
    return await asyncio.to_thread(_answer, query)


async def _ask(query: str, message) -> None:
    """Reply to a study question using only the saved PDFs."""
    # When the answer has to travel into the Study Room topic (the question
    # was asked in General or another topic) the question rides along with it,
    # otherwise nobody in the topic would know what is being answered.
    prefix = f"❓ <i>{escape(query)}</i>\n\n" if _is_outside_topic(message) else ""

    async def say(text: str, **extra):
        if prefix:
            extra.setdefault("parse_mode", "HTML")
        return await _reply(message, prefix + text, **extra)

    if not query.strip():
        await _reply(
            message,
            "🔎 Ask me a question about your notes, e.g. "
            "<i>/ask what is a weak entity</i>",
            parse_mode="HTML",
        )
        return
    # Building the BM25 indexes takes a moment, and this handler runs on the
    # event loop that also has to deliver every other update, so both the
    # indexing and the search go to a worker thread.
    if not await asyncio.to_thread(_ask_indexes):
        await say(
            "📭 I have no readable material yet. Add a course with PDFs "
            "first (/start → Add Course), then ask again."
        )
        return

    # Hybrid: LLM RAG first, BM25 fallback (both run off the event loop)
    answer = await _answer_async(query) if _AI_AVAILABLE else await asyncio.to_thread(_answer, query)
    if answer is None:
        await say(
            "🤷 I couldn't find that in your notes. I only answer from the "
            "PDFs you've added — try the exact wording the slides use, or "
            "check the course is saved."
        )
        return
    # in a private chat, offer the next step right under the answer
    markup = _student_kb() if _message_is_private(message) else None
    await say(answer.render(), parse_mode="HTML", reply_markup=markup)


async def cmd_ask(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """`/ask <question>` — answered straight from the saved material."""
    if not await _is_admin(update, context) and not _is_private(update):
        await _reply(update.effective_message, MEMBER_HINT)
        return
    query = " ".join(context.args or []).strip()
    if not query:
        query = (update.effective_message.text or "").removeprefix("/ask").strip()
    await _ask(query, update.effective_message)


async def cmd_topic(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """`/topic` run *inside* the study-room topic teaches the bot its id.

    Telegram gives a topic a numeric id that every send needs, and there is no
    API to look one up by name. Sending /topic from inside the topic is the
    simplest way to teach it, so the bot learns from wherever the command came
    from.
    """
    message = update.effective_message
    if not await _is_admin(update, context):
        await _reply(message, ADMIN_ONLY_HINT)
        return
    thread = getattr(message, "message_thread_id", None)
    if thread is None:
        current = _thread_id()
        await _reply(
            message,
            f"Send /topic from inside the <b>{QUIZ_TOPIC_NAME}</b> topic so I "
            "can learn its id.\n\n"
            + (
                f"Right now I'm posting to id <code>{current}</code> — send "
                "/topic from the right topic to change it."
                if current is not None
                else f"Or put <code>QUIZ_TOPIC_ID=&lt;id&gt;</code> in .env."
            ),
            parse_mode="HTML",
        )
        return
    storage.save_topic(QUIZ_TOPIC_NAME, int(thread))
    await _reply(
        message,
        f"✅ Noted — everything I post (quizzes, notes and answers) goes in "
        f"<b>{QUIZ_TOPIC_NAME}</b> from now on.",
        parse_mode="HTML",
    )


async def _delete_group_message(bot, message_id: int) -> tuple[str, str]:
    """Delete one of the bot's own messages from the group.

    Returns (result, reason): "ok" (deleted), "gone" (not there, not ours, or
    older than Telegram's 48-hour window), "retry" (network/flood — worth
    another run) or "fail" (refused for some other reason). `reason` carries
    Telegram's own words so the user can be told what actually happened.
    """
    for attempt in (1, 2):
        try:
            await bot.delete_message(_group_chat_id(), message_id)
            return "ok", ""
        except RetryAfter as exc:
            wait = exc.retry_after
            if not isinstance(wait, int):
                wait = max(1, int(wait.total_seconds()))
            if attempt == 1:
                await asyncio.sleep(wait + 0.5)
                continue
            return "retry", str(exc)
        except BadRequest as exc:
            reason = str(exc)
            if any(
                needle in reason.lower()
                for needle in ("not found", "48 hours", "too old", "can't be deleted", "cannot be deleted")
            ):
                return "gone", reason
            return "fail", reason
        except (NetworkError, TimedOut, TelegramError) as exc:
            return "retry", str(exc)
    return "retry", ""


# A bounded sweep, e.g. "/cleanup 1234-1290" or "/cleanup from 1234 to 1290".
_SWEEP_MAX = 300
_SWEEP_RE = re.compile(
    r"^(?:from\s+)?(\d+)\s*(?:-|\.\.|to|until|and)\s*(\d+)$",
    re.IGNORECASE,
)


async def _may_delete_others(bot) -> bool:
    """Could this bot delete a message it did not send?

    Telegram always lets a bot delete its own messages, but deleting anybody
    else's needs more: the admin right in a supergroup, or plain being an
    administrator of a basic group. A sweep only ever means "my own messages",
    so when the bot could do more than that it is not used. Unknown counts as
    "could" — the reply form still works.
    """
    try:
        chat = await bot.get_chat(_group_chat_id())
        me = await bot.get_me()
        member = await bot.get_chat_member(_group_chat_id(), me.id)
    except Exception as exc:  # no/unreachable API: assume the worse
        logger.info("could not check the bot's own rights: %s", exc)
        return True
    if getattr(member, "status", "") not in ("administrator", "creator"):
        return False
    if getattr(chat, "type", "") == "group":
        return True
    return bool(getattr(member, "can_delete_messages", False))


async def _sweep_range(bot, low: int, high: int) -> tuple[int, int, list[str]]:
    """Delete the bot's own messages between two message ids.

    Telegram refuses to delete anything the bot did not send, so a sweep can
    only ever remove the bot's own messages — never anybody else's. Ids the
    bot knows it posted inside the Study Room are skipped, so the current
    quizzes and notes are protected.
    """
    tid = _thread_id()
    protected = {e["id"] for e in storage.load_outbox() if e.get("thread") == tid}
    deleted = skipped = 0
    problems: list[str] = []
    for message_id in range(low, high + 1):
        if message_id in protected:
            skipped += 1
            continue
        result, reason = await _delete_group_message(bot, message_id)
        if result == "ok":
            deleted += 1
        else:
            skipped += 1
            if reason and result != "gone" and reason not in problems:
                problems.append(reason)
        await asyncio.sleep(0.05)  # stay under Telegram's per-chat rate limit
    return deleted, skipped, problems


async def cmd_cleanup(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """`/cleanup` — remove the bot's messages from outside the Study Room.

    Three ways to use it:
      * `/cleanup` — clears the messages the bot remembers sending outside the
        topic (storage.record_sent logs every group send with its topic);
      * `/cleanup 1234-1290` — sweeps a range of message ids, the only way to
        reach messages sent before that log existed;
      * replied to one of the bot's messages — deletes just that message.

    Telegram refuses to delete anything the bot did not send, so none of these
    can ever remove somebody else's message.
    """
    message = update.effective_message
    if not await _is_admin(update, context):
        await _reply(message, ADMIN_ONLY_HINT)
        return

    # /cleanup as a reply deletes exactly the message being replied to. When
    # the writer is known and it is not the bot, refuse outright — a bot with
    # the admin right could otherwise delete a student's message. When it
    # cannot be told, try anyway: Telegram refuses anything the bot did not
    # send, and its own words are shown if it does.
    replied = getattr(message, "reply_to_message", None)
    if replied is not None:
        replied_from = getattr(getattr(replied, "from_user", None), "id", None)
        bot_user_id = _bot_id(update)
        if (
            replied_from is not None
            and bot_user_id is not None
            and replied_from != bot_user_id
        ):
            await _reply(message, "That message isn't mine — I only delete my own.")
            return
        try:
            await context.bot.delete_message(replied.chat_id, replied.message_id)
        except TelegramError as exc:
            await _reply(
                message,
                f"😕 Telegram would not delete that message:\n<i>{escape(str(exc))}</i>\n\n"
                "That is what Telegram says when the message is not mine, is "
                "older than 48 hours, or I am not allowed to delete it there.",
                parse_mode="HTML",
            )
            return
        storage.forget_sent([replied.message_id])
        await _reply(message, "🗑")
        return

    # /cleanup 1234-1290 sweeps an id range — the only way to reach messages
    # sent before I started keeping track of them.
    args = " ".join(context.args or []).strip()
    if args:
        match = _SWEEP_RE.match(args)
        if not match:
            await _reply(
                message,
                "Usage: <code>/cleanup 1234-1290</code>\n"
                "Open any message in web.telegram.org — the number at the end "
                "of the link is its id.",
                parse_mode="HTML",
            )
            return
        low, high = sorted((int(match.group(1)), int(match.group(2))))
        if high - low + 1 > _SWEEP_MAX:
            await _reply(
                message,
                f"That's {high - low + 1} messages — sweep at most "
                f"{_SWEEP_MAX} at a time.",
            )
            return
        if await _may_delete_others(context.bot):
            await _reply(
                message,
                "⚠️ A sweep would be able to remove other people's messages "
                "too, so I won't run one here — a sweep must mean my own "
                "messages only.\n\n"
                "Reply to each of my messages and send /cleanup, and I'll "
                "delete it one by one.",
            )
            return
        await _reply(message, f"🔍 Checking messages {low}–{high}…")
        deleted, skipped, problems = await _sweep_range(context.bot, low, high)
        lines = [
            f"🗑 Deleted {deleted} of my messages in {low}–{high}."
            + (f" ({skipped} ids weren't mine, or are in the topic.)" if skipped else "")
        ]
        for problem in problems[:3]:
            lines.append(f"⚠️ {problem[:200]}")
        await _reply(message, "\n".join(lines))
        return

    tid = _thread_id()
    if tid is None:
        await _reply(
            message,
            f"I don't know the {QUIZ_TOPIC_NAME} topic yet, so I can't tell "
            "which of my messages are outside it. Send /topic inside the "
            "topic, then run /cleanup again.",
        )
        return

    outside = [e for e in storage.load_outbox() if e.get("thread") != tid]
    if not outside:
        await _reply(
            message,
            f"✅ Nothing I remember is outside the {QUIZ_TOPIC_NAME} topic.\n\n"
            "Older messages I sent before I started keeping track can still go:\n"
            "• reply to one of them and send /cleanup again, or\n"
            "• send <code>/cleanup 1234-1290</code> with the id range "
            "(web.telegram.org shows the id at the end of a message's link).",
            parse_mode="HTML",
        )
        return

    deleted = gone = retry = 0
    finished: list[int] = []
    for entry in outside:
        result, _reason = await _delete_group_message(context.bot, entry["id"])
        if result == "ok":
            deleted += 1
            finished.append(entry["id"])
        elif result == "gone":
            gone += 1
            finished.append(entry["id"])
        else:
            retry += 1
            continue
        await asyncio.sleep(0.06)  # stay under Telegram's per-chat rate limit
    storage.forget_sent(finished)

    lines = []
    if deleted:
        lines.append(
            f"🗑 Deleted {deleted} message{'s' if deleted != 1 else ''} from "
            f"outside the {QUIZ_TOPIC_NAME} topic."
        )
    if gone:
        lines.append(
            f"ℹ️ {gone} more were already gone or older than Telegram's "
            "48-hour limit."
        )
    if retry:
        lines.append(f"⚠️ {retry} could not be deleted right now — run /cleanup again shortly.")
    if not lines:
        lines.append(f"✅ Nothing to clean up outside {QUIZ_TOPIC_NAME}.")
    await _reply(message, "\n".join(lines))


# --- Add Course flow ------------------------------------------------------

async def _finish_add(query, context: ContextTypes.DEFAULT_TYPE) -> None:
    if (
        context.user_data.get("state") != ADD_FILES
        or not context.user_data.get("course_id")
    ):
        await _safe_answer(query, "Nothing to finish — use the menu below.", show_alert=True)
        return
    course_id = context.user_data["course_id"]
    course = storage.get_course(course_id)
    if course is None or not course.files:
        await _safe_answer(query, "Send at least one PDF first ✏️", show_alert=True)
        return

    await _safe_answer(query, "Saving…")
    try:
        storage.finalize_course(course_id)
        _drop_ask_index()
    except InsufficientTextError as exc:
        storage.delete_course(course_id)
        _clear_state(context)
        await _edit(
            query,
            f"😕 Couldn't read that material: {exc}\n\n"
            "The course was removed — send PDFs with selectable text and "
            "try again.",
            _menu_kb(),
        )
        return

    n = len(course.files)
    _clear_state(context)
    await _edit(
        query,
        f"✅ Course '{course.name}' saved — {n} PDF{'s' if n != 1 else ''}.\n\n"
        "Press Generate Quiz to post a mixed quiz to "
        f"{GROUP_LABEL}.",
        _menu_kb(),
    )


async def _finish_exam(query, context: ContextTypes.DEFAULT_TYPE, *, fallback=None) -> None:
    """Validate the uploaded paper(s) and save the exam."""
    exam_id = context.user_data.get("exam_id")
    exam = storage.get_exam(exam_id) if exam_id else None
    if exam is None or not exam.files:
        await _reply(fallback, "Send at least one exam PDF first ✏️")
        return

    await _safe_answer(query, "Reading the paper…") if query is not None else None
    try:
        storage.finalize_exam(exam.id)
        _drop_ask_index()
    except InsufficientTextError as exc:
        storage.delete_exam(exam.id)
        _clear_state(context)
        message = (
            f"😕 Couldn't read that exam: {exc}\n\n"
            "It was not saved — send a text PDF (with selectable text) and "
            "try again."
        )
        if query is not None:
            await _edit(query, message, _exams_kb())
        elif fallback is not None:
            await _reply(fallback, message, reply_markup=_exams_kb())
        return

    text, kb = _exams_screen()
    _clear_state(context)
    message = (
        f"✅ Exam '{exam.name}' saved.\n\n"
        "Its questions will now show up in your quizzes.\n\n" + text
    )
    if query is not None:
        await _edit(query, message, kb, parse_mode="HTML")
    elif fallback is not None:
        await _reply(fallback, message, reply_markup=kb, parse_mode="HTML")


# --- Generate Quiz --------------------------------------------------------

def _build_sources() -> list[tuple[str, list]]:
    """Every course that still has readable text, ready for quiz mixing."""
    sources: list[tuple[str, list]] = []
    for c in storage.list_courses():
        try:
            segments = storage.course_segments(c.id)
        except Exception:
            logger.exception("failed reading course %s", c.name)
            continue
        if segments:
            sources.append((c.name, segments))
    return sources


def _build_exam_sources() -> list[tuple[str, list]]:
    """Every uploaded past exam that still has readable text."""
    sources: list[tuple[str, list]] = []
    for e in storage.list_exams():
        try:
            segments = storage.exam_segments(e.id)
        except Exception:
            logger.exception("failed reading exam %s", e.name)
            continue
        if segments:
            sources.append((e.name, segments))
    return sources


def _build_quiz(count: int, seed: int | None = None):
    """Hybrid mixed quiz: LLM-powered when configured, rule-based fallback.

    LLM generates Bloom-aware questions (scenario, comparison, cause-effect) from
    the material; exams still contribute EXAM_SHARE verbatim. If LLM is absent
    or fails validation, the original engine fills the quiz.
    """
    sources = _build_sources()
    exam_sources = _build_exam_sources()
    if not sources and not exam_sources:
        raise InsufficientTextError("No course material to build a quiz from.")

    rng = random.Random(seed)
    exam_quota = int(round(count * EXAM_SHARE)) if exam_sources else 0
    note_sources = sources or exam_sources

    from_exams: list = []
    if exam_quota:
        note_pool = [s for _, segs in sources for s in segs]
        from_exams = questions_from_exams(
            exam_sources, exam_quota, note_pool or None, rng
        )

    note_quota = max(1, count - len(from_exams))

    # Try LLM for the notes portion
    from_notes: list = []
    if _AI_AVAILABLE and sources:
        try:
            # collect all note segments
            all_note_segs: list[Segment] = []
            for _, segs in sources:
                all_note_segs.extend(segs)
            if all_note_segs:
                llm_qs = _ai_questions(all_note_segs, note_quota, seed)
                if llm_qs:
                    from_notes = llm_qs
        except Exception:
            logger.exception("AI quiz generation failed, falling back to rule-based")
            from_notes = []

    # Fallback / supplement with rule-based if LLM gave too few
    if len(from_notes) < note_quota:
        remaining = note_quota - len(from_notes)
        try:
            rb_qs, _skipped = generate_mixed_questions(note_sources, remaining, seed)
            # dedup against LLM questions
            seen = {__import__("re").sub(r"\W+", "", q.text.lower())[:60] for q in from_notes}
            for q in rb_qs:
                key = __import__("re").sub(r"\W+", "", q.text.lower())[:60]
                if key not in seen:
                    from_notes.append(q)
                    seen.add(key)
        except InsufficientTextError:
            if not from_notes and not from_exams:
                raise

    questions = from_notes + from_exams
    if not questions:
        raise InsufficientTextError(
            "None of the courses had enough readable text to build a quiz."
        )
    rng.shuffle(questions)
    # hard cap
    return questions[:count]


async def _post_questions(bot, questions, chat_id=None) -> int:
    """Post quiz polls; returns how many made it through.

    `chat_id` is None for the group topic, or a private chat id when the quiz
    was requested for one person.
    """
    target = QUIZ_CHAT_ID if chat_id is None else chat_id
    in_group = str(target) == str(QUIZ_CHAT_ID)
    sent = 0
    for q in questions:
        kwargs: dict = {
            "chat_id": target,
            "question": q.text,
            "options": q.options,
            "type": PollType.QUIZ,
            "correct_option_id": q.correct_index,
            "is_anonymous": False,
        }
        if q.explanation:
            kwargs["explanation"] = q.explanation
        if in_group:
            kwargs = _thread_kwargs(**kwargs)
        if await _send_poll_with_retry(bot, kwargs):
            sent += 1
        await asyncio.sleep(0.4)  # stay under Telegram's per-chat rate limit
    return sent


async def _run_quiz(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    count: int,
    *,
    edit_query=None,
    msg_ref: tuple | None = None,
    fallback=None,
    dest_chat: int | None = None,
) -> None:
    """Build the mixed quiz and post it.

    Progress/result text is shown by editing the pressed button's message
    (`edit_query`) or the stored prompt message (`msg_ref` — used when the
    quantity was typed), falling back to a fresh reply when edits fail.

    `dest_chat` decides where the polls go: None is the group topic, an int is
    one private chat (an admin who asked for a quiz here, or a student).
    """
    admin = await _is_admin(update, context)
    screen_kb = _menu_screen(admin)[1]
    if not admin:
        # a student never gets the quiz posted in the group
        dest_chat = _chat_id(update)
    in_group = dest_chat is None
    where = GROUP_LABEL if in_group else "this chat"

    async def show(text: str, kb: InlineKeyboardMarkup | None = None) -> None:
        if edit_query is not None:
            await _edit(edit_query, text, kb)
            return
        if msg_ref is not None and _ref_in_topic(msg_ref):
            chat_id, message_id = msg_ref[:2]
            if await _edit_text_safe(context.bot, chat_id, message_id, text, kb):
                return
        if fallback is not None:
            await _reply(fallback, text, reply_markup=kb)

    courses = storage.list_courses()
    if not courses and not storage.list_exams():
        if edit_query is not None:
            await _safe_answer(edit_query, "Add a course first!", show_alert=True)
        elif fallback is not None:
            await _reply(fallback, "Add a course first!")
        return

    await show(
        f"⏳ Building a mixed quiz from {len(courses)} course(s), "
        f"{count} questions…"
    )

    sources = _build_sources()
    if not sources and not _build_exam_sources():
        await show(
            "😕 No readable text found in any course. Re-add the PDFs "
            "(they need selectable text).",
            screen_kb,
        )
        return

    try:
        questions = await asyncio.to_thread(_build_quiz, count, None)
    except InsufficientTextError as exc:
        await show(f"😕 {exc}", screen_kb)
        return

    sent = await _post_questions(context.bot, questions, chat_id=dest_chat)

    names = ", ".join(c.name for c in courses) or "your saved exams"
    if sent:
        exam_qs = sum(1 for q in questions if q.subtype == "exam")
        lines = [
            f"✅ Sent {sent}/{len(questions)} mixed questions from "
            f"{names} to {where} 🎯"
        ]
        if exam_qs:
            lines.append(f"📚 {exam_qs} from your saved exams")
        await show("\n".join(lines), screen_kb)
    else:
        if in_group:
            await show(
                f"❌ Couldn't post to {GROUP_LABEL}. Check QUIZ_CHAT_ID in .env, "
                "that the bot is still an admin, and that the "
                f"'{QUIZ_TOPIC_NAME}' topic exists (send /topic inside it).",
                screen_kb,
            )
        else:
            await show(
                "❌ Couldn't send the quiz to this chat — press /start and "
                "try again.",
                screen_kb,
            )


# --- Generate Note --------------------------------------------------------

def _pick_note(exclude_course_id: int | None = None) -> tuple[storage.Course | None, Note | None]:
    """Pick a random course and build one note from it.

    The course used last time is skipped whenever there is an alternative, so
    the daily note never repeats a subject two days in a row. A course without
    readable text is passed over instead of failing the whole note.
    """
    courses = storage.list_courses()
    if not courses:
        return None, None
    pool = [c for c in courses if c.id != exclude_course_id] or courses
    rng = random.Random()
    rng.shuffle(pool)
    for course in pool:
        try:
            segs = storage.course_segments(course.id)
            if _AI_AVAILABLE:
                try:
                    note = _ai_note(segs, seed=rng.randrange(1 << 30))
                    return course, note
                except Exception:
                    logger.exception("AI note failed for %s, falling back", course.name)
            return course, generate_note(segs, seed=rng.randrange(1 << 30))
        except Exception:
            logger.exception("failed building a note from %s", course.name)
            continue
    return None, None


async def _run_note(
    context: ContextTypes.DEFAULT_TYPE, *, edit_query, dest_chat: int | None = None
) -> None:
    """Post one note from a random course (manual test).

    The note itself goes to the group topic — or to `dest_chat` when an admin
    asked for it in a private chat — exactly like the daily one; the button
    message just confirms what was sent and where.
    """
    in_group = dest_chat is None
    where = GROUP_LABEL if in_group else "this chat"
    if not storage.list_courses():
        await _safe_answer(edit_query, "Add a course first!", show_alert=True)
        return

    await _edit(edit_query, "⏳ Picking a note from a random course…")
    course, note = _pick_note()
    if note is None:
        await _edit(
            edit_query,
            "😕 Couldn't find a usable note. Re-add the PDFs (they need "
            "selectable text).",
            _menu_kb(),
        )
        return
    if not await _send_message_safe(
        context.bot, note.render("Note"), parse_mode="HTML", chat_id=dest_chat
    ):
        if in_group:
            await _edit(
                edit_query,
                f"❌ Couldn't post to {GROUP_LABEL}. Check QUIZ_CHAT_ID in .env, "
                "that the bot is still an admin, and that the "
                f"'{QUIZ_TOPIC_NAME}' topic exists (send /topic inside it).",
                _menu_kb(),
            )
        else:
            await _edit(
                edit_query,
                "❌ Couldn't send the note to this chat — press /start and "
                "try again.",
                _menu_kb(),
            )
        return
    logger.info("note from %s posted to %s", course.name, where)
    await _edit(
        edit_query,
        f"✅ Posted a note from {course.name} to {where} 📖",
        _menu_kb(),
    )


# --- Daily automations ---------------------------------------------------

async def _auto_quiz_job(app: Application) -> None:
    """Post the day's mixed quiz in the group (no chat UI involved)."""
    settings = storage.get_auto_quiz()
    count = int(settings.get("count") or DEFAULT_QUESTION_COUNT)

    sources = _build_sources()
    if not sources:
        logger.warning("Auto quiz skipped: no readable course material.")
        return

    try:
        questions = await asyncio.to_thread(_build_quiz, count, None)
    except InsufficientTextError as exc:
        logger.warning("Auto quiz skipped: %s", exc)
        return

    names = ", ".join(label for label, _ in sources)
    await _send_message_safe(
        app.bot, f"📝 Daily Quiz — {len(questions)} questions from {names}."
    )
    sent = await _post_questions(app.bot, questions)
    logger.info("Auto quiz sent %d/%d questions to %s", sent, len(questions), GROUP_LABEL)


async def _auto_note_job(app: Application) -> None:
    """Post the day's note in the group, from a course used recently."""
    course, note = _pick_note(exclude_course_id=storage.get_note_last_course())
    if note is None:
        logger.warning("Auto note skipped: no readable course material.")
        return

    # remember the course first: if the send fails, tomorrow rotates anyway
    storage.set_note_last_course(course.id)
    if await _send_message_safe(app.bot, note.render("Daily Note"), parse_mode="HTML"):
        logger.info("Auto note sent from %s", course.name)


async def _run_due(app: Application) -> None:
    now = datetime.now()
    today = now.date().isoformat()
    if _is_due(storage.get_auto_quiz(), now):
        # mark before sending so a failure can't post it twice
        storage.save_auto_quiz(last_run=today)
        await _auto_quiz_job(app)
    if _is_due(storage.get_auto_note(), now):
        storage.save_auto_note(last_run=today)
        await _auto_note_job(app)


async def _scheduler(app: Application) -> None:
    """Check the saved schedules every TICK_SECONDS and run what is due.

    Reading the schedules on every tick (instead of locking in timers) means a
    time changed in the menu takes effect at once, a restart does not drop
    the day, and turning something off stops it immediately.
    """
    while True:
        try:
            await _run_due(app)
        except Exception:
            logger.exception("scheduler tick failed")
        await asyncio.sleep(TICK_SECONDS)


# --- button router --------------------------------------------------------

async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    data = query.data or ""
    admin = await _is_admin(update, context)

    if not admin and not _is_private(update):
        # the group's controls belong to its owner and admins
        await _safe_answer(query, MEMBER_HINT, show_alert=True)
        return
    if not admin and not _student_can(data):
        await _safe_answer(query, ADMIN_ONLY_HINT, show_alert=True)
        return

    if data == "menu:add":
        _set_state(context, ADD_NAME)
        await _safe_answer(query)
        await _edit(
            query,
            "What's the course name?\n\nJust type it below (or press /cancel).",
        )

    elif data == "menu:del":
        courses = storage.list_courses()
        if not courses:
            await _safe_answer(query, "No courses yet — press 'Add Course' first.", show_alert=True)
            return
        _set_state(context, DEL_SELECT)
        await _safe_answer(query)
        await _edit(
            query,
            "Which course do you want to delete?",
            _course_list_kb(courses),
        )

    elif data == "menu:gen":
        courses = storage.list_courses()
        if not courses and not storage.list_exams():
            await _safe_answer(query, "Add a course first!", show_alert=True)
            return
        if admin and _is_private(update):
            # asked for in a private chat: post it where? (in the group the
            # answer is obvious, so no question is needed)
            _set_state(context, GEN_DEST, dest_flow="quiz")
            await _safe_answer(query)
            await _edit(query, "Where should I post the quiz?", _dest_kb())
            return
        _set_state(context, GEN_COUNT)
        if query.message:
            context.user_data["gen_msg"] = _screen_ref(query.message)
        await _safe_answer(query)
        await _edit(
            query,
            "How many questions should the quiz have?\n\n"
            f"Tap a preset below, or type any number from 1 to {MAX_QUESTIONS}.",
            _qty_kb(),
        )

    elif data.startswith("gen:") and data[4:].isdigit():
        count = int(data[4:])
        if not 1 <= count <= MAX_QUESTIONS:
            await _safe_answer(query, f"Pick 1-{MAX_QUESTIONS}.", show_alert=True)
            return
        dest_chat = context.user_data.get("dest_chat")
        _clear_state(context)
        await _safe_answer(query)
        await _run_quiz(update, context, count, edit_query=query, dest_chat=dest_chat)

    elif data in ("dest:group", "dest:private"):
        flow = context.user_data.get("dest_flow")
        dest_chat = _chat_id(update) if data == "dest:private" else None
        _clear_state(context)
        await _safe_answer(query)
        if flow == "note":
            await _run_note(context, edit_query=query, dest_chat=dest_chat)
            return
        if dest_chat is not None:
            context.user_data["dest_chat"] = dest_chat
        _set_state(context, GEN_COUNT)
        if query.message:
            context.user_data["gen_msg"] = _screen_ref(query.message)
        await _edit(
            query,
            "How many questions should the quiz have?\n\n"
            f"Tap a preset below, or type any number from 1 to {MAX_QUESTIONS}.",
            _qty_kb(),
        )

    elif data == "menu:squiz":
        dest_chat = _chat_id(update)
        if dest_chat is None:
            await _safe_answer(query, "Press /start first.", show_alert=True)
            return
        _clear_state(context)
        context.user_data["dest_chat"] = dest_chat
        _set_state(context, GEN_COUNT)
        if query.message:
            context.user_data["gen_msg"] = _screen_ref(query.message)
        await _safe_answer(query)
        await _edit(
            query,
            "How many questions should your quiz have?\n\n"
            f"Tap a preset below, or type any number from 1 to {MAX_QUESTIONS}.",
            _qty_kb(),
        )

    elif data == "menu:how":
        await _safe_answer(query)
        await _edit(query, STUDENT_HELP, _student_kb(), parse_mode="HTML")

    elif data == "menu:genote":
        _clear_state(context)
        await _safe_answer(query)
        if admin and _is_private(update):
            _set_state(context, GEN_DEST, dest_flow="note")
            await _edit(query, "Where should I post the note?", _dest_kb())
            return
        await _run_note(context, edit_query=query)

    elif data == "menu:aq":
        _clear_state(context)
        await _safe_answer(query)
        text, kb = _auto_quiz_screen()
        await _edit(query, text, kb)

    elif data == "menu:an":
        _clear_state(context)
        await _safe_answer(query)
        text, kb = _auto_note_screen()
        await _edit(query, text, kb)

    elif data.startswith("aq:time:") or data.startswith("an:time:"):
        prefix, rest = data.split(":", 1)   # "aq" | "time:08:00"
        hhmm = rest.split(":", 1)[1]        # "08:00"
        await _safe_answer(query)
        if prefix == "an":
            settings = _enable_auto_note(hhmm)
            _clear_state(context)
            await _edit(
                query,
                f"✅ Auto Note is ON — every day at {settings['time']} I'll "
                f"post a note from a random course to {GROUP_LABEL}.\n\n"
                "The course changes every day, so no two notes in a row come "
                "from the same one.",
                _menu_kb(),
            )
            return
        if query.message:
            context.user_data["aq_msg"] = _screen_ref(query.message)
        context.user_data["pending_time"] = hhmm
        _set_state(context, AUTO_QUIZ_COUNT)
        await _edit(
            query,
            f"🕐 {hhmm} it is. How many questions should each daily quiz "
            "have?\n\nTap a preset below, or type any number from 1 to "
            f"{MAX_QUESTIONS}.",
            _auto_qty_kb(),
        )

    elif data in ("aq:custom", "an:custom"):
        prefix = data.split(":")[0]
        if query.message:
            context.user_data["aq_msg"] = _screen_ref(query.message)
        _set_state(context, AUTO_QUIZ_TIME if prefix == "aq" else AUTO_NOTE_TIME)
        await _safe_answer(query)
        await _edit(
            query,
            "🕐 Send the time in 24-hour format — for example 07:30 or "
            "19:05 (00:00 to 23:59).",
            _back_kb(),
        )

    elif data in ("aq:off", "an:off"):
        if data.startswith("aq:"):
            storage.save_auto_quiz(enabled=False)
            _clear_state(context)
            await _safe_answer(query, "Auto quiz turned off")
            text, kb = _auto_quiz_screen()
            await _edit(query, "❌ Auto Quiz turned off.\n\n" + text, kb)
        else:
            storage.save_auto_note(enabled=False)
            _clear_state(context)
            await _safe_answer(query, "Auto note turned off")
            text, kb = _auto_note_screen()
            await _edit(query, "❌ Auto Note turned off.\n\n" + text, kb)

    elif data.startswith("aq:count:") and data[9:].isdigit():
        count = int(data[9:])
        hhmm = context.user_data.get("pending_time")
        if not hhmm:
            await _safe_answer(query, "Pick a time first ⏰", show_alert=True)
            return
        if not 1 <= count <= MAX_QUESTIONS:
            await _safe_answer(query, f"Pick 1-{MAX_QUESTIONS}.", show_alert=True)
            return
        _clear_state(context)
        await _safe_answer(query)
        settings = _enable_auto_quiz(hhmm, count)
        await _edit(
            query,
            f"✅ Auto Quiz is ON — every day at {settings['time']} I'll post "
            f"{count} mixed questions to {GROUP_LABEL}.",
            _menu_kb(),
        )

    elif data in ("menu:back",):
        _clear_state(context)
        await _safe_answer(query)
        text, kb = _menu_screen(admin)
        await _edit(query, text, kb, parse_mode="HTML")

    elif data == "menu:cancel":
        _clear_state(context)
        await _safe_answer(query, "Cancelled.")
        text, kb = _menu_screen(admin)
        await _edit(query, text, kb, parse_mode="HTML")

    elif data == "add:done":
        if context.user_data.get("state") == EXAM_FILES:
            await _finish_exam(query, context)
        else:
            await _finish_add(query, context)

    elif data == "menu:exams":
        _clear_state(context)
        text, kb = _exams_screen()
        await _edit(query, text, kb, parse_mode="HTML")

    elif data == "ex:upload":
        _set_state(context, EXAM_NAME)
        await _safe_answer(query)
        await _edit(
            query,
            "📝 <b>New exam</b>\n\n"
            "What is this paper called? e.g. <i>Database Midterm 2024</i>",
            parse_mode="HTML",
        )

    elif data.startswith("ex:del:") and data[6:].isdigit():
        exam = storage.delete_exam(int(data[6:]))
        _drop_ask_index()
        name = exam.name if exam else "Exam"
        text, kb = _exams_screen()
        await _safe_answer(query)
        await _edit(query, f"🗑️ '{name}' deleted.\n\n{text}", kb, parse_mode="HTML")

    elif data.startswith("del:") and data[4:].isdigit():
        course = storage.get_course(int(data[4:]))
        if course is None:
            await _safe_answer(query, "That course no longer exists.", show_alert=True)
            courses = storage.list_courses()
            if courses:
                await _edit(query, "Which course do you want to delete?", _course_list_kb(courses))
            else:
                _clear_state(context)
                await _edit(query, GREETING, _menu_kb(), parse_mode="HTML")
            return
        context.user_data["del_id"] = course.id
        _set_state(context, DEL_CONFIRM)
        await _safe_answer(query)
        await _edit(
            query,
            f"Are you sure you want to delete '{course.name}' and all its PDFs?",
            _confirm_kb(),
        )

    elif data == "del:yes":
        course_id = context.user_data.get("del_id")
        course = storage.delete_course(course_id) if course_id else None
        _drop_ask_index()
        _clear_state(context)
        await _safe_answer(query)
        name = course.name if course else "Course"
        await _edit(query, f"🗑️ '{name}' deleted.", _menu_kb())

    elif data == "del:cancel":
        courses = storage.list_courses()
        _set_state(context, DEL_SELECT)
        await _safe_answer(query, "Cancelled.")
        if courses:
            await _edit(query, "Which course do you want to delete?", _course_list_kb(courses))
        else:
            _clear_state(context)
            await _edit(query, GREETING, _menu_kb(), parse_mode="HTML")

    else:
        await _safe_answer(query, "Unknown action — use /start.", show_alert=True)


# --- message handlers -----------------------------------------------------

def _wants_answer(update: Update, message) -> bool:
    """Should this plain message be treated as a study question?

    Answering every message in the group would flood it, so the bot only
    listens in a private chat, when the message calls it by name, or when it
    replies to one of the bot's own messages. (Telegram's privacy mode already
    hides most group messages from bots; this is the second half of that guard.)
    """
    if update.effective_chat.type == "private":
        return True
    text = message.text or ""
    if _WAKE_WORDS.search(text):
        return True
    replied = getattr(message, "reply_to_message", None)
    return bool(
        replied
        and getattr(replied, "from_user", None)
        and replied.from_user.id == _bot_id(update)
    )


def _question_from_message(update: Update, message) -> str:
    """The question a plain message is asking, or '' when it isn't asking.

    "baymax what is 2NF" -> "what is 2NF"; the wake word and the bot's own
    mention at the end ("... thanks bot") are dropped so they don't pollute the
    search.
    """
    text = (message.text or "").strip()
    if not text or not _wants_answer(update, message):
        return ""
    # a wake word is an explicit call; without one we need a real question
    if _WAKE_WORDS.search(text):
        text = _WAKE_WORDS.sub(" ", text)
    # "thanks bot" / "please bot" trailing the question
    text = re.sub(r"\b(?:thanks|thank you|pls|please)\b\s*(?:bot|baymax)?\s*$",
                  "", text, flags=re.IGNORECASE)
    return text.strip().lstrip(".,;: ")


def _bot_id(update: Update) -> int | None:
    bot = getattr(update, "bot", None)
    return getattr(getattr(bot, "bot", None), "id", None)


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    state = context.user_data.get("state")
    message = update.effective_message
    admin = await _is_admin(update, context)

    if not admin and not _is_private(update):
        # the group is the owner's and admins' bot; members are sent to a
        # private chat for their questions and quizzes
        if state:
            _clear_state(context)
        if _question_from_message(update, message):
            await _reply(message, MEMBER_HINT)
        return

    if state in ADMIN_STATES and not admin:
        _clear_state(context)
        await _reply(message, ADMIN_ONLY_HINT)
        return

    if state == GEN_DEST:
        await _reply(
            message,
            "Tap a button above to choose where it goes 🙂",
            reply_markup=_dest_kb(),
        )
        return

    if state == EXAM_NAME:
        name = (message.text or "").strip()
        if not 1 <= len(name) <= 60:
            await _reply(message, "Please send a name between 1 and 60 characters.")
            return
        exam = storage.create_exam(name)
        _set_state(context, EXAM_FILES, exam_id=exam.id)
        await _reply(
            message,
            f"📄 Now send the exam PDF for '{name}'.\n\n"
            "It must be a text PDF (the answers have to be readable). "
            "Send /done when it's the last one.",
            reply_markup=_done_kb(),
        )
        return

    if state == EXAM_FILES:
        if (message.text or "").strip().lower() in ("/done", "done"):
            await _finish_exam(context, None, fallback=message)
        else:
            await _reply(message, "Please send the exam PDF, then press Done.")
        return

    if state in (AUTO_QUIZ_TIME, AUTO_NOTE_TIME):
        hhmm = _parse_time(message.text or "")
        if hhmm is None:
            await _reply(
                message,
                "🕐 That isn't a 24-hour time. Send it as HH:MM — for "
                "example 07:30 or 19:05 (00:00 to 23:59).",
            )
            return

        msg_ref = context.user_data.get("aq_msg")
        if state == AUTO_NOTE_TIME:
            settings = _enable_auto_note(hhmm)
            _clear_state(context)
            text = (
                f"✅ Auto Note is ON — every day at {settings['time']} I'll "
                f"post a note from a random course to {GROUP_LABEL}.\n\n"
                "The course changes every day, so no two notes in a row come "
                "from the same one."
            )
            if msg_ref and _ref_in_topic(msg_ref) and await _edit_text_safe(
                context.bot, msg_ref[0], msg_ref[1], text, _menu_kb()
            ):
                return
            await _reply(message, text, reply_markup=_menu_kb())
            return

        # auto quiz: the time is set, now ask for the quantity
        context.user_data["pending_time"] = hhmm
        _set_state(context, AUTO_QUIZ_COUNT)
        text = (
            f"🕐 {hhmm} it is. How many questions should each daily quiz "
            "have?\n\nTap a preset below, or type any number from 1 to "
            f"{MAX_QUESTIONS}."
        )
        if msg_ref and _ref_in_topic(msg_ref) and await _edit_text_safe(
            context.bot, msg_ref[0], msg_ref[1], text, _auto_qty_kb()
        ):
            return
        await _reply(message, text, reply_markup=_auto_qty_kb())
        return

    if state == AUTO_QUIZ_COUNT:
        try:
            count = int((message.text or "").strip())
            valid = 1 <= count <= MAX_QUESTIONS
        except ValueError:
            valid = False
        if not valid:
            await _reply(
                message,
                f"Please send a whole number between 1 and {MAX_QUESTIONS}.",
            )
            return
        hhmm = context.user_data.get("pending_time")
        msg_ref = context.user_data.get("aq_msg")
        _clear_state(context)
        if not hhmm:
            await _reply(
                message,
                "That time got lost — pick a time again ⏰",
                reply_markup=_menu_kb(),
            )
            return
        settings = _enable_auto_quiz(hhmm, count)
        text = (
            f"✅ Auto Quiz is ON — every day at {settings['time']} I'll post "
            f"{count} mixed questions to {GROUP_LABEL}."
        )
        if msg_ref and _ref_in_topic(msg_ref) and await _edit_text_safe(
            context.bot, msg_ref[0], msg_ref[1], text, _menu_kb()
        ):
            return
        await _reply(message, text, reply_markup=_menu_kb())
        return

    if state == GEN_COUNT:
        try:
            count = int((message.text or "").strip())
            valid = 1 <= count <= MAX_QUESTIONS
        except ValueError:
            valid = False
        if not valid:
            await _reply(
                message,
                f"Please send a whole number between 1 and {MAX_QUESTIONS}.",
            )
            return
        msg_ref = context.user_data.get("gen_msg")
        dest_chat = context.user_data.get("dest_chat")
        _clear_state(context)
        await _run_quiz(
            update,
            context,
            count,
            msg_ref=msg_ref,
            fallback=message,
            dest_chat=dest_chat,
        )
        return

    if state == ADD_NAME:
        name = (message.text or "").strip()
        if not 1 <= len(name) <= 60:
            await _reply(message, "Please send a name between 1 and 60 characters.")
            return
        course = storage.create_course(name)
        _set_state(context, ADD_FILES, course_id=course.id)
        await _reply(message, 
            f"📚 Now send the materials for '{course.name}'.\n\n"
            "Send as many PDFs as you like, one by one. When you're finished, "
            "press Done below.",
            reply_markup=_done_kb(),
        )
        return

    if state == ADD_FILES and update.effective_chat.type == "private":
        await _reply(message, 
            "Please send PDF files, then press Done.",
            reply_markup=_done_kb(),
        )
        return

    # No state: a plain message is a question when it calls the bot by name
    # ("baymax what is 2NF"), or in a private chat, or when it replies to one
    # of the bot's own messages.
    query = _question_from_message(update, message)
    if query:
        await _ask(query, message)
        return

    # anything else: stay silent — important in groups


async def on_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    doc = message.document
    state = context.user_data.get("state")

    if not await _is_admin(update, context):
        if state in ADMIN_STATES:
            _clear_state(context)
        if _is_private(update):
            await _reply(
                message,
                "🔒 Saving material is for the group owner and admins. "
                "Ask me a question or press /start for a quiz instead 💬",
            )
        return

    uploading_exam = state == EXAM_FILES and context.user_data.get("exam_id")
    uploading_course = state == ADD_FILES and context.user_data.get("course_id")
    if not uploading_exam and not uploading_course:
        if update.effective_chat.type == "private":
            await _reply(message, "Open the menu with /start to save materials 📚")
        return

    is_pdf = doc.mime_type == "application/pdf" or (doc.file_name or "").lower().endswith(".pdf")
    if not is_pdf:
        await _reply(message, "Please send PDF files only 📄")
        return

    try:
        tg_file = await doc.get_file()
        with tempfile.TemporaryDirectory() as tmpdir:
            local = Path(tmpdir) / "incoming.pdf"
            await tg_file.download_to_drive(custom_path=str(local))
            if uploading_exam:
                storage.add_exam_pdf(
                    context.user_data["exam_id"], local, doc.file_name or "exam.pdf"
                )
            else:
                storage.add_pdf(
                    context.user_data["course_id"], local, doc.file_name or "document.pdf"
                )
    except Exception:
        logger.exception("failed saving pdf %s", doc.file_name)
        await _reply(message, "❌ Couldn't save that file — please try again.")
        return

    if uploading_exam:
        exam = storage.get_exam(context.user_data["exam_id"])
        n = len(exam.files) if exam else 0
        await _reply(
            message,
            f"✅ Saved {doc.file_name} — {n} PDF{'s' if n != 1 else ''} so far.\n"
            "Send more, or press Done."
        )
        return

    course = storage.get_course(context.user_data["course_id"])
    n = len(course.files) if course else 0
    await _reply(message, 
        f"✅ Saved {doc.file_name} — {n} PDF{'s' if n != 1 else ''} so far.\n"
        "Send more, or press Done."
    )


# --- entry point ----------------------------------------------------------

async def on_topic_created(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Learn the study-room topic id the moment someone creates that topic.

    Telegram sends a service message with the new topic's name and id; that is
    the one automatic way to map name -> id, so the bot doesn't need to be
    poked with /topic afterwards.
    """
    message = update.effective_message
    created = getattr(message, "forum_topic_created", None)
    if created is None:
        return
    name = getattr(created, "name", "") or ""
    if name.strip().lower() != QUIZ_TOPIC_NAME.strip().lower():
        return
    thread = getattr(message, "message_thread_id", None)
    if thread is None:
        return
    storage.save_topic(QUIZ_TOPIC_NAME, int(thread))
    logger.info("Learned topic '%s' -> %s", name, thread)
    await _send_to_topic(
        context.bot,
        f"✅ '{name}' is ready — quizzes, notes and answers will be posted "
        "here from now on.",
    )


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Keep transient network noise quiet; still surface real bugs."""
    exc = context.error
    if isinstance(exc, TimedOut):
        logger.warning("Network timeout while handling an update (will recover).")
    elif isinstance(exc, BadRequest) and "query is too old" in str(exc).lower():
        logger.info("Ignored a stale button press (queued during a network drop).")
    elif isinstance(exc, NetworkError):
        logger.warning("Network error while handling an update: %s", exc)
    else:
        logger.error("Unhandled error while processing an update", exc_info=exc)


async def post_init(app: Application) -> None:
    """Start the daily quiz/note scheduler once the bot is running."""
    app.create_task(_scheduler(app))
    quiz = storage.get_auto_quiz()
    note = storage.get_auto_note()
    logger.info(
        "Auto quiz %s · auto note %s",
        f"ON at {quiz['time']}" if quiz["enabled"] else "off",
        f"ON at {note['time']}" if note["enabled"] else "off",
    )


def main() -> None:
    if not BOT_TOKEN:
        raise SystemExit(
            "BOT_TOKEN is missing. Copy .env.example to .env and add your "
            "@BotFather token, then run the bot again."
        )
    lock = _acquire_single_instance_lock()
    if lock is None:
        raise SystemExit(
            f"Another copy of the bot is already running (lock: {_lock_path()}).\n"
            "Telegram only delivers updates to one instance at a time, so stop "
            "the other one first — including a copy running on a server or a "
            "second terminal window.\n"
            "If you are sure nothing else is running, delete that file and "
            "start the bot again."
        )
    atexit.register(_release_single_instance_lock, lock)
    app = (
        Application.builder()
        .token(BOT_TOKEN)
        .request(_build_httpx())
        .get_updates_request(_build_httpx())
        .post_init(post_init)
        .build()
    )
    app.add_error_handler(on_error)
    app.add_handler(CommandHandler(["start", "menu"], cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("cancel", cmd_cancel))
    app.add_handler(CommandHandler("ask", cmd_ask))
    app.add_handler(CommandHandler("topic", cmd_topic))
    app.add_handler(CommandHandler(["cleanup", "purge"], cmd_cleanup))
    app.add_handler(
        CallbackQueryHandler(
            on_button, pattern=r"^(menu:|add:|del:|gen:|aq:|an:|ex:|dest:)"
        )
    )
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_handler(MessageHandler(filters.Document.ALL, on_document))
    app.add_handler(
        MessageHandler(filters.StatusUpdate.FORUM_TOPIC_CREATED, on_topic_created)
    )
    logger.info("Bot started — course quiz bot for %s", GROUP_LABEL)
    tid = _thread_id()
    logger.info(
        "Posting to topic '%s'%s",
        QUIZ_TOPIC_NAME,
        f" (id {tid})" if tid is not None else " (id unknown — send /topic inside it)",
    )
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
