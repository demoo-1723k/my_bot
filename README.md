# 📚 Course Quiz Bot (`@my_poller_bot`)

A Telegram bot for the study group **[@practicemakesperfect2](https://t.me/practicemakesperfect2)**:
save PDFs as **courses**, then generate a **mixed quiz** (fill-in-the-blank +
true/false Telegram quiz polls) built from *all* courses and post it to the group —
on demand, or automatically every day at a time you pick. It can also post a
daily **note** from a random course, and **answer questions about your notes**.

Generation is **hybrid — works with or without an LLM**:

| Mode | Quiz | Ask | Note |
|------|------|-----|------|
| **No API key** (zero config) | 7 rule-based types mapped to Bloom's taxonomy (recall → analyze), grounded distractors | Hybrid BM25 + semantic embeddings when `sentence-transformers` is installed, otherwise BM25 | Structured 1–3 bullet synthesis, ranked by causal/comparative relevance |
| **With LLM** (`OPENAI_API_KEY` / `GEMINI_API_KEY` / `GROQ_API_KEY` / Ollama) | LLM generates fluent, scenario/comparison/cause-effect questions from RAG context; validated & deduped, rule-based fills any gap | RAG answering with citation + hallucination guard (word-overlap + `NOT_FOUND`) | LLM synthesis with grounding check |
| **Fallback** | If LLM fails or returns invalid JSON, rule-based engine takes over — the bot never breaks | Same — BM25/hybrid is the safety net | Same |

In every mode every answer cites **course · file · page** so you can re-read the exact part. Text is extracted with `pdfplumber` and cleaned of layout debris; the LLM path adds RAG grounding so nothing is hallucinated.

Everything the bot posts goes into one **forum topic** (`Study Room` by
default), so quizzes, notes and answers stay together and out of the rest of
the group chat.

## Who can use the bot

The bot belongs to the **group owner and its admins**. Everyone else can only
talk to it in a **private chat**.

| | Owner / admins | Everyone else |
|---|---|---|
| Menu in the group | ✅ full | ❌ told to DM the bot |
| Add / delete courses, exams, schedules | ✅ | ❌ |
| `baymax …` / `/ask` in the group | ✅ | ❌ pointed at a private chat |
| Ask questions in a private chat | ✅ | ✅ |
| Get a quiz in a private chat | ✅ (chooses group or private) | ✅ (always their own chat) |
| Notes in a private chat | ✅ (chooses group or private) | ❌ |

Membership is read from Telegram (`getChatMember`) and cached for a minute, so
demoting someone takes effect within a minute. If the check cannot be
completed — no chat attached, network trouble — the bot **fails open** rather
than locking the owner out of their own bot.

An admin who presses **Generate Quiz** or **Generate Note** *in a private chat*
is asked where to post it:

```
Where should I post the quiz?
[ 📌 To the group ]
[ 💬 Here in this chat ]
```

In the group the question is not asked — the group is the obvious answer. A
student never sees the question: their quiz always goes to their own chat.

## The flow

`/start` shows a greeting and a menu with inline buttons:

| Button | What happens |
|---|---|
| **Add Course** | Asks for the course name → you send as many PDFs as you like → press **Done** → course is saved (texts are extracted and cached) |
| **Delete Course** | Lists your courses → pick one → *"Are you sure?"* → **Yes, Delete** / **Cancel** |
| **Generate Quiz** | Asks **how many questions** (presets 5/10/15/25/50 — or type any number 1–50), then builds a quiz mixing every course (round-robin, so big courses don't drown out small ones) and posts it. In a private chat an admin is first asked **where**: the group or that chat |
| **Generate Note** | Picks a **random course** and posts one note from it — the same message the daily note uses, so you can see it before turning the automation on. In a private chat an admin is first asked **where** |
| **⏰ Auto Quiz** | Shows the current status (**ON**/**OFF**), a description, and time buttons in 24-hour format → pick a time (or **⌨️ Custom time** and type it) → asks how many questions → the daily quiz is ready |
| **📝 Auto Note** | Same screen and time picker, but no quantity — at the chosen time a note from a random course is posted every day |
| **📚 Exams** | Save past exam papers *with their answers* — about 30% of every quiz then uses the real questions from them. Tap a saved paper to delete it |

Students get a smaller menu instead: **🎯 Send me a quiz** and **🔎 How to
ask**. The same two buttons appear under every answer in a private chat.

Commands: `/start` · `/help` · `/cancel` · `/ask <question>` · `/topic` · `/cleanup`

## Everything goes in one topic

Quizzes, daily notes and every answer are posted into the **Study Room** topic
rather than the group's General chat.

Telegram gives a topic a numeric id that every message must carry, and there
is no API to look one up by name — so the bot learns it, in any of these ways:

1. **Do nothing.** When someone creates a topic named `Study Room`, Telegram
   sends the bot the new topic's id, and it remembers it.
2. **Send `/topic` from inside the topic** — the bot learns it from wherever
   the command came from (admins only).
3. **Pin it** — put `QUIZ_TOPIC_ID=<id>` in `.env`.

The name is configurable with `QUIZ_TOPIC_NAME`. If the id is not known yet,
the bot logs a hint at startup and falls back to General. A topic id that
Telegram later rejects (topic deleted, wrong id pinned) is **forgotten
automatically** and the next send lands in General instead of failing forever.

### Nothing is posted anywhere else

The topic is not just where quizzes go — it is where **everything** goes:

* a `/start`, `/ask` or `baymax …` asked in **General or another topic** is
  answered *inside* the Study Room, and the question travels with the answer
  (`❓ <question>` above it) so the topic stays readable;
* a menu screen that was opened outside the topic is **moved in** — the new
  screen appears in the topic and the old one is blanked to
  `➡️ Moved to the Study Room topic.` so no flow keeps running in General;
* private chats are never redirected: a student asking the bot directly is
  answered right there.

### `/cleanup` — delete what landed elsewhere

Telegram has no API for "list my own messages", so the bot writes down every
group message it sends together with the topic it landed in
(`data/outbox.json`). **`/cleanup`** (admins only) deletes everything recorded
outside the Study Room:

```/cleanup
🗑 Deleted 6 messages from outside the Study Room topic.
⚠️ 1 could not be deleted right now — run /cleanup again shortly.
```

* **Reply to one of the bot's messages** while sending `/cleanup` to delete
  just that one message. (If the message is not the bot's, it says so instead
  of deleting a student's message.)
* **Older messages**, sent before the bot started keeping track, need either
  the reply trick or a **range**:

```
/cleanup 1234-1290          # or: /cleanup from 1234 to 1290
```

  Open any message in **web.telegram.org** — the last number in its link is
  the message id. The sweep deletes only messages Telegram lets the bot
  delete, i.e. **its own**, and skips ids it knows it posted inside the topic.
  At most 300 ids per run.

  Per the Bot API, a bot may always delete its own messages, but deleting
  anyone else's needs `can_delete_messages` in a supergroup (or plain admin in
  a basic group). If the bot ever gets that right, the sweep is **refused**
  rather than risk a student's message — the reply form still works.
* Telegram only allows a bot to delete its own messages **within 48 hours**;
  older ones are reported as already gone and forgotten.
* No admin right is needed for this: deleting *your own* messages does not
  require the "Delete messages" permission — that right only governs deleting
  **other people's** messages. If Telegram refuses anyway, `/cleanup` shows
  its exact words.
* It is safe to run repeatedly — only messages that are actually outside the
topic are touched.

## How questions are chosen

The generator never invents a question. A sentence becomes a question only if
it states something a student could be right or wrong about:

* **Central terms, not random words.** A word is only used as a blank or as a
  distractor if it recurs across the material (frequency × page spread). A word
  that appears once inside a worked example is never asked about.
* **Real definitions only.** "A search tree is a representation in which nodes
  denote paths" qualifies. "Color change is one rectangle at a time" and
  "The goal is to get one liter of water" do not — a purpose clause is not a
  definition.
* **Seven question kinds** across Bloom's taxonomy — *which term is described as …* (recall), *______ is …* (recall), *number* blank (recall), *true/false* (understand), *scenario/case* — "in a scenario where … which term applies?" (apply), *comparison* — contrasting two concepts (analyze), *cause → effect* (analyze).
* **True/false is capped** at roughly a third of a quiz, so a quiz stays a
   quiz rather than a run of coin flips. New Bloom-level types are capped similarly to keep variety.
* **Layout debris is dropped** — Wingdings bullets, headings, tables of
  contents, figure captions, "Individual Assignment (5%)", running heads, and
  columns bleeding into each other.
* Every question keeps its **course · file · page** in the explanation.

## `/ask` — questions about your notes

```
/ask what is a weak entity
```

The bot searches every course with a **hybrid retriever** — BM25 exact-match plus
dense semantic embeddings (`sentence-transformers` / `all-MiniLM-L6-v2`) fused via
RRF and optional cross-encoder reranking. With no extra install it is BM25 only;
with `pip install sentence-transformers` it understands paraphrases and synonyms.
When an LLM is configured, answering upgrades to full **RAG** (retrieved context →
grounded generation) with a hallucination guard. It replies with:

```
📖 From your material

A weak entity set doesn't have any primary key which can identify each
entity in a set distinctly.

━━━━━━━━━━━━
📄 Database · DB CH3.pdf · p.9
📎 Database · DB CH3.pdf · p.6
```

* The answer is always **grounded in your PDFs** — BM25/hybrid returns a copied
   sentence, RAG returns a synthesized answer but only if it overlaps the retrieved
   context (otherwise `NOT_FOUND`). No hallucination.
* If the material doesn't contain the answer, the bot says so
  ("I couldn't find that in your notes") instead of returning a sentence that
  merely looks related. A question using a word that appears nowhere in the
  notes — "what is quantum entanglement?" — always gets that answer.
* 📎 lines point at other pages that mention the same thing, for cross-reading.

### Asking in the group

`/ask …` works anywhere. In a group **only the owner and admins are answered** —
everyone else is pointed at a private chat — and the bot **only listens when it
is spoken to directly**, so it doesn't read along with everyone's chat:

* say **“baymax what is a weak entity”** — the wake word calls the bot
  (`jarvis`, `jarves` and `bay max` are recognised too, in any spelling);
* **reply to one of the bot's messages** and just type the question;
* send `/ask <question>` — a command always reaches the bot;
* or ask in a **private chat** with the bot, where nothing is posted publicly
  and anyone in the group can do it.

Anything else in the group is ignored. Because of this, privacy mode can stay
**enabled** — you don't need `/setprivacy → Disable` to use `/ask`. You only
need it for the `baymax …` wake word without a reply, or if you send PDFs into
the group itself; both work from a private chat with privacy mode on.

## Exams — using past papers

A past paper already contains questions *and* the examiner's answers. Those are
the best quiz material there is, so the bot reads them instead of re-inventing
anything:

1. **/start → 📚 Exams → Upload exam PDF**
2. Give the paper a name (e.g. *Database Midterm 2024*) and send the PDF.
3. Press **Done**.

The PDF must be a **text PDF** — scanned image-only files are rejected with an
explanation, because their questions cannot be read.

Both layouts are understood:

* **Multiple choice** — `a) 1NF  b) 2NF  c) 3NF` with `Ans: c` becomes the
  question with the paper's *own* options, and the right one selected. The
  letter is resolved to the option's text.
* **Written answers** — `Ans: A candidate key is a minimal superkey…` becomes
  the question with that sentence as the correct option; the wrong options are
  drawn from terms in your own course notes, or from other answers in the same
  paper.

Only **about 30% of each quiz** (`EXAM_SHARE`) comes from exams, so they enrich
the quiz instead of replacing it — the rest is generated from your notes as
usual. A paper with **no** answer key contributes no exam questions at all:
the bot would rather skip it than invent an answer. Those papers are still used
as ordinary material.

Deleting a paper removes it and its folder from disk, and its questions leave
the quizzes at once.

## Auto Quiz & Auto Note

* **Time** — tap a preset or use **⌨️ Custom time** and type it yourself. Times
  must be in 24-hour format: `07:30`, `19:05` (`7:30` is fine too, it is saved
  as `07:30`); `8 pm` and `25:00` are rejected.
* **Firing** — the bot re-reads its saved schedules every 20 seconds, so a time
  changed in the menu takes effect at once. Each task runs **once per day**; if
  the bot was closed when the time came, it sends when it starts instead of
  skipping the day.
* **Turn off** — both automation screens carry **❌ Turn off** while they are
  running. The saved time and question count are kept, so switching it back on
  is a couple of taps.
* **Rotation** — the daily note never uses the same course twice in a row (with
  a single course there is no alternative, so it repeats).
* ⚠️ The bot is a normal local program: it can only post while `bot.py` is
  running, on a machine whose clock is set to your local time. Keep it running
  (or on a small always-on machine) if you rely on the daily posts.

A note looks like this:

```
📖 Daily Note

Photosynthesis is the process by which plants convert light energy
into chemical energy.

💡 The light dependent reactions take place in the thylakoid
membranes of chloroplasts.

━━━━━━━━━━━━
📚 Course: Biology
📄 File: ch4_photosynthesis.pdf
📃 Page: 12
```

The note is a central term with its definition, plus up to two supporting bullets
(ranked: causal explanation > comparison > example > variety) from elsewhere in the
material. When an LLM is configured the note is synthesized from RAG context with
a grounding check; otherwise the structured rule-based note is used.

## AI upgrade — zero-config to LLM-grade

The bot works out of the box with no API key. To go LLM-grade, add **one** of
these to `.env` (first available wins when `LLM_PROVIDER=auto`):

```ini
# Pick one — best to cheapest:
OPENAI_API_KEY=sk-...          # gpt-4o-mini (best quality)
GROQ_API_KEY=gsk_...           # llama-3.3-70b (fast & cheap)
GEMINI_API_KEY=AIza...         # gemini-2.0-flash (generous free tier)
# Local — no key, no cost:
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
```

Optional routing / tuning:

```ini
LLM_PROVIDER=auto              # auto | openai | gemini | groq | ollama | none
LLM_MODEL=gpt-4o-mini          # override model name
LLM_BASE_URL=https://...       # proxy / custom Ollama host
LLM_TIMEOUT=25                 # seconds
```

New modules (all optional, bot runs without them):

- `llm_client.py` — unified client for OpenAI / Gemini / Groq / Ollama; JSON-mode helpers.
- `retriever.py` — `HybridRetriever` (BM25 + `all-MiniLM-L6-v2` dense + cross-encoder rerank via RRF). Install with `pip install sentence-transformers`.
- `ai_generator.py` — `ai_generate_questions` / `ai_answer_question` / `ai_generate_note` — RAG-grounded LLM generation with strict validation, dedup and rule-based fallback. Never hallucinates: every LLM output must overlap retrieved context.

What changes for the user: quizzes gain scenario / comparison / cause-effect questions (Bloom apply/analyze), `/ask` understands paraphrases, notes become multi-bullet syntheses — and without a key the new rule-based Bloom types + structured notes still improve over the old bot.

## Setup

1. Create a bot with [@BotFather](https://t.me/BotFather) and copy the token.
2. Add the bot to your group and make it an **admin** (it needs to send polls).
3. Install and configure:

```bash
python -m venv .venv
source .venv/bin/activate      # PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env           # then paste your token + QUIZ_CHAT_ID into .env
```

4. Run it:

```bash
python bot.py
```

No activation? Call the venv Python directly:
`.venv/Scripts/python.exe bot.py` (PowerShell/Windows) or `.venv/bin/python bot.py`.

First startup takes ~25 seconds on Windows (antivirus scans the Telegram
library) — give it a moment before the `Bot started` log appears.

**Run only one copy at a time.** Telegram gives a bot's updates to a single
poller; a second copy used to fill the log with
`Conflict: terminated by other getUpdates request` and quietly fight the first
one for updates. It now refuses to start instead:

```
Another copy of the bot is already running (lock: …\data\bot.lock).
Telegram only delivers updates to one instance at a time, so stop the other
one first — including a copy running on a server or a second terminal window.
```

That matters after a deploy too: stop the local copy before the server one
starts (or vice versa). The lock file is removed on a clean exit and is
taken over automatically if a copy crashed, so it never blocks a restart.

### ⚠️ Important: privacy mode (optional)

The bot only receives group messages addressed to it unless privacy mode is
disabled. **You do not need to disable it for quizzes, notes or `/ask`** —
commands, inline buttons and replies to the bot all work with privacy mode on.

Disable it (Telegram: **@BotFather → /setprivacy → Disable → your bot**, then
restart the bot) if you want either of these:

* **the `baymax …` wake word in the group** — with privacy mode on, Telegram
  does not forward plain group text, so an admin would have to reply to one of
  the bot's messages or use `/ask` instead;
* **sending course PDFs from inside the group**.

Adding material in a private chat with the bot works either way, and every
question asked in a private chat works either way.

## Storage

Courses live on disk, gitignored:

```
data/
  courses.json        # index: id, name, files
  exams.json          # index of old exam papers
  automation.json     # daily Auto Quiz / Auto Note schedules
  topics.json         # learned forum topic ids (name -> id)
  outbox.json         # every group message the bot sent, and its topic
  bot.lock            # single-instance lock (written at start, removed at exit)
  courses/<id>/       # the PDFs + cached .txt extractions
  exams/<id>/         # one folder per uploaded exam paper
```

Delete a course or an exam from the menu and both the index entry and its
folder go away.

## Tests

```bash
python -m unittest -v
```

Tests build tiny PDFs in memory and cover extraction, question validity
against Telegram's poll limits, the junk-filtering rules (bullets, truncated
sentences, purpose clauses, running heads), storage lifecycle, mixed-quiz
generation, note formatting, BM25 retrieval and `/ask` answers, old-exam
parsing and quiz mixing, forum-topic routing, the wake-word trigger, the
24-hour time input, the daily scheduling rules, the auto quiz / auto note
menu flows, the permission rules (who may press what, where a quiz or note is
posted), the redirect of everything into the Study Room topic and `/cleanup`.

## Project layout

```
bot.py          # menu + inline-button state machine, sends polls/notes, /ask, runs the daily schedules
generator.py    # PDF extraction + 7 rule-based Bloom question types, structured notes, BM25
storage.py      # course CRUD (JSON index + per-course folders) + automation settings
llm_client.py   # unified LLM client (OpenAI/Gemini/Groq/Ollama) with auto-fallback
retriever.py    # HybridRetriever — BM25 + dense + RRF + rerank
ai_generator.py # hybrid LLM layer (RAG quiz/answer/note + validation + fallback)
tests/          # unit tests (self-contained, no sample files needed)
```
