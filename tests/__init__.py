"""Test package.

Force the LLM layer off before any test module imports `bot`: bot.py loads the
project's real .env at import time, so without this guard quiz-building tests
would call whatever LLM_PROVIDER points at (a live cloud endpoint) instead of
staying self-contained. No test asserts LLM behaviour — the suite covers the
rule-based engine, storage, routing and flows.
"""

import os

os.environ["LLM_PROVIDER"] = "none"
os.environ.setdefault("QUIZ_CHAT_ID", "-1002292353150")
os.environ.setdefault("QUIZ_TOPIC_NAME", "Study Room")
os.environ.setdefault("QUIZ_TOPIC_ID", "")
os.environ.setdefault("LOG_USERNAME", "kena_ng")
