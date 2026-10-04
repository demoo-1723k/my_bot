"""Tests for storage.py — course CRUD, PDF persistence, text extraction cache.

Run from the project root:
    python -m unittest -v
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import storage
from generator import InsufficientTextError
from tests.test_generator import _build_pdf, _build_pdf_pages

BIO_LINES = [
    "Mitochondria is the powerhouse of the cell and produces ATP.",
    "The cell membrane is a phospholipid bilayer that controls transport.",
    "Ribosomes are the cell structures where proteins are synthesized.",
    "Osmosis is the movement of water across a semipermeable membrane.",
    "Enzymes are biological catalysts that speed up chemical reactions.",
    "During respiration cells break down glucose to release energy.",
    "The pancreas produces insulin which regulates blood sugar levels.",
]


class StorageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        storage.set_data_dir(self.tmp.name)
        self.addCleanup(storage.set_data_dir, Path(__file__).resolve().parent.parent / "data")

    def _write_pdf(self, name: str, lines: list[str]) -> Path:
        path = Path(self.tmp.name) / name
        path.write_bytes(_build_pdf(lines))
        return path

    def _write_pdf_pages(self, name: str, *pages: list[str]) -> Path:
        path = Path(self.tmp.name) / name
        path.write_bytes(_build_pdf_pages(list(pages)))
        return path

    def test_create_list_delete(self) -> None:
        course = storage.create_course("Biology")
        self.assertEqual(course.name, "Biology")
        self.assertEqual(len(storage.list_courses()), 1)

        again = storage.get_course(course.id)
        self.assertIsNotNone(again)
        self.assertEqual(again.name, "Biology")

        removed = storage.delete_course(course.id)
        self.assertEqual(removed.name, "Biology")
        self.assertEqual(storage.list_courses(), [])
        self.assertIsNone(storage.get_course(course.id))
        self.assertFalse((storage._COURSES_DIR / str(course.id)).exists())

    def test_empty_name_rejected(self) -> None:
        with self.assertRaises(ValueError):
            storage.create_course("   ")

    def test_add_pdf_and_finalize(self) -> None:
        course = storage.create_course("Biology")
        pdf = self._write_pdf("notes.pdf", BIO_LINES)
        storage.add_pdf(course.id, pdf, "notes.pdf")

        course = storage.get_course(course.id)
        self.assertEqual(len(course.files), 1)
        entry = course.files[0]
        self.assertEqual(entry.name, "notes.pdf")
        self.assertIsNone(entry.text)  # not extracted yet
        self.assertTrue((storage._COURSES_DIR / entry.pdf).exists())

        text = storage.finalize_course(course.id)
        self.assertIn("Mitochondria", text)

        # text got cached next to the pdf
        course = storage.get_course(course.id)
        self.assertIsNotNone(course.files[0].text)
        self.assertTrue((storage._COURSES_DIR / course.files[0].text).exists())

        # course_text returns the cached extraction
        self.assertIn("Ribosomes", storage.course_text(course.id))

        # finalize is idempotent
        text2 = storage.finalize_course(course.id)
        self.assertEqual(text.strip(), text2.strip())

    def test_multiple_pdfs_combine_text(self) -> None:
        course = storage.create_course("Mixed")
        storage.add_pdf(course.id, self._write_pdf("a.pdf", BIO_LINES), "a.pdf")
        storage.add_pdf(
            course.id,
            self._write_pdf("b.pdf", ["Photosynthesis is how plants make food from light."]),
            "b.pdf",
        )
        text = storage.finalize_course(course.id)
        self.assertIn("Mitochondria", text)
        self.assertIn("Photosynthesis", text)
        self.assertEqual(len(storage.get_course(course.id).files), 2)

    def test_unreadable_pdf_raises_and_stores_nothing(self) -> None:
        course = storage.create_course("Bad")
        bad = Path(self.tmp.name) / "garbage.pdf"
        bad.write_bytes(b"%PDF-1.4 this is not a real pdf")
        storage.add_pdf(course.id, bad, "garbage.pdf")
        with self.assertRaises(InsufficientTextError):
            storage.finalize_course(course.id)

    def test_path_traversal_is_stripped(self) -> None:
        course = storage.create_course("Evil")
        pdf = self._write_pdf("ok.pdf", BIO_LINES)
        entry = storage.add_pdf(course.id, pdf, "../../../etc/passwd.pdf")
        self.assertNotIn("..", entry.pdf)
        self.assertNotIn("/", entry.pdf.split("/", 1)[-1])
        self.assertTrue((storage._COURSES_DIR / entry.pdf).exists())

    def test_index_persists_across_reload(self) -> None:
        course = storage.create_course("Chem")
        data = storage._load()  # re-reads the JSON from disk
        self.assertEqual(data["courses"][0]["name"], "Chem")
        self.assertEqual(data["next_id"], course.id + 1)

    PAGE_1 = [
        "Photosynthesis is the process by which plants convert light energy into chemical energy.",
        "The light dependent reactions take place inside the thylakoid membranes.",
    ]
    PAGE_2 = [
        "Osmosis is the movement of water across a semipermeable membrane.",
        "Enzymes are biological catalysts that speed up chemical reactions in cells.",
    ]

    def test_course_segments_have_page_numbers(self) -> None:
        course = storage.create_course("Biology")
        pdf = self._write_pdf_pages("book.pdf", self.PAGE_1, self.PAGE_2)
        storage.add_pdf(course.id, pdf, "book.pdf")
        storage.finalize_course(course.id)

        segments = storage.course_segments(course.id)
        self.assertTrue(segments)
        self.assertEqual({s.page for s in segments}, {1, 2})
        self.assertTrue(all(s.course == "Biology" for s in segments))
        self.assertTrue(all(s.filename == "book.pdf" for s in segments))
        page2_text = " ".join(s.text for s in segments if s.page == 2)
        self.assertIn("Osmosis", page2_text)

    def test_segments_legacy_paths(self) -> None:
        """Courses saved before page tracking must still work."""
        course = storage.create_course("Legacy")
        pdf = self._write_pdf_pages("notes.pdf", self.PAGE_1, self.PAGE_2)
        storage.add_pdf(course.id, pdf, "notes.pdf")
        storage.finalize_course(course.id)
        entry = storage.get_course(course.id).files[0]
        pages_file = storage._COURSES_DIR / Path(entry.pdf).with_suffix(".pages.json")
        self.assertTrue(pages_file.exists())

        # 1) no pages.json but PDF present -> re-extract page data
        pages_file.unlink()
        segments = storage.course_segments(course.id)
        self.assertTrue(segments)
        self.assertTrue(all(s.page is not None for s in segments))
        self.assertTrue(pages_file.exists())  # rebuilt

        # 2) no PDF either -> cached txt, page unknown but content present
        (storage._COURSES_DIR / entry.pdf).unlink()
        pages_file.unlink()
        segments = storage.course_segments(course.id)
        self.assertTrue(segments)
        self.assertTrue(all(s.page is None for s in segments))
        self.assertIn("Photosynthesis", " ".join(s.text for s in segments))


class AutomationTest(unittest.TestCase):
    """The daily Auto Quiz / Auto Note schedules live in automation.json."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        storage.set_data_dir(self.tmp.name)
        self.addCleanup(storage.set_data_dir, Path(__file__).resolve().parent.parent / "data")

    def test_everything_starts_off(self) -> None:
        quiz = storage.get_auto_quiz()
        self.assertFalse(quiz["enabled"])
        self.assertIsNone(quiz["time"])
        self.assertIsNone(quiz["last_run"])
        self.assertEqual(quiz["count"], 15)

        note = storage.get_auto_note()
        self.assertFalse(note["enabled"])
        self.assertIsNone(note["time"])
        self.assertIsNone(note["last_run"])

        self.assertIsNone(storage.get_note_last_course())

    def test_auto_quiz_round_trip(self) -> None:
        storage.save_auto_quiz(enabled=True, time="08:30", count=20)
        saved = storage.get_auto_quiz()  # re-read from disk
        self.assertTrue(saved["enabled"])
        self.assertEqual(saved["time"], "08:30")
        self.assertEqual(saved["count"], 20)

    def test_auto_note_round_trip(self) -> None:
        storage.save_auto_note(enabled=True, time="17:00")
        saved = storage.get_auto_note()
        self.assertTrue(saved["enabled"])
        self.assertEqual(saved["time"], "17:00")

    def test_turning_off_keeps_the_saved_time_and_count(self) -> None:
        storage.save_auto_quiz(enabled=True, time="08:30", count=20)
        storage.save_auto_quiz(enabled=False)
        saved = storage.get_auto_quiz()
        self.assertFalse(saved["enabled"])
        self.assertEqual(saved["time"], "08:30")
        self.assertEqual(saved["count"], 20)

    def test_last_run_is_tracked_separately(self) -> None:
        storage.save_auto_quiz(enabled=True, time="08:00", last_run="2026-10-03")
        self.assertEqual(storage.get_auto_quiz()["last_run"], "2026-10-03")
        self.assertEqual(storage.get_auto_note()["last_run"], None)

    def test_note_rotation_is_remembered(self) -> None:
        storage.set_note_last_course(4)
        self.assertEqual(storage.get_note_last_course(), 4)
        storage.set_note_last_course(None)
        self.assertIsNone(storage.get_note_last_course())

    def test_settings_and_courses_do_not_collide(self) -> None:
        course = storage.create_course("Biology")
        storage.save_auto_note(enabled=True, time="17:00")
        self.assertEqual(storage.get_note_last_course(), None)
        self.assertEqual(storage.get_course(course.id).name, "Biology")

    def test_unknown_setting_rejected(self) -> None:
        with self.assertRaises(KeyError):
            storage.save_auto_quiz(tim="08:00")

    def test_corrupt_file_falls_back_to_defaults(self) -> None:
        storage.save_auto_note(enabled=True, time="17:00")
        storage._AUTOMATION.write_text("{not json", encoding="utf-8")
        self.assertFalse(storage.get_auto_note()["enabled"])
        self.assertIsNone(storage.get_auto_note()["time"])


if __name__ == "__main__":
    unittest.main()
