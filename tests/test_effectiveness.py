import io
import unittest

from analyzer import analyze_text
from app import app


class AnalyzerEffectivenessTests(unittest.TestCase):
    def test_clean_sentences_have_no_issues(self) -> None:
        cases = [
            "She has two cats at home.",
            "How are you today?",
            "There is a book on the table.",
            "My brother plays cricket every day.",
            "I went to school yesterday.",
        ]

        for text in cases:
            with self.subTest(text=text):
                result = analyze_text(text)
                self.assertEqual(result["summary"]["misspelled_word_count"], 0)
                self.assertEqual(result["summary"]["grammar_issue_count"], 0)

    def test_mixed_paragraph_detects_and_corrects(self) -> None:
        paragraph = (
            "I goed to school late today. "
            "She have two cat at home. "
            "We is happy to see you. "
            "He dont know the answer. "
            "They was eating lunch now. "
            "My brother play cricket everydays. "
            "The sun rise in the west."
        )

        result = analyze_text(paragraph)

        self.assertGreaterEqual(result["summary"]["misspelled_word_count"], 3)
        self.assertGreaterEqual(result["summary"]["grammar_issue_count"], 8)

        corrected = result["corrected_text"]
        self.assertIn("I went to school late today.", corrected)
        self.assertIn("She has two cats at home.", corrected)
        self.assertIn("He doesn't know the answer.", corrected)
        self.assertIn("My brother plays cricket every day.", corrected)


class ApiIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app.test_client()

    def test_analyze_endpoint_success(self) -> None:
        response = self.client.post("/api/analyze", json={"text": "He dont know the answer."})
        self.assertEqual(response.status_code, 200)

        payload = response.get_json()
        self.assertIsInstance(payload, dict)
        self.assertIn("summary", payload)
        self.assertIn("corrected_text", payload)
        self.assertIn("doesn't", payload["corrected_text"])

    def test_analyze_endpoint_rejects_empty_text(self) -> None:
        response = self.client.post("/api/analyze", json={"text": "   "})
        self.assertEqual(response.status_code, 400)

        payload = response.get_json()
        self.assertEqual(payload["error"], "Please provide non-empty English text for analysis.")

    def test_analyze_file_endpoint_txt_success(self) -> None:
        data = {
            "file": (io.BytesIO(b"I goed to school."), "sample.txt"),
        }
        response = self.client.post("/api/analyze-file", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 200)

        payload = response.get_json()
        self.assertEqual(payload["source_file"], "sample.txt")
        self.assertIn("went", payload["corrected_text"])

    def test_analyze_file_endpoint_rejects_unsupported_extension(self) -> None:
        data = {
            "file": (io.BytesIO(b"hello"), "sample.pdf"),
        }
        response = self.client.post("/api/analyze-file", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)

        payload = response.get_json()
        self.assertEqual(payload["error"], "Only .txt or .md files are supported.")


if __name__ == "__main__":
    unittest.main()
