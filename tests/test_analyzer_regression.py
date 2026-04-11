import unittest

from analyzer import analyze_text


class AnalyzerRegressionTests(unittest.TestCase):
    def test_core_spelling_typos_detected(self) -> None:
        text = "dont recieve definately goverment writting teh"
        result = analyze_text(text)

        misspelled_words = {item["word"].lower() for item in result["misspellings"]}
        self.assertSetEqual(
            misspelled_words,
            {"dont", "recieve", "definately", "goverment", "writting", "teh"},
        )

    def test_user_reported_sample_corrected(self) -> None:
        sample = (
            "I goed to school late today.\n"
            "She have two cat at home.\n"
            "We is happy to see you.\n"
            "He dont know the answer.\n"
            "They was eating lunch now.\n"
            "My brother play cricket everydays.\n"
            "The sun rise in the west."
        )

        result = analyze_text(sample)

        expected = (
            "I went to school late today.\n"
            "She has two cats at home.\n"
            "We are happy to see you.\n"
            "He doesn't know the answer.\n"
            "They are eating lunch now.\n"
            "My brother plays cricket every day.\n"
            "The sun rises in the west."
        )

        self.assertEqual(result["corrected_text"], expected)
        self.assertGreaterEqual(result["summary"]["misspelled_word_count"], 3)
        self.assertGreaterEqual(result["summary"]["grammar_issue_count"], 8)


if __name__ == "__main__":
    unittest.main()
