import unittest

from analyzer import analyze_text


class BasicGrammarRuleTests(unittest.TestCase):
    def _rules(self, text: str) -> set[str]:
        result = analyze_text(text)
        return {issue["rule"] for issue in result["grammar_issues"]}

    def test_repeated_word_detected_and_fixed(self) -> None:
        text = "This is is a test."
        result = analyze_text(text)

        self.assertIn("Repeated word", {i["rule"] for i in result["grammar_issues"]})
        self.assertEqual(result["corrected_text"], "This is a test.")

    def test_article_agreement_detected_and_fixed(self) -> None:
        text = "She has a apple."
        result = analyze_text(text)

        self.assertIn("Article agreement", {i["rule"] for i in result["grammar_issues"]})
        self.assertEqual(result["corrected_text"], "She has an apple.")

    def test_sentence_capitalization_detected_and_fixed(self) -> None:
        text = "she is ready."
        result = analyze_text(text)

        self.assertIn("Sentence capitalization", {i["rule"] for i in result["grammar_issues"]})
        self.assertEqual(result["corrected_text"], "She is ready.")

    def test_pronoun_i_capitalization_detected_and_fixed(self) -> None:
        text = "i am here."
        result = analyze_text(text)

        self.assertIn("Pronoun capitalization", {i["rule"] for i in result["grammar_issues"]})
        self.assertEqual(result["corrected_text"], "I am here.")

    def test_sentence_punctuation_detected_and_fixed(self) -> None:
        text = "We are happy"
        result = analyze_text(text)

        self.assertIn("Sentence punctuation", {i["rule"] for i in result["grammar_issues"]})
        self.assertEqual(result["corrected_text"], "We are happy.")

    def test_day_month_capitalization_detected_and_fixed(self) -> None:
        text = "i met her on monday in january."
        result = analyze_text(text)

        self.assertIn("Capital letters", {i["rule"] for i in result["grammar_issues"]})
        self.assertEqual(result["corrected_text"], "I met her on Monday in January.")

    def test_basic_subject_verb_agreement_detected_and_fixed(self) -> None:
        text = "We is ready."
        result = analyze_text(text)

        self.assertIn("Subject-verb agreement", {i["rule"] for i in result["grammar_issues"]})
        self.assertEqual(result["corrected_text"], "We are ready.")

    def test_do_support_and_infinitive_base_form_fixed(self) -> None:
        text = "She doesn't likes to eats bandanna in the morning."
        result = analyze_text(text)

        self.assertIn("Verb form", {i["rule"] for i in result["grammar_issues"]})
        self.assertIn(
            "She doesn't like to eat banana in the morning.",
            result["corrected_text"],
        )

    def test_theer_and_bandanna_context_spelling(self) -> None:
        text = "theer is no word called bandanna"
        result = analyze_text(text)

        misspellings = {item["word"].lower(): item for item in result["misspellings"]}
        self.assertIn("theer", misspellings)
        self.assertIn("there", misspellings["theer"]["suggestions"])
        self.assertEqual(result["corrected_text"], "There is no word called bandanna.")


if __name__ == "__main__":
    unittest.main()
