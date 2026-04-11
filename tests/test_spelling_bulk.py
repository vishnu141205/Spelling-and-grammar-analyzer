import unittest

from analyzer import analyze_text


class SpellingBulkRegressionTests(unittest.TestCase):
    CASES = [
        ("dont", "He dont know the answer.", "don't"),
        ("cant", "I cant swim.", "can't"),
        ("wont", "I wont do that.", "won't"),
        ("shouldnt", "You shouldnt worry.", "shouldn't"),
        ("couldnt", "They couldnt join us.", "couldn't"),
        ("wouldnt", "She wouldnt agree.", "wouldn't"),
        ("didnt", "We didnt finish it.", "didn't"),
        ("doesnt", "It doesnt matter.", "doesn't"),
        ("isnt", "This isnt right.", "isn't"),
        ("arent", "They arent ready.", "aren't"),
        ("wasnt", "He wasnt here.", "wasn't"),
        ("werent", "You werent late.", "weren't"),
        ("teh", "I saw teh movie.", "the"),
        ("recieve", "Please recieve this file.", "receive"),
        ("definately", "This is definately useful.", "definitely"),
        ("goverment", "The goverment announced a policy.", "government"),
        ("writting", "She is writting a note.", "writing"),
        ("beautifull", "That is beautifull.", "beautiful"),
        ("vegtables", "I eat vegtables daily.", "vegetables"),
        ("yestarday", "I met him yestarday.", "yesterday"),
        ("everydays", "I walk everydays.", "every day"),
        ("goed", "I goed to school.", "went"),
        ("raning", "It is raning now.", "raining"),
        ("tred", "I am tred after work.", "tired"),
        ("ypu", "Can ypu help me?", "you"),
        ("yu", "Yu are kind.", "you"),
        ("ia", "There ia a dog outside.", "is"),
        ("isa", "There isa a book here.", "is"),
        ("bout", "We bout milk yesterday.", "bought"),
        ("ar", "How ar you today?", "are"),
    ]

    def test_bulk_spelling_coverage(self) -> None:
        for typo, sentence, expected in self.CASES:
            with self.subTest(typo=typo):
                result = analyze_text(sentence)
                misspellings = {item["word"].lower(): item for item in result["misspellings"]}

                self.assertIn(typo, misspellings)
                self.assertIn(expected, misspellings[typo]["suggestions"])


if __name__ == "__main__":
    unittest.main()
