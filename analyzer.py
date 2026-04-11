from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher, get_close_matches
from functools import lru_cache
from typing import Any

from wordfreq import top_n_list, zipf_frequency

TOKEN_REGEX = re.compile(
    r"(?P<NEWLINE>\n)|"
    r"(?P<WHITESPACE>[ \t\r]+)|"
    r"(?P<WORD>[A-Za-z]+(?:'[A-Za-z]+)?)|"
    r"(?P<NUMBER>\d+(?:\.\d+)?)|"
    r"(?P<PUNCT>[.,!?;:\"()\[\]{}-])|"
    r"(?P<OTHER>.)"
)

WORD_MATCH_REGEX = re.compile(r"\b[A-Za-z]+(?:'[A-Za-z]+)?\b")
STANDALONE_I_REGEX = re.compile(r"(?<![A-Za-z'])i(?![A-Za-z'])")

REPEATED_WORD_REGEX = re.compile(r"\b([A-Za-z]+)\s+\1\b", re.IGNORECASE)
ARTICLE_REGEX = re.compile(r"\b(a|an)\s+([A-Za-z]+)\b", re.IGNORECASE)
SENTENCE_REGEX = re.compile(r"[^.!?]+[.!?]?", re.MULTILINE)
NAME_INTRO_REGEX = re.compile(
    r"\b(i am|i'm|my name is|this is|he is|she is|mr\.?|mrs\.?|ms\.?|dr\.?)\s+([a-z][a-z'-]*)\b",
    re.IGNORECASE,
)
SENTENCE_JOIN_REGEX = re.compile(
    r"\b((?:i am|i'm|my name is|this is|he is|she is)\b[^.\n,!?]*?)\s+(?=(?:i am|i'm|my name is|this is|he is|she is)\b)",
    re.IGNORECASE,
)
LEADING_GREETING_REGEX = re.compile(r"^(?:\s*(?:hi|hello|hey)\b[,\s]*)+", re.IGNORECASE)
SUBJECT_BE_REGEX = re.compile(
    r"\b(?:(?P<det>a|an|the|this|that|these|those|my|your|our|their|his|her|each|every|many|few|several)\s+)?(?P<head>[A-Za-z]+)\s+(?P<verb>is|are)\b",
    re.IGNORECASE,
)
THERE_BE_REGEX = re.compile(r"\bthere\s+(?P<verb>is|are)\s+(?P<next>[A-Za-z]+)\b", re.IGNORECASE)
A_NUMBER_OF_REGEX = re.compile(r"\ba number of\s+[A-Za-z]+\s+(?P<verb>is|are)\b", re.IGNORECASE)
THE_NUMBER_OF_REGEX = re.compile(r"\bthe number of\s+[A-Za-z]+\s+(?P<verb>is|are)\b", re.IGNORECASE)
COMMA_SPLICE_REGEX = re.compile(
    r"\b(?P<left>(?:I|you|we|they|he|she|it|[A-Z][a-z]+|the\s+[a-z]+|my\s+[a-z]+|our\s+[a-z]+|their\s+[a-z]+)\b[^,!?\n]*\b(?:am|is|are|was|were|have|has|had|do|does|did|can|could|will|would|should|must|go|goes|went|make|makes|made|say|says|said|see|sees|saw|know|knows|knew|think|thinks|thought)\b[^,!?\n]*)\s*,\s*(?P<right>(?:I|you|we|they|he|she|it|[A-Z][a-z]+|the\s+[a-z]+|my\s+[a-z]+|our\s+[a-z]+|their\s+[a-z]+)\b[^.!?\n]*\b(?:am|is|are|was|were|have|has|had|do|does|did|can|could|will|would|should|must|go|goes|went|make|makes|made|say|says|said|see|sees|saw|know|knows|knew|think|thinks|thought)\b[^.!?\n]*)(?P<end>[.!?]|$)",
    re.IGNORECASE,
)
MISSING_COMMA_BEFORE_COORDINATING_CONJ_REGEX = re.compile(
    r"\b(?P<left>(?:I|you|we|they|he|she|it|[A-Z][a-z]+|the\s+[a-z]+|my\s+[a-z]+|our\s+[a-z]+|their\s+[a-z]+)\b[^,.!?\n]*\b(?:am|is|are|was|were|have|has|had|do|does|did|can|could|will|would|should|must|go|goes|went|make|makes|made|say|says|said|see|sees|saw|know|knows|knew|think|thinks|thought)\b[^,.!?\n]*)\s+(?P<conj>and|but|or|nor|for|so|yet)\s+(?P<right>(?:I|you|we|they|he|she|it|[A-Z][a-z]+|the\s+[a-z]+|my\s+[a-z]+|our\s+[a-z]+|their\s+[a-z]+)\b[^.!?\n]*\b(?:am|is|are|was|were|have|has|had|do|does|did|can|could|will|would|should|must|go|goes|went|make|makes|made|say|says|said|see|sees|saw|know|knows|knew|think|thinks|thought)\b[^.!?\n]*)(?P<end>[.!?]|$)",
    re.IGNORECASE,
)
COMMA_BEFORE_THAN_REGEX = re.compile(r",\s+(than\b)", re.IGNORECASE)
THAN_THEN_COMPARISON_REGEX = re.compile(r"\b(?P<left>(?:more\s+)?[A-Za-z]+(?:er)?)\s+then\b", re.IGNORECASE)
REDUNDANT_MORE_COMPARATIVE_REGEX = re.compile(r"\bmore\s+(?P<adj>[A-Za-z]+er)\b", re.IGNORECASE)
SUBJECT_HAVE_REGEX = re.compile(
    r"\b(?P<subject>I|you|we|they|he|she|it|[A-Za-z]+)\s+(?P<verb>have|has)\b",
    re.IGNORECASE,
)
SUBJECT_WAS_WERE_REGEX = re.compile(
    r"\b(?P<subject>I|you|we|they|he|she|it|[A-Za-z]+)\s+(?P<verb>was|were)\b",
    re.IGNORECASE,
)
THIRD_PERSON_DONT_REGEX = re.compile(r"\b(?P<subject>he|she|it)\s+dont\b", re.IGNORECASE)
THIRD_PERSON_DONT_CONTRACTION_REGEX = re.compile(r"\b(?P<subject>he|she|it)\s+don't\b", re.IGNORECASE)
THERE_THEIR_POSSESSIVE_REGEX = re.compile(r"\bthere\s+(?P<noun>[a-z]+)\b", re.IGNORECASE)
PRESENT_PERFECT_WENT_REGEX = re.compile(r"\b(?P<aux>has|have|had)\s+went\b", re.IGNORECASE)
WHEN_PAST_PROGRESSIVE_BASE_REGEX = re.compile(
    r"\b(?P<left>(?:I|you|we|they|he|she|it)\s+(?:was|were)\s+[A-Za-z]+ing\s+when\s+(?:I|you|we|they|he|she|it)\s+)(?P<verb>start)\b",
    re.IGNORECASE,
)
NAME_AS_VERB_REGEX = re.compile(r"\b(?P<noun>[A-Za-z]+)\s+name\s+(?P<name>[A-Z][a-z]+)\b")
STATIVE_PROGRESSIVE_REGEX = re.compile(
    r"\b(?P<subject>I|you|we|they|he|she|it)\s+am\s+not\s+understanding\b",
    re.IGNORECASE,
)
PRESENT_PERFECT_YESTERDAY_REGEX = re.compile(
    r"\b(?P<subject>I|you|we|they|he|she|it)\s+(?:have|has|had)\s+gone(?P<tail>[^.!?\n]*\byesterday\b)",
    re.IGNORECASE,
)
PLURAL_QUANTIFIER_NOUN_REGEX = re.compile(
    r"\b(?P<qty>\d+|two|three|four|five|six|seven|eight|nine|ten|many|several|few)\s+(?P<noun>[A-Za-z]+)\b",
    re.IGNORECASE,
)
THIRD_PERSON_SIMPLE_BASE_REGEX = re.compile(
    r"\b(?P<subject>he|she|it|my\s+[a-z]+|the\s+[a-z]+|[A-Z][a-z]+)\s+(?P<verb>play|go|rise|start|eat|drink|read|write|walk|run|talk|look|need|want|make|take|say|know|like|work|study)\b",
    re.IGNORECASE,
)
PROGRESSIVE_NOW_BE_REGEX = re.compile(
    r"\b(?P<subject>I|you|we|they|he|she|it)\s+(?P<verb>was|were)\s+(?P<prog>[A-Za-z]+ing)(?P<tail>[^.!?\n]*\bnow\b)",
    re.IGNORECASE,
)
DO_SUPPORT_BASE_FORM_REGEX = re.compile(
    r"\b(?P<subject>I|you|we|they|he|she|it|[A-Z][a-z]+)\s+(?P<aux>do|does|did|don't|doesn't|didn't)\s+(?P<verb>[A-Za-z]+)\b",
    re.IGNORECASE,
)
TO_BASE_FORM_REGEX = re.compile(r"\bto\s+(?P<verb>[A-Za-z]+)\b", re.IGNORECASE)

POSSESSIVE_NOUN_HINTS = {
    "house",
    "home",
    "school",
    "teacher",
    "dog",
    "cat",
    "friend",
    "brother",
    "sister",
    "car",
    "book",
    "parents",
    "mother",
    "father",
    "class",
    "team",
    "family",
    "room",
    "bag",
}

QUESTION_WORDS = {
    "who",
    "what",
    "when",
    "where",
    "why",
    "how",
    "whose",
    "which",
}

QUESTION_PHRASES = {"how many", "how much"}

QUESTION_AUXILIARY_STARTERS = {
    "am",
    "is",
    "are",
    "was",
    "were",
    "do",
    "does",
    "did",
    "can",
    "could",
    "will",
    "would",
    "should",
    "have",
    "has",
    "had",
}

MONTH_NAMES = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
}

DAY_NAMES = {
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
}

CAPITALIZED_TIME_WORDS = MONTH_NAMES | DAY_NAMES

SINGULAR_INDEFINITE_PRONOUNS = {
    "another",
    "anybody",
    "anyone",
    "anything",
    "each",
    "either",
    "everybody",
    "everyone",
    "everything",
    "much",
    "neither",
    "nobody",
    "nothing",
    "one",
    "somebody",
    "someone",
    "something",
}

PLURAL_INDEFINITE_PRONOUNS = {
    "both",
    "few",
    "fewer",
    "many",
    "others",
    "several",
    "they",
}

SINGULAR_WORDS_ENDING_WITH_S = {
    "news",
    "series",
    "species",
    "mathematics",
    "physics",
    "economics",
    "politics",
}

IRREGULAR_PLURAL_NOUNS = {
    "people",
    "children",
    "men",
    "women",
    "teeth",
    "feet",
    "mice",
    "geese",
}

SINGULAR_DETERMINERS = {"a", "an", "this", "that", "each", "every"}
PLURAL_DETERMINERS = {"these", "those", "many", "few", "several"}

NUMBER_WORDS = {
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
    "hundred",
    "thousand",
}

NAME_LIKE_MAX_ZIPF = 2.7

COMMON_ALWAYS_VALID = {
    "i",
    "a",
    "an",
    "the",
    "am",
    "is",
    "are",
    "was",
    "were",
    "to",
    "of",
    "in",
    "on",
    "at",
    "my",
    "your",
    "our",
    "their",
    "it's",
    "don't",
    "can't",
    "won't",
    "i'm",
    "we're",
    "they're",
    "you're",
}

COMMON_MISSPELLINGS = {
    "dont": "don't",
    "cant": "can't",
    "wont": "won't",
    "shouldnt": "shouldn't",
    "couldnt": "couldn't",
    "wouldnt": "wouldn't",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "isnt": "isn't",
    "arent": "aren't",
    "wasnt": "wasn't",
    "werent": "weren't",
    "teh": "the",
    "recieve": "receive",
    "definately": "definitely",
    "goverment": "government",
    "writting": "writing",
    "theer": "there",
}

PRONOUN_SUBJECTS = {"i", "you", "we", "they", "he", "she", "it"}

CONTEXTUAL_SPELLING_RULES = {
    "ar": {
        "suggestion": "are",
        "previous_words": {"how", "what", "when", "where", "why", "who", "hi", "hello", "hey"},
        "rule": "Contextual spelling",
        "message": "Did you mean 'are' instead of 'ar' in this question context?",
    },
    "ypu": {
        "suggestion": "you",
        "rule": "Contextual spelling",
        "message": "Did you mean 'you' instead of 'ypu'?",
    },
    "yu": {
        "suggestion": "you",
        "rule": "Contextual spelling",
        "message": "Did you mean 'you' instead of 'yu'?",
    },
    "ia": {
        "suggestion": "is",
        "previous_words": {"there"},
        "next_words": {"a", "an", "the"},
        "rule": "Contextual spelling",
        "message": "Did you mean 'is' instead of 'ia' after 'there'?",
    },
    "isa": {
        "suggestion": "is",
        "previous_words": {"there"},
        "next_words": {"a", "an", "the"},
        "rule": "Contextual spelling",
        "message": "Did you mean 'is' instead of 'isa' after 'there'?",
    },
    "bout": {
        "suggestion": "bought",
        "previous_words": PRONOUN_SUBJECTS,
        "rule": "Contextual spelling",
        "message": "Did you mean 'bought' instead of 'bout' in this context?",
    },
    "bandanna": {
        "suggestion": "banana",
        "previous_words": {"eat", "eats", "ate", "eating"},
        "rule": "Contextual spelling",
        "message": "Did you mean 'banana' in this food context?",
    },
    "raning": {
        "suggestion": "raining",
        "previous_words": {"am", "is", "are", "was", "were", "be", "been", "being"},
        "rule": "Contextual spelling",
        "message": "Did you mean 'raining' instead of 'raning' in this context?",
    },
    "tred": {
        "suggestion": "tired",
        "previous_words": {"am", "is", "are", "was", "were", "be", "been", "being"},
        "rule": "Contextual spelling",
        "message": "Did you mean 'tired' instead of 'tred' in this context?",
    },
    "goed": {
        "suggestion": "went",
        "previous_words": {"i", "you", "we", "they", "he", "she", "it"},
        "next_words": {"to"},
        "rule": "Contextual spelling",
        "message": "Use 'went' as the past tense of 'go'.",
    },
    "everydays": {
        "suggestion": "every day",
        "rule": "Contextual spelling",
        "message": "Use 'every day' for frequency, not 'everydays'.",
    },
    "beautifull": {
        "suggestion": "beautiful",
        "rule": "Contextual spelling",
        "message": "Did you mean 'beautiful' instead of 'beautifull'?",
    },
    "he": {
        "suggestion": "her",
        "previous_words": {"finished"},
        "next_words": {"homework"},
        "rule": "Contextual spelling",
        "message": "Did you mean 'her' instead of 'he' before 'homework'?",
    },
}


@dataclass
class Token:
    lexeme: str
    token_type: str
    line: int
    column: int


@lru_cache(maxsize=1)
def _lexicon() -> set[str]:
    words = {word.lower() for word in top_n_list("en", 80000)}
    words.update(COMMON_ALWAYS_VALID)
    return words


@lru_cache(maxsize=1)
def _lexicon_buckets() -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = defaultdict(list)
    for word in _lexicon():
        if word:
            buckets[word[0]].append(word)
    return buckets


def lexical_analysis(text: str) -> list[Token]:
    tokens: list[Token] = []
    line = 1
    column = 1

    for match in TOKEN_REGEX.finditer(text):
        token_type = match.lastgroup or "OTHER"
        lexeme = match.group(0)
        tokens.append(Token(lexeme=lexeme, token_type=token_type, line=line, column=column))

        if token_type == "NEWLINE":
            line += 1
            column = 1
        else:
            column += len(lexeme)

    return tokens


def _is_valid_word(word: str) -> bool:
    candidate = word.lower()
    if len(candidate) <= 2:
        return True
    if candidate in COMMON_MISSPELLINGS:
        return False
    if candidate in _lexicon():
        return not _is_suspect_misspelling(candidate)
    if _zipf(candidate) <= 2.4:
        return False
    return not _is_suspect_misspelling(candidate)


@lru_cache(maxsize=20000)
def _zipf(word: str) -> float:
    return zipf_frequency(word, "en")


@lru_cache(maxsize=5000)
def _suggest_word(word: str) -> list[str]:
    lower_word = word.lower()
    if not lower_word:
        return []

    bucket = _lexicon_buckets().get(lower_word[0], [])
    target_len = len(lower_word)
    narrowed = [w for w in bucket if abs(len(w) - target_len) <= 2]
    suggestions = [s for s in get_close_matches(lower_word, narrowed, n=10, cutoff=0.76) if s != lower_word]

    common_fix = COMMON_MISSPELLINGS.get(lower_word)
    if common_fix:
        suggestions.insert(0, common_fix)
    suggestions.sort(
        key=lambda candidate: (
            -SequenceMatcher(None, lower_word, candidate).ratio(),
            abs(len(candidate) - target_len),
            -_zipf(candidate),
        )
    )
    deduped: list[str] = []
    for suggestion in suggestions:
        if suggestion not in deduped:
            deduped.append(suggestion)

    if common_fix:
        deduped = [common_fix] + [s for s in deduped if s != common_fix]

    return deduped[:3]


@lru_cache(maxsize=20000)
def _is_suspect_misspelling(word: str) -> bool:
    if word in COMMON_ALWAYS_VALID or len(word) <= 3:
        return False

    suggestions = _suggest_word(word)
    if not suggestions:
        return False

    best = suggestions[0]
    ratio = SequenceMatcher(None, word, best).ratio()
    if ratio < 0.82:
        return False

    return (_zipf(best) - _zipf(word)) >= 1.2


def _contextual_spelling_issue(
    word: str,
    previous_word: str | None,
    next_word: str | None,
) -> dict[str, Any] | None:
    rule = CONTEXTUAL_SPELLING_RULES.get(word.lower())
    if not rule:
        return None

    previous_words = rule.get("previous_words")
    if previous_words and (not previous_word or previous_word.lower() not in previous_words):
        return None

    next_words = rule.get("next_words")
    if next_words and (not next_word or next_word.lower() not in next_words):
        return None

    return {
        "word": word,
        "suggestions": [rule["suggestion"]],
        "rule": rule["rule"],
        "message": rule["message"],
    }


def _detect_grammar_issues(text: str) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []

    for repeated in REPEATED_WORD_REGEX.finditer(text):
        issues.append(
            {
                "rule": "Repeated word",
                "message": f"Repeated word detected: '{repeated.group(1)}'.",
                "severity": "warning",
                "position": repeated.start(),
            }
        )

    for match in ARTICLE_REGEX.finditer(text):
        article = match.group(1).lower()
        next_word = match.group(2)
        starts_with_vowel = next_word[0].lower() in "aeiou"

        if article == "a" and starts_with_vowel:
            issues.append(
                {
                    "rule": "Article agreement",
                    "message": f"Use 'an' before '{next_word}'.",
                    "severity": "info",
                    "position": match.start(),
                }
            )
        elif article == "an" and not starts_with_vowel:
            issues.append(
                {
                    "rule": "Article agreement",
                    "message": f"Use 'a' before '{next_word}'.",
                    "severity": "info",
                    "position": match.start(),
                }
            )

    sentences = [s.strip() for s in SENTENCE_REGEX.findall(text) if s.strip()]
    for sentence in sentences:
        first_char = sentence[0]
        if first_char.isalpha() and first_char.islower():
            issues.append(
                {
                    "rule": "Sentence capitalization",
                    "message": f"Sentence should start with uppercase: '{sentence[:30]}...'.",
                    "severity": "info",
                    "position": text.find(sentence),
                }
            )

    for match in STANDALONE_I_REGEX.finditer(text):
        issues.append(
            {
                "rule": "Pronoun capitalization",
                "message": "Capitalize standalone pronoun 'I'.",
                "severity": "info",
                "position": match.start(),
            }
        )

    for match in re.finditer(r"\b([A-Za-z]+)\b", text):
        word = match.group(1)
        lower_word = word.lower()
        if lower_word in CAPITALIZED_TIME_WORDS and not word[0].isupper():
            issues.append(
                {
                    "rule": "Capital letters",
                    "message": f"Capitalize '{word}' when referring to days or months.",
                    "severity": "info",
                    "position": match.start(),
                }
            )

    for match in NAME_INTRO_REGEX.finditer(text):
        name = match.group(2)
        if (
            name
            and name[0].islower()
            and name.lower() not in NUMBER_WORDS
            and _zipf(name.lower()) < NAME_LIKE_MAX_ZIPF
        ):
            issues.append(
                {
                    "rule": "Name capitalization",
                    "message": f"Capitalize likely person name: '{name}'.",
                    "severity": "info",
                    "position": match.start(2),
                }
            )

    for match in SENTENCE_JOIN_REGEX.finditer(text):
        issues.append(
            {
                "rule": "Punctuation and connector usage",
                "message": "Use punctuation or a connector to join related clauses on the same line.",
                "severity": "info",
                "position": match.end(),
            }
        )

    for match in SUBJECT_BE_REGEX.finditer(text):
        subject_head = match.group("head")
        if subject_head.lower() == "there":
            continue
        if subject_head.lower() in QUESTION_WORDS or subject_head.lower() in QUESTION_AUXILIARY_STARTERS:
            continue

        expected = _expected_be_for_subject(match.group("det"), subject_head)
        actual = match.group("verb").lower()
        if actual != expected:
            issues.append(
                {
                    "rule": "Subject-verb agreement",
                    "message": f"Use '{expected}' with subject '{subject_head}'.",
                    "severity": "warning",
                    "position": match.start("verb"),
                }
            )

    for match in SUBJECT_HAVE_REGEX.finditer(text):
        subject = match.group("subject")
        actual = match.group("verb").lower()
        lower_subject = subject.lower()

        if lower_subject in {"i", "you", "we", "they"}:
            expected = "have"
        elif lower_subject in {"he", "she", "it"}:
            expected = "has"
        else:
            expected = "have" if _is_likely_plural_noun(lower_subject) else "has"

        if actual != expected:
            issues.append(
                {
                    "rule": "Subject-verb agreement",
                    "message": f"Use '{expected}' with subject '{subject}'.",
                    "severity": "warning",
                    "position": match.start("verb"),
                }
            )

    for match in SUBJECT_WAS_WERE_REGEX.finditer(text):
        subject = match.group("subject")
        actual = match.group("verb").lower()
        lower_subject = subject.lower()

        if lower_subject in {"you", "we", "they"}:
            expected = "were"
        elif lower_subject in {"i", "he", "she", "it"}:
            expected = "was"
        else:
            expected = "were" if _is_likely_plural_noun(lower_subject) else "was"

        if actual != expected:
            issues.append(
                {
                    "rule": "Subject-verb agreement",
                    "message": f"Use '{expected}' with subject '{subject}'.",
                    "severity": "warning",
                    "position": match.start("verb"),
                }
            )

    for match in THIRD_PERSON_DONT_REGEX.finditer(text):
        issues.append(
            {
                "rule": "Subject-verb agreement",
                "message": "Use 'doesn't' with he/she/it in simple present.",
                "severity": "warning",
                "position": match.start(),
            }
        )

    for match in THIRD_PERSON_DONT_CONTRACTION_REGEX.finditer(text):
        issues.append(
            {
                "rule": "Subject-verb agreement",
                "message": "Use 'doesn't' with he/she/it in simple present.",
                "severity": "warning",
                "position": match.start(),
            }
        )

    for match in PRESENT_PERFECT_WENT_REGEX.finditer(text):
        issues.append(
            {
                "rule": "Verb tense",
                "message": "Use 'gone' after has/have/had in the present perfect.",
                "severity": "warning",
                "position": match.start(),
            }
        )

    for match in THAN_THEN_COMPARISON_REGEX.finditer(text):
        left = match.group("left")
        if left.lower().endswith("er") or left.lower().startswith("more "):
            issues.append(
                {
                    "rule": "Comparison word choice",
                    "message": "Use 'than' (not 'then') for comparisons.",
                    "severity": "info",
                    "position": match.start(),
                }
            )

    for match in REDUNDANT_MORE_COMPARATIVE_REGEX.finditer(text):
        issues.append(
            {
                "rule": "Comparative form",
                "message": f"Use either 'more {match.group('adj')[:-2]}' or '{match.group('adj')}', not both.",
                "severity": "info",
                "position": match.start(),
            }
        )

    for match in THERE_THEIR_POSSESSIVE_REGEX.finditer(text):
        noun = match.group("noun").lower()
        if noun in POSSESSIVE_NOUN_HINTS:
            issues.append(
                {
                    "rule": "Homophone choice",
                    "message": "Use 'their' for possession, not 'there'.",
                    "severity": "warning",
                    "position": match.start(),
                }
            )

    for match in NAME_AS_VERB_REGEX.finditer(text):
        issues.append(
            {
                "rule": "Word form",
                "message": "Use 'named' when introducing a name (e.g., 'a dog named Bruno').",
                "severity": "info",
                "position": match.start(),
            }
        )

    for match in STATIVE_PROGRESSIVE_REGEX.finditer(text):
        issues.append(
            {
                "rule": "Verb form",
                "message": "Use 'do not understand' instead of 'am not understanding' in standard usage.",
                "severity": "info",
                "position": match.start(),
            }
        )

    for match in PLURAL_QUANTIFIER_NOUN_REGEX.finditer(text):
        noun = match.group("noun")
        if not _is_likely_plural_noun(noun):
            issues.append(
                {
                    "rule": "Noun number agreement",
                    "message": f"Use a plural noun after '{match.group('qty')}' (e.g., '{_pluralize_noun(noun)}').",
                    "severity": "info",
                    "position": match.start("noun"),
                }
            )

    for match in THIRD_PERSON_SIMPLE_BASE_REGEX.finditer(text):
        subject = match.group("subject")
        verb = match.group("verb")
        if subject.lower() in {"he", "she", "it"} or subject.lower().startswith(("my ", "the ")) or subject[0].isupper():
            issues.append(
                {
                    "rule": "Subject-verb agreement",
                    "message": f"Use '{_third_person_singular(verb)}' with singular subject '{subject}'.",
                    "severity": "warning",
                    "position": match.start("verb"),
                }
            )

    for match in PROGRESSIVE_NOW_BE_REGEX.finditer(text):
        issues.append(
            {
                "rule": "Tense consistency",
                "message": "Use present progressive with 'now' (am/is/are + verb-ing).",
                "severity": "info",
                "position": match.start("verb"),
            }
        )

    for match in DO_SUPPORT_BASE_FORM_REGEX.finditer(text):
        verb = match.group("verb")
        if _looks_like_third_person_singular_form(verb):
            issues.append(
                {
                    "rule": "Verb form",
                    "message": f"Use base verb '{_to_base_verb(verb)}' after auxiliary '{match.group('aux')}'.",
                    "severity": "warning",
                    "position": match.start("verb"),
                }
            )

    for match in TO_BASE_FORM_REGEX.finditer(text):
        verb = match.group("verb")
        if _looks_like_third_person_singular_form(verb):
            issues.append(
                {
                    "rule": "Verb form",
                    "message": f"Use base verb '{_to_base_verb(verb)}' after 'to'.",
                    "severity": "info",
                    "position": match.start("verb"),
                }
            )

    for match in WHEN_PAST_PROGRESSIVE_BASE_REGEX.finditer(text):
        issues.append(
            {
                "rule": "Tense consistency",
                "message": "Use past tense in the 'when' clause after a past progressive clause.",
                "severity": "info",
                "position": match.start("verb"),
            }
        )

    for match in THERE_BE_REGEX.finditer(text):
        expected = _expected_be_after_there(match.group("next"))
        actual = match.group("verb").lower()
        if actual != expected:
            issues.append(
                {
                    "rule": "There is/are agreement",
                    "message": f"Use 'there {expected}' with '{match.group('next')}'.",
                    "severity": "warning",
                    "position": match.start("verb"),
                }
            )

    for match in A_NUMBER_OF_REGEX.finditer(text):
        if match.group("verb").lower() != "are":
            issues.append(
                {
                    "rule": "Number phrase agreement",
                    "message": "Use 'are' with the phrase 'a number of ...'.",
                    "severity": "info",
                    "position": match.start("verb"),
                }
            )

    for match in THE_NUMBER_OF_REGEX.finditer(text):
        if match.group("verb").lower() != "is":
            issues.append(
                {
                    "rule": "Number phrase agreement",
                    "message": "Use 'is' with the phrase 'the number of ...'.",
                    "severity": "info",
                    "position": match.start("verb"),
                }
            )

    for match in COMMA_SPLICE_REGEX.finditer(text):
        right_lead = match.group("right").lstrip().lower()
        if any(right_lead.startswith(f"{conj} ") for conj in ("and", "but", "or", "nor", "for", "so", "yet")):
            continue
        issues.append(
            {
                "rule": "Comma splice",
                "message": "A comma alone cannot join two independent clauses. Use a conjunction, semicolon, or period.",
                "severity": "warning",
                "position": match.start("right"),
            }
        )

    for match in MISSING_COMMA_BEFORE_COORDINATING_CONJ_REGEX.finditer(text):
        issues.append(
            {
                "rule": "Comma with coordinating conjunction",
                "message": f"Add a comma before '{match.group('conj').lower()}' when joining independent clauses.",
                "severity": "info",
                "position": match.start("conj"),
            }
        )

    for match in COMMA_BEFORE_THAN_REGEX.finditer(text):
        issues.append(
            {
                "rule": "Comma with than-comparison",
                "message": "Do not use a comma before 'than' in comparisons.",
                "severity": "info",
                "position": match.start(),
            }
        )

    lines = text.splitlines()
    offset = 0
    for line in lines:
        stripped_line = line.rstrip()
        if stripped_line and stripped_line[-1] not in ".!?":
            issues.append(
                {
                    "rule": "Sentence punctuation",
                    "message": "Sentence should end with terminal punctuation (. ! ?).",
                    "severity": "info",
                    "position": offset + len(stripped_line) - 1,
                }
            )
        offset += len(line) + 1

    return issues


def _fix_article_case(source_article: str, replacement: str) -> str:
    if source_article.isupper():
        return replacement.upper()
    if source_article.istitle():
        return replacement.title()
    return replacement


def _capitalize_likely_person_names(text: str) -> str:
    # Heuristic: words after common self/introduction phrases are likely names.
    def replacer(match: re.Match[str]) -> str:
        prefix = match.group(1)
        name = match.group(2)
        if name.lower() in NUMBER_WORDS or _zipf(name.lower()) >= NAME_LIKE_MAX_ZIPF:
            return match.group(0)
        return f"{prefix} {name[0].upper()}{name[1:]}"

    return NAME_INTRO_REGEX.sub(replacer, text)


def _is_question_sentence(sentence_body: str) -> bool:
    candidate = sentence_body.strip().lower()
    if not candidate:
        return False

    candidate = LEADING_GREETING_REGEX.sub("", candidate).strip()
    if not candidate:
        return False

    for phrase in QUESTION_PHRASES:
        if candidate == phrase or candidate.startswith(f"{phrase} "):
            return True

    match = re.match(r"[a-z]+", candidate)
    if not match:
        return False

    first_word = match.group(0)
    if first_word in QUESTION_WORDS:
        return True

    return first_word in QUESTION_AUXILIARY_STARTERS


def _is_likely_plural_noun(word: str) -> bool:
    lower = word.lower()
    if lower in IRREGULAR_PLURAL_NOUNS:
        return True
    if lower in SINGULAR_WORDS_ENDING_WITH_S:
        return False
    return len(lower) > 1 and lower.endswith("s")


def _pluralize_noun(noun: str) -> str:
    lower = noun.lower()
    if lower.endswith("y") and len(lower) > 1 and lower[-2] not in "aeiou":
        return f"{noun[:-1]}ies"
    if lower.endswith(("s", "x", "z", "ch", "sh")):
        return f"{noun}es"
    return f"{noun}s"


def _third_person_singular(verb: str) -> str:
    lower = verb.lower()
    irregular = {"have": "has", "do": "does", "go": "goes"}
    if lower in irregular:
        out = irregular[lower]
    elif lower.endswith("y") and len(lower) > 1 and lower[-2] not in "aeiou":
        out = f"{lower[:-1]}ies"
    elif lower.endswith(("s", "x", "z", "ch", "sh", "o")):
        out = f"{lower}es"
    else:
        out = f"{lower}s"

    if verb.isupper():
        return out.upper()
    if verb.istitle():
        return out.title()
    return out


def _present_be_for_subject(subject: str) -> str:
    lower = subject.lower()
    if lower == "i":
        return "am"
    if lower in {"you", "we", "they"}:
        return "are"
    return "is"


def _to_base_verb(verb: str) -> str:
    lower = verb.lower()
    irregular = {
        "does": "do",
        "has": "have",
        "is": "be",
        "are": "be",
        "was": "be",
        "were": "be",
    }
    if lower in irregular:
        base = irregular[lower]
    elif lower.endswith("ies") and len(lower) > 3:
        base = f"{lower[:-3]}y"
    elif lower.endswith("es") and len(lower) > 2 and lower[:-2].endswith(("s", "x", "z", "ch", "sh", "o")):
        base = lower[:-2]
    elif lower.endswith("s") and len(lower) > 1:
        base = lower[:-1]
    else:
        base = lower

    if verb.isupper():
        return base.upper()
    if verb.istitle():
        return base.title()
    return base


def _looks_like_third_person_singular_form(verb: str) -> bool:
    lower = verb.lower()
    if lower in {"is", "has", "does"}:
        return True
    if lower.endswith("ies") and len(lower) > 3:
        return True
    if lower.endswith("es") and len(lower) > 2 and lower[:-2].endswith(("s", "x", "z", "ch", "sh", "o")):
        return True
    return lower.endswith("s") and len(lower) > 1


def _expected_be_for_subject(det: str | None, head: str) -> str:
    lower_head = head.lower()
    lower_det = det.lower() if det else None

    if lower_head == "i":
        return "am"

    if lower_head in {"you", "we", "they", "these", "those"}:
        return "are"

    if lower_head in {"he", "she", "it", "this", "that"}:
        return "is"

    if lower_head in SINGULAR_INDEFINITE_PRONOUNS:
        return "is"

    if lower_head in PLURAL_INDEFINITE_PRONOUNS:
        return "are"

    if lower_det in SINGULAR_DETERMINERS:
        return "is"

    if lower_det in PLURAL_DETERMINERS:
        return "are"

    return "are" if _is_likely_plural_noun(lower_head) else "is"


def _expected_be_after_there(next_word: str) -> str:
    lower = next_word.lower()

    if lower in {"a", "an", "one", "each", "every", "this", "that"}:
        return "is"

    if lower in {"many", "few", "several", "these", "those"}:
        return "are"

    if lower in NUMBER_WORDS - {"one"}:
        return "are"

    return "are" if _is_likely_plural_noun(lower) else "is"


def _expected_have_for_subject(subject: str) -> str:
    lower_subject = subject.lower()
    if lower_subject in {"i", "you", "we", "they"}:
        return "have"
    if lower_subject in {"he", "she", "it"}:
        return "has"
    return "have" if _is_likely_plural_noun(lower_subject) else "has"


def _expected_was_were_for_subject(subject: str) -> str:
    lower_subject = subject.lower()
    if lower_subject in {"you", "we", "they"}:
        return "were"
    if lower_subject in {"i", "he", "she", "it"}:
        return "was"
    return "were" if _is_likely_plural_noun(lower_subject) else "was"


def _do_support_for_subject(subject: str) -> str:
    return "does" if subject.lower() in {"he", "she", "it"} else "do"


def _apply_grammar_corrections(text: str) -> str:
    corrected = text

    # Remove immediate duplicate words such as "has has".
    corrected = re.sub(r"\b([A-Za-z]+)(\s+)\1\b", r"\1", corrected, flags=re.IGNORECASE)

    # Normalize the standalone pronoun "i" to uppercase.
    corrected = STANDALONE_I_REGEX.sub("I", corrected)

    corrected = re.sub(
        r"\b(" + "|".join(sorted(CAPITALIZED_TIME_WORDS, key=len, reverse=True)) + r")\b",
        lambda m: m.group(1).capitalize(),
        corrected,
        flags=re.IGNORECASE,
    )

    corrected = _capitalize_likely_person_names(corrected)

    corrected = SENTENCE_JOIN_REGEX.sub(r"\1, and ", corrected)

    def fix_subject_be(match: re.Match[str]) -> str:
        det = match.group("det")
        head = match.group("head")
        verb = match.group("verb")

        if head.lower() == "there":
            return match.group(0)
        if head.lower() in QUESTION_WORDS or head.lower() in QUESTION_AUXILIARY_STARTERS:
            return match.group(0)

        expected = _expected_be_for_subject(det, head)
        if verb.lower() == expected:
            return match.group(0)

        if verb.isupper():
            expected = expected.upper()
        elif verb.istitle():
            expected = expected.title()

        if det:
            return f"{det} {head} {expected}"
        return f"{head} {expected}"

    corrected = SUBJECT_BE_REGEX.sub(fix_subject_be, corrected)

    def fix_subject_have(match: re.Match[str]) -> str:
        subject = match.group("subject")
        verb = match.group("verb")
        expected = _expected_have_for_subject(subject)
        if verb.lower() == expected:
            return match.group(0)
        if verb.isupper():
            expected = expected.upper()
        elif verb.istitle():
            expected = expected.title()
        return f"{subject} {expected}"

    corrected = SUBJECT_HAVE_REGEX.sub(fix_subject_have, corrected)

    def fix_subject_was_were(match: re.Match[str]) -> str:
        subject = match.group("subject")
        verb = match.group("verb")
        expected = _expected_was_were_for_subject(subject)
        if verb.lower() == expected:
            return match.group(0)
        if verb.isupper():
            expected = expected.upper()
        elif verb.istitle():
            expected = expected.title()
        return f"{subject} {expected}"

    corrected = SUBJECT_WAS_WERE_REGEX.sub(fix_subject_was_were, corrected)

    def fix_there_be(match: re.Match[str]) -> str:
        verb = match.group("verb")
        next_word = match.group("next")
        expected = _expected_be_after_there(next_word)
        if verb.lower() == expected:
            return match.group(0)

        if verb.isupper():
            expected = expected.upper()
        elif verb.istitle():
            expected = expected.title()

        return f"there {expected} {next_word}"

    corrected = THERE_BE_REGEX.sub(fix_there_be, corrected)
    corrected = THIRD_PERSON_DONT_REGEX.sub(lambda m: f"{m.group('subject')} doesn't", corrected)
    corrected = THIRD_PERSON_DONT_CONTRACTION_REGEX.sub(lambda m: f"{m.group('subject')} doesn't", corrected)
    corrected = PRESENT_PERFECT_WENT_REGEX.sub(lambda m: f"{m.group('aux')} gone", corrected)
    corrected = THAN_THEN_COMPARISON_REGEX.sub(
        lambda m: f"{m.group('left')} than" if (m.group('left').lower().endswith('er') or m.group('left').lower().startswith('more ')) else m.group(0),
        corrected,
    )
    corrected = REDUNDANT_MORE_COMPARATIVE_REGEX.sub(lambda m: m.group("adj"), corrected)
    corrected = THERE_THEIR_POSSESSIVE_REGEX.sub(
        lambda m: f"their {m.group('noun')}" if m.group("noun").lower() in POSSESSIVE_NOUN_HINTS else m.group(0),
        corrected,
    )
    corrected = NAME_AS_VERB_REGEX.sub(lambda m: f"{m.group('noun')} named {m.group('name')}", corrected)
    corrected = STATIVE_PROGRESSIVE_REGEX.sub(
        lambda m: f"{m.group('subject')} {_do_support_for_subject(m.group('subject'))} not understand",
        corrected,
    )
    corrected = PRESENT_PERFECT_YESTERDAY_REGEX.sub(
        lambda m: f"{m.group('subject')} went{m.group('tail')}",
        corrected,
    )
    corrected = PROGRESSIVE_NOW_BE_REGEX.sub(
        lambda m: f"{m.group('subject')} {_present_be_for_subject(m.group('subject'))} {m.group('prog')}{m.group('tail')}",
        corrected,
    )
    corrected = PLURAL_QUANTIFIER_NOUN_REGEX.sub(
        lambda m: f"{m.group('qty')} {_pluralize_noun(m.group('noun'))}"
        if not _is_likely_plural_noun(m.group("noun"))
        else m.group(0),
        corrected,
    )
    corrected = THIRD_PERSON_SIMPLE_BASE_REGEX.sub(
        lambda m: f"{m.group('subject')} {_third_person_singular(m.group('verb'))}",
        corrected,
    )
    corrected = DO_SUPPORT_BASE_FORM_REGEX.sub(
        lambda m: f"{m.group('subject')} {m.group('aux')} {_to_base_verb(m.group('verb'))}"
        if _looks_like_third_person_singular_form(m.group("verb"))
        else m.group(0),
        corrected,
    )
    corrected = TO_BASE_FORM_REGEX.sub(
        lambda m: f"to {_to_base_verb(m.group('verb'))}"
        if _looks_like_third_person_singular_form(m.group("verb"))
        else m.group(0),
        corrected,
    )
    corrected = WHEN_PAST_PROGRESSIVE_BASE_REGEX.sub(lambda m: f"{m.group('left')}started", corrected)
    corrected = A_NUMBER_OF_REGEX.sub(lambda m: m.group(0)[: m.start("verb") - m.start()] + "are", corrected)
    corrected = THE_NUMBER_OF_REGEX.sub(lambda m: m.group(0)[: m.start("verb") - m.start()] + "is", corrected)

    corrected = COMMA_BEFORE_THAN_REGEX.sub(r" \1", corrected)

    def fix_coordinating_conjunction_comma(match: re.Match[str]) -> str:
        left = match.group("left").rstrip()
        conj = match.group("conj")
        right = match.group("right").lstrip()
        end = match.group("end")
        return f"{left}, {conj} {right}{end}"

    corrected = MISSING_COMMA_BEFORE_COORDINATING_CONJ_REGEX.sub(
        fix_coordinating_conjunction_comma, corrected
    )

    def fix_comma_splice(match: re.Match[str]) -> str:
        left = match.group("left").rstrip()
        right = match.group("right").lstrip()
        end = match.group("end")
        if any(right.lower().startswith(f"{conj} ") for conj in ("and", "but", "or", "nor", "for", "so", "yet")):
            return f"{left}, {right}{end}"
        return f"{left}; {right}{end}"

    corrected = COMMA_SPLICE_REGEX.sub(fix_comma_splice, corrected)

    # Align article usage with the first letter of the following word.
    def fix_article(match: re.Match[str]) -> str:
        article = match.group(1)
        next_word = match.group(2)
        required = "an" if next_word[0].lower() in "aeiou" else "a"
        return f"{_fix_article_case(article, required)} {next_word}"

    corrected = re.sub(r"\b(a|an)\s+([A-Za-z]+)\b", fix_article, corrected, flags=re.IGNORECASE)

    # Capitalize line starts and sentence starts after punctuation.
    corrected = re.sub(
        r"(^|\n\s*)([a-z])",
        lambda m: f"{m.group(1)}{m.group(2).upper()}",
        corrected,
        flags=re.MULTILINE,
    )
    corrected = re.sub(
        r"(^|[.!?]\s+)([a-z])",
        lambda m: f"{m.group(1)}{m.group(2).upper()}",
        corrected,
        flags=re.MULTILINE,
    )

    lines = corrected.splitlines(keepends=True)
    if lines:
        fixed_lines: list[str] = []
        for line in lines:
            newline = ""
            body = line
            if line.endswith("\n"):
                body = line[:-1]
                newline = "\n"

            trimmed = body.rstrip()
            if trimmed:
                terminal = trimmed[-1] if trimmed[-1] in ".!?" else ""
                core = trimmed[:-1].rstrip() if terminal else trimmed
                is_question = _is_question_sentence(core)

                if terminal:
                    if terminal != "!" and is_question:
                        body = f"{core}?"
                    else:
                        body = trimmed
                else:
                    body = f"{core}{'?' if is_question else '.'}"
            else:
                body = trimmed if trimmed else body

            fixed_lines.append(f"{body}{newline}")

        corrected = "".join(fixed_lines)

    stripped = corrected.rstrip()
    if stripped and stripped[-1] not in ".!?":
        corrected = f"{stripped}."

    return corrected


def _apply_basic_corrections(text: str, replacements: dict[str, str]) -> str:
    corrected = text

    def replacer(match: re.Match[str]) -> str:
        source = match.group(0)
        replacement = replacements.get(source.lower())
        if not replacement:
            return source
        if source.istitle():
            return replacement.title()
        if source.isupper():
            return replacement.upper()
        return replacement

    corrected = WORD_MATCH_REGEX.sub(replacer, corrected)
    return _apply_grammar_corrections(corrected)


def analyze_text(text: str) -> dict[str, Any]:
    tokens = lexical_analysis(text)
    word_tokens = [t for t in tokens if t.token_type == "WORD"]

    misspelled_instances: list[dict[str, Any]] = []
    frequency_counter: Counter[str] = Counter()
    validity_cache: dict[str, bool] = {}
    suggestion_cache: dict[str, list[str]] = {}

    previous_word: str | None = None

    for index, token in enumerate(word_tokens):
        next_word = word_tokens[index + 1].lexeme if index + 1 < len(word_tokens) else None
        contextual_issue = _contextual_spelling_issue(token.lexeme, previous_word, next_word)
        if contextual_issue is not None:
            frequency_counter[token.lexeme.lower()] += 1
            misspelled_instances.append(
                {
                    "word": contextual_issue["word"],
                    "line": token.line,
                    "column": token.column,
                    "suggestions": contextual_issue["suggestions"],
                    "rule": contextual_issue["rule"],
                    "message": contextual_issue["message"],
                }
            )
            previous_word = token.lexeme
            continue

        lower_lexeme = token.lexeme.lower()
        is_valid = validity_cache.get(lower_lexeme)
        if is_valid is None:
            is_valid = _is_valid_word(token.lexeme)
            validity_cache[lower_lexeme] = is_valid

        if is_valid:
            previous_word = token.lexeme
            continue

        frequency_counter[token.lexeme.lower()] += 1
        misspelled_instances.append(
            {
                "word": token.lexeme,
                "line": token.line,
                "column": token.column,
                "suggestions": suggestion_cache.setdefault(lower_lexeme, _suggest_word(token.lexeme)),
                "rule": "Spelling",
            }
        )
        previous_word = token.lexeme

    unique_misspellings: list[dict[str, Any]] = []
    replacements: dict[str, str] = {}
    seen: set[str] = set()

    for item in misspelled_instances:
        key = item["word"].lower()
        if key in seen:
            continue
        seen.add(key)
        first_suggestion = item["suggestions"][0] if item["suggestions"] else None
        if first_suggestion:
            replacements[key] = first_suggestion
        unique_misspellings.append(
            {
                "word": item["word"],
                "count": frequency_counter[key],
                "suggestions": item["suggestions"],
                "rule": item.get("rule", "Spelling"),
            }
        )

    grammar_issues = _detect_grammar_issues(text)
    corrected_text = _apply_basic_corrections(text, replacements)

    return {
        "summary": {
            "character_count": len(text),
            "token_count": len(tokens),
            "word_count": len(word_tokens),
            "misspelled_word_count": len(misspelled_instances),
            "unique_misspelled_word_count": len(unique_misspellings),
            "grammar_issue_count": len(grammar_issues),
        },
        "tokens": [asdict(token) for token in tokens],
        "misspellings": unique_misspellings,
        "grammar_issues": grammar_issues,
        "corrected_text": corrected_text,
    }
