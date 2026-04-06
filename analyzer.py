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
    if candidate in _lexicon():
        return True
    return zipf_frequency(candidate, "en") > 2.4


@lru_cache(maxsize=5000)
def _suggest_word(word: str) -> list[str]:
    lower_word = word.lower()
    if not lower_word:
        return []

    bucket = _lexicon_buckets().get(lower_word[0], [])
    target_len = len(lower_word)
    narrowed = [w for w in bucket if abs(len(w) - target_len) <= 2]
    suggestions = get_close_matches(lower_word, narrowed, n=8, cutoff=0.76)
    suggestions.sort(
        key=lambda candidate: (
            -zipf_frequency(candidate, "en"),
            -SequenceMatcher(None, lower_word, candidate).ratio(),
            len(candidate),
        )
    )
    return suggestions[:3]


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

    for match in re.finditer(r"(?<![A-Za-z'])i(?![A-Za-z'])", text):
        issues.append(
            {
                "rule": "Pronoun capitalization",
                "message": "Capitalize standalone pronoun 'I'.",
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
            and zipf_frequency(name.lower(), "en") < NAME_LIKE_MAX_ZIPF
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

    if text and not lines:
        stripped = text.strip()
        if stripped and stripped[-1] not in ".!?":
            issues.append(
                {
                    "rule": "Sentence punctuation",
                    "message": "Sentence should end with terminal punctuation (. ! ?).",
                    "severity": "info",
                    "position": max(0, len(text) - 1),
                }
            )

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
        if name.lower() in NUMBER_WORDS or zipf_frequency(name.lower(), "en") >= NAME_LIKE_MAX_ZIPF:
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


def _apply_grammar_corrections(text: str) -> str:
    corrected = text

    # Remove immediate duplicate words such as "has has".
    corrected = re.sub(r"\b([A-Za-z]+)(\s+)\1\b", r"\1", corrected, flags=re.IGNORECASE)

    # Normalize the standalone pronoun "i" to uppercase.
    corrected = re.sub(r"(?<![A-Za-z'])i(?![A-Za-z'])", "I", corrected)

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

    corrected = re.sub(r"\b[A-Za-z]+(?:'[A-Za-z]+)?\b", replacer, corrected)
    return _apply_grammar_corrections(corrected)


def analyze_text(text: str) -> dict[str, Any]:
    tokens = lexical_analysis(text)
    word_tokens = [t for t in tokens if t.token_type == "WORD"]

    misspelled_instances: list[dict[str, Any]] = []
    frequency_counter: Counter[str] = Counter()

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

        if _is_valid_word(token.lexeme):
            previous_word = token.lexeme
            continue

        frequency_counter[token.lexeme.lower()] += 1
        misspelled_instances.append(
            {
                "word": token.lexeme,
                "line": token.line,
                "column": token.column,
                "suggestions": _suggest_word(token.lexeme),
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
