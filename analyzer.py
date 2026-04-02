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


def _apply_grammar_corrections(text: str) -> str:
    corrected = text

    # Remove immediate duplicate words such as "has has".
    corrected = re.sub(r"\b([A-Za-z]+)(\s+)\1\b", r"\1", corrected, flags=re.IGNORECASE)

    # Normalize the standalone pronoun "i" to uppercase.
    corrected = re.sub(r"(?<![A-Za-z'])i(?![A-Za-z'])", "I", corrected)

    corrected = _capitalize_likely_person_names(corrected)

    corrected = SENTENCE_JOIN_REGEX.sub(r"\1, and ", corrected)

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
            if trimmed and trimmed[-1] not in ".!?":
                body = f"{trimmed}."
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
