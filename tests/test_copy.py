"""Copy rules for every word this repo ships.

Four checks, each over the shipped text files (the walk below skips the
corpus, the built index, the recorded fixtures and the eval result files,
which are data rather than copy):

  1. no smart punctuation: em dash, en dash, and the four curly quotes.
     Those six code points are the ones a word processor or an assistant
     inserts silently, and they break a grep for a quote in the source.
  2. no non-ASCII anywhere else either. The parser normalizes filing text
     to ASCII on both sides of every comparison, so a stray U+00A0 in a
     test fixture or a docstring is a comparison that will not fire.
  3. no confidential terms, read from the file named by the environment
     variable COPY_DENYLIST_FILE. The list itself lives outside the repo,
     so a public clone has no variable set and this check reports why it
     did not run instead of failing.
  4. no claim that an answer is established: the words "verified",
     "confidence" and "confidently" are absent from the page, the docs,
     the README and the prompt module. The evidence checks establish that
     a quote is present in the excerpt it cites and which column a figure
     sits in. Whether the answer is right is a separate, graded reading,
     so the reader-facing copy never uses a word that blurs the two.

Matching is word-bounded and case-sensitive. Word boundaries are what let
the identifier column_unverified stay: it reads as "not established", and
"verified" inside it is not a word of its own. That is the one exception.

Failure policy: every check collects every hit and fails once with the
whole list as "path:line: text", so one run names every place to fix
rather than one place per run.

Scope note: the test image holds the source tree the Dockerfile copies
into it, so a run inside `docker compose run --rm tests` scans service/
and tests/ and skips README.md and docs/, which are not in the image. A
run from a checkout scans everything.
"""

import os
import re

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directories that hold data, not copy. The corpus is public filing text
# full of curly quotes; the index and the fixtures are derived from it;
# eval results are recorded model output. None of them is written here.
SKIP_DIRS = {
    ".git", ".venv", ".pytest_cache", ".fastembed", "__pycache__", "node_modules",
    "data", "index", "index-tuning", "models",
}
SKIP_PATHS = {
    os.path.join("eval", "fixtures"),
    os.path.join("eval", "results"),
}

# What counts as shipped text. Anything else in the tree (images, the zip,
# the saved index) is binary or data and is not scanned.
TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".json", ".jsonl", ".yaml", ".yml",
    ".html", ".js", ".css", ".sh", ".ini", ".cfg", ".toml",
}
TEXT_NAMES = {"Dockerfile", ".env.example", ".gitignore", ".dockerignore", "pytest.ini"}

# The two Bauhaus stylesheets carry glyphs as CSS escapes inside content:
# declarations (a caret, a minus sign, a check mark). Those are style, not
# copy, so the declaration is dropped before the file is scanned.
VENDORED_CSS = {
    os.path.join("service", "static", "bauhaus.css"),
    os.path.join("service", "static", "tokens.css"),
}
CSS_CONTENT_RE = re.compile(r"content\s*:[^;}]*")

# Written as escapes so this file passes its own ASCII check.
SMART_PUNCTUATION = {
    "\u2014": "em dash",
    "\u2013": "en dash",
    "\u2018": "left single quote",
    "\u2019": "right single quote",
    "\u201c": "left double quote",
    "\u201d": "right double quote",
}

# Where a word about an answer would be read by a person: the page, the
# written documents, and the text sent to the model.
ANSWER_COPY_PREFIXES = (
    os.path.join("service", "static"),
    "docs" + os.sep,
    os.path.join("service", "prompts.py"),
    "README.md",
)
BANNED_WORDS_RE = re.compile(r"\b(verified|confidence|confidently)\b")
# The one sentence where naming what the checks do not do is the point.
BANNED_WORDS_EXEMPT = "what the checks do not establish"

DENYLIST_ENV = "COPY_DENYLIST_FILE"


def shipped_text_files() -> list[str]:
    """Every text file in the repo, as paths relative to the root. A walk
    rather than `git ls-files` because the test image has no git and no
    .git directory: it holds a copy of the source tree only."""
    found = []
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        rel_dir = os.path.relpath(dirpath, REPO_ROOT)
        rel_dir = "" if rel_dir == "." else rel_dir
        if any(rel_dir == p or rel_dir.startswith(p + os.sep) for p in SKIP_PATHS):
            continue
        for name in sorted(filenames):
            if name in TEXT_NAMES or os.path.splitext(name)[1] in TEXT_SUFFIXES:
                found.append(os.path.join(rel_dir, name) if rel_dir else name)
    return found


def read_lines(rel_path: str) -> list[tuple[int, str]]:
    """Numbered lines of one file, with the vendored stylesheets' content:
    declarations removed."""
    with open(os.path.join(REPO_ROOT, rel_path), encoding="utf-8") as handle:
        text = handle.read()
    if rel_path in VENDORED_CSS:
        text = CSS_CONTENT_RE.sub("content:", text)
    return list(enumerate(text.splitlines(), start=1))


def denylist_terms(path: str) -> list[str]:
    """One term per line; blank lines and # comments ignored."""
    with open(path, encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip() and not line.startswith("#")]


def denylist_pattern(terms: list[str]) -> re.Pattern:
    """Word-bounded and case-sensitive, with runs of whitespace inside a
    term matching any whitespace, so a term broken across two lines by a
    text wrap is still found."""
    parts = [re.escape(term).replace("\\ ", r"\s+") for term in terms]
    return re.compile(r"\b(?:%s)\b" % "|".join(parts))


def banned_word_hits(line: str) -> list[str]:
    """The certainty words in one line, or nothing when the line is the
    sentence that names what the checks do not establish."""
    if BANNED_WORDS_EXEMPT in line:
        return []
    return BANNED_WORDS_RE.findall(line)


def report(hits: list[str]) -> str:
    return "\n".join(hits)


def test_no_smart_punctuation():
    hits = []
    for rel_path in shipped_text_files():
        for number, line in read_lines(rel_path):
            for char, name in SMART_PUNCTUATION.items():
                if char in line:
                    hits.append("%s:%d: %s in %r" % (rel_path, number, name, line.strip()[:120]))
    assert not hits, "smart punctuation in shipped text:\n" + report(hits)


def test_shipped_text_is_ascii():
    hits = []
    for rel_path in shipped_text_files():
        for number, line in read_lines(rel_path):
            outside = sorted({char for char in line if ord(char) > 127})
            if outside:
                codes = " ".join("U+%04X" % ord(char) for char in outside)
                hits.append("%s:%d: %s in %r" % (rel_path, number, codes, line.strip()[:120]))
    assert not hits, "non-ASCII in shipped text:\n" + report(hits)


def test_no_confidential_terms():
    path = os.environ.get(DENYLIST_ENV)
    if not path:
        reason = ("%s is not set, so the confidential-term check did not run. "
                  "The list lives outside this repo; set the variable to its path to run it."
                  % DENYLIST_ENV)
        print(reason)
        pytest.skip(reason)
    if not os.path.exists(path):
        pytest.fail("%s points at %s, which does not exist" % (DENYLIST_ENV, path))
    terms = denylist_terms(path)
    if not terms:
        pytest.fail("%s points at %s, which holds no terms" % (DENYLIST_ENV, path))
    pattern = denylist_pattern(terms)
    hits = []
    for rel_path in shipped_text_files():
        for number, line in read_lines(rel_path):
            found = pattern.findall(line)
            if found:
                hits.append("%s:%d: %s" % (rel_path, number, ", ".join(sorted(set(found)))))
    assert not hits, "confidential terms in shipped text:\n" + report(hits)


def test_no_certainty_wording_about_answers():
    hits = []
    for rel_path in shipped_text_files():
        if not rel_path.startswith(ANSWER_COPY_PREFIXES):
            continue
        for number, line in read_lines(rel_path):
            found = banned_word_hits(line)
            if found:
                hits.append("%s:%d: %s in %r" % (
                    rel_path, number, ", ".join(sorted(set(found))), line.strip()[:120]))
    assert not hits, (
        "the page, the docs, the README or the prompt module claims an answer is established:\n"
        + report(hits))


def test_column_unverified_is_not_a_hit():
    """The identifier the evidence checks use stays legal, and the words on
    their own do not. This is the exception the module docstring names."""
    assert banned_word_hits("columns_unverified counts what was not established") == []
    assert banned_word_hits("the answer is verified") == ["verified"]
    assert banned_word_hits("high confidence in the figure") == ["confidence"]
