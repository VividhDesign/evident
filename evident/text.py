"""Tokenisation for BM25: lowercase, alphanumeric tokens, Lucene's English stopword list,
Snowball (Porter2) stemming. Close to Lucene's EnglishAnalyzer used by Pyserini."""

from __future__ import annotations

import re

import Stemmer

# org.apache.lucene.analysis.en.EnglishAnalyzer.ENGLISH_STOP_WORDS_SET
STOPWORDS = frozenset(
    "a an and are as at be but by for if in into is it no not of on or such that the their then "
    "there these they this to was will with".split()
)
_TOKEN = re.compile(r"[a-z0-9]+")
_stemmer = Stemmer.Stemmer("english")


def tokenize(text: str) -> list[str]:
    tokens = [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]
    return _stemmer.stemWords(tokens)
