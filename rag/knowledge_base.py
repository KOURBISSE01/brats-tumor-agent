"""Base de connaissances RAG — BM25 en pur Python sur rag/docs/*.md.

Aucune dépendance (pas de faiss/sentence-transformers) : tokenisation
insensible aux accents + stopwords FR, scoring Robertson-Spärck Jones.
Les documents sont découpés par titres `## ` (une section = un chunk).
"""
from __future__ import annotations

import math
import re
import unicodedata
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent / "docs"

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "le", "la", "les", "de", "des", "du", "un", "une", "et", "en", "a", "au",
    "aux", "pour", "sur", "dans", "par", "avec", "que", "qui", "est", "sont",
    "ce", "cette", "ces", "son", "sa", "ses", "il", "elle", "on", "se", "au",
    "plus", "sur", "tout", "tous", "toute", "comme", "mais", "ou", "si", "ne",
    "pas", "dans", "the", "of", "and", "est", "etre", "avoir",
}


def tokenize(text: str) -> list[str]:
    """Minuscules, accents supprimés, tokens alphanumériques hors stopwords."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return [t for t in _TOKEN_RE.findall(text) if t not in _STOPWORDS]


class KnowledgeBase:
    """Index BM25 mémoire sur les sections des documents markdown."""

    def __init__(self, docs_dir: Path | None = None):
        self.docs_dir = Path(docs_dir) if docs_dir else DOCS_DIR
        self.chunks: list[dict] = []          # {source, heading, text}
        self._df: dict[str, int] = {}
        self._tf: list[dict[str, int]] = []
        self._dl: list[int] = []
        self._loaded = False
        self.k1 = 1.5
        self.b = 0.75

    # -- construction -------------------------------------------------------
    def _load(self) -> None:
        if self._loaded:
            return
        self.chunks = []
        if self.docs_dir.is_dir():
            for path in sorted(self.docs_dir.glob("*.md")):
                self._split_file(path)
        self._tf = []
        self._dl = []
        self._df = {}
        for chunk in self.chunks:
            tf: dict[str, int] = {}
            for tok in tokenize(chunk["text"] + " " + chunk["heading"]):
                tf[tok] = tf.get(tok, 0) + 1
            self._tf.append(tf)
            self._dl.append(sum(tf.values()))
            for tok in tf:
                self._df[tok] = self._df.get(tok, 0) + 1
        self._loaded = True

    def _split_file(self, path: Path) -> None:
        text = path.read_text(encoding="utf-8")
        heading = path.stem
        current: list[str] = []
        for line in text.splitlines():
            if line.startswith("## "):
                if current:
                    self.chunks.append(self._make_chunk(path.name, heading, current))
                heading = line[3:].strip()
                current = [line]
            else:
                current.append(line)
        if current:
            self.chunks.append(self._make_chunk(path.name, heading, current))

    @staticmethod
    def _make_chunk(source: str, heading: str, lines: list[str]) -> dict:
        return {
            "source": source,
            "heading": heading,
            "text": "\n".join(lines).strip(),
        }

    # -- requête ------------------------------------------------------------
    def query(self, text: str, k: int = 3) -> list[dict]:
        """Top-k sections par score BM25. Chaque hit : {source, heading, text, score}."""
        self._load()
        if not self.chunks:
            return []
        q_tokens = tokenize(text)
        if not q_tokens:
            return []
        n = len(self.chunks)
        avgdl = (sum(self._dl) / n) if n else 0.0
        scores = []
        for i, tf in enumerate(self._tf):
            s = 0.0
            dl = self._dl[i] or 1
            for tok in q_tokens:
                f = tf.get(tok, 0)
                if f == 0:
                    continue
                df = self._df.get(tok, 0)
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                s += idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / avgdl))
            scores.append((s, i))
        scores.sort(key=lambda t: -t[0])
        hits = []
        for s, i in scores:
            if s <= 0:
                break
            hit = dict(self.chunks[i])
            hit["score"] = round(s, 4)
            hits.append(hit)
            if len(hits) >= k:
                break
        return hits


# -- instance singleton ----------------------------------------------------
_kb: KnowledgeBase | None = None


def query(text: str, k: int = 3) -> list[dict]:
    global _kb
    if _kb is None:
        _kb = KnowledgeBase()
    return _kb.query(text, k=k)


def reset() -> None:
    """Recharge l'index (tests / ajout de documents)."""
    global _kb
    _kb = None
