"""Text embeddings: dense numeric vectors for free-text columns."""

from __future__ import annotations

import math
import re
import warnings
from collections import Counter

import numpy as np
import polars as pl

from ..core import Step, register_step

_TOKEN = re.compile(r"(?u)\b\w\w+\b")


def _tokens(text) -> list:
    return _TOKEN.findall(text.lower()) if text else []


def _tfidf(texts, index: dict, idf: list):
    """L2-normalised TF-IDF rows as a scipy CSR matrix (same maths for fit and transform)."""
    from scipy.sparse import csr_matrix

    rows, cols, vals = [], [], []
    for i, text in enumerate(texts):
        counts = Counter(t for t in _tokens(text) if t in index)
        if not counts:
            continue
        weights = {index[t]: n * idf[index[t]] for t, n in counts.items()}
        norm = math.sqrt(sum(w * w for w in weights.values())) or 1.0
        for j, w in weights.items():
            rows.append(i)
            cols.append(j)
            vals.append(w / norm)
    return csr_matrix((vals, (rows, cols)), shape=(len(texts), len(index)))


def _text_columns(df, exclude=()):
    from ..quality.profile import semantic_type
    return [c for c, t in df.schema.items()
            if c not in exclude and t == pl.String and semantic_type(df[c], c) == "text"]


@register_step
class TextEmbeddings(Step):
    """Turn free-text columns into ``<col>_emb0..N`` numeric columns.

    ``method="tfidf_svd"`` (default, no extra dependencies): TF-IDF over the training
    vocabulary, compressed with truncated SVD (latent semantic analysis). Everything learned is
    stored as plain JSON, so saved pipelines reproduce the same vectors.

    ``method="sentence_transformers"``: neural sentence embeddings from a pretrained model
    (``pip install sentence-transformers``); only the model name is stored.

    By default every column whose semantic type is ``text`` (long, varied strings) is embedded.
    """

    name = "text_embeddings"

    def __init__(self, *, columns=None, method="tfidf_svd", n_components=16, max_features=1000,
                 min_df=2, model="all-MiniLM-L6-v2", drop_original=True, seed=0, fit_only=None):
        if method not in ("tfidf_svd", "sentence_transformers"):
            raise ValueError("method must be 'tfidf_svd' or 'sentence_transformers'")
        super().__init__(fit_only=fit_only, columns=columns, method=method,
                         n_components=n_components, max_features=max_features, min_df=min_df,
                         model=model, drop_original=drop_original, seed=seed)
        self._encoder = None

    # -- sentence-transformers ----------------------------------------------
    def _st_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError("method='sentence_transformers' needs: "
                                  "pip install sentence-transformers") from exc
            self._encoder = SentenceTransformer(self.params["model"])
        return self._encoder

    # -- fit / transform -----------------------------------------------------
    def _fit(self, df):
        p = self.params
        target = getattr(self, "_pipeline_target", None)
        cols = ([c for c in p["columns"] if c in df.columns] if p["columns"] is not None
                else _text_columns(df, exclude={target}))
        learned = {}
        for c in cols:
            texts = df[c].cast(pl.String).fill_null("").to_list()
            if p["method"] == "sentence_transformers":
                dim = int(np.asarray(self._st_encoder().encode(["probe"])).shape[1])
                learned[c] = {"dim": dim}
                continue
            doc_freq = Counter(t for text in texts for t in set(_tokens(text)))
            terms = [t for t, n in doc_freq.items() if n >= p["min_df"]]
            terms = sorted(sorted(terms), key=lambda t: -doc_freq[t])[: p["max_features"]]
            if len(terms) < 2:
                continue
            terms.sort()
            n = len(texts)
            idf = [math.log((1 + n) / (1 + doc_freq[t])) + 1.0 for t in terms]
            matrix = _tfidf(texts, {t: i for i, t in enumerate(terms)}, idf)
            k = min(p["n_components"], len(terms) - 1, n - 1)
            if k < 1:
                continue
            from sklearn.decomposition import TruncatedSVD
            with warnings.catch_warnings():     # identical documents: zero variance ratio
                warnings.simplefilter("ignore", RuntimeWarning)
                svd = TruncatedSVD(n_components=k, random_state=p["seed"]).fit(matrix)
            learned[c] = {"vocab": terms, "idf": idf, "components": svd.components_.tolist(), "dim": k}
        return {"columns": learned}

    def _transform(self, df):
        p = self.params
        out, embedded = [], []
        for c, info in self.state_["columns"].items():
            if c not in df.columns:
                continue
            texts = df[c].cast(pl.String).fill_null("").to_list()
            if p["method"] == "sentence_transformers":
                vectors = np.asarray(self._st_encoder().encode(texts), dtype=float)
            else:
                index = {t: i for i, t in enumerate(info["vocab"])}
                vectors = _tfidf(texts, index, info["idf"]) @ np.asarray(info["components"]).T
                vectors = np.asarray(vectors)
            embedded.append(c)
            out += [pl.Series(f"{c}_emb{i}", vectors[:, i]) for i in range(info["dim"])]
        if out:
            df = df.with_columns(out)
        if p["drop_original"] and embedded:
            df = df.drop(embedded)
        return df
