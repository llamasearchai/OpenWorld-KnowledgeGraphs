from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ..data.docs import export_docs


@dataclass
class RetrieverArtifact:
    vectorizer: TfidfVectorizer
    doc_matrix: np.ndarray
    doc_ids: List[int]
    texts: List[str]


def build_retriever(db_path: str, artifact_path: str) -> str:
    df = export_docs(db_path)
    texts = df["text"].astype(str).tolist()
    ids = df["id"].astype(int).tolist()
    vec = TfidfVectorizer()
    mat = vec.fit_transform(texts)
    art = RetrieverArtifact(vectorizer=vec, doc_matrix=mat, doc_ids=ids, texts=texts)
    ap = Path(artifact_path)
    ap.parent.mkdir(parents=True, exist_ok=True)
    with ap.open("wb") as f:
        pickle.dump(art, f)
    return str(ap)


def _load_artifact(artifact_path: str) -> RetrieverArtifact:
    with open(artifact_path, "rb") as f:
        return pickle.load(f)


def query(db_path: str, artifact_path: str, question: str, k: int = 3) -> List[Tuple[int, float, str]]:
    art = _load_artifact(artifact_path)
    qv = art.vectorizer.transform([question])
    sims = (qv @ art.doc_matrix.T).toarray().ravel()
    order = np.argsort(-sims)
    out: List[Tuple[int, float, str]] = []
    for idx in order[: max(0, k)]:
        out.append((art.doc_ids[idx], float(sims[idx]), art.texts[idx]))
    return out

