"""Advanced retrieval methods for enhanced RAG functionality."""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ..data.docs import export_docs

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Structured result for document retrieval."""
    doc_id: int
    score: float
    text: str
    method: str
    rank: int


@dataclass 
class AdvancedRetrieverArtifact:
    """Enhanced retrieval artifact with multiple methods."""
    tfidf_vectorizer: TfidfVectorizer
    tfidf_matrix: np.ndarray
    doc_ids: List[int]
    texts: List[str]
    
    # BM25 components
    bm25_idf: Optional[Dict[str, float]] = None
    bm25_doc_freqs: Optional[List[Dict[str, int]]] = None
    bm25_avgdl: Optional[float] = None
    
    # Metadata
    doc_lengths: Optional[List[int]] = None
    vocabulary: Optional[Dict[str, int]] = None


class BM25Retriever:
    """BM25 retrieval implementation."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization - can be enhanced with better tokenizers."""
        return text.lower().split()
    
    def fit(self, documents: List[str]) -> Tuple[Dict[str, float], List[Dict[str, int]], float]:
        """Fit BM25 parameters on documents."""
        tokenized_docs = [self._tokenize(doc) for doc in documents]
        doc_lengths = [len(doc) for doc in tokenized_docs]
        avgdl = sum(doc_lengths) / len(doc_lengths)
        
        # Calculate document frequencies
        doc_freqs = []
        vocabulary = set()
        for doc in tokenized_docs:
            freq = {}
            for word in doc:
                freq[word] = freq.get(word, 0) + 1
                vocabulary.add(word)
            doc_freqs.append(freq)
        
        # Calculate IDF scores
        N = len(documents)
        idf_scores = {}
        for word in vocabulary:
            containing_docs = sum(1 for doc_freq in doc_freqs if word in doc_freq)
            idf_scores[word] = np.log((N - containing_docs + 0.5) / (containing_docs + 0.5))
        
        return idf_scores, doc_freqs, avgdl
    
    def score(self, query: str, doc_freqs: Dict[str, int], doc_length: int, 
              idf_scores: Dict[str, float], avgdl: float) -> float:
        """Calculate BM25 score for a document given a query."""
        query_tokens = self._tokenize(query)
        score = 0.0
        
        for token in query_tokens:
            if token in doc_freqs and token in idf_scores:
                tf = doc_freqs[token]
                idf = idf_scores[token]
                
                # BM25 formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / avgdl))
                score += idf * (numerator / denominator)
        
        return score


class HybridRetriever:
    """Hybrid retrieval combining multiple methods."""
    
    def __init__(self, tfidf_weight: float = 0.6, bm25_weight: float = 0.4):
        self.tfidf_weight = tfidf_weight
        self.bm25_weight = bm25_weight
        self.bm25 = BM25Retriever()
    
    def retrieve(
        self, 
        artifact: AdvancedRetrieverArtifact, 
        query: str, 
        k: int = 3,
        method: str = "hybrid"
    ) -> List[RetrievalResult]:
        """Retrieve documents using specified method."""
        
        if method == "tfidf":
            return self._tfidf_retrieve(artifact, query, k)
        elif method == "bm25":
            return self._bm25_retrieve(artifact, query, k)
        elif method == "hybrid":
            return self._hybrid_retrieve(artifact, query, k)
        else:
            raise ValueError(f"Unknown retrieval method: {method}")
    
    def _tfidf_retrieve(self, artifact: AdvancedRetrieverArtifact, query: str, k: int) -> List[RetrievalResult]:
        """TF-IDF based retrieval."""
        query_vec = artifact.tfidf_vectorizer.transform([query])
        similarities = (query_vec @ artifact.tfidf_matrix.T).toarray().ravel()
        top_indices = np.argsort(-similarities)[:k]
        
        results = []
        for rank, idx in enumerate(top_indices):
            if similarities[idx] > 0:  # Only include relevant results
                results.append(RetrievalResult(
                    doc_id=artifact.doc_ids[idx],
                    score=float(similarities[idx]),
                    text=artifact.texts[idx],
                    method="tfidf",
                    rank=rank + 1
                ))
        
        return results
    
    def _bm25_retrieve(self, artifact: AdvancedRetrieverArtifact, query: str, k: int) -> List[RetrievalResult]:
        """BM25 based retrieval."""
        if not all([artifact.bm25_idf, artifact.bm25_doc_freqs, artifact.bm25_avgdl]):
            raise ValueError("BM25 components not available in artifact")
        
        scores = []
        for i, doc_freq in enumerate(artifact.bm25_doc_freqs):
            doc_length = artifact.doc_lengths[i] if artifact.doc_lengths else len(artifact.texts[i].split())
            score = self.bm25.score(query, doc_freq, doc_length, artifact.bm25_idf, artifact.bm25_avgdl)
            scores.append((i, score))
        
        # Sort by score and take top k
        scores.sort(key=lambda x: x[1], reverse=True)
        top_scores = scores[:k]
        
        results = []
        for rank, (idx, score) in enumerate(top_scores):
            if score > 0:  # Only include relevant results
                results.append(RetrievalResult(
                    doc_id=artifact.doc_ids[idx],
                    score=score,
                    text=artifact.texts[idx],
                    method="bm25", 
                    rank=rank + 1
                ))
        
        return results
    
    def _hybrid_retrieve(self, artifact: AdvancedRetrieverArtifact, query: str, k: int) -> List[RetrievalResult]:
        """Hybrid retrieval combining TF-IDF and BM25."""
        try:
            tfidf_results = self._tfidf_retrieve(artifact, query, k * 2)  # Get more for fusion
            bm25_results = self._bm25_retrieve(artifact, query, k * 2)
        except ValueError:
            # Fall back to TF-IDF if BM25 not available
            logger.warning("BM25 not available, falling back to TF-IDF only")
            return self._tfidf_retrieve(artifact, query, k)
        
        # Normalize scores to [0, 1] range for combination
        if tfidf_results:
            max_tfidf = max(r.score for r in tfidf_results)
            for result in tfidf_results:
                result.score = result.score / max_tfidf if max_tfidf > 0 else 0
        
        if bm25_results:
            max_bm25 = max(r.score for r in bm25_results)
            for result in bm25_results:
                result.score = result.score / max_bm25 if max_bm25 > 0 else 0
        
        # Combine scores
        combined_scores = {}
        
        for result in tfidf_results:
            combined_scores[result.doc_id] = {
                'tfidf': result.score * self.tfidf_weight,
                'bm25': 0,
                'text': result.text
            }
        
        for result in bm25_results:
            if result.doc_id in combined_scores:
                combined_scores[result.doc_id]['bm25'] = result.score * self.bm25_weight
            else:
                combined_scores[result.doc_id] = {
                    'tfidf': 0,
                    'bm25': result.score * self.bm25_weight,
                    'text': result.text
                }
        
        # Calculate final scores and rank
        final_scores = []
        for doc_id, scores in combined_scores.items():
            final_score = scores['tfidf'] + scores['bm25']
            final_scores.append((doc_id, final_score, scores['text']))
        
        final_scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for rank, (doc_id, score, text) in enumerate(final_scores[:k]):
            if score > 0:
                results.append(RetrievalResult(
                    doc_id=doc_id,
                    score=score,
                    text=text,
                    method="hybrid",
                    rank=rank + 1
                ))
        
        return results


def build_advanced_retriever(db_path: str, artifact_path: str, include_bm25: bool = True) -> str:
    """Build enhanced retrieval artifact with multiple methods."""
    df = export_docs(db_path)
    texts = df["text"].astype(str).tolist()
    ids = df["id"].astype(int).tolist()
    
    # Build TF-IDF components
    tfidf_vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words='english',
        ngram_range=(1, 2),  # Include bigrams
        min_df=1,
        max_df=0.95
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    
    # Initialize artifact
    artifact = AdvancedRetrieverArtifact(
        tfidf_vectorizer=tfidf_vectorizer,
        tfidf_matrix=tfidf_matrix,
        doc_ids=ids,
        texts=texts,
        vocabulary=tfidf_vectorizer.vocabulary_,
        doc_lengths=[len(text.split()) for text in texts]
    )
    
    # Add BM25 components if requested
    if include_bm25:
        try:
            bm25_retriever = BM25Retriever()
            idf_scores, doc_freqs, avgdl = bm25_retriever.fit(texts)
            artifact.bm25_idf = idf_scores
            artifact.bm25_doc_freqs = doc_freqs
            artifact.bm25_avgdl = avgdl
            logger.info("BM25 components added to artifact")
        except Exception as e:
            logger.warning(f"Failed to build BM25 components: {e}")
    
    # Save artifact
    ap = Path(artifact_path)
    ap.parent.mkdir(parents=True, exist_ok=True)
    with ap.open("wb") as f:
        pickle.dump(artifact, f)
    
    logger.info(f"Advanced retriever artifact saved to {ap}")
    return str(ap)


def load_advanced_artifact(artifact_path: str) -> AdvancedRetrieverArtifact:
    """Load advanced retrieval artifact."""
    with open(artifact_path, "rb") as f:
        return pickle.load(f)


def advanced_query(
    db_path: str, 
    artifact_path: str, 
    question: str, 
    k: int = 3,
    method: str = "hybrid"
) -> List[Tuple[int, float, str]]:
    """
    Query using advanced retrieval methods.
    
    Args:
        db_path: Path to database (for compatibility)
        artifact_path: Path to advanced retrieval artifact
        question: Query string
        k: Number of results to return
        method: Retrieval method ('tfidf', 'bm25', 'hybrid')
    
    Returns:
        List of (doc_id, score, text) tuples for compatibility
    """
    artifact = load_advanced_artifact(artifact_path)
    retriever = HybridRetriever()
    
    results = retriever.retrieve(artifact, question, k, method)
    
    # Convert to legacy format for compatibility
    return [(r.doc_id, r.score, r.text) for r in results]