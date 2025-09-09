from __future__ import annotations

import logging
from typing import List

from .backends import get_backend
from ..kg.store import KGStore
from ..rag.retriever import query as retriever_query

logger = logging.getLogger(__name__)


def _build_context_prompt(
    question: str, 
    docs: List[tuple], 
    neighbors: List,
    max_doc_length: int = 300
) -> str:
    """Build a context-aware prompt for the LLM."""
    
    prompt_parts = [
        "You are an AI assistant that answers questions using provided context.",
        "Use the following information to answer the user's question:\n"
    ]
    
    # Add document context
    if docs:
        prompt_parts.append("RETRIEVED DOCUMENTS:")
        for i, (doc_id, score, text) in enumerate(docs, 1):
            # Truncate very long documents
            doc_text = text[:max_doc_length] + "..." if len(text) > max_doc_length else text
            prompt_parts.append(f"{i}. (Score: {score:.3f}) {doc_text}")
        prompt_parts.append("")
    
    # Add knowledge graph context
    if neighbors:
        prompt_parts.append("KNOWLEDGE GRAPH RELATIONSHIPS:")
        for i, triple in enumerate(neighbors, 1):
            prompt_parts.append(f"{i}. {triple.s} {triple.p} {triple.o}")
        prompt_parts.append("")
    
    # Add the question
    prompt_parts.extend([
        "Please answer the following question based on the context above.",
        "If the context doesn't contain relevant information, say so.",
        f"Question: {question}",
        "\nAnswer:"
    ])
    
    return "\n".join(prompt_parts)


def ask(
    question: str,
    *,
    db_path: str,
    artifact_path: str,
    backend: str = "dummy",
    k: int = 3,
    **backend_kwargs
) -> str:
    """
    Ask a question using RAG + Knowledge Graph enhanced context.
    
    Args:
        question: The question to ask
        db_path: Path to the SQLite database
        artifact_path: Path to the TF-IDF retriever artifact
        backend: LLM backend to use ('dummy', 'openai', 'ollama', 'llm')
        k: Number of documents/neighbors to retrieve
        **backend_kwargs: Additional arguments for the LLM backend
    
    Returns:
        Generated response string
    """
    try:
        # Try advanced retrieval first, fall back to basic retrieval
        try:
            from ..rag.advanced_retriever import advanced_query
            docs = advanced_query(db_path, artifact_path, question, k=k, method="hybrid")
            logger.info(f"Retrieved {len(docs)} documents using advanced retrieval")
        except Exception as e:
            logger.debug(f"Advanced retrieval failed, falling back to basic: {e}")
            docs = retriever_query(db_path, artifact_path, question, k=k)
            logger.info(f"Retrieved {len(docs)} documents using basic retrieval")
        
        # Get knowledge graph neighbors
        kg = KGStore(db_path)
        # Use first capitalized token as a guess for neighbor lookup
        tokens = [t.strip(".,?!") for t in question.split() if t[:1].isupper()]
        node = tokens[0] if tokens else ""
        neigh = kg.neighbors(node, k=k) if node else []
        logger.info(f"Found {len(neigh)} KG neighbors for node '{node}'")
        
        # For dummy backend, return deterministic response for tests
        if backend == "dummy":
            return f"DUMMY: {question} | docs={len(docs)} neigh={len(neigh)}"
        
        # Build context-aware prompt
        prompt = _build_context_prompt(question, docs, neigh)
        
        # Get LLM backend and generate response
        llm_backend = get_backend(backend, **backend_kwargs)
        response = llm_backend.generate_response(prompt)
        
        return response
        
    except Exception as e:
        logger.error(f"Error in ask function: {e}")
        return f"Error processing question: {e}"

