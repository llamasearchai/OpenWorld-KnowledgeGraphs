"""Custom exceptions for OpenWorld-KnowledgeGraphs."""

from __future__ import annotations


class OpenWorldKGError(Exception):
    """Base exception for all OpenWorld-KG errors."""
    pass


class DatabaseError(OpenWorldKGError):
    """Database-related errors."""
    pass


class RetrievalError(OpenWorldKGError):
    """Retrieval and search-related errors."""
    pass


class KnowledgeGraphError(OpenWorldKGError):
    """Knowledge graph-related errors."""
    pass


class AgentError(OpenWorldKGError):
    """Agent and LLM backend-related errors."""
    pass


class ConfigurationError(OpenWorldKGError):
    """Configuration and settings-related errors."""
    pass


class ValidationError(OpenWorldKGError):
    """Data validation errors."""
    pass


class FileNotFoundError(OpenWorldKGError):
    """File or path not found errors."""
    pass


class APIError(OpenWorldKGError):
    """API and external service errors."""
    pass