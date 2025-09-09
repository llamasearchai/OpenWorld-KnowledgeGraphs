"""Validation utilities for input data and configurations."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Union

from .exceptions import ValidationError, FileNotFoundError, ConfigurationError


def validate_file_path(path: Union[str, Path], must_exist: bool = True) -> Path:
    """
    Validate and normalize file path.
    
    Args:
        path: File path to validate
        must_exist: Whether the file must already exist
        
    Returns:
        Validated Path object
        
    Raises:
        ValidationError: If path is invalid
        FileNotFoundError: If file doesn't exist when required
    """
    if not path:
        raise ValidationError("Path cannot be empty")
    
    path_obj = Path(path)
    
    if must_exist and not path_obj.exists():
        raise FileNotFoundError(f"File does not exist: {path}")
    
    return path_obj


def validate_database_path(db_path: Union[str, Path], create_parent: bool = True) -> Path:
    """
    Validate database path and optionally create parent directory.
    
    Args:
        db_path: Database file path
        create_parent: Whether to create parent directory if it doesn't exist
        
    Returns:
        Validated Path object
        
    Raises:
        ValidationError: If path is invalid
    """
    if not db_path:
        raise ValidationError("Database path cannot be empty")
    
    path_obj = Path(db_path)
    
    # Validate file extension
    if not path_obj.suffix:
        path_obj = path_obj.with_suffix('.db')
    elif path_obj.suffix not in ['.db', '.sqlite', '.sqlite3']:
        raise ValidationError(f"Invalid database file extension: {path_obj.suffix}")
    
    # Create parent directory if needed
    if create_parent:
        path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    return path_obj


def validate_text_content(text: str, min_length: int = 1, max_length: int = 100000) -> str:
    """
    Validate text content.
    
    Args:
        text: Text to validate
        min_length: Minimum text length
        max_length: Maximum text length
        
    Returns:
        Validated text
        
    Raises:
        ValidationError: If text is invalid
    """
    if not isinstance(text, str):
        raise ValidationError("Text must be a string")
    
    text = text.strip()
    
    if len(text) < min_length:
        raise ValidationError(f"Text too short (minimum {min_length} characters)")
    
    if len(text) > max_length:
        raise ValidationError(f"Text too long (maximum {max_length} characters)")
    
    return text


def validate_positive_integer(value: Any, name: str = "value") -> int:
    """
    Validate positive integer.
    
    Args:
        value: Value to validate
        name: Name of the parameter for error messages
        
    Returns:
        Validated integer
        
    Raises:
        ValidationError: If value is not a positive integer
    """
    if not isinstance(value, int):
        try:
            value = int(value)
        except (ValueError, TypeError):
            raise ValidationError(f"{name} must be an integer")
    
    if value <= 0:
        raise ValidationError(f"{name} must be positive")
    
    return value


def validate_backend_name(backend: str) -> str:
    """
    Validate LLM backend name.
    
    Args:
        backend: Backend name to validate
        
    Returns:
        Validated backend name
        
    Raises:
        ValidationError: If backend name is invalid
    """
    if not backend:
        raise ValidationError("Backend name cannot be empty")
    
    backend = backend.lower().strip()
    valid_backends = {'dummy', 'openai', 'ollama', 'llm'}
    
    if backend not in valid_backends:
        raise ValidationError(f"Invalid backend '{backend}'. Valid options: {valid_backends}")
    
    return backend


def validate_retrieval_method(method: str) -> str:
    """
    Validate retrieval method name.
    
    Args:
        method: Retrieval method to validate
        
    Returns:
        Validated method name
        
    Raises:
        ValidationError: If method is invalid
    """
    if not method:
        raise ValidationError("Retrieval method cannot be empty")
    
    method = method.lower().strip()
    valid_methods = {'tfidf', 'bm25', 'hybrid'}
    
    if method not in valid_methods:
        raise ValidationError(f"Invalid retrieval method '{method}'. Valid options: {valid_methods}")
    
    return method


def validate_openai_config(api_key: str, model: str) -> Dict[str, str]:
    """
    Validate OpenAI configuration.
    
    Args:
        api_key: OpenAI API key
        model: Model name
        
    Returns:
        Validated configuration
        
    Raises:
        ConfigurationError: If configuration is invalid
    """
    if not api_key:
        raise ConfigurationError("OpenAI API key is required")
    
    if not api_key.startswith('sk-'):
        raise ConfigurationError("Invalid OpenAI API key format")
    
    if not model:
        raise ConfigurationError("OpenAI model name is required")
    
    # Common OpenAI models
    valid_models = {
        'gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-4', 'gpt-4-turbo',
        'gpt-4-turbo-preview', 'gpt-4o', 'gpt-4o-mini'
    }
    
    if model not in valid_models:
        # Allow unknown models but log a warning
        import logging
        logging.getLogger(__name__).warning(f"Unknown OpenAI model: {model}")
    
    return {"api_key": api_key, "model": model}


def validate_ollama_config(host: str, model: str) -> Dict[str, str]:
    """
    Validate Ollama configuration.
    
    Args:
        host: Ollama server host URL
        model: Model name
        
    Returns:
        Validated configuration
        
    Raises:
        ConfigurationError: If configuration is invalid
    """
    if not host:
        raise ConfigurationError("Ollama host URL is required")
    
    # Basic URL validation
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    if not url_pattern.match(host):
        raise ConfigurationError(f"Invalid Ollama host URL: {host}")
    
    if not model:
        raise ConfigurationError("Ollama model name is required")
    
    return {"host": host, "model": model}


def validate_question(question: str) -> str:
    """
    Validate user question input.
    
    Args:
        question: User question
        
    Returns:
        Validated question
        
    Raises:
        ValidationError: If question is invalid
    """
    if not question:
        raise ValidationError("Question cannot be empty")
    
    question = question.strip()
    
    if len(question) < 3:
        raise ValidationError("Question too short (minimum 3 characters)")
    
    if len(question) > 1000:
        raise ValidationError("Question too long (maximum 1000 characters)")
    
    return question


def validate_document_paths(paths: List[Union[str, Path]]) -> List[Path]:
    """
    Validate list of document paths.
    
    Args:
        paths: List of document paths
        
    Returns:
        List of validated Path objects
        
    Raises:
        ValidationError: If any path is invalid
        FileNotFoundError: If any file doesn't exist
    """
    if not paths:
        raise ValidationError("At least one document path must be provided")
    
    validated_paths = []
    for path in paths:
        validated_path = validate_file_path(path, must_exist=True)
        
        # Check if it's a readable file
        if not validated_path.is_file():
            raise ValidationError(f"Path is not a file: {path}")
        
        try:
            # Test if file is readable
            with validated_path.open('r', encoding='utf-8') as f:
                f.read(1)  # Just read one character to test
        except (UnicodeDecodeError, PermissionError) as e:
            raise ValidationError(f"Cannot read file {path}: {e}")
        
        validated_paths.append(validated_path)
    
    return validated_paths