"""Logging configuration and utilities."""

from __future__ import annotations

import logging
import logging.config
import sys
from pathlib import Path
from typing import Dict, Optional, Union

import structlog


def setup_logging(
    level: Union[str, int] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    structured: bool = True,
    include_stdlib: bool = True
) -> None:
    """
    Configure application logging.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        structured: Whether to use structured logging with structlog
        include_stdlib: Whether to configure stdlib logging as well
    """
    # Normalize level
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    
    # Configure structured logging with structlog
    if structured:
        # Configure processors
        processors = [
            # Add timestamp
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
        ]
        
        # Add different processors for console vs file
        if log_file:
            # JSON format for file logging
            processors.append(structlog.processors.JSONRenderer())
        else:
            # Human-readable format for console
            processors.extend([
                structlog.dev.ConsoleRenderer(colors=True),
            ])
        
        # Configure structlog
        structlog.configure(
            processors=processors,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    
    # Configure stdlib logging
    if include_stdlib:
        config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'detailed': {
                    'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                    'datefmt': '%Y-%m-%d %H:%M:%S'
                },
                'simple': {
                    'format': '[%(levelname)s] %(name)s: %(message)s'
                }
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'level': level,
                    'formatter': 'simple',
                    'stream': sys.stdout
                }
            },
            'root': {
                'level': level,
                'handlers': ['console']
            },
            'loggers': {
                'openworld_knowledgegraphs': {
                    'level': level,
                    'propagate': True
                },
                # Suppress noisy third-party loggers
                'urllib3': {
                    'level': logging.WARNING
                },
                'requests': {
                    'level': logging.WARNING
                },
                'httpx': {
                    'level': logging.WARNING
                }
            }
        }
        
        # Add file handler if log file specified
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            config['handlers']['file'] = {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': level,
                'formatter': 'detailed',
                'filename': str(log_path),
                'maxBytes': 10 * 1024 * 1024,  # 10MB
                'backupCount': 5
            }
            config['root']['handlers'].append('file')
        
        logging.config.dictConfig(config)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LoggingMixin:
    """Mixin class to add logging capabilities to any class."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        return get_logger(f"{self.__class__.__module__}.{self.__class__.__name__}")


def log_function_call(logger: logging.Logger):
    """Decorator to log function calls with arguments and results."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"{func.__name__} completed successfully")
                return result
            except Exception as e:
                logger.error(f"{func.__name__} failed with error: {e}")
                raise
        return wrapper
    return decorator


def setup_cli_logging(verbose: bool = False, quiet: bool = False) -> None:
    """
    Setup logging for CLI usage.
    
    Args:
        verbose: Enable debug logging
        quiet: Suppress all but error logs
    """
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    
    setup_logging(level=level, structured=False)


def setup_api_logging(log_file: Optional[str] = None, debug: bool = False) -> None:
    """
    Setup logging for API server.
    
    Args:
        log_file: Optional log file path
        debug: Enable debug logging
    """
    level = logging.DEBUG if debug else logging.INFO
    setup_logging(level=level, log_file=log_file, structured=True)


class PerformanceLogger:
    """Logger for performance monitoring."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or get_logger(__name__)
        self._timers: Dict[str, float] = {}
    
    def start_timer(self, name: str) -> None:
        """Start a performance timer."""
        import time
        self._timers[name] = time.perf_counter()
        self.logger.debug(f"Started timer: {name}")
    
    def end_timer(self, name: str) -> float:
        """End a performance timer and log the duration."""
        import time
        if name not in self._timers:
            self.logger.warning(f"Timer '{name}' was not started")
            return 0.0
        
        duration = time.perf_counter() - self._timers[name]
        del self._timers[name]
        
        self.logger.info(f"Timer '{name}' completed in {duration:.3f}s")
        return duration
    
    def log_memory_usage(self, message: str = "Memory usage") -> None:
        """Log current memory usage."""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.logger.info(f"{message}: {memory_mb:.1f} MB")
        except ImportError:
            self.logger.debug("psutil not available for memory monitoring")


def timer_context(name: str, logger: Optional[logging.Logger] = None):
    """Context manager for timing operations."""
    from contextlib import contextmanager
    
    perf_logger = PerformanceLogger(logger)
    
    @contextmanager
    def timer():
        perf_logger.start_timer(name)
        try:
            yield
        finally:
            perf_logger.end_timer(name)
    
    return timer()