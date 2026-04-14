"""Project-specific exception types for consistent error handling."""


class KGSemEmbedError(Exception):
    """Base exception for the project."""


class ConfigurationError(KGSemEmbedError):
    """Raised for invalid configuration values or setup."""


class DataError(KGSemEmbedError):
    """Raised for dataset parsing/loading failures."""


class PipelineStageError(KGSemEmbedError):
    """Raised when a pipeline stage fails execution."""
