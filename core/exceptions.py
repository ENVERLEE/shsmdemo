class ResearchException(Exception):
    """Base exception for research-related errors."""
    pass

class LLMServiceException(ResearchException):
    """Exception raised for errors in the LLM service."""
    pass

class EmbeddingServiceException(ResearchException):
    """Exception raised for errors in the embedding service."""
    pass

class QualityControlException(ResearchException):
    """Exception raised for errors in the quality control service."""
    pass

class ValidationError(ResearchException):
    """Exception raised for validation errors."""
    pass

class TimeoutError(ResearchException):
    """Exception raised when an operation times out."""
    pass

class ConfigurationError(ResearchException):
    """Exception raised for configuration-related errors."""
    pass

class ServiceUnavailableError(ResearchException):
    """Exception raised when a required service is unavailable."""
    pass

class DataProcessingError(ResearchException):
    """Exception raised for errors during data processing."""
    pass
