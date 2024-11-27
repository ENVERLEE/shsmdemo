"""
Research Project - A comprehensive research assistance system
"""

from .core.models import Research, ResearchProject, Reference, EmbeddingData
from .core.types import ResearchRequest, ResearchStatus, QualityLevel, QualityMetrics
from .services.research.service import ResearchService
from .services.quality.service import QualityControlService
from .services.llm.service import LLMService
from .services.embedding.service import EmbeddingService
from .main import ResearchAssistant, MainFunction
__version__ = '0.1.0'
