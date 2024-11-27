from datetime import datetime
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from core.types import (
    ResearchPhase,
    ResearchGap,
    ResearchMetrics,
    Reference,
    TheoreticalFramework,
    QualityLevel,
    ResearchStatus
)
class Research(BaseModel):
    id: str = Field(default_factory=lambda: str(datetime.now().timestamp()))
    query: str
    status: ResearchStatus = ResearchStatus.PENDING
    result: Optional[str] = None
    confidence_score: Optional[float] = None
    quality_score: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ResearchIteration(BaseModel):
    id: str = Field(default_factory=lambda: str(datetime.now().timestamp()))
    research_id: str
    iteration_number: int
    result: str
    confidence_score: float
    quality_score: float
    validity_score: float
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class QualityCheck(BaseModel):
    id: str = Field(default_factory=lambda: str(datetime.now().timestamp()))
    research_id: str
    iteration_id: Optional[str] = None
    quality_level: QualityLevel
    metrics: Dict[str, float]
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class EmbeddingData(BaseModel):
    id: str = Field(default_factory=lambda: str(datetime.now().timestamp()))
    text: str
    vector: List[float]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)

class ResearchStep(BaseModel):
    id: str = Field(default_factory=lambda: str(datetime.now().timestamp()))
    research_id: str
    phase: ResearchPhase
    description: str
    status: ResearchStatus = ResearchStatus.PENDING
    metrics: Optional[ResearchMetrics] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ResearchProject(BaseModel):
    id: str = Field(default_factory=lambda: str(datetime.now().timestamp()))
    query: str
    context: Optional[str] = None
    phase: ResearchPhase = ResearchPhase.PLANNING
    status: ResearchStatus = ResearchStatus.PENDING

    # Literature Review
    references: List[Reference] = Field(default_factory=list)
    research_gaps: List[ResearchGap] = Field(default_factory=list)

    # Methodology
    theoretical_framework: Optional[TheoreticalFramework] = None
    methodology_description: Optional[str] = None

    # Quality Metrics
    metrics: Optional[ResearchMetrics] = None
    quality_level: Optional[QualityLevel] = None

    # Results
    result: Optional[str] = None
    confidence_score: Optional[float] = None
    quality_score: Optional[float] = None

    # Metadata
    steps: List[ResearchStep] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
