from typing import TypeVar, Dict, List, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field

class ResearchStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class QualityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class ResearchDirection(str, Enum):
    EXPLORATORY = "exploratory"  # 탐색적 연구
    DESCRIPTIVE = "descriptive"  # 기술적 연구
    EXPLANATORY = "explanatory"  # 설명적 연구
    EXPERIMENTAL = "experimental"  # 실험적 연구
    THEORETICAL = "theoretical"  # 이론적 연구

class EvaluationCriteria(BaseModel):
    methodology_weight: float = 0.3  # 연구방법론 가중치
    innovation_weight: float = 0.2   # 혁신성 가중치
    validity_weight: float = 0.3     # 타당성 가중치
    reliability_weight: float = 0.2   # 신뢰성 가중치

    min_quality_score: float = 0.7    # 최소 품질 점수
    required_validity_score: float = 0.8  # 필요 타당성 점수

    custom_criteria: Optional[Dict[str, float]] = None  # 추가 평가 기준

class ResearchRequest(BaseModel):
    topic: str  # 연구 주제
    description: str  # 연구 설명
    direction: ResearchDirection  # 연구 방향성
    evaluation_criteria: EvaluationCriteria  # 평가 기준

    context: Optional[str] = None  # 추가 컨텍스트
    max_iterations: Optional[int] = None  # 최대 반복 횟수
    custom_config: Optional[Dict[str, Any]] = None  # 추가 설정

class ResearchResult(BaseModel):
    query: str
    result: str
    confidence_score: float
    quality_score: float
    metadata: Dict[str, Any]
    status: ResearchStatus

class EmbeddingVector(BaseModel):
    text: str
    vector: List[float]
    metadata: Dict[str, Any]

class QualityMetrics(BaseModel):
    coherence_score: float
    relevance_score: float
    completeness_score: float
    overall_score: float
    validity_score: float = Field(
        default=0.0,
        description="Validity score of the research",
        ge=0.0,
        le=1.0
    )
    quality_score: float = Field(
        default=0.0,
        description="Overall quality score combining all metrics",
        ge=0.0,
        le=1.0
    )

    def calculate_quality_score(self):
        """Calculate overall quality score from individual metrics"""
        self.quality_score = (
            self.coherence_score * 0.25 +
            self.relevance_score * 0.25 +
            self.completeness_score * 0.2 +
            self.validity_score * 0.2 +
            self.overall_score * 0.1
        )
        return self.quality_score

class ResearchPhase(str, Enum):
    PLANNING = "planning"
    METHODOLOGY = "methodology"
    QUALITY = "quality"
    OPTIMIZATION = "optimization"
    COMPLETED = "completed"
    FAILED = "failed"

class ValidationCriteria(str, Enum):
    METHODOLOGY = "methodology"
    DATA = "data"
    ANALYSIS = "analysis"
    RESULTS = "results"

class ResearchMetrics(BaseModel):
    methodology_score: float = Field(default=0.0, ge=0.0, le=1.0)
    data_score: float = Field(default=0.0, ge=0.0, le=1.0)
    analysis_score: float = Field(default=0.0, ge=0.0, le=1.0)
    results_score: float = Field(default=0.0, ge=0.0, le=1.0)
    overall_score: float = Field(default=0.0, ge=0.0, le=1.0)

class ResearchGap(BaseModel):
    description: str
    importance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    feasibility_score: float = Field(default=0.0, ge=0.0, le=1.0)
    impact_score: float = Field(default=0.0, ge=0.0, le=1.0)
    cluster: Optional[str] = None

class Reference(BaseModel):
    title: str
    authors: List[str]
    year: int
    abstract: str
    citation_count: Optional[int] = None
    journal_impact_factor: Optional[float] = None
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    url: Optional[str] = None
    doi: Optional[str] = None
    quality_metrics: Optional[QualityMetrics] = None  # 추가
    embedding_data: Optional[EmbeddingVector] = None  # 추가

class TheoreticalFramework(BaseModel):
    description: str
    concepts: List[Dict[str, Any]]  # Dict[str, str]에서 List[Dict[str, Any]]로 변경
    relationships: List[Dict[str, Any]] = Field(default_factory=list)
    hierarchy: Dict[str, List[str]] = Field(default_factory=dict)
    variables: Optional[Dict[str, Any]] = None
    hypotheses: Optional[List[str]] = None
    theoretical_foundations: Optional[List[str]] = None
    measurement_approach: Optional[Dict[str, Any]] = None
    embedding: Optional[List[float]] = None
    completeness_score: float = Field(default=0.0, ge=0.0, le=1.0)

T = TypeVar('T')  # Generic type for flexible type hints
