from typing import Dict, Any, Optional, List
import numpy as np
from dataclasses import dataclass
from core.exceptions import ResearchException
from core.models import Research, ResearchIteration
from core.types import ResearchStatus, ResearchRequest, ResearchDirection, EvaluationCriteria, QualityMetrics
from config.settings import RESEARCH_CONFIG
from services.llm.service import LLMService
from services.embedding.service import EmbeddingService
from services.quality.service import QualityControlService


@dataclass
class Context:
    research: Research
    iterations: List[ResearchIteration]
    current_result: Optional[str] = None
    metrics: Optional[QualityMetrics] = None

class ResearchService:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or RESEARCH_CONFIG
        self.llm_service = LLMService()
        self.embedding_service = EmbeddingService()
        self.quality_service = QualityControlService()
        self._research_cache: Dict[str, Research] = {}

    def conduct_research(self, request: ResearchRequest) -> Research:
        try:
            # Initialize research with structured information
            research = Research(
                query=request.topic,
                status=ResearchStatus.IN_PROGRESS,
                metadata={
                    "evaluation_criteria": request.evaluation_criteria.dict(),
                    "research_direction": request.direction.value,
                    "context": request.context or {}
                }
            )
            self._research_cache[research.id] = research

            # Create context object
            context = Context(
                research=research,
                iterations=[]
            )

            # Prepare research prompt based on direction
            direction_prompts = {
                ResearchDirection.EXPLORATORY: """
                    이 탐색적 연구를 수행하여:
                    1. 주요 변수와 관계 식별
                    2. 가설 생성
                    3. 향후 연구 방향 제시
                    """,
                ResearchDirection.DESCRIPTIVE: """
                    이 기술적 연구를 통해:
                    1. 현상의 상세한 묘사
                    2. 패턴과 특성 분석
                    3. 체계적 분류와 정리
                    """,
                ResearchDirection.EXPLANATORY: """
                    이 설명적 연구에서:
                    1. 인과관계 분석
                    2. 메커니즘 설명
                    3. 이론적 근거 제시
                    """,
                ResearchDirection.EXPERIMENTAL: """
                    이 실험적 연구를 위해:
                    1. 변수 통제 및 조작
                    2. 실험 설계 및 수행
                    3. 결과의 통계적 분석
                    """,
                ResearchDirection.THEORETICAL: """
                    이 이론적 연구에서:
                    1. 기존 이론 분석
                    2. 새로운 이론 구축
                    3. 이론적 통합 시도
                    """
            }

            # Add context to research prompt
            research_prompt = f"""
            연구 주제: {request.topic}

            연구 설명: {request.description}

            연구 방향: {direction_prompts[request.direction]}

            평가 기준:
            - 방법론적 타당성 (가중치: {request.evaluation_criteria.methodology_weight})
            - 혁신성 (가중치: {request.evaluation_criteria.innovation_weight})
            - 타당성 (가중치: {request.evaluation_criteria.validity_weight})
            - 신뢰성 (가중치: {request.evaluation_criteria.reliability_weight})

            추가 평가 기준: {request.evaluation_criteria.custom_criteria or '없음'}

            컨텍스트: {request.context or '없음'}
            """

            # Set iteration parameters
            max_iterations = request.max_iterations or self.config["max_iterations"]
            quality_threshold = request.evaluation_criteria.min_quality_score

            for i in range(max_iterations):
                # Add context from previous iteration
                iter_context = request.context if i == 0 else context.current_result
                result = self.llm_service.generate_research(
                    query=research_prompt,
                    context=iter_context
                )

                # Evaluate quality with custom criteria and context
                metrics = self.quality_service.evaluate_quality(
                    result,
                    evaluation_criteria=request.evaluation_criteria
                )
                context.metrics = metrics

                # Create iteration record with context
                iteration = ResearchIteration(
                    research_id=research.id,
                    iteration_number=i + 1,
                    result=result,
                    confidence_score=metrics.overall_score,
                    quality_score=metrics.quality_score,
                    validity_score=metrics.validity_score,
                    metadata={
                        "direction": request.direction.value,
                        "evaluation_details": metrics.dict(),
                        "context": iter_context
                    }
                )
                context.iterations.append(iteration)

                # Check if quality meets threshold
                if (metrics.quality_score >= quality_threshold and
                    metrics.validity_score >= request.evaluation_criteria.required_validity_score):
                    research.result = result
                    research.confidence_score = metrics.overall_score
                    research.quality_score = metrics.quality_score
                    research.status = ResearchStatus.COMPLETED
                    break

                # If not meeting threshold, get improvements with context
                context.current_result = self.quality_service.suggest_improvements(
                    result,
                    metrics,
                    evaluation_criteria=request.evaluation_criteria
                )

            # Handle completion or fallback
            if research.status != ResearchStatus.COMPLETED:
                best_iteration = max(context.iterations, key=lambda x: x.quality_score)
                research.result = best_iteration.result
                research.confidence_score = best_iteration.confidence_score
                research.quality_score = best_iteration.quality_score
                research.status = ResearchStatus.COMPLETED

            research.metadata.update({
                "iterations": len(context.iterations),
                "final_evaluation": context.metrics if context.metrics is not None else {},
                "research_direction": request.direction.value,
                "final_context": context.current_result
            })

            return research

        except Exception as e:
            if research:
                research.status = ResearchStatus.FAILED
                research.metadata["error"] = str(e)
            raise ResearchException(f"Error conducting research: {str(e)}")

    def get_research(self, research_id: str) -> Optional[Research]:
        """Retrieve research by ID from cache."""
        return self._research_cache.get(research_id)

    def enhance_with_embeddings(self, research: Research) -> Research:
        try:
            if research.result:
                embeddings = self.embedding_service.create_embeddings(
                    research.result,
                    metadata={"research_id": research.id}
                )

                embedded_vectors = []
                for embedding in embeddings:
                    vector = embedding.vector
                    # numpy array인 경우 tolist() 메서드 호출
                    if isinstance(vector, np.ndarray):
                        vector = vector.tolist()
                    # 그 외의 경우 list로 변환
                    elif not isinstance(vector, list):
                        vector = list(vector)

                    embedded_vectors.append({
                        "text": embedding.text,
                        "vector": vector,
                        "metadata": {"research_id": research.id}
                    })

                research.metadata["embeddings"] = embedded_vectors

            return research
        except Exception as e:
            raise ResearchException(f"Error enhancing research with embeddings: {str(e)}")
