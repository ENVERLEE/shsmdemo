from typing import Dict, Any, Optional
from core.exceptions import QualityControlException
from core.types import QualityLevel, QualityMetrics, EvaluationCriteria
from config.settings import QUALITY_CONFIG
from services.llm.service import LLMService

class QualityControlService:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or QUALITY_CONFIG
        self.llm_service = LLMService()

    def evaluate_quality(
        self,
        text: str,
        evaluation_criteria: Optional[EvaluationCriteria] = None
    ) -> QualityMetrics:
        try:
            # Get quality scores from LLM with evaluation criteria
            scores = self.llm_service.check_quality(
                text,
                evaluation_criteria=evaluation_criteria
            )

            metrics = QualityMetrics(
                coherence_score=scores.get("methodology", 0.0),
                relevance_score=scores.get("validity", 0.0),
                completeness_score=scores.get("reliability", 0.0),
                overall_score=scores.get("overall", 0.0)
            )

            return metrics
        except Exception as e:
            raise QualityControlException(f"Error evaluating quality: {str(e)}")

    def determine_quality_level(
        self,
        metrics: QualityMetrics,
        evaluation_criteria: Optional[EvaluationCriteria] = None
    ) -> QualityLevel:
        try:
            overall_score = metrics.overall_score
            threshold = (
                evaluation_criteria.min_quality_score
                if evaluation_criteria
                else self.config["validation_threshold"]
            )

            if overall_score >= threshold * 1.2:
                return QualityLevel.HIGH
            elif overall_score >= threshold:
                return QualityLevel.MEDIUM
            else:
                return QualityLevel.LOW
        except Exception as e:
            raise QualityControlException(f"Error determining quality level: {str(e)}")

    def suggest_improvements(
        self,
        text: str,
        metrics: QualityMetrics,
        evaluation_criteria: Optional[EvaluationCriteria] = None
    ) -> str:
        try:
            # Generate feedback based on metrics and criteria
            feedback = self._generate_feedback(metrics, evaluation_criteria)

            # Get improvement suggestions from LLM
            improved_text = self.llm_service.improve_text(
                text,
                feedback,
                evaluation_criteria=evaluation_criteria
            )

            return improved_text
        except Exception as e:
            raise QualityControlException(f"Error suggesting improvements: {str(e)}")

    def _generate_feedback(
        self,
        metrics: QualityMetrics,
        evaluation_criteria: Optional[EvaluationCriteria] = None
    ) -> str:
        """Generate feedback based on quality metrics and evaluation criteria."""
        feedback_points = []

        threshold = (
            evaluation_criteria.min_quality_score
            if evaluation_criteria
            else self.config["min_confidence_score"]
        )

        if metrics.coherence_score < threshold:
            feedback_points.append("Improve methodology and logical flow")

        if metrics.relevance_score < threshold:
            feedback_points.append("Enhance validity and relevance")

        if metrics.completeness_score < threshold:
            feedback_points.append("Increase reliability and completeness")

        if evaluation_criteria and evaluation_criteria.custom_criteria:
            for criterion, weight in evaluation_criteria.custom_criteria.items():
                feedback_points.append(f"Consider {criterion} (weight: {weight})")

        return ". ".join(feedback_points) if feedback_points else "Minor improvements suggested"
