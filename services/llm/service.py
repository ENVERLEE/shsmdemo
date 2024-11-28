from typing import Dict, Any, Optional, Union, List
from langchain_community.llms.mlx_pipeline import MLXPipeline
from langchain_community.chat_models.mlx import ChatMLX
from langchain.chains import LLMChain
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from core.exceptions import LLMServiceException
from config.settings import LLM_CONFIG
from services.llm.prompts import RESEARCH_PROMPT, QUALITY_CHECK_PROMPT, IMPROVEMENT_PROMPT
from core.types import ResearchDirection, EvaluationCriteria

class LLMService:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or LLM_CONFIG

        # Initialize MLX model
        self.llm = MLXPipeline.from_model_id(
            model_id=self.config['model_id'],
            pipeline_kwargs={
                "max_tokens": self.config['max_tokens'],
                "temp": self.config['temperature'],
                "top_p": self.config['top_p']
            }
        )

        self.chat_model = ChatMLX(llm=self.llm)

    def _format_response(
        self,
        response: Union[str, List[Union[str, Dict[str, Any]]], BaseMessage]
    ) -> str:
        """Helper method to format MLX response to string"""
        print("LLM Response:", response)  # Print the raw LLM response
        if isinstance(response, BaseMessage):
            return str(response.content).strip()
        if isinstance(response, str):
            return response.strip()
        elif isinstance(response, list):
            return str(response[0]).strip() if response else ""
        return str(response).strip()

    def generate_research(
        self,
        query: str,
        context: Optional[str] = None,
        direction: Optional[ResearchDirection] = None
    ) -> str:
        try:
            direction_guidance = ""
            if direction:
                direction_guidance = f"\nResearch Direction: {direction.value}\n"
                direction_map = {
                    ResearchDirection.EXPLORATORY: "Focus on identifying variables and generating hypotheses.",
                    ResearchDirection.DESCRIPTIVE: "Focus on detailed description and pattern analysis.",
                    ResearchDirection.EXPLANATORY: "Focus on causal relationships and mechanisms.",
                    ResearchDirection.EXPERIMENTAL: "Focus on experimental design and variable control.",
                    ResearchDirection.THEORETICAL: "Focus on theoretical analysis and integration."
                }
                direction_guidance += direction_map.get(direction, "")

            prompt_content = RESEARCH_PROMPT.format(
                query=query,
                context=context or "",
                direction_guidance=direction_guidance
            )

            message = HumanMessage(content=prompt_content)
            response = self.chat_model.invoke([message])
            formatted = self._format_response(response)
            print("Formatted Response:", formatted)  # Print formatted response
            return formatted

        except Exception as e:
            raise LLMServiceException(f"Error generating research: {str(e)}")

    def check_quality(
        self,
        text: str,
        evaluation_criteria: Optional[EvaluationCriteria] = None
    ) -> Dict[str, float]:
        """
        Check quality of text using custom evaluation criteria.

        Args:
            text: The text to evaluate
            evaluation_criteria: Optional evaluation criteria with weights
        """
        try:
            # Prepare evaluation criteria prompt
            criteria_prompt = QUALITY_CHECK_PROMPT.format(text=text)  # Remove str() conversion
            if evaluation_criteria:
                criteria_prompt = f"""Evaluate the quality based on these weighted criteria:
                1. Methodology (weight: {evaluation_criteria.methodology_weight})
                2. Innovation (weight: {evaluation_criteria.innovation_weight})
                3. Validity (weight: {evaluation_criteria.validity_weight})
                4. Reliability (weight: {evaluation_criteria.reliability_weight})

                Additional criteria: {evaluation_criteria.custom_criteria or 'None'}

                Text: {text}

                Provide numerical scores (0.0-1.0) for each criterion and detailed justification.
                """

            message = HumanMessage(content=criteria_prompt)
            response = self.chat_model.invoke([message])
            response_text = self._format_response(response)
            print("Quality Check Response:", response_text)  # Print quality check response

            scores = self._parse_quality_scores(
                response_text,
                evaluation_criteria
            )
            return scores

        except Exception as e:
            raise LLMServiceException(f"Error checking quality: {str(e)}")

    def improve_text(
        self,
        text: str,
        feedback: str,
        evaluation_criteria: Optional[EvaluationCriteria] = None
    ) -> str:
        """
        Improve text based on feedback and evaluation criteria.

        Args:
            text: The text to improve
            feedback: Feedback for improvement
            evaluation_criteria: Optional evaluation criteria
        """
        try:
            improvement_guidance = ""
            if evaluation_criteria:
                improvement_guidance = f"""
                Consider these criteria weights:
                - Methodology: {evaluation_criteria.methodology_weight}
                - Innovation: {evaluation_criteria.innovation_weight}
                - Validity: {evaluation_criteria.validity_weight}
                - Reliability: {evaluation_criteria.reliability_weight}

                Minimum required scores:
                - Quality: {evaluation_criteria.min_quality_score}
                - Validity: {evaluation_criteria.required_validity_score}
                """

            prompt = IMPROVEMENT_PROMPT.format(  # Remove str() conversion
                text=text,
                feedback=feedback,
                improvement_guidance=improvement_guidance
            )

            message = HumanMessage(content=prompt)
            response = self.chat_model.invoke([message])
            formatted = self._format_response(response)
            print("Improvement Response:", formatted)  # Print improvement response
            return formatted

        except Exception as e:
            raise LLMServiceException(f"Error improving text: {str(e)}")

    def _parse_quality_scores(
        self,
        response: str,
        evaluation_criteria: Optional[EvaluationCriteria] = None
    ) -> Dict[str, float]:
        """
        Parse quality check response to extract numerical scores.

        Args:
            response: The raw response text
            evaluation_criteria: Optional evaluation criteria with weights
        """
        try:
            scores = {
                "methodology": 0.0,
                "innovation": 0.0,
                "validity": 0.0,
                "reliability": 0.0,
                "overall": 0.0
            }

            for line in response.split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip().lower()
                    if key in scores:
                        try:
                            scores[key] = float(value.strip())
                        except ValueError:
                            pass

            # Calculate weighted overall score if criteria provided
            if evaluation_criteria:
                scores["overall"] = (
                    scores["methodology"] * evaluation_criteria.methodology_weight +
                    scores["innovation"] * evaluation_criteria.innovation_weight +
                    scores["validity"] * evaluation_criteria.validity_weight +
                    scores["reliability"] * evaluation_criteria.reliability_weight
                )
            else:
                scores["overall"] = sum(scores.values()) / len(scores)

            return scores

        except Exception as e:
            raise LLMServiceException(f"Error parsing quality scores: {str(e)}")
