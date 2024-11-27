from typing import Dict, Any, List, Optional
from core.models import ResearchProject
from core.exceptions import ResearchException
from core.types import QualityLevel as CoreQualityLevel
from services.llm.service import LLMService
from services.embedding.service import EmbeddingService
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import json
import re
from datetime import datetime

from enum import Enum

class QualityLevel(Enum):
    EXCELLENT = "excellent"
    VERY_GOOD = "very good"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"

class QualityService:
    def __init__(self, llm_service: LLMService, embedding_service: EmbeddingService):
        self.llm_service = llm_service
        self.embedding_service = embedding_service
        self.quality_model = None
        self.scaler = StandardScaler()

    def initialize_quality_model(self, training_data: List[Dict[str, Any]]):
        """Initialize and train the quality evaluation model"""
        try:
            # Extract features and labels from training data
            X = []
            y = []

            for item in training_data:
                # Create embeddings for text content
                text_content = f"""
                Methodology: {item['methodology']}
                Results: {item['results']}
                Theoretical Framework: {item.get('theoretical_framework', '')}
                """
                embedding = self.embedding_service.create_embeddings(text_content)

                # Extract numeric features
                numeric_features = [
                    item['sample_size'],
                    item['effect_size'],
                    item['power'],
                    item['methodology_score'],
                    item['innovation_score'],
                    item['internal_validity'],
                    item['external_validity'],
                    item['construct_validity'],
                    item['statistical_validity']
                ]

                # Combine embeddings with numeric features
                features = np.concatenate([
                    embedding[0].vector,
                    numeric_features
                ])

                X.append(features)
                y.append(item['quality_score'])

            # Convert to numpy arrays
            X = np.array(X)
            y = np.array(y)

            # Scale features
            X = self.scaler.fit_transform(X)

            # Train model with optimized hyperparameters
            self.quality_model = RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            self.quality_model.fit(X, y)

        except Exception as e:
            raise ResearchException(f"Error initializing quality model: {str(e)}")

    def evaluate_research_quality(
        self,
        project: ResearchProject,
        metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate overall research quality"""
        try:
            evaluation = {}

            # Methodology Quality
            methodology_score = self._evaluate_methodology(
                project.methodology_description or "",
                metrics
            )
            evaluation["methodology_score"] = methodology_score

            # Innovation Assessment
            innovation_score = self._evaluate_innovation(
                project.query,
                project.methodology_description or "",
                project.result
            )
            evaluation["innovation_score"] = innovation_score

            # Validity Assessment
            validity_scores = self._evaluate_validity(project, metrics)
            evaluation["validity_scores"] = validity_scores

            # Calculate Overall Quality Score
            if self.quality_model:
                # Create text content for embedding
                text_content = f"""
                Methodology: {project.methodology_description}
                Results: {project.result if project.result else 'Not available'}
                Theoretical Framework: {project.theoretical_framework.description if project.theoretical_framework else 'Not available'}
                """
                embedding = self.embedding_service.create_embeddings(text_content)

                # Create feature vector
                numeric_features = [
                    metrics.get('sample_size', 0),
                    metrics.get('effect_size', 0),
                    metrics.get('power', 0),
                    methodology_score,
                    innovation_score,
                    validity_scores['internal'],
                    validity_scores['external'],
                    validity_scores['construct'],
                    validity_scores['statistical']
                ]

                features = np.concatenate([
                    embedding[0].vector,
                    numeric_features
                ])

                # Scale features
                features_scaled = self.scaler.transform(features.reshape(1, -1))

                # Predict quality score
                quality_score = float(self.quality_model.predict(features_scaled)[0])

                # Ensure score is within bounds
                quality_score = max(0.0, min(1.0, quality_score))
                evaluation["quality_score"] = quality_score

                # Determine quality level
                evaluation["quality_level"] = self._determine_quality_level(quality_score)

                # Generate improvement suggestions
                suggestions = self._generate_improvement_suggestions(
                    project,
                    evaluation
                )
                evaluation["improvement_suggestions"] = suggestions

            return evaluation

        except Exception as e:
            raise ResearchException(f"Error evaluating research quality: {str(e)}")

    def _evaluate_methodology(
        self,
        methodology: str,
        metrics: Dict[str, Any]
    ) -> float:
        """Evaluate methodology quality"""
        try:
            prompt = f"""
            Evaluate this research methodology:
            {methodology}

            Consider these aspects and provide a score (0.0-1.0) for each:
            1. Research Design:
               - Appropriateness for research question
               - Control of confounding variables
               - Sampling strategy

            2. Statistical Rigor:
               - Sample size: {metrics.get('sample_size')}
               - Power: {metrics.get('power')}
               - Effect size: {metrics.get('effect_size')}

            3. Data Collection:
               - Methods appropriateness
               - Quality control measures
               - Data validation procedures

            4. Analysis Techniques:
               - Statistical methods
               - Data processing procedures
               - Result interpretation approach

            5. Bias Control:
               - Selection bias
               - Measurement bias
               - Reporting bias

            Format response as JSON with scores and justifications for each aspect.
            """

            response = self.llm_service.generate_research(prompt)
            scores = json.loads(response)

            # Calculate weighted average
            weights = {
                "research_design": 0.25,
                "statistical_rigor": 0.25,
                "data_collection": 0.2,
                "analysis_techniques": 0.2,
                "bias_control": 0.1
            }

            methodology_score = sum(
                scores[aspect]["score"] * weight
                for aspect, weight in weights.items()
            )

            return float(methodology_score)

        except Exception as e:
            raise ResearchException(f"Error evaluating methodology: {str(e)}")

    def _evaluate_innovation(
        self,
        research_question: str,
        methodology: str,
        results: Optional[str]
    ) -> float:
        """Evaluate research innovation"""
        try:
            prompt = f"""
            Evaluate the innovation level of this research:

            Research Question: {research_question}
            Methodology: {methodology}
            Results: {results if results else 'Not available'}

            Score these aspects (0.0-1.0):
            1. Novelty:
               - Originality of research question
               - Uniqueness of approach
               - New combinations of existing ideas

            2. Methodological Innovation:
               - Novel methods or techniques
               - Creative adaptations of existing methods
               - Integration of multiple approaches

            3. Potential Impact:
               - Theoretical contributions
               - Practical applications
               - Influence on future research

            4. Technical Advancement:
               - Use of cutting-edge techniques
               - Technical sophistication
               - Implementation quality

            Format response as JSON with scores and justifications for each aspect.
            """

            response = self.llm_service.generate_research(prompt)
            scores = json.loads(response)

            # Calculate weighted average
            weights = {
                "novelty": 0.3,
                "methodological_innovation": 0.3,
                "potential_impact": 0.25,
                "technical_advancement": 0.15
            }

            innovation_score = sum(
                scores[aspect]["score"] * weight
                for aspect, weight in weights.items()
            )

            return float(innovation_score)

        except Exception as e:
            raise ResearchException(f"Error evaluating innovation: {str(e)}")

    def _evaluate_validity(
        self,
        project: ResearchProject,
        metrics: Dict[str, Any]
    ) -> Dict[str, float]:
        """Evaluate different types of validity"""
        try:
            validity_prompts = {
                "internal": """
                Evaluate internal validity considering:
                1. Control of confounding variables
                2. Causal relationship strength
                3. Experimental control
                4. Measurement accuracy
                5. Time-related threats
                """,
                "external": """
                Evaluate external validity considering:
                1. Population generalizability
                2. Environmental generalizability
                3. Temporal generalizability
                4. Ecological validity
                5. Replication potential
                """,
                "construct": """
                Evaluate construct validity considering:
                1. Operational definitions
                2. Measurement appropriateness
                3. Theoretical alignment
                4. Construct independence
                5. Method variance
                """,
                "statistical": """
                Evaluate statistical validity considering:
                1. Statistical power
                2. Effect size significance
                3. Assumption verification
                4. Analysis appropriateness
                5. Type I/II error control
                """
            }

            validity_scores = {}
            for validity_type, criteria in validity_prompts.items():
                prompt = f"""
                {criteria}

                Research Details:
                Question: {project.query}
                Methodology: {project.methodology_description}
                Results: {project.result if project.result else 'Not available'}

                Metrics:
                - Sample Size: {metrics.get('sample_size')}
                - Effect Size: {metrics.get('effect_size')}
                - Power: {metrics.get('power')}

                Provide a detailed evaluation and score (0.0-1.0) in JSON format.
                Include justifications for the score.
                """

                response = self.llm_service.generate_research(prompt)
                evaluation = json.loads(response)

                validity_scores[validity_type] = float(evaluation["score"])

            return validity_scores

        except Exception as e:
            raise ResearchException(f"Error evaluating validity: {str(e)}")

    def _determine_quality_level(self, quality_score: float) -> CoreQualityLevel:
        """Determine quality level based on score"""

        if quality_score >= 0.9:
            return CoreQualityLevel["EXCELLENT"]
        elif quality_score >= 0.8:
            return CoreQualityLevel["VERY_GOOD"]
        elif quality_score >= 0.7:
            return CoreQualityLevel["GOOD"]
        elif quality_score >= 0.6:
            return CoreQualityLevel["FAIR"]
        else:
            return CoreQualityLevel["POOR"]

    def _generate_improvement_suggestions(
        self,
        project: ResearchProject,
        evaluation: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate specific improvement suggestions based on evaluation"""
        try:
            prompt = f"""
            Based on the quality evaluation results:
            {json.dumps(evaluation, indent=2)}

            Generate specific improvement suggestions for:
            1. Methodology:
               - Research design
               - Data collection
               - Analysis methods

            2. Validity:
               - Internal validity
               - External validity
               - Construct validity
               - Statistical validity

            3. Innovation:
               - Novelty
               - Technical advancement
               - Impact potential

            4. Overall Quality:
               - Major strengths
               - Critical weaknesses
               - Priority improvements

            Format response as JSON with:
            - Category
            - Specific suggestions
            - Priority level (1-5)
            - Implementation difficulty (1-5)
            - Expected impact (1-5)
            """

            response = self.llm_service.generate_research(prompt)
            suggestions = json.loads(response)

            # Sort suggestions by priority and expected impact
            sorted_suggestions = sorted(
                suggestions,
                key=lambda x: (x["priority_level"], x["expected_impact"]),
                reverse=True
            )

            return sorted_suggestions

        except Exception as e:
            raise ResearchException(f"Error generating improvement suggestions: {str(e)}")
