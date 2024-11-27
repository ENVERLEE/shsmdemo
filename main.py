from typing import Optional

from services.research.service import ResearchService
from services.research.literature_review import LiteratureReviewService
from services.research.methodology import MethodologyService
from services.research.quality import QualityService
from services.llm.service import LLMService
from services.embedding.service import EmbeddingService
from core.types import (
    ResearchRequest,
    ResearchStatus,
    ResearchDirection,
    EvaluationCriteria
)
from core.models import ResearchProject
from utils.helpers import save_json_file, ensure_directory
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
import json
from datetime import datetime
import sys
import os

class ResearchAssistant:
    def __init__(self):
        self.research_service = ResearchService()
        self.output_dir = Path("output")
        ensure_directory(str(self.output_dir))

    def process_research_request(
        self,
        topic: str,
        description: str,
        direction: ResearchDirection,
        evaluation_criteria: Optional[EvaluationCriteria] = None,
        context: Optional[str] = None,
        max_iterations: Optional[int] = None
    ):
        """Process a research request and save results."""
        try:
            # Create default evaluation criteria if not provided
            if evaluation_criteria is None:
                evaluation_criteria = EvaluationCriteria()

            # Create research request
            request = ResearchRequest(
                topic=topic,
                description=description,
                direction=direction,
                evaluation_criteria=evaluation_criteria,
                context=context,
                max_iterations=max_iterations
            )

            # Conduct research
            print(f"Starting research for topic: {topic}")
            print(f"Research direction: {direction.value}")
            research = self.research_service.conduct_research(request)

            # Enhance with embeddings if research was successful
            if research.status == ResearchStatus.COMPLETED:
                research = self.research_service.enhance_with_embeddings(research)

            # Save results
            output_file = self.output_dir / f"research_{research.id}.json"
            save_json_file(
                research.dict(),
                str(self.output_dir / f"research_{research.id}.json")
            )

            print(f"Research completed. Results saved to: {output_file}")
            print(f"Status: {research.status}")
            print(f"Confidence Score: {research.confidence_score}")
            print(f"Quality Score: {research.quality_score}")

            return research

        except Exception as e:
            print(f"Error processing research request: {str(e)}")
            raise

class MainFunction:
    @staticmethod
    def get_user_input() -> tuple[str, str, ResearchDirection, Optional[str]]:
        """Get research inputs from user"""
        print("\n=== Research Assistant CLI ===")

        # Get research topic
        print("Enter your research topic (or 'quit' to exit):")
        topic = input("> ").strip()

        if topic.lower() == 'quit':
            sys.exit(0)

        # Get research description
        print("\nEnter detailed research description:")
        description = input("> ").strip()

        # Get research direction
        print("\nSelect research direction:")
        for i, direction in enumerate(ResearchDirection):
            print(f"{i+1}. {direction.value} - {direction.name}")

        while True:
            try:
                direction_idx = int(input("> ").strip()) - 1
                direction = list(ResearchDirection)[direction_idx]
                break
            except (ValueError, IndexError):
                print("Invalid selection. Please try again.")

        # Get additional context (optional)
        print("\nEnter additional context (optional, press Enter to skip):")
        context = input("> ").strip()

        return topic, description, direction, context if context else None

    @staticmethod
    def get_evaluation_criteria() -> EvaluationCriteria:
        """Get custom evaluation criteria from user"""
        print("\n=== Evaluation Criteria Setup ===")
        print("Enter weights for each criterion (0.0-1.0):")

        try:
            methodology = float(input("Methodology weight (default 0.3): ") or 0.3)
            innovation = float(input("Innovation weight (default 0.2): ") or 0.2)
            validity = float(input("Validity weight (default 0.3): ") or 0.3)
            reliability = float(input("Reliability weight (default 0.2): ") or 0.2)

            min_quality = float(input("Minimum quality score (default 0.7): ") or 0.7)
            required_validity = float(input("Required validity score (default 0.8): ") or 0.8)

            return EvaluationCriteria(
                methodology_weight=methodology,
                innovation_weight=innovation,
                validity_weight=validity,
                reliability_weight=reliability,
                min_quality_score=min_quality,
                required_validity_score=required_validity
            )
        except ValueError:
            print("Invalid input. Using default values.")
            return EvaluationCriteria()

    @staticmethod
    def process_and_display_results(
        assistant: ResearchAssistant,
        topic: str,
        description: str,
        direction: ResearchDirection,
        evaluation_criteria: EvaluationCriteria,
        context: Optional[str] = None
    ):
        """Process research request and display results with quality feedback"""
        try:
            print("\nProcessing your research request...")
            research = assistant.process_research_request(
                topic=topic,
                description=description,
                direction=direction,
                evaluation_criteria=evaluation_criteria,
                context=context
            )

            print("\n=== Research Results ===")
            print(f"Topic: {research.query}")
            print(f"Direction: {research.metadata['research_direction']}")
            print(f"\nAnalysis:\n{research.result}")

            quality_info = []
            if research.quality_score:
                print("\n=== Quality Assessment ===")
                metrics = research.metadata.get("final_evaluation", {})
                for criterion, score in metrics.items():
                    quality_info.append({
                        "criterion": criterion,
                        "score": f"{score:.2f}"
                    })
                    print(f"{criterion}: {score:.2f}")

            research_dict = {
                "topic": research.query,
                "direction": research.metadata["research_direction"],
                "result": research.result,
                "evaluation": research.metadata.get("final_evaluation", {}),
                "iterations": research.metadata.get("iterations", 0)
            }

            return quality_info, research_dict
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            return None, None

def main():
    assistant = ResearchAssistant()
    main_func = MainFunction()

    while True:
        # Get research inputs
        topic, description, direction, context = main_func.get_user_input()

        # Get evaluation criteria
        evaluation_criteria = main_func.get_evaluation_criteria()

        # Process and display results
        quality_info, research_dict = main_func.process_and_display_results(
            assistant,
            topic,
            description,
            direction,
            evaluation_criteria,
            context
        )

        if quality_info and research_dict:
            print("\nResearch completed successfully!")

        print("\nPress Enter to start new research or type 'quit' to exit:")
        if input().lower() == 'quit':
            break

if __name__ == "__main__":
    main()
