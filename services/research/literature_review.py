from typing import List, Dict, Any, Optional
from core.models import ResearchProject, Reference
from core.exceptions import ResearchException
from core.types import (
    ResearchStatus,
    ResearchGap,
    QualityLevel,
    EmbeddingVector,
    Reference,
    QualityMetrics
)
from services.llm.service import LLMService
from services.embedding.service import EmbeddingService
import numpy as np
from sklearn.cluster import KMeans
from datetime import datetime
import logging
from logging import Logger

logger = logging.getLogger(__name__)
import requests
import json
from dataclasses import dataclass

@dataclass
class Context:
    research_project: Optional[ResearchProject] = None

class LiteratureReviewService:
    def __init__(self, llm_service: LLMService, embedding_service: EmbeddingService, context: Optional[Context] = None):
        self.llm_service = llm_service
        self.embedding_service = embedding_service
        self.context = context or Context()
        self.perplexity_api_key = self._load_api_key()
        self.perplexity_api_url = "https://api.perplexity.ai/chat/completions"

        if self.context.research_project:
            self.current_project = self.context.research_project

    def _load_api_key(self) -> str:
        """Load Perplexity API key from environment"""
        import os
        api_key = os.getenv("PERPLEXITY_API_KEY")
        if not api_key:
            raise ResearchException("Perplexity API key not found in environment")
        return api_key

    def collect_initial_papers(self, query: str, limit: int = 100) -> List[Reference]:
        try:
            headers = {
                "Authorization": f"Bearer {self.perplexity_api_key}",
                "Content-Type": "application/json"
            }

            messages = [
                {
                    "role": "system",
                    "content": "Return academic papers in a structured format. For each paper include: Title, Authors (separated by commas), Year, Abstract, Citations count, Journal name, DOI, and URL. Separate papers with '---'. Format each paper as follows:\nTitle: [title]\nAuthors: [authors]\nYear: [year]\nAbstract: [abstract]\nCitations: [count]\nJournal: [journal]\nDOI: [doi]\nURL: [url]"
                },
                {
                    "role": "user",
                    "content": f"Find detailed academic papers about: {query}"
                }
            ]

            if self.context.research_project and self.context.research_project.context:
                messages.append({
                    "role": "user",
                    "content": f"Consider the research context: {self.context.research_project.context}"
                })

            # Print search query
            print("\n=== Perplexity API Search ===")
            print(f"Query: {query}")
            print("============================\n")

            request_body = {
                "model": "llama-3.1-sonar-small-128k-online",
                "messages": messages,
                "temperature": 0.2,
                "top_p": 0.9,
                "max_tokens": 2048,
                "return_citations": True,
                "search_domain_filter": ["arxiv.org", "scholar.google.com", "science.org"],
                "stream": False
            }

            response = requests.post(
                self.perplexity_api_url,
                headers=headers,
                json=request_body
            )

            if response.status_code != 200:
                print(f"Error: API request failed with status code {response.status_code}")
                print(f"Response: {response.text}")
                raise ResearchException(f"Perplexity API error: {response.text}")

            response_data = response.json()
            papers_data = self._parse_assistant_response(response_data["choices"][0]["message"]["content"])
            citations = response_data.get("citations", [])

            # Print found papers
            print("\n=== Search Results ===")
            for idx, paper in enumerate(papers_data, 1):
                print(f"\nPaper {idx}:")
                print(f"Title: {paper.get('title', 'N/A')}")
                print(f"Authors: {paper.get('authors', 'N/A')}")
                print(f"Year: {paper.get('year', 'N/A')}")
                print(f"Journal: {paper.get('journal', 'N/A')}")
                print(f"Citations: {paper.get('citations', 'N/A')}")
                print(f"DOI: {paper.get('doi', 'N/A')}")
                print(f"URL: {paper.get('url', 'N/A')}")
                print("Abstract:", paper.get('abstract', 'N/A')[:200] + "..." if paper.get('abstract') else 'N/A')
                print("-" * 50)
            print(f"\nTotal papers found: {len(papers_data)}")
            print("============================\n")

            papers = []
            for paper_data in papers_data[:limit]:
                paper_text = f"{paper_data['title']}. {paper_data.get('abstract', '')}"
                paper_embedding = self.embedding_service.create_embeddings(paper_text)
                query_embedding = self.embedding_service.create_embeddings(query)

                if self.context.research_project and self.context.research_project.context:
                    context_embedding = self.embedding_service.create_embeddings(self.context.research_project.context)
                    context_score = np.dot(paper_embedding[0].vector, context_embedding[0].vector)
                else:
                    context_score = 0.0

                relevance_score = np.dot(
                    paper_embedding[0].vector,
                    query_embedding[0].vector
                )

                paper = Reference(
                    title=paper_data["title"].strip(),
                    authors=[author.strip() for author in paper_data.get("authors", "").split(",")],
                    year=int(paper_data.get("year", 0)),
                    abstract=paper_data.get("abstract", "").strip(),
                    citation_count=int(paper_data.get("citations", "0").replace(",", "")),
                    journal_impact_factor=self._get_journal_impact_factor(paper_data.get("journal", "")),
                    relevance_score=float(relevance_score + context_score),
                    url=paper_data.get("url", "").strip(),
                    doi=paper_data.get("doi", "").strip()
                )
                papers.append(paper)

            return papers

        except Exception as e:
            print(f"\nError occurred during paper collection: {str(e)}")
            raise ResearchException(f"Error collecting papers: {str(e)}")

    def _parse_assistant_response(self, content: str) -> List[dict]:
        import re

        content = re.sub(r'\s+', ' ', content)
        content = content.replace('\n', ' ').strip()

        patterns = {
            'paper_separator': r'---',
            'title': r'Title:\s*(?P<title>(?:(?!Authors:|Year:|Abstract:|Citations:|Journal:|DOI:|URL:).)+)',
            'authors': r'Authors:\s*(?P<authors>(?:(?!Year:|Abstract:|Citations:|Journal:|DOI:|URL:).)+)',
            'year': r'Year:\s*(?P<year>(?:(?!Abstract:|Citations:|Journal:|DOI:|URL:).)+)',
            'abstract': r'Abstract:\s*(?P<abstract>(?:(?!Citations:|Journal:|DOI:|URL:).)+)',
            'citations': r'Citations:\s*(?P<citations>(?:(?!Journal:|DOI:|URL:).)+)',
            'journal': r'Journal:\s*(?P<journal>(?:(?!DOI:|URL:).)+)',
            'doi': r'DOI:\s*(?P<doi>(?:(?!URL:).)+)',
            'url': r'URL:\s*(?P<url>.+?)(?=---|$)',
        }

        papers_raw = re.split(patterns['paper_separator'], content)
        papers_data = []

        for paper_raw in papers_raw:
            if not paper_raw.strip():
                continue

            paper_data = {}

            for field, pattern in patterns.items():
                if field == 'paper_separator':
                    continue

                match = re.search(pattern, paper_raw, re.IGNORECASE | re.DOTALL)
                if match:
                    value = match.group(field).strip()
                    value = re.sub(r'\s+', ' ', value)
                    paper_data[field] = value
                else:
                    paper_data[field] = ""

            try:
                year_str = re.sub(r'[^\d]', '', paper_data.get('year', ''))
                paper_data['year'] = int(year_str) if year_str and 1900 <= int(year_str) <= 2024 else 0

                citations_str = re.sub(r'[^\d]', '', paper_data.get('citations', ''))
                paper_data['citations'] = int(citations_str) if citations_str else 0

                doi_match = re.search(r'10\.\d{4,9}/[-._;()/:\w]+', paper_data.get('doi', ''))
                paper_data['doi'] = doi_match.group(0) if doi_match else ""

                url_match = re.search(r'https?://\S+', paper_data.get('url', ''))
                paper_data['url'] = url_match.group(0) if url_match else ""

                authors = paper_data.get('authors', '')
                authors = re.split(r',\s*(?:and\s+)?|\s+and\s+', authors)
                authors = [author.strip() for author in authors if author.strip()]
                paper_data['authors'] = ', '.join(authors)

            except Exception as e:
                print(f"Error cleaning paper data: {str(e)}")
                continue

            if paper_data.get('title') and paper_data.get('abstract'):
                papers_data.append(paper_data)

        return papers_data

    def _get_journal_impact_factor(self, journal_name: str) -> float:
        impact_factors = {
            "Nature": 49.962,
            "Science": 47.728,
            "Cell": 41.582,
            "PNAS": 11.205,
            "PLoS ONE": 3.240,
            "Scientific Reports": 4.996
        }
        return impact_factors.get(journal_name, 1.0)

    def analyze_research_gaps(self, papers: List[Reference]) -> List[Dict[str, Any]]:
        try:
            gap_texts = []
            for paper in papers:
                context = ""
                if self.context.research_project and self.context.research_project.context:
                    context = f"\nResearch Context: {self.context.research_project.context}"

                gap_prompt = f"""
                Based on this paper:
                Title: {paper.title}
                Abstract: {paper.abstract}
                Year: {paper.year}
                {context}

                Analyze potential research gaps through this structured approach:

                Step 1: Initial Gap Analysis
                For each gap category below, identify at least one specific example and explain its significance:
                - Methodological gaps (focus on research design, data collection, or analytical approaches)
                - Theoretical gaps (examine underlying assumptions, conceptual frameworks, or theoretical foundations)
                - Application gaps (consider practical implementations or real-world applications)

                Step 2: Contextual Evaluation
                Compare this paper's approach with:
                - Contemporary research trends in the field
                - Similar studies from different contexts
                - Alternative methodological approaches

                Step 3: Specific Gap Documentation
                For each identified gap, provide:
                a) Concrete evidence from the paper supporting its existence
                b) Potential impact on the field if addressed
                c) Specific challenges in addressing this gap
                d) At least one novel approach to address it

                Step 4: Priority Assessment
                Rank the top 3 most critical gaps based on:
                - Scientific impact
                - Practical feasibility
                - Innovation potential
                - Resource requirements

                Format your response as a structured analysis following these steps, ensuring each gap is unique and specifically related to this paper's context."
                """

                gap_response = self.llm_service.generate_research(gap_prompt)
                gap_texts.append({
                    "text": gap_response,
                    "paper": paper.title,
                    "year": paper.year,
                    "citations": paper.citation_count
                })

            gap_embeddings = []
            for gap in gap_texts:
                embedding = self.embedding_service.create_embeddings(gap["text"])
                gap_embeddings.append(embedding[0].vector)

            n_clusters = min(5, len(gap_embeddings))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(gap_embeddings)

            research_gaps = []
            for i in range(n_clusters):
                cluster_gaps = [gap for j, gap in enumerate(gap_texts) if clusters[j] == i]

                cluster_gaps.sort(
                    key=lambda x: (x["citations"], x["year"]),
                    reverse=True
                )

                context = ""
                if self.context.research_project and self.context.research_project.context:
                    context = f"\nResearch Context: {self.context.research_project.context}"

                summary_prompt = f"""
                Analyze these related research gaps:
                {json.dumps(cluster_gaps, indent=2)}
                {context}

                Provide:
                1. A concise summary of the common gap theme
                2. The significance of this research gap
                3. Potential approaches to address it
                4. Required resources or expertise
                5. Potential challenges
                """

                cluster_summary = self.llm_service.generate_research(summary_prompt)

                scores = self._score_research_gap(
                    cluster_summary,
                    cluster_gaps
                )

                research_gaps.append({
                    "description": cluster_summary,
                    "papers": [gap["paper"] for gap in cluster_gaps],
                    "importance_score": scores["importance"],
                    "feasibility_score": scores["feasibility"],
                    "impact_score": scores["impact"],
                    "novelty_score": scores["novelty"],
                    "resource_requirements": scores["resources"],
                    "cluster_id": f"cluster_{i}"
                })

            return sorted(
                research_gaps,
                key=lambda x: (
                    x["importance_score"] *
                    x["feasibility_score"] *
                    x["impact_score"] *
                    x["novelty_score"]
                ),
                reverse=True
            )

        except Exception as e:
            raise ResearchException(f"Error analyzing research gaps: {str(e)}")

    def _score_research_gap(
        self,
        gap_summary: str,
        cluster_gaps: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        try:
            context = ""
            if self.context.research_project and self.context.research_project.context:
                context = f"\nResearch Context: {self.context.research_project.context}"

            scoring_prompt = f"""
            Evaluate this research gap and provide scores (0.0-1.0) for each criterion:

            Research Gap Summary:
            {gap_summary}

            Related Papers:
            {json.dumps([{
                "title": gap["paper"],
                "year": gap["year"],
                "citations": gap["citations"]
            } for gap in cluster_gaps], indent=2)}
            {context}

            Score these aspects:
            1. Importance: How critical is addressing this gap?
            2. Feasibility: How feasible is it to address this gap?
            3. Impact: What potential impact would addressing this gap have?
            4. Novelty: How original/innovative is this research direction?
            5. Resources: How resource-intensive would this research be? (0.0 = very intensive, 1.0 = minimal resources)

            Format response as JSON with scores and brief justifications.
            """

            response = self.llm_service.generate_research(scoring_prompt)

            scores = json.loads(response)

            required_fields = ["importance", "feasibility", "impact", "novelty", "resources"]
            for field in required_fields:
                if field not in scores:
                    raise ResearchException(f"Missing score field: {field}")
                if not isinstance(scores[field], (int, float)):
                    raise ResearchException(f"Invalid score format for {field}")
                if not 0 <= scores[field] <= 1:
                    raise ResearchException(f"Score out of range for {field}")

            return scores

        except Exception as e:
            raise ResearchException(f"Error scoring research gap: {str(e)}")

    def _normalize_citations(self, citations: int) -> float:
        if citations <= 0:
            return 0.0
        return min(1.0, citations / 1000)

    def _normalize_impact_factor(self, impact_factor: float) -> float:
        if impact_factor <= 0:
            return 0.0
        return min(1.0, impact_factor / 10)

    def _calculate_recency_score(self, year: int) -> float:
        current_year = datetime.now().year
        years_old = current_year - year
        if years_old <= 0:
            return 1.0
        elif years_old >= 10:
            return 0.0
        return 1.0 - (years_old / 10)

    def _generate_sample_papers(self, query: str, limit: int) -> List[Reference]:
        papers = []
        for i in range(limit):
            papers.append(Reference(
                title=f"Sample Paper {i}",
                authors=["Author A", "Author B"],
                year=2020 + (i % 5),
                abstract=f"Sample abstract for paper {i}",
                citation_count=100 * (i % 10),
                journal_impact_factor=2.0 + (i % 8),
                relevance_score=0.8,
                url=f"https://example.com/paper{i}",
                doi=f"10.1234/sample.{i}"
            ))
        return papers
        def evaluate_paper_quality(self, papers: List[Reference]) -> List[Reference]:
            try:
                evaluated_papers = []
                for paper in papers:
                    citation_score = self._normalize_citations(paper.citation_count or 0)
                    impact_score = self._normalize_impact_factor(paper.journal_impact_factor or 0)
                    year_score = self._calculate_recency_score(paper.year)

                    paper_embedding = self.embedding_service.create_embeddings(
                        f"{paper.title}. {', '.join(paper.authors)}"
                    )

                    # numpy array를 일반 list로 변환
                    vector = paper_embedding[0].vector
                    if isinstance(vector, np.ndarray):
                        vector = vector.tolist()

                    if self.context.research_project and self.context.research_project.context:
                        context_embedding = self.embedding_service.create_embeddings(
                            self.context.research_project.context
                        )
                        context_score = np.dot(paper_embedding[0].vector, context_embedding[0].vector)
                    else:
                        context_score = 0.0

                    paper_score = (
                        citation_score * 0.3 +
                        impact_score * 0.3 +
                        year_score * 0.2 +
                        paper.relevance_score * 0.2
                    )

                    quality_metrics = QualityMetrics(
                        coherence_score=citation_score,
                        relevance_score=paper.relevance_score,
                        completeness_score=impact_score,
                        overall_score=paper_score,
                        validity_score=context_score
                    )
                    quality_metrics.calculate_quality_score()

                    embedding_data = EmbeddingVector(
                        text=f"{paper.title}. {', '.join(paper.authors)}",
                        vector=vector,  # 이미 변환된 vector 사용
                        metadata={
                            "year_score": year_score,
                            "citation_score": citation_score,
                            "impact_score": impact_score,
                            "context_score": context_score
                        }
                    )

                    evaluated_paper = Reference(
                        **paper.dict(),
                        quality_metrics=quality_metrics,
                        embedding_data=embedding_data
                    )
                    evaluated_papers.append(evaluated_paper)

                return sorted(evaluated_papers, key=lambda x: x.quality_metrics.quality_score, reverse=True)

            except Exception as e:
                raise ResearchException(f"Error evaluating papers: {str(e)}")
