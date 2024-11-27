from langchain.prompts import PromptTemplate

RESEARCH_PROMPT = PromptTemplate(
    input_variables=["query", "context"],
    template="""Based on the following query and context, provide a comprehensive research analysis:

Query: {query}

Context: {context}

Please provide a detailed analysis that:
1. Addresses the main points of the query
2. Incorporates relevant information from the context
3. Maintains logical flow and coherence
4. Supports conclusions with evidence
"""
)

QUALITY_CHECK_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""Evaluate the quality of the following text based on these criteria:
1. Coherence and logical flow
2. Relevance to the topic
3. Completeness of analysis
4. Evidence-based reasoning

Text: {text}

Please provide numerical scores (0.0-1.0) for each criterion and an explanation.
"""
)

IMPROVEMENT_PROMPT = PromptTemplate(
    input_variables=["text", "feedback"],
    template="""Improve the following text based on the provided feedback:

Original Text: {text}

Feedback: {feedback}

Please provide an improved version that addresses the feedback while maintaining the original intent.
"""
)
