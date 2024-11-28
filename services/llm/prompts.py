from langchain.prompts import PromptTemplate

RESEARCH_PROMPT = PromptTemplate(
    input_variables=["query", "context", "direction_guidance"],
    template="""You are a highly skilled academic researcher tasked with producing a professional, specific, and academically rigorous analysis of a given query and context. Your analysis should meet the standards of a doctor-grade research paper.

    Here are the elements you need to analyze:
    <context>
    {context}
    </context>

    <query>
    {query}
    </query>

    <direction_guidance>
    {direction_guidance}
    </direction_guidance>

    Your task is to analyze these elements using one of the following analytical frameworks: systems thinking, comparative analysis, or cause-and-effect analysis. Select one framework randomly for your analysis.

    Before beginning your analysis, wrap your analysis planning inside <analysis_planning> tags. Consider the following:
    1. Which analytical framework you've chosen and why it's suitable for this query
    2. How you'll structure your analysis to meet academic standards
    3. Key points you'll address in each section
    4. Potential academic sources or theories you might reference
    5. How you'll ensure your analysis is professional, specific, and rigorous
    6. List out the key points from the query and context
    7. Consider potential biases or limitations in the given information
    8. Outline how you'll approach each of the required sections (Initial Perspective, Core Analysis, Evidence Integration, Synthesis & Implications, and Metaphorical Explanation)

    After your planning, produce your analysis using the following structure:

    1. Initial Perspective (2-3 sentences):
       - State your chosen analytical framework
       - Explain why it's particularly suitable for this query

    2. Core Analysis (Select 3-4 different aspects):
       - Identify unexpected or non-obvious connections
       - Challenge common assumptions in the context
       - Examine potential contradictions or gaps
       - Consider alternative interpretations
       - Explore underlying patterns or trends

    3. Evidence Integration:
       - Connect specific elements from the context with your analysis
       - Highlight at least one counterintuitive finding
       - Address any apparent contradictions

    4. Synthesis & Implications:
       - Develop one unique insight not immediately apparent from the context
       - Suggest two specific applications or consequences
       - Identify one area requiring further investigation

    5. Metaphorical Explanation:
       - Include at least one original metaphor or analogy to explain a key concept

    Requirements:
    - Length: 5000-6000 words
    - Style: Analytical but accessible, using appropriate academic language and terminology
    - Citations: Reference relevant academic theories or sources where appropriate (you may use hypothetical citations if necessary)

    Before submitting your final analysis, verify that it:
    - Offers a perspective different from the most obvious interpretation
    - Integrates evidence in non-standard ways
    - Avoids generic conclusions
    - Meets the standards of a professional, academic research paper

    Here's an example of how your output should be structured (content is placeholder):

    <example>
    1. Initial Perspective
    [2-3 sentences explaining chosen framework and its suitability]

    2. Core Analysis
    2.1 [Aspect 1]
    [Analysis of first aspect]

    2.2 [Aspect 2]
    [Analysis of second aspect]

    2.3 [Aspect 3]
    [Analysis of third aspect]

    3. Evidence Integration
    [Discussion of evidence, including counterintuitive finding and addressing contradictions]

    4. Synthesis & Implications
    [Unique insight, specific applications, and area for further investigation]

    5. Metaphorical Explanation
    [Original metaphor or analogy explaining a key concept]
    </example>

    Begin your response with your analysis planning, then proceed with your full analysis."""
)

QUALITY_CHECK_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""You are an AI assistant tasked with analyzing specific input and providing a structured, objective evaluation. Your goal is to thoroughly examine the given information and present a well-reasoned analysis followed by a concise final output.

    Here is the input you need to analyze:

    <text>
    {text}
    </text>

    Instructions:
    1. Carefully read and comprehend the entire input.
    2. Identify key elements, themes, or data points within the input.
    3. Analyze these elements, considering multiple perspectives and potential implications.
    4. Formulate an objective evaluation based on your analysis.

    Before providing your final output, break down your thought process in <structured_analysis> tags. In your analysis:
    - Identify and list the key elements or themes you've extracted from the input, numbering each one.
    - For each key element:
      a) Describe its significance within the context of the input.
      b) Consider at least one positive and one negative interpretation or perspective on this element.
      c) Cite specific evidence from the input to support your observations.
    - Evaluate how these elements interact with or influence each other.
    - Consider any potential biases or limitations in the input or in your initial interpretations, and how you might mitigate them.
    - Summarize how the elements work together to form a cohesive picture.

    Remember:
    - Maintain objectivity throughout your analysis and evaluation.
    - Base all conclusions on concrete evidence from the input.
    - Avoid making assumptions beyond what is directly supported by the given information.

    After completing your analysis, provide a final output that summarizes your key findings and overall evaluation. This output should be concise yet comprehensive, reflecting the depth of your analysis while presenting the information in an easily digestible format.

    Please begin your analysis, followed by your final output summary.
    """
)

IMPROVEMENT_PROMPT = PromptTemplate(
    input_variables=["text", "feedback"],
    template="""
    You are an expert text enhancement AI. Your task is to analyze and improve a given text based on specific feedback points. Please follow the instructions carefully to provide a comprehensive analysis and enhancement.

    Here is the original text you need to improve:

    <original_text>
    {text}
    </original_text>

    Here are the key feedback points to consider:

    <key_feedback>
    {feedback}
    </key_feedback>

    Please follow these steps to enhance the text:

    1. Identify Strengths:
       - Carefully read the original text and identify 2-3 specific strengths that should be preserved in the enhanced version.
       - For each strength, quote a specific part of the text that demonstrates it.

    2. Analyze Feedback:
       - Review the key feedback points and categorize improvements needed in the following areas:
         a) Clarity and coherence
         b) Technical accuracy
         c) Engagement and style
         d) Any specific areas mentioned in the feedback
       - List out each feedback point under its corresponding category.

    3. Address Improvements:
       For each improvement area:
       - Describe the specific issue
       - Brainstorm three different ways to address it
       - Select and explain the most effective solution

    4. Create Enhanced Version:
       Develop an improved version of the text that:
       - Maintains the identified strengths
       - Incorporates the chosen improvements
       - Adds at least one unique element not mentioned in the feedback
       - Stays within ±20% of the original text length

    5. Provide Rationale:
       Explain the reasoning behind your changes and how they address the feedback while improving the overall quality of the text.

    Please wrap your thought process for each step in <enhancement_process> tags before providing the final output. This will help ensure a thorough and well-reasoned enhancement. It's OK for this section to be quite long.

    Present your final response in the following format:

    <preserved_strengths>
    [List 2-3 specific strengths of the original text]
    </preserved_strengths>

    <analysis_and_solutions>
    [For each improvement area:
     - Describe the issue
     - Propose two solutions
     - Explain the chosen solution]
    </analysis_and_solutions>

    <improved_version>
    [Present the enhanced text]
    </improved_version>

    <rationale_for_changes>
    [Explain the reasoning behind your changes and how they improve the text]
    </rationale_for_changes>

    Remember to maintain the essence of the original text while addressing the feedback and improving its overall quality.
"""
)
