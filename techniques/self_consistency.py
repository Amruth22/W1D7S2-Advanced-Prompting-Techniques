"""
Self-Consistency Prompts
Multiple sampling prompts for consistent and reliable answers
"""

# General Self-Consistency Prompt
GENERAL_CONSISTENCY = """Please answer this question carefully and accurately.

Question: {question}

Think through this step by step and provide your best answer. Be clear and specific in your response."""

# Mathematical Self-Consistency Prompt
MATH_CONSISTENCY = """Please solve this mathematical problem carefully.

Problem: {problem}

Show your work step by step and provide the final answer. Double-check your calculations to ensure accuracy."""

# Reasoning Self-Consistency Prompt
REASONING_CONSISTENCY = """Please analyze this problem using careful reasoning.

Problem: {problem}

Think through this logically, consider different aspects, and provide a well-reasoned conclusion."""

# Factual Self-Consistency Prompt
FACTUAL_CONSISTENCY = """Please provide accurate factual information about this topic.

Topic: {topic}

Give precise, factual information based on your knowledge. Be specific and avoid speculation."""

# Decision Self-Consistency Prompt
DECISION_CONSISTENCY = """Please help make a decision about this situation.

Situation: {situation}

Consider the options carefully, weigh the pros and cons, and provide a clear recommendation with reasoning."""

# Analysis Self-Consistency Prompt
ANALYSIS_CONSISTENCY = """Please provide a thorough analysis of this topic.

Topic: {topic}

Examine different aspects, consider various perspectives, and provide a comprehensive analysis."""

# Explanation Self-Consistency Prompt
EXPLANATION_CONSISTENCY = """Please explain this concept clearly.

Concept: {concept}

Provide a clear, accurate explanation that would help someone understand this concept well."""

# Comparison Self-Consistency Prompt
COMPARISON_CONSISTENCY = """Please compare these options objectively.

Options to compare: {options}

Analyze the similarities, differences, advantages, and disadvantages of each option."""

# Prediction Self-Consistency Prompt
PREDICTION_CONSISTENCY = """Please make a reasoned prediction about this scenario.

Scenario: {scenario}

Based on available information and logical reasoning, what do you predict will happen? Explain your reasoning."""

# Evaluation Self-Consistency Prompt
EVALUATION_CONSISTENCY = """Please evaluate this situation objectively.

Situation: {situation}

Assess the strengths, weaknesses, opportunities, and potential issues. Provide a balanced evaluation."""