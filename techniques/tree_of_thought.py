"""
Tree-of-Thought (ToT) Prompts
Multiple reasoning path exploration prompts
"""

# Complex Problem Solving ToT Prompt
COMPLEX_PROBLEM_SOLVING = """I need to solve this complex problem by exploring multiple reasoning paths.

Problem: {problem}

Let me explore different approaches:

Approach 1: {approach1_description}
- Step 1: {step1}
- Step 2: {step2}
- Step 3: {step3}
- Evaluation: {evaluation1}

Approach 2: {approach2_description}
- Step 1: {step1}
- Step 2: {step2}
- Step 3: {step3}
- Evaluation: {evaluation2}

Approach 3: {approach3_description}
- Step 1: {step1}
- Step 2: {step2}
- Step 3: {step3}
- Evaluation: {evaluation3}

Now let me compare these approaches and select the best path:"""

# Creative Brainstorming ToT Prompt
CREATIVE_BRAINSTORMING = """I need to generate creative solutions by exploring multiple idea paths.

Challenge: {challenge}

Let me explore different creative directions:

Creative Direction 1: {direction1}
- Initial idea: {idea1}
- Development: {development1}
- Potential: {potential1}

Creative Direction 2: {direction2}
- Initial idea: {idea2}
- Development: {development2}
- Potential: {potential2}

Creative Direction 3: {direction3}
- Initial idea: {idea3}
- Development: {development3}
- Potential: {potential3}

Now let me evaluate and refine the most promising ideas:"""

# Strategic Decision Making ToT Prompt
STRATEGIC_DECISION_MAKING = """I need to make a strategic decision by exploring multiple strategic paths.

Decision: {decision}

Let me explore different strategic options:

Strategy A: {strategy_a}
- Implementation approach: {implementation_a}
- Expected outcomes: {outcomes_a}
- Risks and challenges: {risks_a}
- Success probability: {probability_a}

Strategy B: {strategy_b}
- Implementation approach: {implementation_b}
- Expected outcomes: {outcomes_b}
- Risks and challenges: {risks_b}
- Success probability: {probability_b}

Strategy C: {strategy_c}
- Implementation approach: {implementation_c}
- Expected outcomes: {outcomes_c}
- Risks and challenges: {risks_c}
- Success probability: {probability_c}

Now let me compare these strategies and recommend the best approach:"""

# Multi-Perspective Analysis ToT Prompt
MULTI_PERSPECTIVE_ANALYSIS = """I need to analyze this topic from multiple perspectives.

Topic: {topic}

Let me explore different viewpoints:

Perspective 1: {perspective1}
- Key arguments: {arguments1}
- Supporting evidence: {evidence1}
- Implications: {implications1}

Perspective 2: {perspective2}
- Key arguments: {arguments2}
- Supporting evidence: {evidence2}
- Implications: {implications2}

Perspective 3: {perspective3}
- Key arguments: {arguments3}
- Supporting evidence: {evidence3}
- Implications: {implications3}

Now let me synthesize these perspectives and identify common ground:"""

# Research Hypothesis Testing ToT Prompt
RESEARCH_HYPOTHESIS_TESTING = """I need to test this hypothesis by exploring multiple research paths.

Hypothesis: {hypothesis}

Let me explore different research approaches:

Research Path 1: {path1}
- Methodology: {methodology1}
- Expected findings: {findings1}
- Limitations: {limitations1}

Research Path 2: {path2}
- Methodology: {methodology2}
- Expected findings: {findings2}
- Limitations: {limitations2}

Research Path 3: {path3}
- Methodology: {methodology3}
- Expected findings: {findings3}
- Limitations: {limitations3}

Now let me evaluate which research approach would be most effective:"""

# Solution Optimization ToT Prompt
SOLUTION_OPTIMIZATION = """I need to optimize this solution by exploring multiple improvement paths.

Current Solution: {current_solution}

Let me explore different optimization approaches:

Optimization Path 1: {optimization1}
- Improvements: {improvements1}
- Benefits: {benefits1}
- Trade-offs: {tradeoffs1}

Optimization Path 2: {optimization2}
- Improvements: {improvements2}
- Benefits: {benefits2}
- Trade-offs: {tradeoffs2}

Optimization Path 3: {optimization3}
- Improvements: {improvements3}
- Benefits: {benefits3}
- Trade-offs: {tradeoffs3}

Now let me determine the best optimization strategy:"""

# Scenario Planning ToT Prompt
SCENARIO_PLANNING = """I need to plan for different scenarios by exploring multiple future paths.

Situation: {situation}

Let me explore different scenarios:

Scenario 1: {scenario1}
- Key assumptions: {assumptions1}
- Likely developments: {developments1}
- Required responses: {responses1}
- Preparation needed: {preparation1}

Scenario 2: {scenario2}
- Key assumptions: {assumptions2}
- Likely developments: {developments2}
- Required responses: {responses2}
- Preparation needed: {preparation2}

Scenario 3: {scenario3}
- Key assumptions: {assumptions3}
- Likely developments: {developments3}
- Required responses: {responses3}
- Preparation needed: {preparation3}

Now let me develop a comprehensive plan that addresses all scenarios:"""

# Innovation Development ToT Prompt
INNOVATION_DEVELOPMENT = """I need to develop an innovation by exploring multiple development paths.

Innovation Goal: {goal}

Let me explore different development approaches:

Development Path 1: {path1}
- Core concept: {concept1}
- Technical approach: {technical1}
- Market potential: {market1}
- Feasibility: {feasibility1}

Development Path 2: {path2}
- Core concept: {concept2}
- Technical approach: {technical2}
- Market potential: {market2}
- Feasibility: {feasibility2}

Development Path 3: {path3}
- Core concept: {concept3}
- Technical approach: {technical3}
- Market potential: {market3}
- Feasibility: {feasibility3}

Now let me select and refine the most promising innovation path:"""

# Risk Assessment ToT Prompt
RISK_ASSESSMENT = """I need to assess risks by exploring multiple risk scenarios.

Situation: {situation}

Let me explore different risk paths:

Risk Scenario 1: {risk1}
- Probability: {probability1}
- Impact: {impact1}
- Mitigation strategies: {mitigation1}
- Contingency plans: {contingency1}

Risk Scenario 2: {risk2}
- Probability: {probability2}
- Impact: {impact2}
- Mitigation strategies: {mitigation2}
- Contingency plans: {contingency2}

Risk Scenario 3: {risk3}
- Probability: {probability3}
- Impact: {impact3}
- Mitigation strategies: {mitigation3}
- Contingency plans: {contingency3}

Now let me develop a comprehensive risk management strategy:"""

# Learning Path Exploration ToT Prompt
LEARNING_PATH_EXPLORATION = """I need to explore different learning paths for this topic.

Learning Goal: {goal}

Let me explore different learning approaches:

Learning Path 1: {path1}
- Learning method: {method1}
- Resources needed: {resources1}
- Timeline: {timeline1}
- Expected outcomes: {outcomes1}

Learning Path 2: {path2}
- Learning method: {method2}
- Resources needed: {resources2}
- Timeline: {timeline2}
- Expected outcomes: {outcomes2}

Learning Path 3: {path3}
- Learning method: {method3}
- Resources needed: {resources3}
- Timeline: {timeline3}
- Expected outcomes: {outcomes3}

Now let me design the optimal learning strategy:"""