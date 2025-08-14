"""
Chain-of-Thought (CoT) Implementation
Enables step-by-step reasoning for complex problem solving
"""

from typing import List, Dict, Any, Optional, Tuple
import re
import json
from dataclasses import dataclass
from enum import Enum
from utils.gemini_client import GeminiClient, TEMPLATES


class ReasoningType(Enum):
    """Types of reasoning approaches"""
    MATHEMATICAL = "mathematical"
    LOGICAL = "logical"
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    SCIENTIFIC = "scientific"


@dataclass
class ReasoningStep:
    """Represents a single reasoning step"""
    step_number: int
    description: str
    reasoning: str
    result: Optional[str] = None


class ChainOfThought:
    """Chain-of-Thought reasoning implementation"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Chain-of-Thought reasoner
        
        Args:
            api_key: Gemini API key
        """
        self.client = GeminiClient(api_key)
        self.reasoning_templates = self._load_reasoning_templates()
    
    def solve_math_problem(
        self, 
        problem: str,
        show_work: bool = True,
        verify_answer: bool = True
    ) -> Dict[str, Any]:
        """
        Solve mathematical problems with step-by-step reasoning
        
        Args:
            problem: Mathematical problem to solve
            show_work: Whether to show detailed work
            verify_answer: Whether to verify the final answer
            
        Returns:
            Solution with reasoning steps
        """
        prompt = self._build_math_prompt(problem, show_work)
        
        response = self.client.generate_response(
            prompt, 
            temperature=0.2,
            thinking_budget=10000
        )
        
        steps = self._parse_reasoning_steps(response)
        final_answer = self._extract_final_answer(response)
        
        result = {
            "problem": problem,
            "reasoning_steps": steps,
            "final_answer": final_answer,
            "full_response": response,
            "reasoning_type": ReasoningType.MATHEMATICAL.value
        }
        
        if verify_answer:
            verification = self._verify_math_answer(problem, final_answer, steps)
            result["verification"] = verification
        
        return result
    
    def solve_logical_puzzle(
        self, 
        puzzle: str,
        constraints: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Solve logical puzzles with systematic reasoning
        
        Args:
            puzzle: Logical puzzle description
            constraints: Additional constraints
            
        Returns:
            Solution with logical reasoning
        """
        prompt = self._build_logical_prompt(puzzle, constraints)
        
        response = self.client.generate_response(
            prompt, 
            temperature=0.3,
            thinking_budget=15000
        )
        
        steps = self._parse_reasoning_steps(response)
        solution = self._extract_solution(response)
        
        return {
            "puzzle": puzzle,
            "constraints": constraints or [],
            "reasoning_steps": steps,
            "solution": solution,
            "full_response": response,
            "reasoning_type": ReasoningType.LOGICAL.value
        }
    
    def analyze_problem(
        self, 
        problem: str,
        context: Optional[str] = None,
        reasoning_type: ReasoningType = ReasoningType.ANALYTICAL
    ) -> Dict[str, Any]:
        """
        Analyze complex problems with structured reasoning
        
        Args:
            problem: Problem to analyze
            context: Additional context
            reasoning_type: Type of reasoning to apply
            
        Returns:
            Analysis with reasoning steps
        """
        prompt = self._build_analysis_prompt(problem, context, reasoning_type)
        
        response = self.client.generate_response(
            prompt, 
            temperature=0.4,
            thinking_budget=12000
        )
        
        steps = self._parse_reasoning_steps(response)
        analysis = self._extract_analysis(response)
        
        return {
            "problem": problem,
            "context": context,
            "reasoning_steps": steps,
            "analysis": analysis,
            "full_response": response,
            "reasoning_type": reasoning_type.value
        }
    
    def creative_problem_solving(
        self, 
        challenge: str,
        constraints: Optional[List[str]] = None,
        num_solutions: int = 3
    ) -> Dict[str, Any]:
        """
        Generate creative solutions with reasoning
        
        Args:
            challenge: Creative challenge
            constraints: Solution constraints
            num_solutions: Number of solutions to generate
            
        Returns:
            Creative solutions with reasoning
        """
        prompt = self._build_creative_prompt(challenge, constraints, num_solutions)
        
        response = self.client.generate_response(
            prompt, 
            temperature=0.7,
            thinking_budget=10000
        )
        
        solutions = self._parse_creative_solutions(response)
        
        return {
            "challenge": challenge,
            "constraints": constraints or [],
            "solutions": solutions,
            "full_response": response,
            "reasoning_type": ReasoningType.CREATIVE.value
        }
    
    def scientific_reasoning(
        self, 
        hypothesis: str,
        observations: List[str],
        experiment_design: bool = False
    ) -> Dict[str, Any]:
        """
        Apply scientific reasoning to hypotheses
        
        Args:
            hypothesis: Scientific hypothesis
            observations: Relevant observations
            experiment_design: Whether to design experiments
            
        Returns:
            Scientific analysis with reasoning
        """
        prompt = self._build_scientific_prompt(hypothesis, observations, experiment_design)
        
        response = self.client.generate_response(
            prompt, 
            temperature=0.3,
            thinking_budget=15000
        )
        
        steps = self._parse_reasoning_steps(response)
        conclusion = self._extract_scientific_conclusion(response)
        
        result = {
            "hypothesis": hypothesis,
            "observations": observations,
            "reasoning_steps": steps,
            "conclusion": conclusion,
            "full_response": response,
            "reasoning_type": ReasoningType.SCIENTIFIC.value
        }
        
        if experiment_design:
            experiments = self._extract_experiments(response)
            result["proposed_experiments"] = experiments
        
        return result
    
    def multi_step_reasoning(
        self, 
        question: str,
        reasoning_depth: int = 5,
        verify_each_step: bool = False
    ) -> Dict[str, Any]:
        """
        Perform deep multi-step reasoning
        
        Args:
            question: Complex question requiring multiple steps
            reasoning_depth: Number of reasoning levels
            verify_each_step: Whether to verify each step
            
        Returns:
            Deep reasoning analysis
        """
        prompt = self._build_multi_step_prompt(question, reasoning_depth)
        
        response = self.client.generate_response(
            prompt, 
            temperature=0.3,
            thinking_budget=20000
        )
        
        steps = self._parse_reasoning_steps(response)
        final_conclusion = self._extract_final_answer(response)
        
        result = {
            "question": question,
            "reasoning_depth": reasoning_depth,
            "reasoning_steps": steps,
            "final_conclusion": final_conclusion,
            "full_response": response,
            "step_count": len(steps)
        }
        
        if verify_each_step:
            verifications = self._verify_reasoning_steps(steps)
            result["step_verifications"] = verifications
        
        return result
    
    def _build_math_prompt(self, problem: str, show_work: bool) -> str:
        """Build prompt for mathematical reasoning"""
        base_prompt = TEMPLATES["chain_of_thought"].format(problem=problem)
        
        if show_work:
            base_prompt += """

Please show your work clearly:
1. Identify what we know
2. Identify what we need to find
3. Choose the appropriate method/formula
4. Perform calculations step by step
5. Check your answer for reasonableness

Work through this systematically:"""
        
        return base_prompt
    
    def _build_logical_prompt(self, puzzle: str, constraints: Optional[List[str]]) -> str:
        """Build prompt for logical reasoning"""
        prompt = f"""Let's solve this logical puzzle step by step.

Puzzle: {puzzle}"""
        
        if constraints:
            prompt += f"\n\nConstraints:\n" + "\n".join(f"- {c}" for c in constraints)
        
        prompt += """

Step-by-step logical reasoning:
1. Analyze the given information
2. Identify key relationships and rules
3. Apply logical deduction
4. Test potential solutions
5. Verify the solution meets all constraints

Let me work through this systematically:"""
        
        return prompt
    
    def _build_analysis_prompt(
        self, 
        problem: str, 
        context: Optional[str], 
        reasoning_type: ReasoningType
    ) -> str:
        """Build prompt for analytical reasoning"""
        prompt = f"""Let's analyze this problem using {reasoning_type.value} reasoning.

Problem: {problem}"""
        
        if context:
            prompt += f"\n\nContext: {context}"
        
        prompt += f"""

{reasoning_type.value.title()} analysis approach:
1. Break down the problem into components
2. Examine each component systematically
3. Identify patterns, relationships, and dependencies
4. Consider multiple perspectives
5. Synthesize findings into conclusions

Step-by-step analysis:"""
        
        return prompt
    
    def _build_creative_prompt(
        self, 
        challenge: str, 
        constraints: Optional[List[str]], 
        num_solutions: int
    ) -> str:
        """Build prompt for creative problem solving"""
        prompt = f"""Let's approach this creative challenge systematically.

Challenge: {challenge}"""
        
        if constraints:
            prompt += f"\n\nConstraints:\n" + "\n".join(f"- {c}" for c in constraints)
        
        prompt += f"""

Creative problem-solving approach:
1. Understand the challenge deeply
2. Brainstorm without judgment
3. Apply different creative techniques
4. Evaluate ideas against constraints
5. Develop {num_solutions} distinct solutions

Let me generate creative solutions step by step:"""
        
        return prompt
    
    def _build_scientific_prompt(
        self, 
        hypothesis: str, 
        observations: List[str], 
        experiment_design: bool
    ) -> str:
        """Build prompt for scientific reasoning"""
        prompt = f"""Let's apply scientific reasoning to this hypothesis.

Hypothesis: {hypothesis}

Observations:"""
        
        for i, obs in enumerate(observations, 1):
            prompt += f"\n{i}. {obs}"
        
        prompt += """

Scientific reasoning approach:
1. Analyze the hypothesis for testability
2. Examine supporting and contradicting evidence
3. Consider alternative explanations
4. Evaluate the strength of evidence
5. Draw logical conclusions"""
        
        if experiment_design:
            prompt += "\n6. Design experiments to test the hypothesis"
        
        prompt += "\n\nStep-by-step scientific analysis:"
        
        return prompt
    
    def _build_multi_step_prompt(self, question: str, depth: int) -> str:
        """Build prompt for multi-step reasoning"""
        return f"""Let's approach this complex question with deep, multi-level reasoning.

Question: {question}

I'll use {depth} levels of reasoning depth, where each level builds upon the previous:

Level 1: Initial analysis and understanding
Level 2: Deeper examination of components
Level 3: Pattern recognition and relationships
Level 4: Synthesis and integration
Level 5: Final conclusions and implications

Let me work through each level systematically:"""
    
    def _parse_reasoning_steps(self, response: str) -> List[ReasoningStep]:
        """Parse reasoning steps from response"""
        steps = []
        lines = response.split('\n')
        current_step = None
        step_counter = 0
        
        for line in lines:
            line = line.strip()
            
            # Look for step indicators
            step_match = re.match(r'^(?:Step\s*)?(\d+)[:.]?\s*(.+)', line, re.IGNORECASE)
            if step_match:
                if current_step:
                    steps.append(current_step)
                
                step_counter += 1
                current_step = ReasoningStep(
                    step_number=step_counter,
                    description=step_match.group(2),
                    reasoning=""
                )
            elif current_step and line:
                current_step.reasoning += line + " "
        
        if current_step:
            steps.append(current_step)
        
        return steps
    
    def _extract_final_answer(self, response: str) -> str:
        """Extract final answer from response"""
        # Look for common answer patterns
        patterns = [
            r'(?:final answer|answer|result|solution):\s*(.+?)(?:\n|$)',
            r'(?:therefore|thus|so),?\s*(.+?)(?:\n|$)',
            r'(?:the answer is|equals?)\s*(.+?)(?:\n|$)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()
        
        # If no pattern found, return last meaningful line
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        return lines[-1] if lines else "No clear answer found"
    
    def _extract_solution(self, response: str) -> str:
        """Extract solution from logical puzzle response"""
        return self._extract_final_answer(response)
    
    def _extract_analysis(self, response: str) -> Dict[str, str]:
        """Extract analysis components from response"""
        analysis = {
            "summary": "",
            "key_findings": [],
            "recommendations": []
        }
        
        # Simple extraction - in practice, this could be more sophisticated
        lines = response.split('\n')
        current_section = "summary"
        
        for line in lines:
            line = line.strip()
            if "finding" in line.lower() or "conclusion" in line.lower():
                current_section = "key_findings"
            elif "recommend" in line.lower() or "suggest" in line.lower():
                current_section = "recommendations"
            elif line:
                if current_section == "summary":
                    analysis["summary"] += line + " "
                elif current_section == "key_findings":
                    analysis["key_findings"].append(line)
                elif current_section == "recommendations":
                    analysis["recommendations"].append(line)
        
        return analysis
    
    def _parse_creative_solutions(self, response: str) -> List[Dict[str, str]]:
        """Parse creative solutions from response"""
        solutions = []
        lines = response.split('\n')
        current_solution = None
        
        for line in lines:
            line = line.strip()
            
            # Look for solution indicators
            solution_match = re.match(r'^(?:solution\s*)?(\d+)[:.]?\s*(.+)', line, re.IGNORECASE)
            if solution_match:
                if current_solution:
                    solutions.append(current_solution)
                
                current_solution = {
                    "title": solution_match.group(2),
                    "description": "",
                    "reasoning": ""
                }
            elif current_solution and line:
                if "reasoning" in line.lower() or "because" in line.lower():
                    current_solution["reasoning"] += line + " "
                else:
                    current_solution["description"] += line + " "
        
        if current_solution:
            solutions.append(current_solution)
        
        return solutions
    
    def _extract_scientific_conclusion(self, response: str) -> Dict[str, str]:
        """Extract scientific conclusion from response"""
        conclusion = {
            "verdict": "",
            "confidence": "",
            "evidence_strength": "",
            "alternative_hypotheses": []
        }
        
        # Simple extraction logic
        lines = response.split('\n')
        for line in lines:
            line = line.strip().lower()
            if "conclusion" in line or "verdict" in line:
                conclusion["verdict"] = line
            elif "confidence" in line:
                conclusion["confidence"] = line
            elif "evidence" in line:
                conclusion["evidence_strength"] = line
            elif "alternative" in line:
                conclusion["alternative_hypotheses"].append(line)
        
        return conclusion
    
    def _extract_experiments(self, response: str) -> List[Dict[str, str]]:
        """Extract proposed experiments from response"""
        experiments = []
        lines = response.split('\n')
        
        for line in lines:
            if "experiment" in line.lower():
                experiments.append({
                    "description": line.strip(),
                    "purpose": "Test hypothesis",
                    "expected_outcome": "TBD"
                })
        
        return experiments
    
    def _verify_math_answer(
        self, 
        problem: str, 
        answer: str, 
        steps: List[ReasoningStep]
    ) -> Dict[str, Any]:
        """Verify mathematical answer"""
        verification_prompt = f"""Please verify this mathematical solution:

Problem: {problem}
Proposed Answer: {answer}

Steps taken:
{chr(10).join(f"{i+1}. {step.description}: {step.reasoning}" for i, step in enumerate(steps))}

Is this solution correct? Please check:
1. Are the steps logically sound?
2. Are the calculations accurate?
3. Does the final answer make sense?

Verification result:"""
        
        verification_response = self.client.generate_response(
            verification_prompt, 
            temperature=0.1
        )
        
        return {
            "is_correct": "correct" in verification_response.lower(),
            "verification_details": verification_response,
            "confidence": self._extract_confidence(verification_response)
        }
    
    def _verify_reasoning_steps(self, steps: List[ReasoningStep]) -> List[Dict[str, Any]]:
        """Verify individual reasoning steps"""
        verifications = []
        
        for step in steps:
            verification_prompt = f"""Please verify this reasoning step:

Step {step.step_number}: {step.description}
Reasoning: {step.reasoning}

Is this step logically sound and well-reasoned? Please provide:
1. Validity assessment
2. Potential issues or improvements
3. Confidence level

Verification:"""
            
            verification_response = self.client.generate_response(
                verification_prompt, 
                temperature=0.2
            )
            
            verifications.append({
                "step_number": step.step_number,
                "is_valid": "valid" in verification_response.lower(),
                "verification_details": verification_response,
                "confidence": self._extract_confidence(verification_response)
            })
        
        return verifications
    
    def _extract_confidence(self, response: str) -> float:
        """Extract confidence level from response"""
        confidence_patterns = [
            r'confidence[:\s]*(\d+)%',
            r'(\d+)%\s*confident',
            r'confidence[:\s]*(\d+\.\d+)'
        ]
        
        for pattern in confidence_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                return value / 100 if value > 1 else value
        
        # Default confidence based on response characteristics
        if "certain" in response.lower() or "definitely" in response.lower():
            return 0.9
        elif "likely" in response.lower() or "probably" in response.lower():
            return 0.7
        elif "possible" in response.lower() or "might" in response.lower():
            return 0.5
        else:
            return 0.6
    
    def _load_reasoning_templates(self) -> Dict[str, str]:
        """Load reasoning templates for different problem types"""
        return {
            "mathematical": "Let's solve this step by step using mathematical reasoning.",
            "logical": "Let's approach this with systematic logical analysis.",
            "analytical": "Let's break this down analytically.",
            "creative": "Let's explore creative solutions systematically.",
            "scientific": "Let's apply the scientific method to this problem."
        }