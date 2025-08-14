"""
Advanced Prompting Techniques for Gemini 2.5 Flash
Main implementation file with all technique orchestration
"""

import asyncio
import statistics
import os
import argparse
import time
from typing import List, Dict, Any, Optional
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import all prompts
from techniques import few_shot
from techniques import chain_of_thought
from techniques import tree_of_thought
from techniques import self_consistency
from techniques import meta_prompting


class AdvancedPromptingGemini:
    """Main class for advanced prompting techniques with Gemini 2.5 Flash"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Advanced Prompting system
        
        Args:
            api_key: Gemini API key (if None, uses GEMINI_API_KEY from .env)
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file or pass it as a parameter.")
        
        self.client = genai.Client(api_key=self.api_key)
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    
    def generate_response(
        self, 
        prompt: str, 
        temperature: float = 0.7,
        thinking_budget: int = 0,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate a single response from Gemini
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            thinking_budget: Thinking budget for reasoning
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response text
        """
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)]
            )
        ]
        
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
            temperature=temperature
        )
        
        if max_tokens:
            config.max_output_tokens = max_tokens
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config
        )
        
        return response.candidates[0].content.parts[0].text
    
    async def generate_multiple_responses(
        self, 
        prompt: str, 
        num_responses: int = 3,
        temperature: float = 0.8,
        thinking_budget: int = 0
    ) -> List[str]:
        """
        Generate multiple responses asynchronously
        
        Args:
            prompt: Input prompt
            num_responses: Number of responses to generate
            temperature: Sampling temperature
            thinking_budget: Thinking budget for reasoning
            
        Returns:
            List of generated responses
        """
        tasks = []
        for _ in range(num_responses):
            task = asyncio.create_task(
                self._async_generate(prompt, temperature, thinking_budget)
            )
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        return responses
    
    async def _async_generate(self, prompt: str, temperature: float, thinking_budget: int) -> str:
        """Async wrapper for generate_response"""
        return self.generate_response(prompt, temperature, thinking_budget)
    
    # ==================== FEW-SHOT LEARNING ====================
    
    def few_shot_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """Perform sentiment analysis using few-shot learning"""
        prompt = few_shot.SENTIMENT_CLASSIFICATION.format(text=text)
        response = self.generate_response(prompt, temperature=0.3)
        
        return {
            "text": text,
            "sentiment": response.strip(),
            "technique": "Few-shot Learning",
            "prompt_used": "SENTIMENT_CLASSIFICATION"
        }
    
    def few_shot_named_entity_recognition(self, text: str) -> Dict[str, Any]:
        """Extract named entities using few-shot learning"""
        prompt = few_shot.NAMED_ENTITY_RECOGNITION.format(text=text)
        response = self.generate_response(prompt, temperature=0.2)
        
        return {
            "text": text,
            "entities": response.strip(),
            "technique": "Few-shot Learning",
            "prompt_used": "NAMED_ENTITY_RECOGNITION"
        }
    
    def few_shot_text_classification(self, text: str) -> Dict[str, Any]:
        """Classify text using few-shot learning"""
        prompt = few_shot.TEXT_CLASSIFICATION.format(text=text)
        response = self.generate_response(prompt, temperature=0.3)
        
        return {
            "text": text,
            "classification": response.strip(),
            "technique": "Few-shot Learning",
            "prompt_used": "TEXT_CLASSIFICATION"
        }
    
    def few_shot_math_solver(self, problem: str) -> Dict[str, Any]:
        """Solve math problems using few-shot learning"""
        prompt = few_shot.MATH_WORD_PROBLEMS.format(problem=problem)
        response = self.generate_response(prompt, temperature=0.2)
        
        return {
            "problem": problem,
            "solution": response.strip(),
            "technique": "Few-shot Learning",
            "prompt_used": "MATH_WORD_PROBLEMS"
        }
    
    def few_shot_translation(self, text: str, target_language: str) -> Dict[str, Any]:
        """Translate text using few-shot learning"""
        prompt = few_shot.LANGUAGE_TRANSLATION.format(text=text, target_language=target_language)
        response = self.generate_response(prompt, temperature=0.3)
        
        return {
            "original_text": text,
            "target_language": target_language,
            "translation": response.strip(),
            "technique": "Few-shot Learning",
            "prompt_used": "LANGUAGE_TRANSLATION"
        }
    
    def few_shot_code_generation(self, task: str) -> Dict[str, Any]:
        """Generate code using few-shot learning"""
        prompt = few_shot.CODE_GENERATION.format(task=task)
        response = self.generate_response(prompt, temperature=0.4)
        
        return {
            "task": task,
            "generated_code": response.strip(),
            "technique": "Few-shot Learning",
            "prompt_used": "CODE_GENERATION"
        }
    
    # ==================== CHAIN-OF-THOUGHT ====================
    
    def chain_of_thought_math_solver(self, problem: str) -> Dict[str, Any]:
        """Solve math problems using chain-of-thought reasoning"""
        prompt = chain_of_thought.MATH_PROBLEM_SOLVING.format(problem=problem)
        response = self.generate_response(prompt, temperature=0.3, thinking_budget=10000)
        
        return {
            "problem": problem,
            "step_by_step_solution": response.strip(),
            "technique": "Chain-of-Thought",
            "prompt_used": "MATH_PROBLEM_SOLVING"
        }
    
    def chain_of_thought_logical_reasoning(self, problem: str) -> Dict[str, Any]:
        """Solve logical problems using chain-of-thought reasoning"""
        prompt = chain_of_thought.LOGICAL_REASONING.format(problem=problem)
        response = self.generate_response(prompt, temperature=0.3, thinking_budget=12000)
        
        return {
            "problem": problem,
            "logical_reasoning": response.strip(),
            "technique": "Chain-of-Thought",
            "prompt_used": "LOGICAL_REASONING"
        }
    
    def chain_of_thought_complex_analysis(self, problem: str) -> Dict[str, Any]:
        """Analyze complex problems using chain-of-thought reasoning"""
        prompt = chain_of_thought.COMPLEX_ANALYSIS.format(problem=problem)
        response = self.generate_response(prompt, temperature=0.4, thinking_budget=15000)
        
        return {
            "problem": problem,
            "detailed_analysis": response.strip(),
            "technique": "Chain-of-Thought",
            "prompt_used": "COMPLEX_ANALYSIS"
        }
    
    def chain_of_thought_decision_making(self, decision: str) -> Dict[str, Any]:
        """Make decisions using chain-of-thought reasoning"""
        prompt = chain_of_thought.DECISION_MAKING.format(decision=decision)
        response = self.generate_response(prompt, temperature=0.4, thinking_budget=10000)
        
        return {
            "decision": decision,
            "reasoning_process": response.strip(),
            "technique": "Chain-of-Thought",
            "prompt_used": "DECISION_MAKING"
        }
    
    def chain_of_thought_problem_solving(self, problem: str) -> Dict[str, Any]:
        """Solve general problems using chain-of-thought reasoning"""
        prompt = chain_of_thought.PROBLEM_SOLVING.format(problem=problem)
        response = self.generate_response(prompt, temperature=0.4, thinking_budget=12000)
        
        return {
            "problem": problem,
            "solution_process": response.strip(),
            "technique": "Chain-of-Thought",
            "prompt_used": "PROBLEM_SOLVING"
        }
    
    # ==================== TREE-OF-THOUGHT ====================
    
    async def tree_of_thought_complex_problem(self, problem: str) -> Dict[str, Any]:
        """Solve complex problems using tree-of-thought exploration"""
        # Generate multiple approaches
        approaches = [
            "Direct analytical approach",
            "Creative problem-solving approach", 
            "Systematic breakdown approach"
        ]
        
        results = []
        for i, approach in enumerate(approaches, 1):
            prompt = f"""Problem: {problem}

I'll use approach {i}: {approach}

Let me work through this step by step:
1. First, I'll analyze the problem from this perspective
2. Then I'll develop a solution strategy
3. Finally, I'll evaluate the effectiveness

Working through approach {i}:"""
            
            response = await self._async_generate(prompt, temperature=0.6, thinking_budget=8000)
            results.append({
                "approach": approach,
                "solution": response.strip(),
                "approach_number": i
            })
        
        # Select best approach
        best_approach = await self._select_best_approach(problem, results)
        
        return {
            "problem": problem,
            "explored_approaches": results,
            "best_approach": best_approach,
            "technique": "Tree-of-Thought",
            "methodology": "Multiple path exploration"
        }
    
    async def tree_of_thought_creative_brainstorming(self, challenge: str) -> Dict[str, Any]:
        """Generate creative solutions using tree-of-thought"""
        creative_directions = [
            "Innovative technology solution",
            "Human-centered design approach",
            "Sustainable and eco-friendly solution"
        ]
        
        ideas = []
        for direction in creative_directions:
            prompt = f"""Creative Challenge: {challenge}

Exploring creative direction: {direction}

Let me brainstorm innovative ideas:
1. Initial concept development
2. Creative enhancement and refinement  
3. Practical implementation considerations

Creative exploration:"""
            
            response = await self._async_generate(prompt, temperature=0.8, thinking_budget=6000)
            ideas.append({
                "direction": direction,
                "creative_ideas": response.strip()
            })
        
        return {
            "challenge": challenge,
            "creative_directions": ideas,
            "technique": "Tree-of-Thought",
            "methodology": "Creative exploration"
        }
    
    async def tree_of_thought_strategic_planning(self, goal: str) -> Dict[str, Any]:
        """Develop strategic plans using tree-of-thought"""
        strategies = [
            "Aggressive growth strategy",
            "Conservative steady approach",
            "Innovation-focused strategy"
        ]
        
        strategic_plans = []
        for strategy in strategies:
            prompt = f"""Strategic Goal: {goal}

Developing strategy: {strategy}

Strategic planning process:
1. Situation analysis and current state
2. Strategic approach and key initiatives
3. Implementation roadmap and milestones
4. Risk assessment and mitigation

Strategic development:"""
            
            response = await self._async_generate(prompt, temperature=0.5, thinking_budget=10000)
            strategic_plans.append({
                "strategy_type": strategy,
                "strategic_plan": response.strip()
            })
        
        return {
            "goal": goal,
            "strategic_options": strategic_plans,
            "technique": "Tree-of-Thought",
            "methodology": "Strategic exploration"
        }
    
    async def _select_best_approach(self, problem: str, approaches: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select the best approach from multiple options"""
        evaluation_prompt = f"""Problem: {problem}

I have explored these different approaches:

{chr(10).join([f"Approach {a['approach_number']}: {a['approach']}" + chr(10) + f"Solution: {a['solution'][:200]}..." for a in approaches])}

Please evaluate these approaches and select the best one based on:
1. Effectiveness in solving the problem
2. Feasibility of implementation
3. Completeness of the solution
4. Innovation and creativity

Best approach selection:"""
        
        evaluation = await self._async_generate(evaluation_prompt, temperature=0.3, thinking_budget=5000)
        
        return {
            "evaluation": evaluation.strip(),
            "selection_criteria": ["effectiveness", "feasibility", "completeness", "innovation"]
        }
    
    # ==================== SELF-CONSISTENCY ====================
    
    async def self_consistency_answer(self, question: str, num_samples: int = 5) -> Dict[str, Any]:
        """Get consistent answers using multiple sampling"""
        prompt = self_consistency.GENERAL_CONSISTENCY.format(question=question)
        
        # Generate multiple responses
        responses = await self.generate_multiple_responses(
            prompt, num_samples, temperature=0.7, thinking_budget=5000
        )
        
        # Analyze consistency
        consistency_analysis = await self._analyze_consistency(question, responses)
        
        return {
            "question": question,
            "all_responses": responses,
            "consistency_analysis": consistency_analysis,
            "final_answer": consistency_analysis["most_consistent_answer"],
            "technique": "Self-Consistency",
            "num_samples": num_samples
        }
    
    async def self_consistency_math_solver(self, problem: str, num_samples: int = 5) -> Dict[str, Any]:
        """Solve math problems with self-consistency"""
        prompt = self_consistency.MATH_CONSISTENCY.format(problem=problem)
        
        responses = await self.generate_multiple_responses(
            prompt, num_samples, temperature=0.4, thinking_budget=8000
        )
        
        consistency_analysis = await self._analyze_math_consistency(problem, responses)
        
        return {
            "problem": problem,
            "all_solutions": responses,
            "consistency_analysis": consistency_analysis,
            "final_answer": consistency_analysis["most_consistent_answer"],
            "technique": "Self-Consistency",
            "num_samples": num_samples
        }
    
    async def self_consistency_reasoning(self, problem: str, num_samples: int = 4) -> Dict[str, Any]:
        """Perform reasoning with self-consistency"""
        prompt = self_consistency.REASONING_CONSISTENCY.format(problem=problem)
        
        responses = await self.generate_multiple_responses(
            prompt, num_samples, temperature=0.6, thinking_budget=10000
        )
        
        consistency_analysis = await self._analyze_reasoning_consistency(problem, responses)
        
        return {
            "problem": problem,
            "all_reasoning": responses,
            "consistency_analysis": consistency_analysis,
            "final_conclusion": consistency_analysis["most_consistent_answer"],
            "technique": "Self-Consistency",
            "num_samples": num_samples
        }
    
    async def _analyze_consistency(self, question: str, responses: List[str]) -> Dict[str, Any]:
        """Analyze consistency across multiple responses"""
        analysis_prompt = f"""Question: {question}

I have these {len(responses)} different responses:

{chr(10).join([f"Response {i+1}: {resp}" for i, resp in enumerate(responses)])}

Please analyze these responses for consistency:
1. What are the common themes or answers?
2. What are the main differences?
3. Which response seems most accurate and complete?
4. What is the most consistent answer across all responses?

Consistency analysis:"""
        
        analysis = await self._async_generate(analysis_prompt, temperature=0.2, thinking_budget=5000)
        
        return {
            "analysis": analysis.strip(),
            "response_count": len(responses),
            "most_consistent_answer": self._extract_most_consistent_answer(analysis)
        }
    
    async def _analyze_math_consistency(self, problem: str, responses: List[str]) -> Dict[str, Any]:
        """Analyze consistency for math problems"""
        analysis_prompt = f"""Math Problem: {problem}

I have these {len(responses)} different solutions:

{chr(10).join([f"Solution {i+1}: {resp}" for i, resp in enumerate(responses)])}

Please analyze these solutions for consistency:
1. Do they arrive at the same final answer?
2. Are the mathematical steps correct?
3. Which solution is most accurate?
4. What is the most reliable answer?

Mathematical consistency analysis:"""
        
        analysis = await self._async_generate(analysis_prompt, temperature=0.1, thinking_budget=5000)
        
        return {
            "analysis": analysis.strip(),
            "solution_count": len(responses),
            "most_consistent_answer": self._extract_most_consistent_answer(analysis)
        }
    
    async def _analyze_reasoning_consistency(self, problem: str, responses: List[str]) -> Dict[str, Any]:
        """Analyze consistency for reasoning problems"""
        analysis_prompt = f"""Reasoning Problem: {problem}

I have these {len(responses)} different reasoning approaches:

{chr(10).join([f"Reasoning {i+1}: {resp}" for i, resp in enumerate(responses)])}

Please analyze these reasoning approaches for consistency:
1. What logical conclusions are consistent across responses?
2. Which reasoning is most sound and complete?
3. Are there any contradictions to resolve?
4. What is the most reliable conclusion?

Reasoning consistency analysis:"""
        
        analysis = await self._async_generate(analysis_prompt, temperature=0.2, thinking_budget=8000)
        
        return {
            "analysis": analysis.strip(),
            "reasoning_count": len(responses),
            "most_consistent_answer": self._extract_most_consistent_answer(analysis)
        }
    
    def _extract_most_consistent_answer(self, analysis: str) -> str:
        """Extract the most consistent answer from analysis"""
        lines = analysis.split('\n')
        for line in lines:
            if "most consistent" in line.lower() or "most reliable" in line.lower():
                return line.strip()
        
        # If no specific line found, return last meaningful line
        meaningful_lines = [line.strip() for line in lines if line.strip() and len(line.strip()) > 10]
        return meaningful_lines[-1] if meaningful_lines else "Analysis inconclusive"
    
    # ==================== META-PROMPTING ====================
    
    def meta_prompt_optimization(self, task: str, current_prompt: str) -> Dict[str, Any]:
        """Optimize prompts using meta-prompting"""
        prompt = meta_prompting.PROMPT_OPTIMIZATION.format(
            task=task, 
            current_prompt=current_prompt
        )
        response = self.generate_response(prompt, temperature=0.4, thinking_budget=8000)
        
        return {
            "original_task": task,
            "original_prompt": current_prompt,
            "optimized_prompt": response.strip(),
            "technique": "Meta-Prompting",
            "optimization_type": "Prompt Optimization"
        }
    
    def meta_task_analysis(self, task: str) -> Dict[str, Any]:
        """Analyze tasks for better prompting"""
        prompt = meta_prompting.TASK_ANALYSIS.format(task=task)
        response = self.generate_response(prompt, temperature=0.3, thinking_budget=6000)
        
        return {
            "task": task,
            "task_analysis": response.strip(),
            "technique": "Meta-Prompting",
            "analysis_type": "Task Analysis"
        }
    
    def meta_prompt_generation(self, task: str, audience: str, output_type: str, context: str) -> Dict[str, Any]:
        """Generate optimal prompts using meta-prompting"""
        prompt = meta_prompting.PROMPT_GENERATION.format(
            task=task,
            audience=audience,
            output_type=output_type,
            context=context
        )
        response = self.generate_response(prompt, temperature=0.4, thinking_budget=7000)
        
        return {
            "task": task,
            "audience": audience,
            "output_type": output_type,
            "context": context,
            "generated_prompt": response.strip(),
            "technique": "Meta-Prompting",
            "generation_type": "Prompt Generation"
        }
    
    def meta_prompt_evaluation(self, task: str, prompt_to_evaluate: str) -> Dict[str, Any]:
        """Evaluate prompts using meta-prompting"""
        prompt = meta_prompting.PROMPT_EVALUATION.format(
            task=task,
            prompt=prompt_to_evaluate
        )
        response = self.generate_response(prompt, temperature=0.2, thinking_budget=5000)
        
        return {
            "task": task,
            "evaluated_prompt": prompt_to_evaluate,
            "evaluation_results": response.strip(),
            "technique": "Meta-Prompting",
            "evaluation_type": "Prompt Evaluation"
        }


# ==================== INDIVIDUAL TECHNIQUE TESTERS ====================

def test_few_shot_learning(gemini: AdvancedPromptingGemini):
    """Test Few-shot Learning techniques"""
    print("🎯 TESTING FEW-SHOT LEARNING")
    print("=" * 50)
    
    # Test 1: Sentiment Analysis
    print("\n1. Sentiment Analysis:")
    result = gemini.few_shot_sentiment_analysis("This new smartphone is absolutely incredible! Best purchase ever!")
    print(f"Text: {result['text']}")
    print(f"Sentiment: {result['sentiment']}")
    
    print("\n" + "✅ Few-shot Learning test completed!")
    print("💡 Try other examples: --technique few-shot")


def test_chain_of_thought(gemini: AdvancedPromptingGemini):
    """Test Chain-of-Thought reasoning"""
    print("🔗 TESTING CHAIN-OF-THOUGHT")
    print("=" * 50)
    
    # Test 1: Math Problem
    print("\n1. Math Problem Solving:")
    result = gemini.chain_of_thought_math_solver("A pizza costs $12 and is cut into 8 slices. If I eat 3 slices, what's the cost of the pizza I ate?")
    print(f"Problem: {result['problem']}")
    print(f"Solution:\n{result['step_by_step_solution']}")
    
    print("\n" + "✅ Chain-of-Thought test completed!")
    print("💡 Try other examples: --technique chain-of-thought")


async def test_tree_of_thought(gemini: AdvancedPromptingGemini):
    """Test Tree-of-Thought exploration"""
    print("🌳 TESTING TREE-OF-THOUGHT")
    print("=" * 50)
    
    # Test 1: Complex Problem
    print("\n1. Complex Problem Solving:")
    result = await gemini.tree_of_thought_complex_problem("How can we reduce food waste in restaurants?")
    print(f"Problem: {result['problem']}")
    print(f"Approaches explored: {len(result['explored_approaches'])}")
    for i, approach in enumerate(result['explored_approaches'], 1):
        print(f"\nApproach {i}: {approach['approach']}")
        print(f"Solution preview: {approach['solution'][:150]}...")
    
    print("\n" + "✅ Tree-of-Thought test completed!")
    print("💡 Try other examples: --technique tree-of-thought")


async def test_self_consistency(gemini: AdvancedPromptingGemini):
    """Test Self-Consistency technique"""
    print("🎯 TESTING SELF-CONSISTENCY")
    print("=" * 50)
    
    # Test 1: General Question (reduced samples for rate limit)
    print("\n1. General Question with Multiple Samples:")
    result = await gemini.self_consistency_answer("What are the key benefits of exercise?", num_samples=3)
    print(f"Question: {result['question']}")
    print(f"Number of samples: {result['num_samples']}")
    print(f"Final Answer: {result['final_answer']}")
    
    print("\n" + "✅ Self-Consistency test completed!")
    print("💡 Try other examples: --technique self-consistency")


def test_meta_prompting(gemini: AdvancedPromptingGemini):
    """Test Meta-Prompting techniques"""
    print("🧠 TESTING META-PROMPTING")
    print("=" * 50)
    
    # Test 1: Prompt Optimization
    print("\n1. Prompt Optimization:")
    result = gemini.meta_prompt_optimization(
        task="Analyze customer feedback",
        current_prompt="Tell me if this feedback is good or bad: {feedback}"
    )
    print(f"Original Task: {result['original_task']}")
    print(f"Original Prompt: {result['original_prompt']}")
    print(f"Optimized Prompt:\n{result['optimized_prompt']}")
    
    print("\n" + "✅ Meta-Prompting test completed!")
    print("💡 Try other examples: --technique meta-prompting")


def print_available_techniques():
    """Print all available techniques"""
    print("🚀 AVAILABLE TECHNIQUES:")
    print("=" * 50)
    print("1. few-shot        - Few-shot Learning examples")
    print("2. chain-of-thought - Step-by-step reasoning")
    print("3. tree-of-thought  - Multiple path exploration")
    print("4. self-consistency - Multiple sampling validation")
    print("5. meta-prompting   - Self-improving prompts")
    print("\nUsage: python main.py --technique <technique_name>")
    print("Example: python main.py --technique few-shot")


def print_technique_examples(technique: str):
    """Print examples for a specific technique"""
    examples = {
        "few-shot": [
            "python main.py --technique few-shot --example sentiment",
            "python main.py --technique few-shot --example math",
            "python main.py --technique few-shot --example translation",
            "python main.py --technique few-shot --example code"
        ],
        "chain-of-thought": [
            "python main.py --technique chain-of-thought --example math",
            "python main.py --technique chain-of-thought --example logic",
            "python main.py --technique chain-of-thought --example decision"
        ],
        "tree-of-thought": [
            "python main.py --technique tree-of-thought --example problem",
            "python main.py --technique tree-of-thought --example creative",
            "python main.py --technique tree-of-thought --example strategy"
        ],
        "self-consistency": [
            "python main.py --technique self-consistency --example general",
            "python main.py --technique self-consistency --example math",
            "python main.py --technique self-consistency --example reasoning"
        ],
        "meta-prompting": [
            "python main.py --technique meta-prompting --example optimize",
            "python main.py --technique meta-prompting --example analyze",
            "python main.py --technique meta-prompting --example generate"
        ]
    }
    
    if technique in examples:
        print(f"\n💡 EXAMPLES FOR {technique.upper()}:")
        print("-" * 40)
        for example in examples[technique]:
            print(f"  {example}")
    else:
        print(f"❌ No examples found for technique: {technique}")


async def run_specific_technique(technique: str, example: str = None):
    """Run a specific technique test"""
    try:
        gemini = AdvancedPromptingGemini()
        
        if technique == "few-shot":
            if example == "sentiment":
                result = gemini.few_shot_sentiment_analysis("I absolutely love this new coffee shop!")
                print(f"Sentiment: {result['sentiment']}")
            elif example == "math":
                result = gemini.few_shot_math_solver("If a book costs $15 and I buy 4 books, how much do I spend?")
                print(f"Solution: {result['solution']}")
            elif example == "translation":
                result = gemini.few_shot_translation("Good morning", "Spanish")
                print(f"Translation: {result['translation']}")
            elif example == "code":
                result = gemini.few_shot_code_generation("Create a function to find the maximum number in a list")
                print(f"Generated Code:\n{result['generated_code']}")
            else:
                test_few_shot_learning(gemini)
                
        elif technique == "chain-of-thought":
            if example == "math":
                result = gemini.chain_of_thought_math_solver("A train travels 200 km in 2.5 hours. What is its average speed?")
                print(f"Solution:\n{result['step_by_step_solution']}")
            elif example == "logic":
                result = gemini.chain_of_thought_logical_reasoning("If all roses are flowers, and some flowers are red, can we conclude that some roses are red?")
                print(f"Logic:\n{result['logical_reasoning']}")
            elif example == "decision":
                result = gemini.chain_of_thought_decision_making("Should I learn Python or JavaScript as my first programming language?")
                print(f"Decision Process:\n{result['reasoning_process']}")
            else:
                test_chain_of_thought(gemini)
                
        elif technique == "tree-of-thought":
            if example == "problem":
                result = await gemini.tree_of_thought_complex_problem("How can we make cities more sustainable?")
                print(f"Best Approach: {result['best_approach']['evaluation']}")
            elif example == "creative":
                result = await gemini.tree_of_thought_creative_brainstorming("Design a mobile app for mental health")
                print(f"Creative Ideas: {len(result['creative_directions'])} directions explored")
            elif example == "strategy":
                result = await gemini.tree_of_thought_strategic_planning("Launch a new eco-friendly product line")
                print(f"Strategic Options: {len(result['strategic_options'])} strategies developed")
            else:
                await test_tree_of_thought(gemini)
                
        elif technique == "self-consistency":
            if example == "general":
                result = await gemini.self_consistency_answer("What makes a good leader?", num_samples=3)
                print(f"Consistent Answer: {result['final_answer']}")
            elif example == "math":
                result = await gemini.self_consistency_math_solver("Calculate 15% tip on a $80 bill", num_samples=3)
                print(f"Math Answer: {result['final_answer']}")
            elif example == "reasoning":
                result = await gemini.self_consistency_reasoning("Why is teamwork important in the workplace?", num_samples=3)
                print(f"Reasoning: {result['final_conclusion']}")
            else:
                await test_self_consistency(gemini)
                
        elif technique == "meta-prompting":
            if example == "optimize":
                result = gemini.meta_prompt_optimization("Classify emails", "Is this email spam? {email}")
                print(f"Optimized: {result['optimized_prompt']}")
            elif example == "analyze":
                result = gemini.meta_task_analysis("Create a workout plan for beginners")
                print(f"Analysis: {result['task_analysis']}")
            elif example == "generate":
                result = gemini.meta_prompt_generation("Summarize articles", "journalists", "bullet points", "news website")
                print(f"Generated: {result['generated_prompt']}")
            else:
                test_meta_prompting(gemini)
        else:
            print(f"❌ Unknown technique: {technique}")
            print_available_techniques()
            
    except Exception as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            print("⚠️  RATE LIMIT EXCEEDED")
            print("The free tier has a limit of 10 requests per minute.")
            print("Please wait a minute before trying again.")
            print("💡 Tip: Use specific examples to test individual features:")
            print_technique_examples(technique)
        else:
            print(f"❌ Error: {e}")


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Advanced Prompting Techniques for Gemini 2.5 Flash",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --list                           # List all techniques
  python main.py --technique few-shot             # Test few-shot learning
  python main.py --technique chain-of-thought     # Test chain-of-thought
  python main.py --technique few-shot --example sentiment  # Specific example
        """
    )
    
    parser.add_argument(
        "--technique", "-t",
        choices=["few-shot", "chain-of-thought", "tree-of-thought", "self-consistency", "meta-prompting"],
        help="Choose a specific technique to test"
    )
    
    parser.add_argument(
        "--example", "-e",
        help="Run a specific example for the chosen technique"
    )
    
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available techniques"
    )
    
    args = parser.parse_args()
    
    if args.list:
        print_available_techniques()
        return
    
    if not args.technique:
        print("🚀 ADVANCED PROMPTING TECHNIQUES FOR GEMINI 2.5 FLASH")
        print("=" * 60)
        print("\n⚠️  No technique specified!")
        print("Use --technique to choose a specific technique to avoid rate limits.")
        print("\n")
        print_available_techniques()
        return
    
    print(f"🚀 TESTING: {args.technique.upper()}")
    if args.example:
        print(f"📝 Example: {args.example}")
    print("=" * 60)
    
    # Run the specific technique
    if args.technique in ["tree-of-thought", "self-consistency"]:
        asyncio.run(run_specific_technique(args.technique, args.example))
    else:
        asyncio.run(run_specific_technique(args.technique, args.example))


if __name__ == "__main__":
    main()