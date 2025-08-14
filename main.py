"""
Advanced Prompting Techniques for Gemini 2.5 Flash
Main implementation file with all technique orchestration
"""

import asyncio
import statistics
import os
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
            api_key: Gemini API key
        """
        self.api_key = api_key
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.5-flash"
    
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
    
    # ==================== COMBINED TECHNIQUES ====================
    
    async def combined_advanced_solver(self, problem: str) -> Dict[str, Any]:
        """Combine multiple techniques for comprehensive problem solving"""
        results = {}
        
        # 1. Chain-of-Thought Analysis
        cot_result = self.chain_of_thought_complex_analysis(problem)
        results["chain_of_thought"] = cot_result
        
        # 2. Tree-of-Thought Exploration
        tot_result = await self.tree_of_thought_complex_problem(problem)
        results["tree_of_thought"] = tot_result
        
        # 3. Self-Consistency Verification
        sc_result = await self.self_consistency_reasoning(problem, num_samples=3)
        results["self_consistency"] = sc_result
        
        # 4. Meta-Analysis of the problem
        meta_result = self.meta_task_analysis(problem)
        results["meta_analysis"] = meta_result
        
        # 5. Final Synthesis
        synthesis = await self._synthesize_combined_results(problem, results)
        
        return {
            "problem": problem,
            "individual_results": results,
            "final_synthesis": synthesis,
            "techniques_used": ["Chain-of-Thought", "Tree-of-Thought", "Self-Consistency", "Meta-Prompting"],
            "methodology": "Combined Advanced Techniques"
        }
    
    async def _synthesize_combined_results(self, problem: str, results: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize results from multiple techniques"""
        synthesis_prompt = f"""Problem: {problem}

I have analyzed this problem using multiple advanced techniques:

1. Chain-of-Thought Analysis:
{results['chain_of_thought']['detailed_analysis'][:300]}...

2. Tree-of-Thought Exploration:
Best approach: {results['tree_of_thought']['best_approach']['evaluation'][:300]}...

3. Self-Consistency Verification:
Final conclusion: {results['self_consistency']['final_conclusion'][:300]}...

4. Meta-Analysis:
Task analysis: {results['meta_analysis']['task_analysis'][:300]}...

Please synthesize these different analyses into:
1. A comprehensive understanding of the problem
2. The most reliable solution or approach
3. Key insights from combining multiple techniques
4. Confidence level in the final recommendation

Final synthesis:"""
        
        synthesis = await self._async_generate(synthesis_prompt, temperature=0.3, thinking_budget=10000)
        
        return {
            "synthesis": synthesis.strip(),
            "techniques_combined": 4,
            "confidence_level": "High (multiple technique validation)"
        }


# ==================== DEMONSTRATION FUNCTIONS ====================

async def demonstrate_all_techniques():
    """Demonstrate all advanced prompting techniques"""
    
    # Initialize the system
    gemini = AdvancedPromptingGemini()
    
    print("🚀 Advanced Prompting Techniques for Gemini 2.5 Flash")
    print("=" * 60)
    
    # Few-shot Learning Examples
    print("\n📚 FEW-SHOT LEARNING EXAMPLES:")
    print("-" * 40)
    
    # Sentiment Analysis
    sentiment_result = gemini.few_shot_sentiment_analysis("This movie was absolutely fantastic! I loved every minute of it.")
    print(f"Sentiment Analysis: {sentiment_result['sentiment']}")
    
    # Math Problem
    math_result = gemini.few_shot_math_solver("If John has 25 apples and gives 8 to his sister, how many apples does he have left?")
    print(f"Math Solution: {math_result['solution']}")
    
    # Chain-of-Thought Examples
    print("\n🔗 CHAIN-OF-THOUGHT EXAMPLES:")
    print("-" * 40)
    
    # Complex Math
    cot_math = gemini.chain_of_thought_math_solver("A train travels 120 km in 2 hours, then 180 km in 3 hours. What is its average speed?")
    print(f"CoT Math Solution: {cot_math['step_by_step_solution'][:200]}...")
    
    # Logical Reasoning
    cot_logic = gemini.chain_of_thought_logical_reasoning("If all cats are animals, and some animals are pets, can we conclude that some cats are pets?")
    print(f"CoT Logic: {cot_logic['logical_reasoning'][:200]}...")
    
    # Tree-of-Thought Examples
    print("\n🌳 TREE-OF-THOUGHT EXAMPLES:")
    print("-" * 40)
    
    # Complex Problem
    tot_result = await gemini.tree_of_thought_complex_problem("How can we reduce plastic waste in our city?")
    print(f"ToT Best Approach: {tot_result['best_approach']['evaluation'][:200]}...")
    
    # Creative Brainstorming
    creative_result = await gemini.tree_of_thought_creative_brainstorming("Design a mobile app that helps people learn new languages")
    print(f"ToT Creative Ideas: {len(creative_result['creative_directions'])} directions explored")
    
    # Self-Consistency Examples
    print("\n🎯 SELF-CONSISTENCY EXAMPLES:")
    print("-" * 40)
    
    # General Question
    sc_result = await gemini.self_consistency_answer("What are the main benefits of renewable energy?", num_samples=3)
    print(f"SC Final Answer: {sc_result['final_answer'][:200]}...")
    
    # Math Problem
    sc_math = await gemini.self_consistency_math_solver("Calculate the area of a circle with radius 7 meters", num_samples=3)
    print(f"SC Math Answer: {sc_math['final_answer'][:100]}...")
    
    # Meta-Prompting Examples
    print("\n🧠 META-PROMPTING EXAMPLES:")
    print("-" * 40)
    
    # Prompt Optimization
    meta_opt = gemini.meta_prompt_optimization(
        "Classify customer feedback", 
        "Tell me if this feedback is positive or negative: {feedback}"
    )
    print(f"Optimized Prompt: {meta_opt['optimized_prompt'][:200]}...")
    
    # Task Analysis
    meta_analysis = gemini.meta_task_analysis("Create a study plan for learning Python programming")
    print(f"Task Analysis: {meta_analysis['task_analysis'][:200]}...")
    
    # Combined Techniques Example
    print("\n🔥 COMBINED TECHNIQUES EXAMPLE:")
    print("-" * 40)
    
    combined_result = await gemini.combined_advanced_solver("How can artificial intelligence be used to improve healthcare outcomes?")
    print(f"Combined Analysis: {combined_result['final_synthesis']['synthesis'][:300]}...")
    print(f"Techniques Used: {', '.join(combined_result['techniques_used'])}")
    
    print("\n✅ Demonstration Complete!")
    print("All advanced prompting techniques have been successfully demonstrated.")


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(demonstrate_all_techniques())