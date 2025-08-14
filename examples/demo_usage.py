"""
Demo Usage Examples
Simple examples showing how to use each technique
"""

import asyncio
import sys
import os

# Add parent directory to path to import main
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import AdvancedPromptingGemini


async def demo_few_shot_learning():
    """Demo Few-shot Learning techniques"""
    print("🎯 FEW-SHOT LEARNING DEMO")
    print("=" * 50)
    
    gemini = AdvancedPromptingGemini()
    
    # Sentiment Analysis
    print("\n1. Sentiment Analysis:")
    result = gemini.few_shot_sentiment_analysis("I absolutely love this new smartphone! It's incredible.")
    print(f"Text: {result['text']}")
    print(f"Sentiment: {result['sentiment']}")
    
    # Named Entity Recognition
    print("\n2. Named Entity Recognition:")
    result = gemini.few_shot_named_entity_recognition("Elon Musk founded SpaceX in 2002 in California.")
    print(f"Text: {result['text']}")
    print(f"Entities: {result['entities']}")
    
    # Math Problem Solving
    print("\n3. Math Problem Solving:")
    result = gemini.few_shot_math_solver("A pizza is cut into 8 slices. If Maria eats 3 slices and John eats 2 slices, how many slices are left?")
    print(f"Problem: {result['problem']}")
    print(f"Solution: {result['solution']}")
    
    # Code Generation
    print("\n4. Code Generation:")
    result = gemini.few_shot_code_generation("Create a function to reverse a string")
    print(f"Task: {result['task']}")
    print(f"Generated Code: {result['generated_code']}")


async def demo_chain_of_thought():
    """Demo Chain-of-Thought reasoning"""
    print("\n🔗 CHAIN-OF-THOUGHT DEMO")
    print("=" * 50)
    
    gemini = AdvancedPromptingGemini()
    
    # Math Problem with Step-by-Step
    print("\n1. Complex Math Problem:")
    result = gemini.chain_of_thought_math_solver("A rectangular garden is 15 meters long and 8 meters wide. If we want to put a fence around it and the fence costs $12 per meter, how much will the fence cost?")
    print(f"Problem: {result['problem']}")
    print(f"Step-by-Step Solution:\n{result['step_by_step_solution']}")
    
    # Logical Reasoning
    print("\n2. Logical Reasoning:")
    result = gemini.chain_of_thought_logical_reasoning("All birds can fly. Penguins are birds. Can penguins fly? Explain the logical issue with this reasoning.")
    print(f"Problem: {result['problem']}")
    print(f"Logical Analysis:\n{result['logical_reasoning']}")
    
    # Decision Making
    print("\n3. Decision Making:")
    result = gemini.chain_of_thought_decision_making("Should I buy a car or use public transportation in a big city?")
    print(f"Decision: {result['decision']}")
    print(f"Reasoning Process:\n{result['reasoning_process']}")


async def demo_tree_of_thought():
    """Demo Tree-of-Thought exploration"""
    print("\n🌳 TREE-OF-THOUGHT DEMO")
    print("=" * 50)
    
    gemini = AdvancedPromptingGemini()
    
    # Complex Problem Solving
    print("\n1. Complex Problem Solving:")
    result = await gemini.tree_of_thought_complex_problem("How can we make online education more engaging and effective?")
    print(f"Problem: {result['problem']}")
    print(f"Number of approaches explored: {len(result['explored_approaches'])}")
    for i, approach in enumerate(result['explored_approaches'], 1):
        print(f"\nApproach {i}: {approach['approach']}")
        print(f"Solution preview: {approach['solution'][:150]}...")
    print(f"\nBest Approach Evaluation:\n{result['best_approach']['evaluation']}")
    
    # Creative Brainstorming
    print("\n2. Creative Brainstorming:")
    result = await gemini.tree_of_thought_creative_brainstorming("Design an innovative solution for food waste reduction")
    print(f"Challenge: {result['challenge']}")
    for direction in result['creative_directions']:
        print(f"\nCreative Direction: {direction['direction']}")
        print(f"Ideas: {direction['creative_ideas'][:200]}...")


async def demo_self_consistency():
    """Demo Self-Consistency technique"""
    print("\n🎯 SELF-CONSISTENCY DEMO")
    print("=" * 50)
    
    gemini = AdvancedPromptingGemini()
    
    # General Question
    print("\n1. General Question with Multiple Samples:")
    result = await gemini.self_consistency_answer("What are the most important factors for a successful startup?", num_samples=4)
    print(f"Question: {result['question']}")
    print(f"Number of samples: {result['num_samples']}")
    print(f"Final Consistent Answer: {result['final_answer']}")
    print(f"Consistency Analysis: {result['consistency_analysis']['analysis'][:300]}...")
    
    # Math Problem
    print("\n2. Math Problem with Consistency Check:")
    result = await gemini.self_consistency_math_solver("If a car travels at 60 mph for 2.5 hours, how far does it travel?", num_samples=3)
    print(f"Problem: {result['problem']}")
    print(f"Final Answer: {result['final_answer']}")
    print(f"All Solutions:")
    for i, solution in enumerate(result['all_solutions'], 1):
        print(f"  Solution {i}: {solution[:100]}...")


async def demo_meta_prompting():
    """Demo Meta-Prompting techniques"""
    print("\n🧠 META-PROMPTING DEMO")
    print("=" * 50)
    
    gemini = AdvancedPromptingGemini()
    
    # Prompt Optimization
    print("\n1. Prompt Optimization:")
    result = gemini.meta_prompt_optimization(
        task="Summarize research papers",
        current_prompt="Summarize this paper: {paper_text}"
    )
    print(f"Original Task: {result['original_task']}")
    print(f"Original Prompt: {result['original_prompt']}")
    print(f"Optimized Prompt:\n{result['optimized_prompt']}")
    
    # Task Analysis
    print("\n2. Task Analysis:")
    result = gemini.meta_task_analysis("Create a comprehensive marketing strategy for a new product launch")
    print(f"Task: {result['task']}")
    print(f"Analysis:\n{result['task_analysis']}")
    
    # Prompt Generation
    print("\n3. Prompt Generation:")
    result = gemini.meta_prompt_generation(
        task="Analyze customer reviews",
        audience="business analysts",
        output_type="structured report",
        context="e-commerce platform"
    )
    print(f"Task: {result['task']}")
    print(f"Generated Prompt:\n{result['generated_prompt']}")


async def demo_combined_techniques():
    """Demo Combined Techniques"""
    print("\n🔥 COMBINED TECHNIQUES DEMO")
    print("=" * 50)
    
    gemini = AdvancedPromptingGemini()
    
    print("\nSolving a complex problem using ALL techniques:")
    result = await gemini.combined_advanced_solver("How can we address the global climate change crisis effectively?")
    
    print(f"Problem: {result['problem']}")
    print(f"Techniques Used: {', '.join(result['techniques_used'])}")
    print(f"\nFinal Synthesis:\n{result['final_synthesis']['synthesis']}")
    print(f"Confidence Level: {result['final_synthesis']['confidence_level']}")


async def run_all_demos():
    """Run all demonstration examples"""
    print("🚀 ADVANCED PROMPTING TECHNIQUES - COMPLETE DEMO")
    print("=" * 60)
    
    await demo_few_shot_learning()
    await demo_chain_of_thought()
    await demo_tree_of_thought()
    await demo_self_consistency()
    await demo_meta_prompting()
    await demo_combined_techniques()
    
    print("\n✅ ALL DEMOS COMPLETED SUCCESSFULLY!")
    print("You've seen all advanced prompting techniques in action.")


if __name__ == "__main__":
    # Run all demos
    asyncio.run(run_all_demos())