"""
Demo Usage Examples - Rate Limit Friendly
Simple examples showing how to use each technique individually
"""

import asyncio
import sys
import os
import argparse

# Add parent directory to path to import main
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import AdvancedPromptingGemini


def demo_few_shot_examples():
    """Demo individual Few-shot Learning examples"""
    print("🎯 FEW-SHOT LEARNING EXAMPLES")
    print("=" * 50)
    
    gemini = AdvancedPromptingGemini()
    
    print("\nChoose an example to run:")
    print("1. Sentiment Analysis")
    print("2. Named Entity Recognition") 
    print("3. Math Problem")
    print("4. Code Generation")
    print("5. Translation")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == "1":
        result = gemini.few_shot_sentiment_analysis("This new restaurant is absolutely amazing! Best food I've ever had.")
        print(f"\nText: {result['text']}")
        print(f"Sentiment: {result['sentiment']}")
        
    elif choice == "2":
        result = gemini.few_shot_named_entity_recognition("Tim Cook is the CEO of Apple Inc. based in Cupertino, California.")
        print(f"\nText: {result['text']}")
        print(f"Entities: {result['entities']}")
        
    elif choice == "3":
        result = gemini.few_shot_math_solver("A store sells apples for $2 per pound. If I buy 3.5 pounds, how much do I pay?")
        print(f"\nProblem: {result['problem']}")
        print(f"Solution: {result['solution']}")
        
    elif choice == "4":
        result = gemini.few_shot_code_generation("Create a function to check if a number is prime")
        print(f"\nTask: {result['task']}")
        print(f"Generated Code:\n{result['generated_code']}")
        
    elif choice == "5":
        result = gemini.few_shot_translation("How are you today?", "French")
        print(f"\nOriginal: {result['original_text']}")
        print(f"Translation to {result['target_language']}: {result['translation']}")
        
    else:
        print("Invalid choice!")


def demo_chain_of_thought_examples():
    """Demo individual Chain-of-Thought examples"""
    print("🔗 CHAIN-OF-THOUGHT EXAMPLES")
    print("=" * 50)
    
    gemini = AdvancedPromptingGemini()
    
    print("\nChoose an example to run:")
    print("1. Math Problem")
    print("2. Logical Reasoning")
    print("3. Decision Making")
    print("4. Problem Solving")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        result = gemini.chain_of_thought_math_solver("A car rental costs $30 per day plus $0.25 per mile. If I rent for 3 days and drive 150 miles, what's the total cost?")
        print(f"\nProblem: {result['problem']}")
        print(f"Step-by-Step Solution:\n{result['step_by_step_solution']}")
        
    elif choice == "2":
        result = gemini.chain_of_thought_logical_reasoning("All birds can fly. Penguins are birds. But penguins cannot fly. What's wrong with this reasoning?")
        print(f"\nProblem: {result['problem']}")
        print(f"Logical Analysis:\n{result['logical_reasoning']}")
        
    elif choice == "3":
        result = gemini.chain_of_thought_decision_making("Should I buy a laptop or a desktop computer for programming?")
        print(f"\nDecision: {result['decision']}")
        print(f"Reasoning Process:\n{result['reasoning_process']}")
        
    elif choice == "4":
        result = gemini.chain_of_thought_problem_solving("How can I improve my time management skills?")
        print(f"\nProblem: {result['problem']}")
        print(f"Solution Process:\n{result['solution_process']}")
        
    else:
        print("Invalid choice!")


async def demo_tree_of_thought_examples():
    """Demo individual Tree-of-Thought examples"""
    print("🌳 TREE-OF-THOUGHT EXAMPLES")
    print("=" * 50)
    
    gemini = AdvancedPromptingGemini()
    
    print("\nChoose an example to run:")
    print("1. Complex Problem Solving")
    print("2. Creative Brainstorming")
    print("3. Strategic Planning")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        result = await gemini.tree_of_thought_complex_problem("How can we reduce plastic pollution in oceans?")
        print(f"\nProblem: {result['problem']}")
        print(f"Approaches explored: {len(result['explored_approaches'])}")
        for i, approach in enumerate(result['explored_approaches'], 1):
            print(f"\nApproach {i}: {approach['approach']}")
            print(f"Solution: {approach['solution'][:200]}...")
        print(f"\nBest Approach Evaluation:\n{result['best_approach']['evaluation']}")
        
    elif choice == "2":
        result = await gemini.tree_of_thought_creative_brainstorming("Design an app to help people reduce food waste")
        print(f"\nChallenge: {result['challenge']}")
        for direction in result['creative_directions']:
            print(f"\nCreative Direction: {direction['direction']}")
            print(f"Ideas: {direction['creative_ideas'][:300]}...")
            
    elif choice == "3":
        result = await gemini.tree_of_thought_strategic_planning("Expand a local bakery business")
        print(f"\nGoal: {result['goal']}")
        for plan in result['strategic_options']:
            print(f"\nStrategy: {plan['strategy_type']}")
            print(f"Plan: {plan['strategic_plan'][:300]}...")
            
    else:
        print("Invalid choice!")


async def demo_self_consistency_examples():
    """Demo individual Self-Consistency examples"""
    print("🎯 SELF-CONSISTENCY EXAMPLES")
    print("=" * 50)
    
    gemini = AdvancedPromptingGemini()
    
    print("\nChoose an example to run:")
    print("1. General Question")
    print("2. Math Problem")
    print("3. Reasoning Problem")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    # Using smaller sample sizes to avoid rate limits
    if choice == "1":
        result = await gemini.self_consistency_answer("What are the most important skills for success in the 21st century?", num_samples=3)
        print(f"\nQuestion: {result['question']}")
        print(f"Number of samples: {result['num_samples']}")
        print(f"Final Consistent Answer: {result['final_answer']}")
        
    elif choice == "2":
        result = await gemini.self_consistency_math_solver("If I save $50 per month, how much will I have after 2 years?", num_samples=3)
        print(f"\nProblem: {result['problem']}")
        print(f"Final Answer: {result['final_answer']}")
        print(f"\nAll Solutions:")
        for i, solution in enumerate(result['all_solutions'], 1):
            print(f"Solution {i}: {solution[:150]}...")
            
    elif choice == "3":
        result = await gemini.self_consistency_reasoning("Why is continuous learning important in today's world?", num_samples=3)
        print(f"\nProblem: {result['problem']}")
        print(f"Final Conclusion: {result['final_conclusion']}")
        
    else:
        print("Invalid choice!")


def demo_meta_prompting_examples():
    """Demo individual Meta-Prompting examples"""
    print("🧠 META-PROMPTING EXAMPLES")
    print("=" * 50)
    
    gemini = AdvancedPromptingGemini()
    
    print("\nChoose an example to run:")
    print("1. Prompt Optimization")
    print("2. Task Analysis")
    print("3. Prompt Generation")
    print("4. Prompt Evaluation")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        result = gemini.meta_prompt_optimization(
            task="Write product reviews",
            current_prompt="Write a review for this product: {product}"
        )
        print(f"\nOriginal Task: {result['original_task']}")
        print(f"Original Prompt: {result['original_prompt']}")
        print(f"Optimized Prompt:\n{result['optimized_prompt']}")
        
    elif choice == "2":
        result = gemini.meta_task_analysis("Create a social media marketing strategy")
        print(f"\nTask: {result['task']}")
        print(f"Analysis:\n{result['task_analysis']}")
        
    elif choice == "3":
        result = gemini.meta_prompt_generation(
            task="Analyze customer feedback",
            audience="product managers",
            output_type="structured report",
            context="e-commerce platform"
        )
        print(f"\nTask: {result['task']}")
        print(f"Generated Prompt:\n{result['generated_prompt']}")
        
    elif choice == "4":
        result = gemini.meta_prompt_evaluation(
            task="Classify support tickets",
            prompt_to_evaluate="What category does this support ticket belong to: {ticket}"
        )
        print(f"\nTask: {result['task']}")
        print(f"Evaluated Prompt: {result['evaluated_prompt']}")
        print(f"Evaluation Results:\n{result['evaluation_results']}")
        
    else:
        print("Invalid choice!")


def print_main_menu():
    """Print the main menu"""
    print("🚀 ADVANCED PROMPTING TECHNIQUES - DEMO")
    print("=" * 60)
    print("\nChoose a technique to explore:")
    print("1. Few-shot Learning")
    print("2. Chain-of-Thought")
    print("3. Tree-of-Thought")
    print("4. Self-Consistency")
    print("5. Meta-Prompting")
    print("6. Exit")
    print("\n💡 Tip: Each technique has multiple examples to choose from!")
    print("⚠️  Note: Free tier has rate limits - test one at a time.")


async def main():
    """Main interactive demo"""
    parser = argparse.ArgumentParser(description="Interactive Demo for Advanced Prompting Techniques")
    parser.add_argument("--technique", "-t", 
                       choices=["few-shot", "chain-of-thought", "tree-of-thought", "self-consistency", "meta-prompting"],
                       help="Run a specific technique directly")
    
    args = parser.parse_args()
    
    if args.technique:
        # Run specific technique directly
        if args.technique == "few-shot":
            demo_few_shot_examples()
        elif args.technique == "chain-of-thought":
            demo_chain_of_thought_examples()
        elif args.technique == "tree-of-thought":
            await demo_tree_of_thought_examples()
        elif args.technique == "self-consistency":
            await demo_self_consistency_examples()
        elif args.technique == "meta-prompting":
            demo_meta_prompting_examples()
        return
    
    # Interactive menu
    while True:
        print_main_menu()
        choice = input("\nEnter your choice (1-6): ").strip()
        
        try:
            if choice == "1":
                demo_few_shot_examples()
            elif choice == "2":
                demo_chain_of_thought_examples()
            elif choice == "3":
                await demo_tree_of_thought_examples()
            elif choice == "4":
                await demo_self_consistency_examples()
            elif choice == "5":
                demo_meta_prompting_examples()
            elif choice == "6":
                print("\n👋 Thanks for exploring Advanced Prompting Techniques!")
                break
            else:
                print("❌ Invalid choice! Please enter 1-6.")
                continue
                
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print("\n⚠️  RATE LIMIT EXCEEDED")
                print("The free tier has a limit of 10 requests per minute.")
                print("Please wait about a minute before trying again.")
                print("💡 Tip: Try a different technique or wait a bit.")
            else:
                print(f"❌ Error: {e}")
        
        input("\nPress Enter to continue...")
        print("\n" + "="*60)


if __name__ == "__main__":
    asyncio.run(main())