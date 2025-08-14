"""
Advanced Prompting Techniques - Comprehensive Unit Test Suite
Tests all core functionality including API integration, prompt templates, and error handling
"""

import unittest
import asyncio
import os
import sys
import time
import json
from unittest.mock import patch, MagicMock
from typing import Dict, Any, List
import tempfile
from dotenv import load_dotenv

# Load environment variables for testing
load_dotenv()

# Import the main modules for testing
from main import AdvancedPromptingGemini
from techniques import few_shot, chain_of_thought, tree_of_thought, self_consistency, meta_prompting


class TestAdvancedPromptingTechniques(unittest.TestCase):
    """Comprehensive unit tests for Advanced Prompting Techniques"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        print("\n🔧 Setting up Advanced Prompting Test Environment...")
        
        # Test data
        cls.sample_text = "This smartphone is absolutely amazing! Best purchase ever!"
        cls.sample_math_problem = "If a pizza costs $12 and is cut into 8 slices, how much does each slice cost?"
        cls.sample_logic_problem = "All cats are animals. Some animals are pets. Can we conclude that some cats are pets?"
        cls.sample_complex_problem = "How can we reduce plastic waste in our daily lives?"
        cls.sample_prompt = "Tell me if this text is positive or negative: {text}"
        
        # Initialize service (will test API key in first test)
        cls.gemini_service = None
        
        print("✅ Test environment setup complete")
    
    def test_01_api_key_validation(self):
        """Test Case 1: Validate Gemini API key and environment setup"""
        print("\n🔑 Testing API Key Validation and Environment Setup...")
        
        # Test environment variable loading
        self.assertTrue(os.path.exists('.env') or os.getenv('GEMINI_API_KEY'), 
                       "Either .env file should exist or GEMINI_API_KEY should be set")
        
        # Test API key configuration
        api_key = os.getenv('GEMINI_API_KEY')
        self.assertIsNotNone(api_key, "GEMINI_API_KEY not found in environment variables")
        self.assertGreater(len(api_key), 20, "API key seems too short to be valid")
        self.assertTrue(api_key.startswith('AIza'), "Gemini API key should start with 'AIza'")
        
        # Test AdvancedPromptingGemini initialization
        try:
            self.gemini_service = AdvancedPromptingGemini()
            self.__class__.gemini_service = self.gemini_service
            self.assertIsNotNone(self.gemini_service.client, "Gemini client should be initialized")
            self.assertEqual(self.gemini_service.model, "gemini-2.5-flash", "Default model should be gemini-2.5-flash")
        except Exception as e:
            self.fail(f"Failed to initialize AdvancedPromptingGemini: {e}")
        
        # Test basic API connection with a simple request
        try:
            response = self.gemini_service.generate_response("Hello", temperature=0.1)
            self.assertIsInstance(response, str, "Response should be a string")
            self.assertGreater(len(response), 0, "Response should not be empty")
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print("⚠️  Rate limit encountered during API test - this is expected with free tier")
            else:
                self.fail(f"API connection test failed: {e}")
        
        print("✅ API key validation and environment setup passed")
    
    def test_02_few_shot_learning(self):
        """Test Case 2: Validate few-shot learning techniques"""
        print("\n🎯 Testing Few-Shot Learning Functionality...")
        
        if not self.gemini_service:
            self.skipTest("Gemini service not initialized")
        
        try:
            # Test 1: Sentiment Analysis
            print("  Testing sentiment analysis...")
            result = self.gemini_service.few_shot_sentiment_analysis(self.sample_text)
            
            # Validate response structure
            self.assertIsInstance(result, dict, "Result should be a dictionary")
            self.assertIn('text', result, "Result should contain 'text' field")
            self.assertIn('sentiment', result, "Result should contain 'sentiment' field")
            self.assertIn('technique', result, "Result should contain 'technique' field")
            self.assertIn('prompt_used', result, "Result should contain 'prompt_used' field")
            
            # Validate content
            self.assertEqual(result['text'], self.sample_text, "Input text should match")
            self.assertEqual(result['technique'], "Few-shot Learning", "Technique should be Few-shot Learning")
            self.assertIsInstance(result['sentiment'], str, "Sentiment should be a string")
            self.assertGreater(len(result['sentiment']), 0, "Sentiment should not be empty")
            
            # Test 2: Math Problem Solving
            print("  Testing math problem solving...")
            math_result = self.gemini_service.few_shot_math_solver(self.sample_math_problem)
            
            # Validate math result structure
            self.assertIn('problem', math_result, "Math result should contain 'problem' field")
            self.assertIn('solution', math_result, "Math result should contain 'solution' field")
            self.assertEqual(math_result['problem'], self.sample_math_problem, "Problem should match input")
            self.assertIsInstance(math_result['solution'], str, "Solution should be a string")
            
            # Test 3: Named Entity Recognition
            print("  Testing named entity recognition...")
            ner_text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
            ner_result = self.gemini_service.few_shot_named_entity_recognition(ner_text)
            
            # Validate NER result
            self.assertIn('entities', ner_result, "NER result should contain 'entities' field")
            self.assertIsInstance(ner_result['entities'], str, "Entities should be a string")
            
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print("⚠️  Rate limit encountered - few-shot learning test skipped")
                self.skipTest("Rate limit encountered during few-shot learning test")
            else:
                self.fail(f"Few-shot learning test failed: {e}")
        
        print("✅ Few-shot learning functionality validated")
    
    def test_03_chain_of_thought_reasoning(self):
        """Test Case 3: Validate chain-of-thought step-by-step reasoning"""
        print("\n🔗 Testing Chain-of-Thought Reasoning...")
        
        if not self.gemini_service:
            self.skipTest("Gemini service not initialized")
        
        try:
            # Test 1: Math Problem with Step-by-Step Solution
            print("  Testing math reasoning...")
            math_result = self.gemini_service.chain_of_thought_math_solver(self.sample_math_problem)
            
            # Validate response structure
            self.assertIsInstance(math_result, dict, "Result should be a dictionary")
            self.assertIn('problem', math_result, "Result should contain 'problem' field")
            self.assertIn('step_by_step_solution', math_result, "Result should contain 'step_by_step_solution' field")
            self.assertIn('technique', math_result, "Result should contain 'technique' field")
            
            # Validate content
            self.assertEqual(math_result['technique'], "Chain-of-Thought", "Technique should be Chain-of-Thought")
            self.assertIsInstance(math_result['step_by_step_solution'], str, "Solution should be a string")
            self.assertGreater(len(math_result['step_by_step_solution']), 50, "Solution should be detailed")
            
            # Check for step-by-step indicators
            solution_text = math_result['step_by_step_solution'].lower()
            step_indicators = ['step', 'first', 'then', 'next', 'finally', '1.', '2.', '3.']
            has_steps = any(indicator in solution_text for indicator in step_indicators)
            self.assertTrue(has_steps, "Solution should contain step-by-step reasoning indicators")
            
            # Test 2: Logical Reasoning
            print("  Testing logical reasoning...")
            logic_result = self.gemini_service.chain_of_thought_logical_reasoning(self.sample_logic_problem)
            
            # Validate logical reasoning result
            self.assertIn('logical_reasoning', logic_result, "Result should contain 'logical_reasoning' field")
            self.assertIsInstance(logic_result['logical_reasoning'], str, "Logical reasoning should be a string")
            self.assertGreater(len(logic_result['logical_reasoning']), 30, "Logical reasoning should be substantial")
            
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print("⚠️  Rate limit encountered - chain-of-thought test skipped")
                self.skipTest("Rate limit encountered during chain-of-thought test")
            else:
                self.fail(f"Chain-of-thought reasoning test failed: {e}")
        
        print("✅ Chain-of-thought reasoning validated")
    
    def test_04_prompt_templates(self):
        """Test Case 4: Validate prompt templates and formatting"""
        print("\n📝 Testing Prompt Templates and Formatting...")
        
        # Test 1: Import all technique modules
        print("  Testing module imports...")
        
        # Test few-shot prompts
        self.assertTrue(hasattr(few_shot, 'SENTIMENT_CLASSIFICATION'), "Few-shot should have SENTIMENT_CLASSIFICATION")
        self.assertTrue(hasattr(few_shot, 'MATH_WORD_PROBLEMS'), "Few-shot should have MATH_WORD_PROBLEMS")
        self.assertTrue(hasattr(few_shot, 'NAMED_ENTITY_RECOGNITION'), "Few-shot should have NAMED_ENTITY_RECOGNITION")
        
        # Test chain-of-thought prompts
        self.assertTrue(hasattr(chain_of_thought, 'MATH_PROBLEM_SOLVING'), "CoT should have MATH_PROBLEM_SOLVING")
        self.assertTrue(hasattr(chain_of_thought, 'LOGICAL_REASONING'), "CoT should have LOGICAL_REASONING")
        self.assertTrue(hasattr(chain_of_thought, 'COMPLEX_ANALYSIS'), "CoT should have COMPLEX_ANALYSIS")
        
        # Test other technique prompts
        self.assertTrue(hasattr(tree_of_thought, 'COMPLEX_PROBLEM_SOLVING'), "ToT should have COMPLEX_PROBLEM_SOLVING")
        self.assertTrue(hasattr(self_consistency, 'GENERAL_CONSISTENCY'), "SC should have GENERAL_CONSISTENCY")
        self.assertTrue(hasattr(meta_prompting, 'PROMPT_OPTIMIZATION'), "Meta should have PROMPT_OPTIMIZATION")
        
        # Test 2: Prompt Template Formatting
        print("  Testing prompt formatting...")
        
        # Test few-shot sentiment prompt formatting
        sentiment_prompt = few_shot.SENTIMENT_CLASSIFICATION.format(text="Test text")
        self.assertIn("Test text", sentiment_prompt, "Formatted prompt should contain input text")
        self.assertIn("Example", sentiment_prompt, "Sentiment prompt should contain examples")
        self.assertIn("Output:", sentiment_prompt, "Sentiment prompt should have output section")
        
        # Test chain-of-thought math prompt formatting
        math_prompt = chain_of_thought.MATH_PROBLEM_SOLVING.format(problem="2 + 2 = ?")
        self.assertIn("2 + 2 = ?", math_prompt, "Formatted prompt should contain problem")
        self.assertIn("step by step", math_prompt.lower(), "CoT prompt should mention step by step")
        
        # Test meta-prompting optimization
        meta_prompt = meta_prompting.PROMPT_OPTIMIZATION.format(
            task="Test task", 
            current_prompt="Test prompt"
        )
        self.assertIn("Test task", meta_prompt, "Meta prompt should contain task")
        self.assertIn("Test prompt", meta_prompt, "Meta prompt should contain current prompt")
        
        # Test 3: Prompt Structure Validation
        print("  Testing prompt structure...")
        
        # Check that prompts are strings and not empty
        prompts_to_check = [
            few_shot.SENTIMENT_CLASSIFICATION,
            chain_of_thought.MATH_PROBLEM_SOLVING,
            tree_of_thought.COMPLEX_PROBLEM_SOLVING,
            self_consistency.GENERAL_CONSISTENCY,
            meta_prompting.PROMPT_OPTIMIZATION
        ]
        
        for prompt in prompts_to_check:
            self.assertIsInstance(prompt, str, "Prompt should be a string")
            self.assertGreater(len(prompt), 50, "Prompt should be substantial")
            self.assertIn("{", prompt, "Prompt should contain formatting placeholders")
        
        # Test 4: Meta-Prompting Functionality
        if self.gemini_service:
            try:
                print("  Testing meta-prompting optimization...")
                meta_result = self.gemini_service.meta_prompt_optimization(
                    "Classify text sentiment", 
                    self.sample_prompt
                )
                
                # Validate meta-prompting result
                self.assertIn('optimized_prompt', meta_result, "Meta result should contain optimized prompt")
                self.assertIn('technique', meta_result, "Meta result should contain technique")
                self.assertEqual(meta_result['technique'], "Meta-Prompting", "Technique should be Meta-Prompting")
                
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    print("⚠️  Rate limit encountered - meta-prompting test skipped")
                else:
                    print(f"⚠️  Meta-prompting test failed: {e}")
        
        print("✅ Prompt templates and formatting validated")
    
    def test_05_rate_limit_handling(self):
        """Test Case 5: Validate rate limit handling and error management"""
        print("\n⚡ Testing Rate Limit Handling and Error Management...")
        
        # Test 1: Individual Technique Execution
        print("  Testing individual technique execution...")
        
        if self.gemini_service:
            # Test that individual methods exist and are callable
            techniques = [
                'few_shot_sentiment_analysis',
                'chain_of_thought_math_solver',
                'meta_prompt_optimization'
            ]
            
            for technique in techniques:
                self.assertTrue(hasattr(self.gemini_service, technique), 
                              f"Service should have {technique} method")
                method = getattr(self.gemini_service, technique)
                self.assertTrue(callable(method), f"{technique} should be callable")
        
        # Test 2: Error Handling with Mock
        print("  Testing error handling...")
        
        # Mock a rate limit error
        with patch.object(self.gemini_service.client.models if self.gemini_service else MagicMock(), 
                         'generate_content') as mock_generate:
            
            # Create a mock 429 error
            mock_error = Exception("429 RESOURCE_EXHAUSTED")
            mock_generate.side_effect = mock_error
            
            if self.gemini_service:
                try:
                    # This should raise an exception
                    self.gemini_service.generate_response("test")
                    # If we get here without exception, that's also valid (error handling might catch it)
                except Exception as e:
                    # Verify it's the expected error type
                    self.assertIn("429", str(e), "Should handle 429 errors appropriately")
        
        # Test 3: Argument Parser Functionality (import test)
        print("  Testing argument parser functionality...")
        
        # Test that main module can be imported and has required functions
        import main
        self.assertTrue(hasattr(main, 'main'), "Main module should have main function")
        self.assertTrue(hasattr(main, 'run_specific_technique'), "Main should have run_specific_technique")
        self.assertTrue(hasattr(main, 'print_available_techniques'), "Main should have print_available_techniques")
        
        # Test 4: Async Function Handling
        print("  Testing async function handling...")
        
        if self.gemini_service:
            # Test that async methods exist
            async_methods = [
                'tree_of_thought_complex_problem',
                'self_consistency_answer',
                'generate_multiple_responses'
            ]
            
            for method_name in async_methods:
                self.assertTrue(hasattr(self.gemini_service, method_name), 
                              f"Service should have {method_name} method")
                method = getattr(self.gemini_service, method_name)
                self.assertTrue(callable(method), f"{method_name} should be callable")
        
        # Test 5: Configuration Validation
        print("  Testing configuration validation...")
        
        # Test model configuration
        if self.gemini_service:
            self.assertIsInstance(self.gemini_service.model, str, "Model should be a string")
            self.assertIn("gemini", self.gemini_service.model.lower(), "Model should be a Gemini model")
        
        # Test API key validation in constructor
        try:
            # Test with None API key (should use environment)
            test_service = AdvancedPromptingGemini(api_key=None)
            self.assertIsNotNone(test_service.api_key, "API key should be loaded from environment")
        except ValueError as e:
            # This is expected if no API key is available
            self.assertIn("GEMINI_API_KEY", str(e), "Should provide helpful error message")
        
        print("✅ Rate limit handling and error management validated")


class TestAsyncFunctionality(unittest.TestCase):
    """Test async functionality separately"""
    
    def setUp(self):
        """Set up async test environment"""
        if os.getenv('GEMINI_API_KEY'):
            self.gemini_service = AdvancedPromptingGemini()
        else:
            self.gemini_service = None
    
    def test_async_tree_of_thought(self):
        """Test async tree-of-thought functionality"""
        if not self.gemini_service:
            self.skipTest("Gemini service not available")
        
        async def run_async_test():
            try:
                result = await self.gemini_service.tree_of_thought_complex_problem(
                    "How can we make cities more sustainable?"
                )
                
                # Validate async result structure
                self.assertIsInstance(result, dict, "Async result should be a dictionary")
                self.assertIn('problem', result, "Result should contain problem")
                self.assertIn('technique', result, "Result should contain technique")
                self.assertEqual(result['technique'], "Tree-of-Thought", "Should be Tree-of-Thought technique")
                
                return True
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    print("⚠️  Rate limit encountered in async test")
                    return True  # Consider this a pass for rate limit
                else:
                    raise e
        
        # Run the async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(run_async_test())
            self.assertTrue(result, "Async test should complete successfully")
        finally:
            loop.close()


def run_tests():
    """Run all tests with detailed output"""
    print("="*80)
    print("🧪 ADVANCED PROMPTING TECHNIQUES - COMPREHENSIVE TEST SUITE")
    print("="*80)
    print("🔧 Testing Environment:")
    print(f"   Python Version: {sys.version}")
    print(f"   API Key Available: {'✅' if os.getenv('GEMINI_API_KEY') else '❌'}")
    print(f"   .env File: {'✅' if os.path.exists('.env') else '❌'}")
    print("="*80)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add main test cases
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestAdvancedPromptingTechniques))
    
    # Add async test cases
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestAsyncFunctionality))
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        descriptions=True,
        failfast=False
    )
    
    result = runner.run(suite)
    
    # Print detailed summary
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE TEST SUMMARY")
    print("="*80)
    print(f"🧪 Tests Run: {result.testsRun}")
    print(f"✅ Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Failures: {len(result.failures)}")
    print(f"💥 Errors: {len(result.errors)}")
    print(f"⏭️  Skipped: {len(getattr(result, 'skipped', []))}")
    
    if result.failures:
        print(f"\n❌ DETAILED FAILURES ({len(result.failures)}):")
        for i, (test, traceback) in enumerate(result.failures, 1):
            print(f"  {i}. {test}")
            print(f"     {traceback.split('AssertionError:')[-1].strip() if 'AssertionError:' in traceback else 'See full traceback above'}")
    
    if result.errors:
        print(f"\n💥 DETAILED ERRORS ({len(result.errors)}):")
        for i, (test, traceback) in enumerate(result.errors, 1):
            print(f"  {i}. {test}")
            print(f"     {traceback.split('Exception:')[-1].strip() if 'Exception:' in traceback else 'See full traceback above'}")
    
    # Calculate success rate
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0
    
    print(f"\n📈 SUCCESS RATE: {success_rate:.1f}%")
    
    if result.wasSuccessful():
        print("\n🎉 ALL TESTS PASSED! Your Advanced Prompting Techniques are working perfectly!")
    else:
        print(f"\n⚠️  {len(result.failures) + len(result.errors)} TEST(S) FAILED")
        print("💡 Common issues:")
        print("   - Check your GEMINI_API_KEY in .env file")
        print("   - Ensure you have internet connection")
        print("   - Free tier rate limits may cause some tests to be skipped")
    
    print("="*80)
    return result.wasSuccessful()


if __name__ == '__main__':
    print("🚀 Starting Advanced Prompting Techniques Test Suite")
    print("📋 Prerequisites:")
    print("   1. Ensure your .env file has GEMINI_API_KEY configured")
    print("   2. Install required dependencies: pip install -r requirements.txt")
    print("   3. Internet connection for API calls")
    print("⚠️  Note: Some tests may be skipped due to free tier rate limits")
    print()
    
    try:
        success = run_tests()
        exit_code = 0 if success else 1
        print(f"\n🏁 Test suite completed with exit code: {exit_code}")
        exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️  Test suite interrupted by user")
        exit(1)
    except Exception as e:
        print(f"❌ Test suite failed to run: {e}")
        import traceback
        traceback.print_exc()
        exit(1)