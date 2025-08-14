"""
Advanced Prompting Techniques - FAST Unit Test Suite
Optimized tests using mocked API calls for speed (~10-15 seconds total)
"""

import unittest
import asyncio
import os
import sys
import time
from unittest.mock import patch, MagicMock, Mock
from typing import Dict, Any, List
from dotenv import load_dotenv

# Load environment variables for testing
load_dotenv()

# Import test mocks and fixtures
from test_mocks import MockGeminiClient, MockGeminiResponses, TEST_FIXTURES, get_expected_response

# Import the main modules for testing
from main import AdvancedPromptingGemini
from techniques import few_shot, chain_of_thought, tree_of_thought, self_consistency, meta_prompting


class TestAdvancedPromptingFast(unittest.TestCase):
    """Fast unit tests using mocked API calls"""
    
    @classmethod
    def setUpClass(cls):
        """Set up fast test environment with mocks"""
        print("\n⚡ Setting up FAST Test Environment with Mocks...")
        
        # Test data
        cls.sample_text = TEST_FIXTURES["sample_texts"][0]
        cls.sample_math_problem = TEST_FIXTURES["sample_math_problems"][0]
        cls.sample_logic_problem = TEST_FIXTURES["sample_logic_problems"][0]
        cls.sample_ner_text = TEST_FIXTURES["sample_ner_texts"][0]
        
        # Mock responses
        cls.mock_responses = MockGeminiResponses()
        
        print("✅ Fast test environment setup complete")
    
    def setUp(self):
        """Set up individual test with fresh mocks"""
        self.mock_client = MockGeminiClient()
        
        # Create patcher for Gemini client
        self.client_patcher = patch('google.genai.Client')
        self.mock_genai_client = self.client_patcher.start()
        self.mock_genai_client.return_value.models = self.mock_client
        
        # Initialize service with mocked client
        self.gemini_service = AdvancedPromptingGemini()
        self.gemini_service.client.models = self.mock_client
    
    def tearDown(self):
        """Clean up patches"""
        self.client_patcher.stop()
    
    def test_01_api_key_validation_fast(self):
        """Test Case 1: Fast API key validation (mostly environment checks)"""
        print("\n🔑 Testing API Key Validation (Fast)...")
        
        # Test environment variable loading (no API call)
        api_key = os.getenv('GEMINI_API_KEY')
        if api_key:
            self.assertIsNotNone(api_key, "GEMINI_API_KEY should be available")
            self.assertGreater(len(api_key), 20, "API key should be substantial")
            self.assertTrue(api_key.startswith('AIza'), "Gemini API key should start with 'AIza'")
        
        # Test service initialization (mocked)
        self.assertIsNotNone(self.gemini_service, "Service should initialize")
        self.assertIsNotNone(self.gemini_service.client, "Client should be set")
        self.assertEqual(self.gemini_service.model, "gemini-2.5-flash", "Model should be correct")
        
        # Test mock API call (instant response)
        response = self.gemini_service.generate_response("Hello test", temperature=0.1)
        self.assertIsInstance(response, str, "Response should be string")
        self.assertGreater(len(response), 0, "Response should not be empty")
        
        print("✅ Fast API key validation passed")
    
    def test_02_few_shot_learning_fast(self):
        """Test Case 2: Fast few-shot learning with mocked responses"""
        print("\n🎯 Testing Few-Shot Learning (Fast)...")
        
        # Test 1: Sentiment Analysis (mocked)
        result = self.gemini_service.few_shot_sentiment_analysis(self.sample_text)
        
        # Validate response structure
        self.assertIsInstance(result, dict, "Result should be dictionary")
        required_fields = ['text', 'sentiment', 'technique', 'prompt_used']
        for field in required_fields:
            self.assertIn(field, result, f"Result should contain '{field}' field")
        
        # Validate content
        self.assertEqual(result['text'], self.sample_text, "Input text should match")
        self.assertEqual(result['technique'], "Few-shot Learning", "Technique should be correct")
        self.assertIsInstance(result['sentiment'], str, "Sentiment should be string")
        
        # Test 2: Math Problem (mocked)
        math_result = self.gemini_service.few_shot_math_solver(self.sample_math_problem)
        
        self.assertIn('problem', math_result, "Math result should have problem")
        self.assertIn('solution', math_result, "Math result should have solution")
        self.assertEqual(math_result['problem'], self.sample_math_problem, "Problem should match")
        
        # Test 3: Named Entity Recognition (mocked)
        ner_result = self.gemini_service.few_shot_named_entity_recognition(self.sample_ner_text)
        
        self.assertIn('entities', ner_result, "NER should have entities")
        self.assertIsInstance(ner_result['entities'], str, "Entities should be string")
        
        # Verify mock was called (performance check)
        self.assertGreater(self.mock_client.call_count, 0, "Mock client should be called")
        
        print("✅ Fast few-shot learning validated")
    
    def test_03_chain_of_thought_fast(self):
        """Test Case 3: Fast chain-of-thought with mocked responses"""
        print("\n🔗 Testing Chain-of-Thought (Fast)...")
        
        # Test 1: Math Problem with Steps (mocked)
        math_result = self.gemini_service.chain_of_thought_math_solver(self.sample_math_problem)
        
        # Validate structure
        required_fields = ['problem', 'step_by_step_solution', 'technique']
        for field in required_fields:
            self.assertIn(field, math_result, f"Result should contain '{field}'")
        
        # Validate content
        self.assertEqual(math_result['technique'], "Chain-of-Thought", "Technique should be CoT")
        self.assertIsInstance(math_result['step_by_step_solution'], str, "Solution should be string")
        self.assertGreater(len(math_result['step_by_step_solution']), 20, "Solution should be detailed")
        
        # Check for step indicators in mocked response
        solution = math_result['step_by_step_solution'].lower()
        step_indicators = ['step', 'first', 'calculate', 'therefore']
        has_steps = any(indicator in solution for indicator in step_indicators)
        self.assertTrue(has_steps, "Solution should contain reasoning steps")
        
        # Test 2: Logical Reasoning (mocked)
        logic_result = self.gemini_service.chain_of_thought_logical_reasoning(self.sample_logic_problem)
        
        self.assertIn('logical_reasoning', logic_result, "Should have logical reasoning")
        self.assertIsInstance(logic_result['logical_reasoning'], str, "Reasoning should be string")
        
        print("✅ Fast chain-of-thought validated")
    
    def test_04_prompt_templates_fast(self):
        """Test Case 4: Fast prompt template validation (no API calls)"""
        print("\n📝 Testing Prompt Templates (Fast - No API)...")
        
        # Test 1: Module Imports (instant)
        modules_to_test = [
            (few_shot, ['SENTIMENT_CLASSIFICATION', 'MATH_WORD_PROBLEMS', 'NAMED_ENTITY_RECOGNITION']),
            (chain_of_thought, ['MATH_PROBLEM_SOLVING', 'LOGICAL_REASONING', 'COMPLEX_ANALYSIS']),
            (tree_of_thought, ['COMPLEX_PROBLEM_SOLVING', 'CREATIVE_BRAINSTORMING']),
            (self_consistency, ['GENERAL_CONSISTENCY', 'MATH_CONSISTENCY']),
            (meta_prompting, ['PROMPT_OPTIMIZATION', 'TASK_ANALYSIS'])
        ]
        
        for module, attributes in modules_to_test:
            for attr in attributes:
                self.assertTrue(hasattr(module, attr), f"{module.__name__} should have {attr}")
        
        # Test 2: Prompt Formatting (instant)
        test_cases = [
            (few_shot.SENTIMENT_CLASSIFICATION, {"text": "Test text"}, ["Test text", "Example", "Output:"]),
            (chain_of_thought.MATH_PROBLEM_SOLVING, {"problem": "2+2=?"}, ["2+2=?", "step by step"]),
            (meta_prompting.PROMPT_OPTIMIZATION, {"task": "Test", "current_prompt": "Test prompt"}, ["Test", "Test prompt"])
        ]
        
        for template, format_args, expected_content in test_cases:
            formatted = template.format(**format_args)
            self.assertIsInstance(formatted, str, "Formatted prompt should be string")
            for content in expected_content:
                self.assertIn(content.lower(), formatted.lower(), f"Should contain '{content}'")
        
        # Test 3: Prompt Structure Validation (instant)
        prompts_to_validate = [
            few_shot.SENTIMENT_CLASSIFICATION,
            chain_of_thought.MATH_PROBLEM_SOLVING,
            tree_of_thought.COMPLEX_PROBLEM_SOLVING,
            self_consistency.GENERAL_CONSISTENCY,
            meta_prompting.PROMPT_OPTIMIZATION
        ]
        
        for prompt in prompts_to_validate:
            self.assertIsInstance(prompt, str, "Prompt should be string")
            self.assertGreater(len(prompt), 50, "Prompt should be substantial")
            self.assertIn("{", prompt, "Prompt should have placeholders")
        
        # Test 4: Meta-prompting with Mock (fast)
        meta_result = self.gemini_service.meta_prompt_optimization("Test task", "Test prompt")
        
        self.assertIn('optimized_prompt', meta_result, "Should have optimized prompt")
        self.assertEqual(meta_result['technique'], "Meta-Prompting", "Should be meta-prompting")
        
        print("✅ Fast prompt template validation passed")
    
    def test_05_error_handling_fast(self):
        """Test Case 5: Fast error handling with mocked errors"""
        print("\n⚡ Testing Error Handling (Fast)...")
        
        # Test 1: Service Method Existence (instant)
        required_methods = [
            'few_shot_sentiment_analysis',
            'chain_of_thought_math_solver', 
            'meta_prompt_optimization',
            'generate_response'
        ]
        
        for method_name in required_methods:
            self.assertTrue(hasattr(self.gemini_service, method_name), f"Should have {method_name}")
            method = getattr(self.gemini_service, method_name)
            self.assertTrue(callable(method), f"{method_name} should be callable")
        
        # Test 2: Mock Rate Limit Error Handling
        with patch.object(self.mock_client, 'generate_content') as mock_generate:
            # Simulate 429 error
            mock_error = Exception("429 RESOURCE_EXHAUSTED: Rate limit exceeded")
            mock_generate.side_effect = mock_error
            
            try:
                self.gemini_service.generate_response("test")
                # If no exception, that's also valid (error might be handled gracefully)
            except Exception as e:
                self.assertIn("429", str(e), "Should handle 429 errors appropriately")
        
        # Test 3: Configuration Validation (instant)
        self.assertIsInstance(self.gemini_service.model, str, "Model should be string")
        self.assertIn("gemini", self.gemini_service.model.lower(), "Should be Gemini model")
        
        # Test 4: Async Method Existence (instant)
        async_methods = [
            'tree_of_thought_complex_problem',
            'self_consistency_answer',
            'generate_multiple_responses'
        ]
        
        for method_name in async_methods:
            self.assertTrue(hasattr(self.gemini_service, method_name), f"Should have {method_name}")
            method = getattr(self.gemini_service, method_name)
            self.assertTrue(callable(method), f"{method_name} should be callable")
        
        print("✅ Fast error handling validated")
    
    def test_06_performance_validation(self):
        """Test Case 6: Performance validation with mocks"""
        print("\n🚀 Testing Performance with Mocks...")
        
        start_time = time.time()
        
        # Run multiple operations that would normally be slow
        operations = [
            lambda: self.gemini_service.few_shot_sentiment_analysis("Test text 1"),
            lambda: self.gemini_service.few_shot_math_solver("What is 5 + 5?"),
            lambda: self.gemini_service.chain_of_thought_math_solver("Calculate 10 * 2"),
            lambda: self.gemini_service.meta_prompt_optimization("Test", "Test prompt")
        ]
        
        results = []
        for operation in operations:
            result = operation()
            results.append(result)
            self.assertIsInstance(result, dict, "Each operation should return dict")
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # With mocks, this should be very fast (< 1 second)
        self.assertLess(execution_time, 2.0, "Mocked operations should be fast")
        
        # Verify all operations completed
        self.assertEqual(len(results), 4, "All operations should complete")
        
        # Check mock call count
        self.assertGreater(self.mock_client.call_count, 0, "Mock should be called")
        
        print(f"✅ Performance test passed - {len(operations)} operations in {execution_time:.2f}s")


class TestAsyncFunctionalityFast(unittest.TestCase):
    """Fast async functionality tests with mocks"""
    
    def setUp(self):
        """Set up async test with mocks"""
        self.mock_client = MockGeminiClient()
        
        # Create patcher
        self.client_patcher = patch('google.genai.Client')
        self.mock_genai_client = self.client_patcher.start()
        self.mock_genai_client.return_value.models = self.mock_client
        
        # Initialize service
        self.gemini_service = AdvancedPromptingGemini()
        self.gemini_service.client.models = self.mock_client
    
    def tearDown(self):
        """Clean up patches"""
        self.client_patcher.stop()
    
    def test_async_functionality_fast(self):
        """Test async functionality with mocks (fast)"""
        print("\n🔄 Testing Async Functionality (Fast)...")
        
        async def run_async_test():
            start_time = time.time()
            
            # Test tree-of-thought (mocked)
            tot_result = await self.gemini_service.tree_of_thought_complex_problem(
                "How to reduce plastic waste?"
            )
            
            # Test self-consistency (mocked)
            sc_result = await self.gemini_service.self_consistency_answer(
                "What are benefits of exercise?", num_samples=3
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Validate results
            self.assertIsInstance(tot_result, dict, "ToT result should be dict")
            self.assertIn('technique', tot_result, "Should have technique field")
            
            self.assertIsInstance(sc_result, dict, "SC result should be dict")
            self.assertIn('technique', sc_result, "Should have technique field")
            
            # Should be fast with mocks
            self.assertLess(execution_time, 3.0, "Async operations should be fast with mocks")
            
            print(f"✅ Async test completed in {execution_time:.2f}s")
            return True
        
        # Run async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(run_async_test())
            self.assertTrue(result, "Async test should complete successfully")
        finally:
            loop.close()


class TestRealAPIIntegration(unittest.TestCase):
    """Minimal real API tests for critical path validation"""
    
    def setUp(self):
        """Set up real API test (only if API key available)"""
        self.api_key = os.getenv('GEMINI_API_KEY')
        if self.api_key:
            self.gemini_service = AdvancedPromptingGemini()
        else:
            self.gemini_service = None
    
    def test_critical_path_real_api(self):
        """Test critical path with real API (1 call only)"""
        if not self.gemini_service:
            self.skipTest("No API key available for real API test")
        
        print("\n🌐 Testing Critical Path with Real API (1 call)...")
        
        try:
            # Single real API call to verify everything works
            result = self.gemini_service.generate_response(
                "Respond with exactly: 'API test successful'", 
                temperature=0.0
            )
            
            self.assertIsInstance(result, str, "Real API should return string")
            self.assertGreater(len(result), 0, "Real API should return content")
            
            print("✅ Real API integration test passed")
            
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print("⚠️  Rate limit hit in real API test - this is expected")
                self.skipTest("Rate limit encountered in real API test")
            else:
                self.fail(f"Real API test failed: {e}")


def run_fast_tests():
    """Run fast test suite with performance timing"""
    print("="*80)
    print("⚡ ADVANCED PROMPTING TECHNIQUES - FAST TEST SUITE")
    print("="*80)
    print("🎯 Goal: Complete all tests in ~10-15 seconds using mocks")
    print("🔧 Strategy: Mock API calls + minimal real API validation")
    print("="*80)
    
    overall_start = time.time()
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add fast test cases (mocked)
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestAdvancedPromptingFast))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestAsyncFunctionalityFast))
    
    # Add minimal real API test
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestRealAPIIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        descriptions=True,
        failfast=False
    )
    
    result = runner.run(suite)
    
    overall_end = time.time()
    total_time = overall_end - overall_start
    
    # Print performance summary
    print("\n" + "="*80)
    print("⚡ FAST TEST PERFORMANCE SUMMARY")
    print("="*80)
    print(f"🕐 Total Execution Time: {total_time:.2f} seconds")
    print(f"🧪 Tests Run: {result.testsRun}")
    print(f"✅ Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Failures: {len(result.failures)}")
    print(f"💥 Errors: {len(result.errors)}")
    print(f"⏭️  Skipped: {len(getattr(result, 'skipped', []))}")
    
    # Performance analysis
    if total_time < 15:
        print(f"🚀 EXCELLENT: Tests completed in {total_time:.1f}s (Target: <15s)")
    elif total_time < 30:
        print(f"✅ GOOD: Tests completed in {total_time:.1f}s (Acceptable)")
    else:
        print(f"⚠️  SLOW: Tests took {total_time:.1f}s (Consider more mocking)")
    
    # Calculate success rate
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0
    print(f"📈 SUCCESS RATE: {success_rate:.1f}%")
    
    if result.wasSuccessful():
        print("\n🎉 ALL FAST TESTS PASSED!")
        print("Your Advanced Prompting Techniques are working perfectly!")
    else:
        print(f"\n⚠️  {len(result.failures) + len(result.errors)} TEST(S) FAILED")
    
    print("="*80)
    return result.wasSuccessful()


if __name__ == '__main__':
    print("⚡ Starting FAST Advanced Prompting Techniques Test Suite")
    print("🎯 Target: Complete all tests in ~10-15 seconds")
    print("🔧 Method: Mocked API calls + minimal real API validation")
    print()
    
    try:
        success = run_fast_tests()
        exit_code = 0 if success else 1
        exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️  Fast test suite interrupted by user")
        exit(1)
    except Exception as e:
        print(f"❌ Fast test suite failed to run: {e}")
        import traceback
        traceback.print_exc()
        exit(1)