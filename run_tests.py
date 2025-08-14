#!/usr/bin/env python3
"""
Test Runner for Advanced Prompting Techniques
Easy-to-use script for running all tests with different options
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def check_environment():
    """Check if the environment is properly set up"""
    print("🔍 Checking Environment Setup...")
    
    issues = []
    
    # Check .env file
    if not os.path.exists('.env'):
        issues.append("❌ .env file not found")
    else:
        print("✅ .env file found")
    
    # Check API key
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        issues.append("❌ GEMINI_API_KEY not set in environment")
    elif len(api_key) < 20:
        issues.append("⚠️  GEMINI_API_KEY seems too short")
    else:
        print("✅ GEMINI_API_KEY configured")
    
    # Check required modules
    required_modules = ['google.genai', 'dotenv', 'asyncio']
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module} available")
        except ImportError:
            issues.append(f"❌ {module} not installed")
    
    return issues


def run_quick_test():
    """Run a quick smoke test"""
    print("\n🚀 Running Quick Smoke Test...")
    
    try:
        from main import AdvancedPromptingGemini
        
        # Test initialization
        gemini = AdvancedPromptingGemini()
        print("✅ Service initialization successful")
        
        # Test basic functionality
        result = gemini.generate_response("Hello", temperature=0.1)
        if result and len(result) > 0:
            print("✅ Basic API call successful")
            return True
        else:
            print("❌ API call returned empty response")
            return False
            
    except Exception as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            print("⚠️  Rate limit encountered - this is normal for free tier")
            return True
        else:
            print(f"❌ Quick test failed: {e}")
            return False


def run_full_tests(verbose=True, failfast=False):
    """Run the full test suite"""
    print("\n🧪 Running Full Test Suite...")
    
    # Prepare command
    cmd = [sys.executable, "test_advanced_prompting.py"]
    
    if verbose:
        cmd.append("-v")
    
    if failfast:
        cmd.append("--failfast")
    
    # Run tests
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Failed to run tests: {e}")
        return False


def run_fast_tests(verbose=True, failfast=False):
    """Run the fast test suite with mocked API calls"""
    print("\n⚡ Running FAST Test Suite (Mocked API calls)...")
    
    # Prepare command
    cmd = [sys.executable, "test_advanced_prompting_fast.py"]
    
    if verbose:
        cmd.append("-v")
    
    if failfast:
        cmd.append("--failfast")
    
    # Run tests
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Failed to run fast tests: {e}")
        return False


def run_specific_test(test_name):
    """Run a specific test case"""
    print(f"\n🎯 Running Specific Test: {test_name}")
    
    # Map friendly names to actual test methods
    test_mapping = {
        "api": "test_01_api_key_validation",
        "few-shot": "test_02_few_shot_learning", 
        "chain-of-thought": "test_03_chain_of_thought_reasoning",
        "prompts": "test_04_prompt_templates",
        "rate-limits": "test_05_rate_limit_handling",
        "async": "TestAsyncFunctionality"
    }
    
    actual_test = test_mapping.get(test_name, test_name)
    
    cmd = [
        sys.executable, "-m", "unittest", 
        f"test_advanced_prompting.TestAdvancedPromptingTechniques.{actual_test}",
        "-v"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Failed to run specific test: {e}")
        return False


def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(
        description="Test Runner for Advanced Prompting Techniques",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                    # Run environment check + full tests
  python run_tests.py --quick            # Run quick smoke test only
  python run_tests.py --check            # Check environment setup only
  python run_tests.py --test api         # Run specific test
  python run_tests.py --failfast         # Stop on first failure
        """
    )
    
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Run quick smoke test only"
    )
    
    parser.add_argument(
        "--fast", "-F",
        action="store_true",
        help="Run fast test suite with mocked API calls (~10-15 seconds)"
    )
    
    parser.add_argument(
        "--check", "-c",
        action="store_true", 
        help="Check environment setup only"
    )
    
    parser.add_argument(
        "--test", "-t",
        choices=["api", "few-shot", "chain-of-thought", "prompts", "rate-limits", "async"],
        help="Run a specific test case"
    )
    
    parser.add_argument(
        "--failfast", "-f",
        action="store_true",
        help="Stop on first test failure"
    )
    
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full test suite with real API calls (slow ~100+ seconds)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=True,
        help="Verbose output (default: True)"
    )
    
    args = parser.parse_args()
    
    print("🧪 ADVANCED PROMPTING TECHNIQUES - TEST RUNNER")
    print("=" * 60)
    
    # Always check environment first
    issues = check_environment()
    
    if issues:
        print(f"\n⚠️  Environment Issues Found ({len(issues)}):")
        for issue in issues:
            print(f"   {issue}")
        
        if not args.check:
            print("\n💡 Fix these issues before running tests:")
            print("   1. Create .env file: cp .env.example .env")
            print("   2. Add your API key to .env file")
            print("   3. Install dependencies: pip install -r requirements.txt")
        
        if "GEMINI_API_KEY" in str(issues):
            return 1
    
    if args.check:
        print(f"\n✅ Environment check complete - {len(issues)} issues found")
        return 0 if len(issues) == 0 else 1
    
    # Run tests based on arguments
    success = True
    
    if args.quick:
        success = run_quick_test()
    elif args.fast:
        success = run_fast_tests(verbose=args.verbose, failfast=args.failfast)
    elif args.test:
        success = run_specific_test(args.test)
    else:
        # Default: Run quick test first, then fast tests (not slow full tests)
        if run_quick_test():
            print("\n💡 Running FAST tests by default (use --full for complete suite)")
            success = run_fast_tests(verbose=args.verbose, failfast=args.failfast)
        else:
            print("❌ Quick test failed - skipping test suite")
            success = False
    
    # Final summary
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("Your Advanced Prompting Techniques implementation is working correctly.")
    else:
        print("❌ SOME TESTS FAILED")
        print("Check the output above for details on what needs to be fixed.")
    
    print("=" * 60)
    return 0 if success else 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️  Test runner interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Test runner failed: {e}")
        sys.exit(1)