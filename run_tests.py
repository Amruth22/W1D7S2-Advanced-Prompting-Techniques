#!/usr/bin/env python3
"""
Test Runner for Advanced Prompting Techniques
Provides multiple test execution modes for different use cases
"""

import os
import sys
import subprocess
import time
from dotenv import load_dotenv

def print_banner():
    """Print test runner banner"""
    print("=" * 70)
    print("🧪 ADVANCED PROMPTING TECHNIQUES - TEST RUNNER")
    print("=" * 70)

def check_environment():
    """Check environment setup"""
    load_dotenv()
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key or not api_key.startswith('AIza'):
        print("❌ GEMINI_API_KEY not found or invalid!")
        print("\n📋 Setup Instructions:")
        print("1. Copy .env.example to .env: cp .env.example .env")
        print("2. Get API key: https://aistudio.google.com/")
        print("3. Add to .env: GEMINI_API_KEY=your-api-key-here")
        return False
    
    print(f"✅ API Key configured: {api_key[:10]}...{api_key[-5:]}")
    return True

def run_quick_tests():
    """Run quick tests with optimized performance"""
    print("\n🚀 QUICK TEST MODE")
    print("- Reduced API calls for faster execution")
    print("- Essential component validation")
    print("- Expected time: ~8-12 seconds")
    print("-" * 50)
    
    # Set environment variable for quick mode
    env = os.environ.copy()
    env['QUICK_TEST_MODE'] = 'true'
    env['MAX_API_CALLS_PER_TEST'] = '1'
    env['API_TIMEOUT'] = '10'
    
    start_time = time.time()
    result = subprocess.run([sys.executable, 'testsss.py'], env=env)
    total_time = time.time() - start_time
    
    print(f"\n⏱️  Quick tests completed in {total_time:.2f} seconds")
    return result.returncode == 0

def run_full_tests():
    """Run comprehensive tests with full API integration"""
    print("\n🔬 FULL TEST MODE")
    print("- Comprehensive API integration testing")
    print("- All advanced prompting techniques")
    print("- Expected time: ~20-30 seconds")
    print("-" * 50)
    
    # Set environment variable for full mode
    env = os.environ.copy()
    env['QUICK_TEST_MODE'] = 'false'
    env['MAX_API_CALLS_PER_TEST'] = '2'
    env['API_TIMEOUT'] = '30'
    
    start_time = time.time()
    result = subprocess.run([sys.executable, 'testsss.py'], env=env)
    total_time = time.time() - start_time
    
    print(f"\n⏱️  Full tests completed in {total_time:.2f} seconds")
    return result.returncode == 0

def run_legacy_tests():
    """Run legacy comprehensive tests (if available)"""
    legacy_files = ['test_advanced_prompting_fast.py', 'test_advanced_prompting.py']
    
    for legacy_file in legacy_files:
        if os.path.exists(legacy_file):
            print(f"\n📚 LEGACY TEST MODE - {legacy_file}")
            print("- Comprehensive mocked/real tests")
            print("- Original test implementation")
            print("-" * 50)
            
            start_time = time.time()
            result = subprocess.run([sys.executable, legacy_file])
            total_time = time.time() - start_time
            
            print(f"\n⏱️  Legacy tests completed in {total_time:.2f} seconds")
            return result.returncode == 0
    
    print("❌ Legacy test files not found")
    print("Available: test_advanced_prompting_fast.py, test_advanced_prompting.py")
    return False

def run_specific_test(test_name):
    """Run a specific test"""
    print(f"\n🎯 SPECIFIC TEST: {test_name}")
    print("-" * 50)
    
    env = os.environ.copy()
    env['QUICK_TEST_MODE'] = 'true'  # Use quick mode for specific tests
    
    cmd = [
        sys.executable, '-m', 'unittest', 
        f'testsss.CoreAdvancedPromptingTests.{test_name}', 
        '-v'
    ]
    
    start_time = time.time()
    result = subprocess.run(cmd, env=env)
    total_time = time.time() - start_time
    
    print(f"\n⏱️  Test {test_name} completed in {total_time:.2f} seconds")
    return result.returncode == 0

def run_technique_demo(technique):
    """Run a specific technique demonstration"""
    print(f"\n🎯 TECHNIQUE DEMO: {technique}")
    print("-" * 50)
    
    cmd = [sys.executable, 'main.py', '--technique', technique]
    
    start_time = time.time()
    result = subprocess.run(cmd)
    total_time = time.time() - start_time
    
    print(f"\n⏱️  Technique demo completed in {total_time:.2f} seconds")
    return result.returncode == 0

def show_usage():
    """Show usage instructions"""
    print("\n📖 USAGE:")
    print("python run_tests.py [mode]")
    print("\n🎯 Available modes:")
    print("  quick     - Fast validation (~8-12s)")
    print("  full      - Comprehensive testing (~20-30s)")
    print("  legacy    - Original test files")
    print("  specific  - Run specific test")
    print("  demo      - Run technique demonstration")
    print("\n💡 Examples:")
    print("  python run_tests.py quick")
    print("  python run_tests.py full")
    print("  python run_tests.py specific test_01_gemini_client_integration")
    print("  python run_tests.py demo few-shot")
    print("\n🔧 Environment Variables:")
    print("  QUICK_TEST_MODE=true/false")
    print("  MAX_API_CALLS_PER_TEST=1-3")
    print("  API_TIMEOUT=10-30")
    print("\n🧪 Available Tests:")
    print("  test_01_gemini_client_integration")
    print("  test_02_component_structure_validation")
    print("  test_03_prompt_templates_and_implementation")
    print("  test_04_advanced_prompting_methods_and_configuration")
    print("  test_05_integration_workflow_and_production_readiness")
    print("\n🚀 Available Techniques:")
    print("  few-shot, chain-of-thought, tree-of-thought")
    print("  self-consistency, meta-prompting")

def main():
    """Main test runner function"""
    print_banner()
    
    # Check environment
    if not check_environment():
        return False
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        show_usage()
        return False
    
    mode = sys.argv[1].lower()
    
    if mode == 'quick':
        return run_quick_tests()
    elif mode == 'full':
        return run_full_tests()
    elif mode == 'legacy':
        return run_legacy_tests()
    elif mode == 'specific':
        if len(sys.argv) < 3:
            print("❌ Please specify test name for specific mode")
            print("Example: python run_tests.py specific test_01_gemini_client_integration")
            return False
        return run_specific_test(sys.argv[2])
    elif mode == 'demo':
        if len(sys.argv) < 3:
            print("❌ Please specify technique for demo mode")
            print("Example: python run_tests.py demo few-shot")
            return False
        return run_technique_demo(sys.argv[2])
    else:
        print(f"❌ Unknown mode: {mode}")
        show_usage()
        return False

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n🎉 Tests completed successfully!")
        else:
            print("\n❌ Tests failed or incomplete")
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test runner error: {e}")
        sys.exit(1)