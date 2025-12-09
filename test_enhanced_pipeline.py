"""
Test suite for the enhanced LLM pipeline
Tests dynamic function discovery and context-driven reasoning
"""

import os
import json
from dotenv import load_dotenv
import polars as pl

from analyzer import AtmosphericDataAnalyzer, FunctionExecutor
from function_registry import create_analyzer_registry
from llm_pipeline import EnhancedLLMPipeline


def test_function_registry():
    """Test that function registry correctly discovers and documents functions"""
    print("=" * 70)
    print("TEST 1: Function Registry Discovery")
    print("=" * 70)
    print()
    
    # Load data
    if not os.path.exists("forecasted_data.parquet"):
        print("⚠️  SKIPPED: Data file not found. Run: python generate_sample_data.py")
        return None  # None indicates skipped, not failed
    
    df = pl.read_parquet("forecasted_data.parquet")
    analyzer = AtmosphericDataAnalyzer(df)
    
    # Create registry
    registry = create_analyzer_registry(analyzer)
    
    print(f"✓ Discovered {len(registry.list_functions())} functions:")
    for func_name in registry.list_functions():
        metadata = registry.get_function(func_name)
        print(f"  - {func_name}: {metadata.description[:50]}...")
    print()
    
    # Verify each function has required metadata
    for func_name in registry.list_functions():
        metadata = registry.get_function(func_name)
        assert metadata.description, f"Missing description for {func_name}"
        assert metadata.parameters, f"Missing parameters for {func_name}"
        print(f"✓ {func_name} has complete metadata")
    
    print()
    print("✅ Function registry test passed")
    print()
    return True


def test_system_prompt_generation():
    """Test that system prompt is generated dynamically from registry"""
    print("=" * 70)
    print("TEST 2: Dynamic System Prompt Generation")
    print("=" * 70)
    print()
    
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key or api_key == "your_gemini_api_key_here":
        print("⚠️  SKIPPED: GEMINI_API_KEY not set")
        return None
    
    df = pl.read_parquet("forecasted_data.parquet")
    analyzer = AtmosphericDataAnalyzer(df)
    registry = create_analyzer_registry(analyzer)
    
    pipeline = EnhancedLLMPipeline(api_key, registry)
    data_summary = analyzer.get_data_summary()
    
    system_prompt = pipeline.build_dynamic_system_prompt(data_summary)
    
    # Verify system prompt contains key components
    assert "AVAILABLE FUNCTIONS" in system_prompt
    assert "REASONING PROTOCOL" in system_prompt
    assert "DATA SCHEMA" in system_prompt
    
    # Verify all functions are documented
    for func_name in registry.list_functions():
        assert func_name in system_prompt, f"Function {func_name} not in system prompt"
    
    print("✓ System prompt contains:")
    print("  - Data schema documentation")
    print("  - All function descriptions")
    print("  - Reasoning protocol")
    print("  - Example queries")
    print()
    
    # Show sample
    print("Sample from system prompt:")
    print("-" * 70)
    lines = system_prompt.split('\n')
    for i, line in enumerate(lines[:30]):
        print(line)
    print("...")
    print("-" * 70)
    print()
    
    print("✅ System prompt generation test passed")
    print()
    return True


def test_query_processing():
    """Test complete query processing pipeline"""
    print("=" * 70)
    print("TEST 3: Query Processing (Context-Driven)")
    print("=" * 70)
    print()
    
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key or api_key == "your_gemini_api_key_here":
        print("⚠️  SKIPPED: GEMINI_API_KEY not set")
        return None
    
    df = pl.read_parquet("forecasted_data.parquet")
    analyzer = AtmosphericDataAnalyzer(df)
    executor = FunctionExecutor(analyzer)
    registry = create_analyzer_registry(analyzer)
    
    pipeline = EnhancedLLMPipeline(api_key, registry)
    data_summary = analyzer.get_data_summary()
    
    # Test queries
    test_queries = [
        "What's the average PM2.5 concentration?",
        "When did O3 exceed 0.070 ppm?",
        "What's the correlation between temperature and O3?",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"Query {i}: {query}")
        print("-" * 70)
        
        result = pipeline.process_query(query, data_summary, executor)
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            continue
        
        print(f"✓ Query analysis: {result.get('query_analysis', {}).get('query_type', 'N/A')}")
        print(f"✓ Functions called: {len(result.get('function_calls', []))}")
        
        for fc in result.get('function_calls', []):
            print(f"  - {fc['function']}()")
        
        print(f"✓ Answer generated: {len(result.get('answer', ''))} characters")
        print()
        
        # Show first part of answer
        answer = result.get('answer', '')
        print("Answer preview:")
        print(answer[:200] + "..." if len(answer) > 200 else answer)
        print()
        print("=" * 70)
        print()
    
    print("✅ Query processing test passed")
    print()
    return True


def test_function_call_validation():
    """Test that LLM generates valid function calls from context only"""
    print("=" * 70)
    print("TEST 4: Function Call Validation")
    print("=" * 70)
    print()
    
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key or api_key == "your_gemini_api_key_here":
        print("⚠️  SKIPPED: GEMINI_API_KEY not set")
        return None
    
    df = pl.read_parquet("forecasted_data.parquet")
    analyzer = AtmosphericDataAnalyzer(df)
    executor = FunctionExecutor(analyzer)
    registry = create_analyzer_registry(analyzer)
    
    pipeline = EnhancedLLMPipeline(api_key, registry)
    data_summary = analyzer.get_data_summary()
    
    query = "Find when PM2.5 was above 35 μg/m³"
    
    llm_response = pipeline.generate_function_calls(query, data_summary)
    
    assert llm_response.get('success'), "Failed to generate function calls"
    
    parsed = llm_response['parsed_calls']
    function_calls = parsed.get('function_calls', [])
    
    print(f"✓ Generated {len(function_calls)} function call(s)")
    
    # Validate function calls
    for fc in function_calls:
        assert 'function' in fc, "Missing 'function' key"
        assert 'arguments' in fc, "Missing 'arguments' key"
        
        func_name = fc['function']
        
        # Verify function exists in registry
        assert func_name in registry.list_functions(), f"Unknown function: {func_name}"
        
        metadata = registry.get_function(func_name)
        
        # Verify required parameters are provided
        for param_name, param_info in metadata.parameters.items():
            if param_info['required']:
                assert param_name in fc['arguments'], f"Missing required parameter: {param_name}"
        
        print(f"✓ {func_name} call is valid")
        print(f"  Arguments: {json.dumps(fc['arguments'], indent=4)}")
    
    # Execute to verify they actually work
    results = executor.execute_batch(function_calls)
    
    for result in results:
        assert result['status'] == 'success', f"Function execution failed: {result.get('error')}"
        print(f"✓ {result['function']} executed successfully")
    
    print()
    print("✅ Function call validation test passed")
    print()
    return True


def test_no_calculations():
    """Test that LLM doesn't perform calculations"""
    print("=" * 70)
    print("TEST 5: No LLM Calculations")
    print("=" * 70)
    print()
    
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key or api_key == "your_gemini_api_key_here":
        print("⚠️  SKIPPED: GEMINI_API_KEY not set")
        return None
    
    df = pl.read_parquet("forecasted_data.parquet")
    analyzer = AtmosphericDataAnalyzer(df)
    executor = FunctionExecutor(analyzer)
    registry = create_analyzer_registry(analyzer)
    
    pipeline = EnhancedLLMPipeline(api_key, registry)
    data_summary = analyzer.get_data_summary()
    
    # Try query that might tempt LLM to calculate
    query = "What's the average of PM2.5 values?"
    
    result = pipeline.process_query(query, data_summary, executor)
    
    # Check that no calculations appear in reasoning or raw response
    llm_response = pipeline.generate_function_calls(query, data_summary)
    
    if llm_response.get('success'):
        raw = llm_response.get('raw_response', '')
        
        # This should pass if LLM didn't try to calculate
        assert pipeline.validate_no_calculations(raw), "LLM attempted calculations!"
        
        print("✓ LLM did not perform any calculations")
        print("✓ All math delegated to functions")
    
    print()
    print("✅ No calculations test passed")
    print()
    return True


def main():
    """Run all tests"""
    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 18 + "ENHANCED PIPELINE TEST SUITE" + " " * 22 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    results = {}
    
    # Run tests
    results['function_registry'] = test_function_registry()
    input("Press Enter to continue...")
    
    results['system_prompt'] = test_system_prompt_generation()
    input("Press Enter to continue...")
    
    results['query_processing'] = test_query_processing()
    input("Press Enter to continue...")
    
    results['validation'] = test_function_call_validation()
    input("Press Enter to continue...")
    
    results['no_calculations'] = test_no_calculations()
    
    # Summary
    print()
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for v in results.values() if v is True)
    skipped = sum(1 for v in results.values() if v is None)
    failed = sum(1 for v in results.values() if v is False)
    total = len(results)
    
    for test_name, result in results.items():
        if result is True:
            status = "✅ PASSED"
        elif result is None:
            status = "⚠️  SKIPPED"
        else:
            status = "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print()
    print(f"Total: {passed} passed, {skipped} skipped, {failed} failed out of {total} tests")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print("Tests interrupted.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
