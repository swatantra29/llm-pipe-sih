"""
Demonstration of Enhanced LLM Pipeline
Shows how the pipeline works from context with dynamic function discovery
"""

import os
import json
from dotenv import load_dotenv
import polars as pl

from analyzer import AtmosphericDataAnalyzer, FunctionExecutor
from function_registry import create_analyzer_registry
from llm_pipeline import EnhancedLLMPipeline


def print_header(title: str):
    """Print a formatted header"""
    print()
    print("=" * 70)
    print(f"{title.center(70)}")
    print("=" * 70)
    print()


def demo_dynamic_discovery():
    """Demonstrate dynamic function discovery"""
    print_header("DEMO 1: Dynamic Function Discovery")
    
    print("Loading data and creating analyzer...")
    df = pl.read_parquet("forecasted_data.parquet")
    analyzer = AtmosphericDataAnalyzer(df)
    
    print("Creating function registry (automatic discovery)...")
    registry = create_analyzer_registry(analyzer)
    
    print(f"\n✓ Automatically discovered {len(registry.list_functions())} functions:\n")
    
    for func_name in registry.list_functions():
        metadata = registry.get_function(func_name)
        print(f"  📋 {func_name}")
        print(f"     Purpose: {metadata.description}")
        print(f"     Parameters: {len(metadata.parameters)}")
        print(f"     Examples: {len(metadata.examples)}")
        if metadata.domain_knowledge:
            print(f"     Domain knowledge: {metadata.domain_knowledge[:60]}...")
        print()
    
    print("✅ Functions are discovered and documented automatically!")
    print("   No hardcoding required - everything from introspection.")


def demo_context_driven_reasoning():
    """Demonstrate context-driven query processing"""
    print_header("DEMO 2: Context-Driven Reasoning")
    
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key or api_key == "your_gemini_api_key_here":
        print("⚠️  Skipping: GEMINI_API_KEY not set in .env file")
        return
    
    print("Initializing enhanced pipeline...")
    df = pl.read_parquet("forecasted_data.parquet")
    analyzer = AtmosphericDataAnalyzer(df)
    executor = FunctionExecutor(analyzer)
    registry = create_analyzer_registry(analyzer)
    pipeline = EnhancedLLMPipeline(api_key, registry)
    data_summary = analyzer.get_data_summary()
    
    print("✓ Pipeline initialized with dynamic system prompt")
    print()
    
    query = "What's the average PM2.5 concentration?"
    
    print(f"User Query: \"{query}\"\n")
    print("Step 1: LLM analyzes query from context only...")
    
    llm_response = pipeline.generate_function_calls(query, data_summary)
    
    if not llm_response.get('success'):
        print(f"Error: {llm_response.get('error')}")
        return
    
    parsed = llm_response['parsed_calls']
    
    print("\n✓ Query Analysis (generated from context):")
    if 'query_analysis' in parsed:
        for key, value in parsed['query_analysis'].items():
            print(f"   {key}: {value}")
    
    print(f"\n✓ Reasoning: {parsed.get('reasoning', 'N/A')}")
    
    print(f"\n✓ Function Calls Generated:")
    for fc in parsed.get('function_calls', []):
        print(f"   Function: {fc['function']}")
        print(f"   Arguments: {json.dumps(fc['arguments'], indent=6)}")
        print(f"   Purpose: {fc.get('purpose', 'N/A')}")
    
    print("\n✅ LLM determined what to do entirely from context!")
    print("   No hardcoded rules - just system prompt + user query")


def demo_multi_step_processing():
    """Demonstrate multi-step query processing"""
    print_header("DEMO 3: Multi-Step Query Processing")
    
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key or api_key == "your_gemini_api_key_here":
        print("⚠️  Skipping: GEMINI_API_KEY not set in .env file")
        return
    
    df = pl.read_parquet("forecasted_data.parquet")
    analyzer = AtmosphericDataAnalyzer(df)
    executor = FunctionExecutor(analyzer)
    registry = create_analyzer_registry(analyzer)
    pipeline = EnhancedLLMPipeline(api_key, registry)
    data_summary = analyzer.get_data_summary()
    
    # Complex query requiring multiple functions
    query = "What's the correlation between temperature and O3, and what are the peak O3 values?"
    
    print(f"Complex Query: \"{query}\"\n")
    print("This query requires:")
    print("  1. Correlation analysis (compute_correlation)")
    print("  2. Peak detection (identify_peaks)")
    print()
    
    print("Processing...")
    result = pipeline.process_query(query, data_summary, executor)
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    
    print(f"\n✓ Functions called: {len(result['function_calls'])}")
    for fc in result['function_calls']:
        print(f"   - {fc['function']}()")
    
    print("\n✓ All functions executed successfully")
    
    print("\nFinal Answer:")
    print("-" * 70)
    answer = result['answer']
    # Print first 500 chars
    print(answer[:500])
    if len(answer) > 500:
        print("...")
    print("-" * 70)
    
    print("\n✅ Multi-step query processed successfully!")


def demo_self_contained():
    """Demonstrate self-contained operation"""
    print_header("DEMO 4: Self-Contained & Context-Independent")
    
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key or api_key == "your_gemini_api_key_here":
        print("⚠️  Skipping: GEMINI_API_KEY not set in .env file")
        return
    
    print("Key Features of Enhanced Pipeline:\n")
    
    print("✓ Dynamic Function Discovery")
    print("  - Functions automatically introspected from analyzer")
    print("  - Signatures extracted automatically")
    print("  - No manual documentation needed\n")
    
    print("✓ Auto-Generated System Prompts")
    print("  - Prompt built dynamically from function registry")
    print("  - Includes parameter types, descriptions, examples")
    print("  - Domain knowledge included automatically\n")
    
    print("✓ Context-Driven Reasoning")
    print("  - LLM works from system prompt + user query only")
    print("  - No hardcoded logic for query interpretation")
    print("  - Reasoning protocol guides LLM's thought process\n")
    
    print("✓ Structured Output")
    print("  - Query analysis phase (intent, variables, scope)")
    print("  - Planning phase (which functions to call)")
    print("  - Execution phase (run functions)")
    print("  - Synthesis phase (natural language answer)\n")
    
    print("✓ Validation")
    print("  - Ensures LLM doesn't perform calculations")
    print("  - Validates function calls are well-formed")
    print("  - Checks parameters match function signatures\n")
    
    print("✅ The pipeline is self-contained and works from context!")
    print("   Add new functions → They're auto-discovered")
    print("   Change data schema → System prompt auto-updates")
    print("   No code changes needed for new capabilities")


def main():
    """Run all demonstrations"""
    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 12 + "ENHANCED LLM PIPELINE DEMONSTRATION" + " " * 21 + "║")
    print("║" + " " * 15 + "Context-Driven Function Calling" + " " * 22 + "║")
    print("╚" + "=" * 68 + "╝")
    
    if not os.path.exists("forecasted_data.parquet"):
        print("\n❌ Error: forecasted_data.parquet not found")
        print("Please run: python generate_sample_data.py\n")
        return
    
    try:
        demo_dynamic_discovery()
        input("\nPress Enter to continue to next demo...")
        
        demo_context_driven_reasoning()
        input("\nPress Enter to continue to next demo...")
        
        demo_multi_step_processing()
        input("\nPress Enter to continue to next demo...")
        
        demo_self_contained()
        
        print()
        print("=" * 70)
        print("ALL DEMONSTRATIONS COMPLETED")
        print("=" * 70)
        print()
        print("Next steps:")
        print("  1. Try the API: python api.py")
        print("  2. Test endpoints: python test_enhanced_pipeline.py")
        print("  3. Use /query/enhanced endpoint for context-driven queries")
        print()
        
    except KeyboardInterrupt:
        print("\n\nDemonstration interrupted.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
