"""
Example: Complete pipeline walkthrough
Demonstrates how the tool-augmented LLM system works
"""

import json
from analyzer import AtmosphericDataAnalyzer, FunctionExecutor
from llm_client import GeminiLLM
import polars as pl
import os
from dotenv import load_dotenv


def example_1_direct_function_execution():
    """Example 1: Using functions directly without LLM"""
    print("=" * 70)
    print("EXAMPLE 1: Direct Function Execution (No LLM)")
    print("=" * 70)
    print()
    
    # Load data
    df = pl.read_parquet("forecasted_data.parquet")
    analyzer = AtmosphericDataAnalyzer(df)
    executor = FunctionExecutor(analyzer)
    
    # Example: Get PM2.5 statistics
    print("Task: Calculate PM2.5 statistics")
    print()
    
    function_calls = [
        {
            "function": "compute_statistics",
            "arguments": {
                "column": "PM25_ugm3",
                "metrics": ["mean", "max", "min", "std"]
            }
        }
    ]
    
    print("Function Call:")
    print(json.dumps(function_calls[0], indent=2))
    print()
    
    results = executor.execute_batch(function_calls)
    
    print("Result:")
    print(json.dumps(results[0], indent=2))
    print()


def example_2_correlation_analysis():
    """Example 2: Temperature-O3 correlation"""
    print("=" * 70)
    print("EXAMPLE 2: Correlation Analysis")
    print("=" * 70)
    print()
    
    df = pl.read_parquet("forecasted_data.parquet")
    analyzer = AtmosphericDataAnalyzer(df)
    executor = FunctionExecutor(analyzer)
    
    print("Task: Analyze temperature-O3 relationship")
    print()
    
    function_calls = [
        {
            "function": "compute_correlation",
            "arguments": {
                "col1": "temperature_C",
                "col2": "O3_ppm",
                "lag_hours": 0
            }
        }
    ]
    
    results = executor.execute_batch(function_calls)
    
    result = results[0]['output']
    print(f"Correlation coefficient: {result['correlation']:.3f}")
    print(f"P-value: {result['p_value']:.4f}")
    print(f"Sample size: {result['n_samples']}")
    print()
    
    print("Interpretation:")
    if result['correlation'] > 0.5:
        print("  → Strong positive correlation: Higher temperature = More O3")
        print("  → Consistent with photochemical O3 formation")
    print()


def example_3_exceedance_detection():
    """Example 3: Find PM2.5 exceedances"""
    print("=" * 70)
    print("EXAMPLE 3: Exceedance Detection")
    print("=" * 70)
    print()
    
    df = pl.read_parquet("forecasted_data.parquet")
    analyzer = AtmosphericDataAnalyzer(df)
    executor = FunctionExecutor(analyzer)
    
    print("Task: Find when PM2.5 exceeded EPA threshold (35 μg/m³)")
    print()
    
    function_calls = [
        {
            "function": "find_exceedance_events",
            "arguments": {
                "column": "PM25_ugm3",
                "threshold": 35.0,
                "operator": ">",
                "duration_hours": 1
            }
        }
    ]
    
    results = executor.execute_batch(function_calls)
    
    events = results[0]['output']['events']
    print(f"Found {len(events)} exceedance event(s)")
    print()
    
    for i, event in enumerate(events, 1):
        print(f"Event {i}:")
        print(f"  Start: {event['start']}")
        print(f"  Duration: {event['duration_hours']} hours")
        print(f"  Peak: {event['peak_value']:.1f} μg/m³")
        print()


def example_4_lag_analysis():
    """Example 4: SO2 → PM2.5 lag analysis"""
    print("=" * 70)
    print("EXAMPLE 4: Precursor-Product Lag Analysis")
    print("=" * 70)
    print()
    
    df = pl.read_parquet("forecasted_data.parquet")
    analyzer = AtmosphericDataAnalyzer(df)
    executor = FunctionExecutor(analyzer)
    
    print("Task: Find lag between SO2 spikes and PM2.5 increases")
    print("(SO2 oxidizes to sulfate aerosol, contributing to PM2.5)")
    print()
    
    function_calls = [
        {
            "function": "compute_crossover_lag",
            "arguments": {
                "precursor_col": "SO2_ppm",
                "product_col": "PM25_ugm3",
                "max_lag_hours": 48
            }
        }
    ]
    
    results = executor.execute_batch(function_calls)
    
    result = results[0]['output']
    print(f"Optimal lag: {result['lag_hours']} hours")
    print(f"Correlation at lag: {result['correlation_coefficient']:.3f}")
    print()
    
    print("Interpretation:")
    print(f"  → PM2.5 increases {result['lag_hours']} hours after SO2 peaks")
    print(f"  → Typical atmospheric oxidation time: 6-24 hours")
    print()


def example_5_llm_pipeline():
    """Example 5: Complete LLM pipeline"""
    print("=" * 70)
    print("EXAMPLE 5: LLM-Powered Natural Language Query")
    print("=" * 70)
    print()
    
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key or api_key == "your_gemini_api_key_here":
        print("⚠️  Skipping: GEMINI_API_KEY not set in .env file")
        print()
        return
    
    # Initialize
    df = pl.read_parquet("forecasted_data.parquet")
    analyzer = AtmosphericDataAnalyzer(df)
    executor = FunctionExecutor(analyzer)
    llm_client = GeminiLLM(api_key)
    
    data_summary = analyzer.get_data_summary()
    system_prompt = llm_client.build_system_prompt(data_summary)
    
    # User query
    user_query = "What's the average O3 concentration, and when does it typically peak during the day?"
    
    print(f"User Query: {user_query}")
    print()
    
    # Step 1: LLM generates function calls
    print("Step 1: LLM analyzes query and generates function calls...")
    llm_response = llm_client.generate_function_calls(user_query, system_prompt)
    
    if not llm_response.get("success"):
        print(f"Error: {llm_response.get('error')}")
        return
    
    parsed_calls = llm_response['parsed_calls']
    function_calls = parsed_calls.get('function_calls', [])
    
    print(f"Reasoning: {parsed_calls.get('reasoning', 'N/A')}")
    print()
    print("Generated function calls:")
    for fc in function_calls:
        print(f"  - {fc['function']}({', '.join(f'{k}={v}' for k, v in fc['arguments'].items())})")
    print()
    
    # Step 2: Execute functions
    print("Step 2: Executing functions...")
    function_results = executor.execute_batch(function_calls)
    
    for result in function_results:
        if result['status'] == 'success':
            print(f"  ✓ {result['function']} succeeded")
        else:
            print(f"  ✗ {result['function']} failed: {result['error']}")
    print()
    
    # Step 3: LLM synthesizes results
    print("Step 3: LLM synthesizes natural language response...")
    synthesis = llm_client.synthesize_results(user_query, function_results, system_prompt)
    
    print()
    print("=" * 70)
    print("FINAL ANSWER:")
    print("=" * 70)
    print(synthesis)
    print()


def main():
    """Run all examples"""
    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "ATMOSPHERIC DATA ANALYSIS EXAMPLES" + " " * 19 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    # Check if data exists
    if not os.path.exists("forecasted_data.parquet"):
        print("❌ Error: forecasted_data.parquet not found")
        print()
        print("Please run: python generate_sample_data.py")
        return
    
    try:
        # Run examples
        example_1_direct_function_execution()
        input("Press Enter to continue to next example...")
        print()
        
        example_2_correlation_analysis()
        input("Press Enter to continue to next example...")
        print()
        
        example_3_exceedance_detection()
        input("Press Enter to continue to next example...")
        print()
        
        example_4_lag_analysis()
        input("Press Enter to continue to next example...")
        print()
        
        example_5_llm_pipeline()
        
        print("=" * 70)
        print("All examples completed!")
        print("=" * 70)
        print()
        
    except KeyboardInterrupt:
        print()
        print("Examples interrupted.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
