"""
Demonstration of Enhanced Pipeline Core Features
Shows dynamic function discovery without requiring API key
"""

import polars as pl
from analyzer import AtmosphericDataAnalyzer, FunctionExecutor
from function_registry import create_analyzer_registry


def demo_function_discovery():
    """Show how functions are automatically discovered"""
    print("=" * 70)
    print("DEMO: Dynamic Function Discovery")
    print("=" * 70)
    print()
    
    # Load data
    print("1. Loading data...")
    df = pl.read_parquet("forecasted_data.parquet")
    analyzer = AtmosphericDataAnalyzer(df)
    print(f"   ✓ Loaded {len(df)} records\n")
    
    # Create registry - this is where the magic happens
    print("2. Creating function registry (automatic discovery)...")
    registry = create_analyzer_registry(analyzer)
    print(f"   ✓ Discovered {len(registry.list_functions())} functions automatically\n")
    
    # Show what was discovered
    print("3. Discovered functions:\n")
    for i, func_name in enumerate(registry.list_functions(), 1):
        metadata = registry.get_function(func_name)
        print(f"   {i}. {func_name}")
        print(f"      Description: {metadata.description}")
        print(f"      Parameters: {len(metadata.parameters)}")
        
        # Show parameter details
        for param_name, param_info in list(metadata.parameters.items())[:3]:
            req_str = "required" if param_info['required'] else "optional"
            print(f"        - {param_name}: {param_info['type']} ({req_str})")
        
        if len(metadata.parameters) > 3:
            print(f"        ... and {len(metadata.parameters) - 3} more parameters")
        
        print(f"      Examples: {len(metadata.examples)}")
        if metadata.examples:
            print(f"        - {metadata.examples[0]}")
        
        if metadata.domain_knowledge:
            print(f"      Domain Knowledge: {metadata.domain_knowledge[:60]}...")
        print()
    
    print("=" * 70)
    print("KEY INSIGHT: No manual documentation needed!")
    print("Functions are introspected and documented automatically.")
    print("=" * 70)


def demo_system_prompt_generation():
    """Show auto-generated system prompt"""
    print()
    print("=" * 70)
    print("DEMO: Auto-Generated System Prompt")
    print("=" * 70)
    print()
    
    df = pl.read_parquet("forecasted_data.parquet")
    analyzer = AtmosphericDataAnalyzer(df)
    registry = create_analyzer_registry(analyzer)
    
    print("1. Generating system prompt from registry...")
    prompt_section = registry.generate_system_prompt_section()
    print(f"   ✓ Generated {len(prompt_section)} characters of documentation\n")
    
    print("2. Sample of auto-generated documentation:\n")
    print("-" * 70)
    
    # Show first function's documentation
    lines = prompt_section.split('\n')
    for line in lines[:40]:
        print(line)
    print("...")
    print("-" * 70)
    print()
    
    print("=" * 70)
    print("KEY INSIGHT: System prompt builds itself from code!")
    print("Add function → Register it → Prompt updates automatically")
    print("=" * 70)


def demo_function_execution():
    """Show that functions actually work"""
    print()
    print("=" * 70)
    print("DEMO: Function Execution")
    print("=" * 70)
    print()
    
    df = pl.read_parquet("forecasted_data.parquet")
    analyzer = AtmosphericDataAnalyzer(df)
    executor = FunctionExecutor(analyzer)
    
    print("Executing discovered functions with real data:\n")
    
    # Test 1: Statistics
    print("1. compute_statistics for PM2.5:")
    result = executor.execute_batch([{
        "function": "compute_statistics",
        "arguments": {
            "column": "PM25_ugm3",
            "metrics": ["mean", "max", "min", "std"]
        }
    }])
    
    if result[0]['status'] == 'success':
        output = result[0]['output']
        print(f"   ✓ Mean: {output['mean']:.2f} μg/m³")
        print(f"   ✓ Max: {output['max']:.2f} μg/m³ at {output['max_timestamp']}")
        print(f"   ✓ Min: {output['min']:.2f} μg/m³")
        print(f"   ✓ Std: {output['std']:.2f} μg/m³")
    print()
    
    # Test 2: Correlation
    print("2. compute_correlation between temperature and O3:")
    result = executor.execute_batch([{
        "function": "compute_correlation",
        "arguments": {
            "col1": "temperature_C",
            "col2": "O3_ppm"
        }
    }])
    
    if result[0]['status'] == 'success':
        output = result[0]['output']
        print(f"   ✓ Correlation: {output['correlation']:.3f}")
        print(f"   ✓ P-value: {output['p_value']:.4f}")
        print(f"   ✓ Sample size: {output['n_samples']}")
        
        if output['correlation'] > 0.5:
            print(f"   → Strong positive correlation!")
            print(f"   → Higher temperature = More O3 (photochemical formation)")
    print()
    
    # Test 3: Exceedances
    print("3. find_exceedance_events for PM2.5 above EPA threshold:")
    result = executor.execute_batch([{
        "function": "find_exceedance_events",
        "arguments": {
            "column": "PM25_ugm3",
            "threshold": 35.0,
            "operator": ">",
            "duration_hours": 1
        }
    }])
    
    if result[0]['status'] == 'success':
        output = result[0]['output']
        print(f"   ✓ Found {output['total_events']} exceedance events")
        
        for i, event in enumerate(output['events'][:3], 1):
            print(f"   Event {i}:")
            print(f"     Start: {event['start']}")
            print(f"     Duration: {event['duration_hours']} hours")
            print(f"     Peak: {event['peak_value']:.1f} μg/m³")
    print()
    
    print("=" * 70)
    print("KEY INSIGHT: Functions are real and produce actual results!")
    print("The pipeline connects LLM reasoning to real data analysis.")
    print("=" * 70)


def demo_registry_export():
    """Show registry can be exported"""
    print()
    print("=" * 70)
    print("DEMO: Registry Export")
    print("=" * 70)
    print()
    
    df = pl.read_parquet("forecasted_data.parquet")
    analyzer = AtmosphericDataAnalyzer(df)
    registry = create_analyzer_registry(analyzer)
    
    print("1. Exporting registry as JSON...")
    json_str = registry.to_json()
    print(f"   ✓ Exported {len(json_str)} characters\n")
    
    print("2. Sample of exported JSON:\n")
    print("-" * 70)
    print(json_str[:500])
    print("...")
    print("-" * 70)
    print()
    
    print("=" * 70)
    print("KEY INSIGHT: Registry is machine-readable!")
    print("Can be exported for documentation, APIs, or other tools.")
    print("=" * 70)


def main():
    """Run all demonstrations"""
    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 10 + "ENHANCED PIPELINE CORE FEATURES DEMO" + " " * 22 + "║")
    print("║" + " " * 18 + "(No API Key Required)" + " " * 29 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    print("This demo shows the core architecture improvements:")
    print("  1. Dynamic function discovery via introspection")
    print("  2. Auto-generated system prompts")
    print("  3. Real function execution with actual data")
    print("  4. Machine-readable registry export")
    print()
    
    input("Press Enter to start...")
    
    try:
        demo_function_discovery()
        input("\nPress Enter to continue...")
        
        demo_system_prompt_generation()
        input("\nPress Enter to continue...")
        
        demo_function_execution()
        input("\nPress Enter to continue...")
        
        demo_registry_export()
        
        print()
        print("=" * 70)
        print("DEMONSTRATIONS COMPLETED")
        print("=" * 70)
        print()
        print("Summary:")
        print("  ✓ Functions automatically discovered from code")
        print("  ✓ System prompts generated dynamically")
        print("  ✓ LLM can work entirely from context")
        print("  ✓ No hardcoded query logic needed")
        print()
        print("Next steps:")
        print("  - Set GEMINI_API_KEY in .env to enable LLM queries")
        print("  - Run: python demo_enhanced.py (full pipeline with LLM)")
        print("  - Run: python test_enhanced_pipeline.py (test suite)")
        print("  - Start API and try: POST /query/enhanced")
        print()
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
