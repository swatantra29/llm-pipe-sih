# Enhanced LLM Pipeline Documentation

## Overview

The enhanced LLM pipeline is a context-driven system that works primarily from information provided in system prompts and user queries, with minimal hardcoded logic. It features dynamic function discovery and automatic documentation generation.

## Key Features

### 1. Dynamic Function Discovery

Functions are automatically discovered through introspection rather than manual registration:

```python
from analyzer import AtmosphericDataAnalyzer
from function_registry import create_analyzer_registry

# Load analyzer
analyzer = AtmosphericDataAnalyzer(df)

# Automatically discover and register all functions
registry = create_analyzer_registry(analyzer)

# All functions are now available with full metadata
print(f"Discovered {len(registry.list_functions())} functions")
```

**What gets discovered:**
- Function signatures (parameters, types, defaults)
- Parameter requirements (required vs. optional)
- Return types
- Descriptions and examples
- Domain-specific knowledge

### 2. Auto-Generated System Prompts

System prompts are built dynamically from the function registry:

```python
from llm_pipeline import EnhancedLLMPipeline

pipeline = EnhancedLLMPipeline(api_key, registry)
system_prompt = pipeline.build_dynamic_system_prompt(data_summary)

# System prompt includes:
# - Data schema from actual dataframe
# - All function signatures and documentation
# - Reasoning protocols
# - Domain knowledge
# - Examples
```

**Benefits:**
- Add new function → Register it → Automatically appears in system prompt
- Change function signature → Documentation updates automatically
- No manual prompt maintenance needed

### 3. Context-Driven Reasoning

The LLM determines what to do entirely from context:

```python
# User query
query = "What's the correlation between temperature and O3?"

# LLM analyzes query from context only (no hardcoded rules)
result = pipeline.process_query(query, data_summary, executor)

# Result includes:
# - Query analysis (intent, variables, scope, conditions)
# - Reasoning (why these functions were chosen)
# - Function calls (generated from context)
# - Execution results
# - Natural language synthesis
```

**Reasoning Protocol:**

1. **Phase 1: Query Analysis**
   - Identify core question
   - Extract target variables
   - Determine temporal scope
   - Identify conditions
   - Classify query type

2. **Phase 2: Function Planning**
   - Map requirements to available functions
   - Determine execution order
   - Validate parameters

3. **Phase 3: Execution**
   - Run functions with real data
   - Collect results

4. **Phase 4: Synthesis**
   - Generate natural language answer
   - Cite specific results
   - Provide scientific context

### 4. Self-Documenting

The system maintains its own documentation:

```python
# Export registry as JSON
json_doc = registry.to_json()

# Get function metadata
metadata = registry.get_function("compute_statistics")
print(f"Description: {metadata.description}")
print(f"Parameters: {metadata.parameters}")
print(f"Examples: {metadata.examples}")
print(f"Domain knowledge: {metadata.domain_knowledge}")
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Query                                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│             Function Registry (Dynamic Discovery)             │
│  - Introspects analyzer methods                              │
│  - Extracts signatures, types, docs                          │
│  - Maintains metadata (descriptions, examples, domain info)   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│          Enhanced LLM Pipeline (Context-Driven)              │
│                                                               │
│  1. Build dynamic system prompt from registry                │
│  2. LLM analyzes query (intent, variables, scope)            │
│  3. LLM plans function calls (from context only)             │
│  4. Execute functions with real data                         │
│  5. LLM synthesizes natural language answer                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Natural Language Answer                    │
│  - Direct response to query                                   │
│  - Evidence from function results                             │
│  - Scientific context                                         │
│  - Data summary and caveats                                   │
└─────────────────────────────────────────────────────────────┘
```

## Adding New Functions

With the enhanced pipeline, adding functions is straightforward:

### Step 1: Implement the function

```python
# In analyzer.py
def compute_aqi(self, column: str, pollutant_type: str) -> Dict[str, Any]:
    """Calculate Air Quality Index"""
    # Implementation
    return {"aqi": aqi_value, "category": category}
```

### Step 2: Register with metadata

```python
# In function_registry.py (create_analyzer_registry function)
registry.register(
    analyzer.compute_aqi,
    description="Calculate Air Quality Index from pollutant concentration",
    examples=[
        "Calculate AQI for PM2.5",
        "Get AQI category for O3 levels"
    ],
    domain_knowledge="AQI scale: 0-50 Good, 51-100 Moderate, 101-150 Unhealthy for sensitive groups, 151-200 Unhealthy, 201-300 Very Unhealthy, 301+ Hazardous"
)
```

### Step 3: Done!

The function is now:
- ✅ Available to the LLM
- ✅ Documented in system prompt
- ✅ Callable via `/execute/functions`
- ✅ Usable in natural language queries

No code changes needed elsewhere!

## API Endpoints

### Enhanced Pipeline Endpoints

#### `POST /query/enhanced`

Full pipeline with detailed results:

```bash
curl -X POST http://localhost:8000/query/enhanced \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the correlation between temperature and O3?"
  }'
```

Response includes:
- Query analysis (intent, variables, scope, conditions, type)
- Reasoning (why these functions were chosen)
- Function calls (generated from context)
- Function results (actual data)
- Natural language answer (synthesis)

#### `POST /query/enhanced/simple`

Simplified endpoint (answer only):

```bash
curl -X POST "http://localhost:8000/query/enhanced/simple?query=What%20is%20the%20average%20PM2.5?"
```

Response:
```json
{
  "answer": "The average PM2.5 concentration is 14.71 μg/m³..."
}
```

### Legacy Endpoints (Still Available)

- `POST /query` - Original pipeline (backward compatible)
- `POST /query/simple` - Original simple query
- `POST /execute/functions` - Direct function execution
- `GET /data/summary` - Dataset metadata
- `GET /data/columns` - Available columns

## Examples

### Example 1: Simple Statistical Query

**Query:** "What's the average O3 concentration?"

**LLM Analysis:**
```json
{
  "query_analysis": {
    "core_question": "Calculate mean O3 concentration",
    "target_variables": ["O3_ppm"],
    "temporal_scope": "entire dataset",
    "conditions": [],
    "query_type": "statistical"
  },
  "reasoning": "User requests a single statistical metric (mean) for O3. The compute_statistics function handles this.",
  "function_calls": [
    {
      "function": "compute_statistics",
      "arguments": {
        "column": "O3_ppm",
        "metrics": ["mean"]
      }
    }
  ]
}
```

### Example 2: Correlation Analysis

**Query:** "What's the relationship between temperature and ozone?"

**LLM Analysis:**
```json
{
  "query_analysis": {
    "core_question": "Analyze correlation between temperature and O3",
    "target_variables": ["temperature_C", "O3_ppm"],
    "temporal_scope": "entire dataset",
    "conditions": [],
    "query_type": "correlation"
  },
  "reasoning": "User asks about relationship between two variables. Use compute_correlation to calculate Pearson correlation coefficient.",
  "function_calls": [
    {
      "function": "compute_correlation",
      "arguments": {
        "col1": "temperature_C",
        "col2": "O3_ppm",
        "lag_hours": 0
      }
    }
  ]
}
```

### Example 3: Complex Multi-Step Query

**Query:** "What are the peak O3 values and how do they correlate with temperature?"

**LLM Analysis:**
```json
{
  "query_analysis": {
    "core_question": "Identify O3 peaks and analyze temperature correlation",
    "target_variables": ["O3_ppm", "temperature_C"],
    "temporal_scope": "entire dataset",
    "conditions": [],
    "query_type": "temporal and correlation"
  },
  "reasoning": "Query requires two operations: 1) identify peaks in O3, 2) compute correlation with temperature.",
  "function_calls": [
    {
      "function": "identify_peaks",
      "arguments": {
        "column": "O3_ppm",
        "peak_type": "max",
        "prominence": 0.1
      }
    },
    {
      "function": "compute_correlation",
      "arguments": {
        "col1": "temperature_C",
        "col2": "O3_ppm"
      }
    }
  ]
}
```

## Testing

### Run Full Test Suite

```bash
python test_enhanced_pipeline.py
```

Tests:
1. ✅ Function registry discovery
2. ✅ System prompt generation
3. ✅ Query processing
4. ✅ Function call validation
5. ✅ No LLM calculations

### Run Demonstrations

```bash
# Core features (no API key needed)
python demo_no_api.py

# Full pipeline with LLM (requires API key)
python demo_enhanced.py
```

## Comparison: Legacy vs Enhanced

| Feature | Legacy Pipeline | Enhanced Pipeline |
|---------|----------------|-------------------|
| Function Discovery | Manual | Automatic (introspection) |
| System Prompt | Static, hardcoded | Dynamic, auto-generated |
| Adding Functions | Update 3+ files | Register in 1 place |
| Query Logic | Mixed (some hardcoded) | Pure context-driven |
| Documentation | Manual maintenance | Self-documenting |
| Reasoning | Implicit | Explicit multi-phase |
| Extensibility | Moderate | High |

## Benefits

### For Developers

✅ **Less Code:** Add functions by registering metadata, not updating multiple files

✅ **Self-Documenting:** Documentation flows from code automatically

✅ **Easier Maintenance:** Changes propagate automatically

✅ **Better Testing:** Registry can be validated independently

### For the LLM

✅ **More Context:** Richer, more detailed function documentation

✅ **Clear Protocol:** Explicit reasoning phases guide thinking

✅ **Validation:** Parameter requirements enforced

✅ **Examples:** Concrete examples help with edge cases

### For Users

✅ **More Capable:** LLM can handle complex multi-step queries

✅ **More Transparent:** See query analysis and reasoning

✅ **More Reliable:** Validation ensures correct function calls

✅ **More Flexible:** Easy to extend with new capabilities

## Design Philosophy

The enhanced pipeline follows key principles:

1. **Work from Context:** LLM determines actions from system prompt and user query, not hardcoded logic

2. **Single Source of Truth:** Function signatures are the source of truth; documentation derives from them

3. **Explicit Reasoning:** Multi-phase protocol makes LLM's thinking visible and debuggable

4. **Fail Fast:** Validate function calls before execution to catch errors early

5. **Scientific Rigor:** All calculations done by verified functions, not LLM

## Future Enhancements

Potential improvements:

- **Adaptive Planning:** LLM can request intermediate results and adjust plan
- **Function Dependencies:** Auto-detect when functions need to be chained
- **Caching:** Reuse function results for similar queries
- **Explanation:** Generate explanations of reasoning process
- **Learning:** Track successful query patterns to improve future performance

## Conclusion

The enhanced pipeline demonstrates how LLMs can work effectively from context with minimal hardcoded logic. By combining:

- Dynamic function discovery
- Auto-generated documentation
- Context-driven reasoning
- Explicit multi-phase protocols

...we create a system that's both powerful and maintainable, with clear separation between the LLM's role (understanding and planning) and the system's role (execution and validation).
