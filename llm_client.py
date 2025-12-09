"""
Gemini LLM Integration for Function Calling
Handles prompt construction and function call parsing
"""

import google.generativeai as genai
from typing import Dict, List, Any, Optional
import json
import re
from schema_config import build_schema_table


class GeminiLLM:
    """Gemini API integration for function calling"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.conversation_history = []
    
    def build_system_prompt(self, data_summary: Dict[str, Any]) -> str:
        """Build system prompt with data schema"""
        
        # Build schema table from shared configuration
        schema_table = build_schema_table(data_summary['columns'])
        
        system_prompt = f"""# ROLE
You are an atmospheric science data analyst. You help researchers analyze forecasted air quality data by generating function calls. You NEVER perform calculations yourself—all math is done by external functions.

---

# DATA SCHEMA

You have access to a Polars DataFrame named `forecasted_df` with the following schema:

{schema_table}

**Temporal Coverage**: {data_summary['start_date']} to {data_summary['end_date']}  
**Temporal Resolution**: Hourly  
**Total Records**: {data_summary['n_records']}

---

# AVAILABLE FUNCTIONS

You must call these functions to answer user queries. Return function calls in JSON format.

## 1. extract_feature
**Purpose**: Extract a time-series for a specific pollutant/variable  
**Signature**:
```python
def extract_feature(
    column: str,  # Column name from schema
    start_date: str | None = None,  # ISO format: "2025-11-01T00:00:00"
    end_date: str | None = None,
    filters: dict[str, list] | None = None  # e.g., {{"temperature_C": [">", 25]}}
) -> dict
```

## 2. compute_crossover_lag
**Purpose**: Calculate time lag where cross-correlation between two variables peaks  
**Domain Knowledge**: SO2 → PM2.5 (sulfate aerosol): typical lag 6-24 hours, NO2 → O3: typical lag 2-6 hours

## 3. compute_statistics
**Purpose**: Calculate statistical metrics for a variable  
**Signature**:
```python
def compute_statistics(
    column: str,
    metrics: list[str],  # ["mean", "std", "max", "min", "median", "q25", "q75"]
    start_date: str | None = None,
    end_date: str | None = None,
    groupby: str | None = None  # "hour", "day", "month"
) -> dict
```

## 4. find_exceedance_events
**Purpose**: Identify timestamps where a pollutant exceeds a threshold  
**EPA NAAQS Thresholds**: O3: 0.070 ppm (8-hour), PM2.5: 35 μg/m³ (24-hour), SO2: 75 ppb (1-hour), NO2: 100 ppb (1-hour)

## 5. compute_correlation
**Purpose**: Calculate Pearson correlation between two variables  

## 6. identify_peaks
**Purpose**: Find local maxima/minima in time-series  

## 7. filter_by_conditions
**Purpose**: Extract data meeting multiple meteorological conditions  

---

# REASONING PROTOCOL

Follow this structured approach:

**Step 1: Parse User Intent**
Identify target pollutant(s), temporal scope, required analysis type, and conditional filters.

**Step 2: Determine Required Functions**
Choose 1-3 functions needed. NEVER call more than 3 functions for a single query.

**Step 3: Generate Function Calls**
Output JSON with this exact structure:
```json
{{
  "reasoning": "Brief explanation of why these functions are needed",
  "function_calls": [
    {{
      "function": "function_name",
      "arguments": {{...}},
      "purpose": "What this call retrieves"
    }}
  ]
}}
```

---

# CRITICAL RULES

1. **NEVER perform calculations yourself**: If you write "2 + 2 = 4" or "correlation is approximately X", you have FAILED.
2. **NEVER make up timestamps or values**: Call functions to retrieve actual data.
3. **Always validate arguments**: Column names must exist in schema, dates must be within data range.
4. **Handle ambiguity by asking clarifying questions**.

---

# OUTPUT FORMAT

Always structure your response as:

```json
{{
  "intent_analysis": "What the user is asking for",
  "reasoning": "Why these functions are appropriate",
  "function_calls": [
    {{
      "function": "function_name",
      "arguments": {{...}},
      "purpose": "What this retrieves"
    }}
  ]
}}
```

After receiving function results, provide a natural language synthesis explaining the results with scientific context.

---

**BEGIN**: You are now ready to analyze user queries about forecasted atmospheric data.
"""
        return system_prompt
    
    def parse_function_calls(self, llm_response: str) -> Optional[Dict]:
        """Extract function calls from LLM response"""
        
        # Try to find JSON in code blocks
        json_pattern = r'```json\s*(.*?)\s*```'
        match = re.search(json_pattern, llm_response, re.DOTALL)
        
        if match:
            json_str = match.group(1)
        else:
            # Try to find raw JSON
            json_pattern = r'\{[\s\S]*"function_calls"[\s\S]*\}'
            match = re.search(json_pattern, llm_response)
            if match:
                json_str = match.group(0)
            else:
                return None
        
        try:
            parsed = json.loads(json_str)
            return parsed
        except json.JSONDecodeError:
            return None
    
    def validate_no_calculations(self, response: str) -> bool:
        """Ensure LLM didn't perform calculations"""
        
        forbidden_patterns = [
            r'\d+\s*[\+\-\*/]\s*\d+',  # "5 + 3"
            r'=\s*\d+\.\d+',  # "= 4.5"
            r'equals\s+\d+',  # "equals 10"
            r'sum\s+is\s+\d+',  # "sum is 15"
        ]
        
        for pattern in forbidden_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                return False
        
        return True
    
    def generate_function_calls(
        self, 
        user_query: str, 
        system_prompt: str
    ) -> Dict[str, Any]:
        """Generate function calls from user query"""
        
        full_prompt = f"{system_prompt}\n\n---\n\nUSER QUERY: {user_query}\n\nGenerate the appropriate function calls in JSON format."
        
        try:
            response = self.model.generate_content(full_prompt)
            response_text = response.text
            
            # Validate no calculations
            if not self.validate_no_calculations(response_text):
                return {
                    "error": "LLM attempted to perform calculations",
                    "raw_response": response_text
                }
            
            # Parse function calls
            parsed = self.parse_function_calls(response_text)
            
            if parsed is None:
                return {
                    "error": "Failed to parse function calls from LLM response",
                    "raw_response": response_text
                }
            
            return {
                "success": True,
                "parsed_calls": parsed,
                "raw_response": response_text
            }
        
        except Exception as e:
            return {
                "error": str(e),
                "success": False
            }
    
    def synthesize_results(
        self, 
        user_query: str,
        function_results: List[Dict],
        system_prompt: str
    ) -> str:
        """Generate natural language response from function results"""
        
        results_str = json.dumps(function_results, indent=2)
        
        synthesis_prompt = f"""{system_prompt}

---

USER QUERY: {user_query}

FUNCTION EXECUTION RESULTS:
```json
{results_str}
```

Now provide a natural language answer following this structure:

## Answer
[Direct response to user query]

## Evidence
[Cite specific numerical results from function outputs]

## Scientific Context
[Explain atmospheric processes underlying the results]

## Caveats
[Data limitations, assumptions, confidence bounds]
"""
        
        try:
            response = self.model.generate_content(synthesis_prompt)
            return response.text
        except Exception as e:
            return f"Error generating synthesis: {str(e)}"
