"""
Enhanced LLM Pipeline with Dynamic Function Discovery and Multi-Step Reasoning
Works primarily from context in system and user prompts
"""

import google.generativeai as genai
from typing import Dict, List, Any, Optional
import json
import re
from function_registry import FunctionRegistry
from schema_config import build_schema_table


class EnhancedLLMPipeline:
    """
    Enhanced LLM pipeline that:
    1. Automatically discovers available functions
    2. Generates system prompts dynamically
    3. Implements multi-step reasoning
    4. Works primarily from context without hardcoded logic
    """
    
    def __init__(self, api_key: str, function_registry: FunctionRegistry, model_name: str = "gemini-2.5-flash"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.registry = function_registry
        self.conversation_history = []
    
    def build_dynamic_system_prompt(self, data_summary: Dict[str, Any]) -> str:
        """Build system prompt dynamically from function registry and data schema"""
        
        # Build schema table from shared configuration
        schema_table = build_schema_table(data_summary['columns'])
        
        # Generate functions section from registry
        functions_section = self.registry.generate_system_prompt_section()
        
        system_prompt = f"""# ROLE
You are an atmospheric science data analyst AI. You analyze air quality data by generating function calls based on user queries. You work entirely from context provided in this prompt and the user's query.

**CRITICAL**: You NEVER perform calculations yourself. All mathematical operations are done by external functions.

---

# DATA SCHEMA

You have access to a dataset with the following schema:

{schema_table}

**Temporal Coverage**: {data_summary['start_date']} to {data_summary['end_date']}  
**Temporal Resolution**: Hourly  
**Total Records**: {data_summary['n_records']}

---

# AVAILABLE FUNCTIONS

You have access to the following functions. Analyze the user query and determine which function(s) to call based on the descriptions, parameters, and domain knowledge provided.

{functions_section}

---

# REASONING PROTOCOL

When you receive a user query, follow this structured reasoning process:

**Phase 1: Query Analysis**
- Identify the core question or request
- Extract target pollutants/variables mentioned
- Determine temporal scope (specific dates, time ranges, or entire dataset)
- Identify any conditional filters (e.g., "when temperature > 30°C")
- Classify the query type (statistical, correlation, exceedance, temporal pattern, conditional)

**Phase 2: Function Planning**
- Map query requirements to available functions
- Determine if single or multiple functions needed
- Plan function execution order if dependent
- Validate all required parameters can be satisfied

**Phase 3: Function Call Generation**
- Generate JSON with exact function signatures
- Ensure all required parameters are provided
- Use data schema column names exactly as shown
- Include reasoning for each function call

**Phase 4: Result Synthesis** (after function execution)
- Interpret numerical results with scientific context
- Explain atmospheric processes underlying the results
- Cite specific values from function outputs
- Note any data limitations or assumptions

---

# OUTPUT FORMAT

Always respond in this JSON structure:

```json
{{
  "query_analysis": {{
    "core_question": "What the user is asking",
    "target_variables": ["column1", "column2"],
    "temporal_scope": "date range or 'entire dataset'",
    "conditions": ["list any filtering conditions"],
    "query_type": "statistical|correlation|exceedance|temporal|conditional"
  }},
  "reasoning": "Detailed explanation of why these functions are appropriate",
  "function_calls": [
    {{
      "function": "function_name",
      "arguments": {{
        "param1": "value1",
        "param2": "value2"
      }},
      "purpose": "What this specific call retrieves"
    }}
  ]
}}
```

---

# CRITICAL RULES

1. **NO CALCULATIONS**: Never write arithmetic (e.g., "2 + 2 = 4", "average is approximately X"). Let functions do all math.
2. **EXACT COLUMN NAMES**: Use column names exactly as shown in schema (case-sensitive).
3. **VALIDATE PARAMETERS**: Ensure dates are within data range, columns exist in schema.
4. **MINIMAL FUNCTIONS**: Use 1-3 functions maximum per query. Choose the most direct path.
5. **WORK FROM CONTEXT**: All information needed is in this prompt. Don't make assumptions beyond what's provided.

---

# EXAMPLES

**Query**: "What's the average O3 concentration?"

**Response**:
```json
{{
  "query_analysis": {{
    "core_question": "Calculate mean O3 concentration",
    "target_variables": ["O3_ppm"],
    "temporal_scope": "entire dataset",
    "conditions": [],
    "query_type": "statistical"
  }},
  "reasoning": "User requests a single statistical metric (mean) for O3. The compute_statistics function handles this with metrics=['mean'].",
  "function_calls": [
    {{
      "function": "compute_statistics",
      "arguments": {{
        "column": "O3_ppm",
        "metrics": ["mean"]
      }},
      "purpose": "Calculate mean O3 concentration over entire dataset"
    }}
  ]
}}
```

**Query**: "When did PM2.5 exceed EPA standards?"

**Response**:
```json
{{
  "query_analysis": {{
    "core_question": "Identify PM2.5 exceedance events",
    "target_variables": ["PM25_ugm3"],
    "temporal_scope": "entire dataset",
    "conditions": ["threshold: 35 μg/m³ (EPA NAAQS)"],
    "query_type": "exceedance"
  }},
  "reasoning": "EPA NAAQS for PM2.5 is 35 μg/m³ (24-hour standard). Use find_exceedance_events with operator '>' to identify all instances.",
  "function_calls": [
    {{
      "function": "find_exceedance_events",
      "arguments": {{
        "column": "PM25_ugm3",
        "threshold": 35.0,
        "operator": ">",
        "duration_hours": 1
      }},
      "purpose": "Find all timestamps where PM2.5 exceeded 35 μg/m³"
    }}
  ]
}}
```

---

**BEGIN**: You are now ready to analyze user queries about atmospheric data. Remember to work entirely from the context provided in this prompt and generate appropriate function calls based on the user's query.
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
                # Try to find query_analysis pattern
                json_pattern = r'\{[\s\S]*"query_analysis"[\s\S]*\}'
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
            r'\d+\s*[\+\-\*/÷×]\s*\d+',  # "5 + 3", "10 / 2", "3 × 4"
            r'=\s*\d+\.?\d*',  # "= 4.5", "= 4"
            r'equals\s+\d+',  # "equals 10"
            r'sum\s+is\s+\d+',  # "sum is 15"
            r'average\s+is\s+approximately\s+\d+',  # "average is approximately 15"
            r'\d+\s*%\s*of\s*\d+',  # "50% of 100"
            r'total\s+is\s+\d+',  # "total is 100"
        ]
        
        # Check for calculation patterns
        for pattern in forbidden_patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                # Get context around match to check if it's a false positive
                start = max(0, match.start() - 20)
                end = min(len(response), match.end() + 20)
                context = response[start:end].lower()
                
                # Allow if it's part of a date, timestamp, or reference
                if any(word in context for word in ['date', 'timestamp', ':', '/', 'iso', 'utc']):
                    continue
                
                # This looks like an actual calculation
                return False
        
        return True
    
    def generate_function_calls(
        self, 
        user_query: str, 
        data_summary: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate function calls from user query using dynamic system prompt.
        Works entirely from context - no hardcoded logic.
        """
        
        # Build dynamic system prompt from registry
        system_prompt = self.build_dynamic_system_prompt(data_summary)
        
        full_prompt = f"{system_prompt}\n\n---\n\nUSER QUERY: {user_query}\n\nGenerate the appropriate function calls following the JSON format specified above."
        
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
        query_analysis: Dict[str, Any],
        function_results: List[Dict],
        data_summary: Dict[str, Any]
    ) -> str:
        """Generate natural language response from function results"""
        
        results_str = json.dumps(function_results, indent=2)
        analysis_str = json.dumps(query_analysis, indent=2)
        
        synthesis_prompt = f"""You are an atmospheric science data analyst. A user asked a question, you generated function calls, and those functions have been executed.

USER QUERY: {user_query}

YOUR ANALYSIS:
{analysis_str}

FUNCTION EXECUTION RESULTS:
{results_str}

Now provide a natural language answer following this structure:

## Answer
[Direct, concise response to the user's question using the function results]

## Evidence
[Cite specific numerical values from the function outputs to support your answer]

## Scientific Context
[Explain the atmospheric science processes or relationships underlying these results]

## Data Summary
[Note the data range, sample size, and any relevant metadata from the results]

Remember:
- Use exact values from function outputs (don't round or approximate)
- Cite units correctly (μg/m³, ppm, °C, etc.)
- Provide scientific interpretation when relevant
- Be concise but complete
"""
        
        try:
            response = self.model.generate_content(synthesis_prompt)
            return response.text
        except Exception as e:
            return f"Error generating synthesis: {str(e)}"
    
    def process_query(
        self,
        user_query: str,
        data_summary: Dict[str, Any],
        executor
    ) -> Dict[str, Any]:
        """
        Complete pipeline: query -> function calls -> execution -> synthesis.
        Works entirely from context with no hardcoded logic.
        """
        
        # Step 1: Generate function calls from query
        llm_response = self.generate_function_calls(user_query, data_summary)
        
        if not llm_response.get("success"):
            return {
                "query": user_query,
                "error": llm_response.get("error", "Unknown error"),
                "raw_response": llm_response.get("raw_response")
            }
        
        parsed_calls = llm_response['parsed_calls']
        function_calls = parsed_calls.get('function_calls', [])
        query_analysis = parsed_calls.get('query_analysis', {})
        
        # Step 2: Execute functions
        function_results = executor.execute_batch(function_calls)
        
        # Step 3: Synthesize results into natural language
        synthesis = self.synthesize_results(
            user_query=user_query,
            query_analysis=query_analysis,
            function_results=function_results,
            data_summary=data_summary
        )
        
        return {
            "query": user_query,
            "query_analysis": query_analysis,
            "reasoning": parsed_calls.get('reasoning'),
            "function_calls": function_calls,
            "function_results": function_results,
            "answer": synthesis
        }
