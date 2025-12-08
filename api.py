"""
FastAPI Application - Atmospheric Data Analysis API
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import polars as pl
from datetime import datetime
import os
from dotenv import load_dotenv

from analyzer import AtmosphericDataAnalyzer, FunctionExecutor
from llm_client import GeminiLLM

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Atmospheric Data Analysis API",
    description="Tool-augmented LLM for air quality data analysis",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
analyzer: Optional[AtmosphericDataAnalyzer] = None
executor: Optional[FunctionExecutor] = None
llm_client: Optional[GeminiLLM] = None
data_summary: Optional[Dict] = None


# Pydantic models
class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query about atmospheric data")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "When did PM2.5 exceed 35 μg/m³ in November?"
            }
        }


class FunctionCallRequest(BaseModel):
    function_calls: List[Dict[str, Any]] = Field(..., description="List of function calls to execute")
    
    class Config:
        json_schema_extra = {
            "example": {
                "function_calls": [
                    {
                        "function": "compute_statistics",
                        "arguments": {
                            "column": "O3_ppm",
                            "metrics": ["mean", "max", "min"]
                        }
                    }
                ]
            }
        }


class QueryResponse(BaseModel):
    query: str
    intent_analysis: Optional[str] = None
    reasoning: Optional[str] = None
    function_calls: Optional[List[Dict]] = None
    function_results: Optional[List[Dict]] = None
    answer: Optional[str] = None
    error: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    """Initialize data and LLM client on startup"""
    global analyzer, executor, llm_client, data_summary
    
    # Load data
    data_path = os.getenv("DATA_PATH", "forecasted_data.parquet")
    
    if not os.path.exists(data_path):
        print(f"WARNING: Data file not found at {data_path}")
        print("API will start but queries will fail until data is loaded")
        return
    
    try:
        df = pl.read_parquet(data_path)
        analyzer = AtmosphericDataAnalyzer(df)
        executor = FunctionExecutor(analyzer)
        data_summary = analyzer.get_data_summary()
        
        print(f"✓ Loaded data: {data_summary['n_records']} records")
        print(f"  Date range: {data_summary['start_date']} to {data_summary['end_date']}")
        
        # Initialize LLM
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            llm_client = GeminiLLM(api_key)
            print("✓ Gemini LLM initialized")
        else:
            print("WARNING: GEMINI_API_KEY not found in environment")
            print("Set it in .env file or environment variables")
    
    except Exception as e:
        print(f"ERROR during startup: {e}")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Atmospheric Data Analysis API",
        "data_loaded": analyzer is not None,
        "llm_available": llm_client is not None
    }


@app.get("/data/summary")
async def get_data_summary():
    """Get dataset metadata"""
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    return data_summary


@app.get("/data/columns")
async def get_columns():
    """Get available columns"""
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    return {
        "columns": data_summary['columns'],
        "schema": data_summary['schema']
    }


@app.post("/execute/functions", response_model=Dict)
async def execute_functions(request: FunctionCallRequest):
    """Execute function calls directly (bypass LLM)"""
    if executor is None:
        raise HTTPException(status_code=503, detail="Analyzer not initialized")
    
    try:
        results = executor.execute_batch(request.function_calls)
        return {
            "function_calls": request.function_calls,
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query_data(request: QueryRequest):
    """
    Main endpoint: Natural language query → LLM → Function calls → Results → Synthesis
    """
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    if llm_client is None:
        raise HTTPException(status_code=503, detail="LLM client not initialized. Check GEMINI_API_KEY")
    
    try:
        # Step 1: Generate system prompt
        system_prompt = llm_client.build_system_prompt(data_summary)
        
        # Step 2: LLM generates function calls
        llm_response = llm_client.generate_function_calls(
            user_query=request.query,
            system_prompt=system_prompt
        )
        
        if not llm_response.get("success"):
            return QueryResponse(
                query=request.query,
                error=llm_response.get("error", "Unknown error"),
                answer=llm_response.get("raw_response")
            )
        
        parsed_calls = llm_response['parsed_calls']
        function_calls = parsed_calls.get('function_calls', [])
        
        # Step 3: Execute functions
        function_results = executor.execute_batch(function_calls)
        
        # Step 4: LLM synthesizes results
        synthesis = llm_client.synthesize_results(
            user_query=request.query,
            function_results=function_results,
            system_prompt=system_prompt
        )
        
        return QueryResponse(
            query=request.query,
            intent_analysis=parsed_calls.get('intent_analysis'),
            reasoning=parsed_calls.get('reasoning'),
            function_calls=function_calls,
            function_results=function_results,
            answer=synthesis
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/simple")
async def query_simple(query: str = Query(..., description="Natural language query")):
    """Simplified query endpoint (returns only the answer)"""
    if analyzer is None or llm_client is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        system_prompt = llm_client.build_system_prompt(data_summary)
        
        llm_response = llm_client.generate_function_calls(
            user_query=query,
            system_prompt=system_prompt
        )
        
        if not llm_response.get("success"):
            return {"error": llm_response.get("error")}
        
        parsed_calls = llm_response['parsed_calls']
        function_calls = parsed_calls.get('function_calls', [])
        
        function_results = executor.execute_batch(function_calls)
        
        synthesis = llm_client.synthesize_results(
            user_query=query,
            function_results=function_results,
            system_prompt=system_prompt
        )
        
        return {"answer": synthesis}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
