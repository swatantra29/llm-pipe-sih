"""
Test client for the Atmospheric Data Analysis API
"""

import requests
import json


class APIClient:
    """Client for testing the API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def health_check(self):
        """Check API status"""
        response = requests.get(f"{self.base_url}/")
        return response.json()
    
    def get_data_summary(self):
        """Get dataset metadata"""
        response = requests.get(f"{self.base_url}/data/summary")
        return response.json()
    
    def get_columns(self):
        """Get available columns"""
        response = requests.get(f"{self.base_url}/data/columns")
        return response.json()
    
    def execute_functions(self, function_calls: list):
        """Execute functions directly"""
        response = requests.post(
            f"{self.base_url}/execute/functions",
            json={"function_calls": function_calls}
        )
        return response.json()
    
    def query(self, query: str):
        """Send natural language query"""
        response = requests.post(
            f"{self.base_url}/query",
            json={"query": query}
        )
        return response.json()
    
    def query_simple(self, query: str):
        """Send query and get simple answer"""
        response = requests.post(
            f"{self.base_url}/query/simple",
            params={"query": query}
        )
        return response.json()


def test_basic_functions():
    """Test direct function execution"""
    client = APIClient()
    
    print("=== Testing Direct Function Execution ===\n")
    
    # Test 1: Compute statistics
    print("1. Computing O3 statistics...")
    result = client.execute_functions([
        {
            "function": "compute_statistics",
            "arguments": {
                "column": "O3_ppm",
                "metrics": ["mean", "max", "min", "std"]
            }
        }
    ])
    print(json.dumps(result, indent=2))
    print()
    
    # Test 2: Find exceedances
    print("2. Finding PM2.5 exceedances...")
    result = client.execute_functions([
        {
            "function": "find_exceedance_events",
            "arguments": {
                "column": "PM25_ugm3",
                "threshold": 35.0,
                "operator": ">",
                "duration_hours": 1
            }
        }
    ])
    print(json.dumps(result, indent=2))
    print()
    
    # Test 3: Compute correlation
    print("3. Computing temperature-O3 correlation...")
    result = client.execute_functions([
        {
            "function": "compute_correlation",
            "arguments": {
                "col1": "temperature_C",
                "col2": "O3_ppm"
            }
        }
    ])
    print(json.dumps(result, indent=2))
    print()


def test_natural_language_queries():
    """Test natural language queries with LLM"""
    client = APIClient()
    
    print("=== Testing Natural Language Queries ===\n")
    
    queries = [
        "What's the average PM2.5 concentration?",
        "When did PM2.5 exceed 35 μg/m³?",
        "What's the correlation between temperature and O3?",
        "What's the lag between SO2 and PM2.5?",
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"{i}. Query: {query}")
        try:
            result = client.query_simple(query)
            if "error" in result:
                print(f"   Error: {result['error']}")
            else:
                print(f"   Answer: {result['answer'][:200]}...")
        except Exception as e:
            print(f"   Exception: {e}")
        print()


if __name__ == "__main__":
    client = APIClient()
    
    # Check API status
    print("Checking API status...")
    try:
        status = client.health_check()
        print(json.dumps(status, indent=2))
        print()
        
        if not status.get("data_loaded"):
            print("⚠ Data not loaded. Run generate_sample_data.py first!")
            exit(1)
        
        # Show data summary
        print("Data summary:")
        summary = client.get_data_summary()
        print(json.dumps(summary, indent=2))
        print()
        
        # Test direct function execution
        test_basic_functions()
        
        # Test natural language queries (requires Gemini API key)
        if status.get("llm_available"):
            test_natural_language_queries()
        else:
            print("⚠ LLM not available. Set GEMINI_API_KEY to test natural language queries.")
    
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure the server is running:")
        print("   python api.py")
