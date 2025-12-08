"""
Quick setup and verification script
"""

import os
import sys


def check_environment():
    """Check if environment is properly configured"""
    print("🔍 Checking environment...")
    
    issues = []
    
    # Check .env file
    if not os.path.exists(".env"):
        issues.append("❌ .env file not found. Copy .env.example to .env and add your API key")
    else:
        print("✓ .env file found")
        
        # Check if GEMINI_API_KEY is set
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or api_key == "your_gemini_api_key_here":
            issues.append("❌ GEMINI_API_KEY not set in .env file")
        else:
            print("✓ GEMINI_API_KEY is set")
    
    # Check if data file exists
    if not os.path.exists("forecasted_data.parquet"):
        issues.append("⚠️  forecasted_data.parquet not found. Run: python generate_sample_data.py")
    else:
        print("✓ Data file found")
    
    # Check dependencies
    try:
        import polars
        import fastapi
        import google.generativeai
        print("✓ All dependencies installed")
    except ImportError as e:
        issues.append(f"❌ Missing dependency: {e.name}. Run: pip install -r requirements.txt")
    
    return issues


def main():
    print("=" * 60)
    print("Atmospheric Data Analysis API - Setup Check")
    print("=" * 60)
    print()
    
    issues = check_environment()
    
    print()
    if issues:
        print("⚠️  Issues found:")
        for issue in issues:
            print(f"  {issue}")
        print()
        print("Please fix the above issues and run this script again.")
        return 1
    else:
        print("✅ All checks passed!")
        print()
        print("Next steps:")
        print("  1. If you haven't generated sample data:")
        print("     python generate_sample_data.py")
        print()
        print("  2. Start the API server:")
        print("     python api.py")
        print()
        print("  3. In another terminal, test the API:")
        print("     python test_client.py")
        print()
        print("  4. Or visit the interactive docs:")
        print("     http://localhost:8000/docs")
        return 0


if __name__ == "__main__":
    sys.exit(main())
