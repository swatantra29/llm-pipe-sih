# Atmospheric Data Analysis API - Quick Setup Script
# Run this in PowerShell to set up everything automatically

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  Atmospheric Data Analysis API - Setup Script" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check Python
Write-Host "Step 1: Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "  ✓ Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Python not found! Please install Python 3.8 or higher." -ForegroundColor Red
    exit 1
}

# Step 2: Install dependencies
Write-Host ""
Write-Host "Step 2: Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ Dependencies installed successfully" -ForegroundColor Green
} else {
    Write-Host "  ✗ Failed to install dependencies" -ForegroundColor Red
    exit 1
}

# Step 3: Check .env file
Write-Host ""
Write-Host "Step 3: Checking configuration..." -ForegroundColor Yellow
if (Test-Path ".env") {
    $envContent = Get-Content ".env" -Raw
    if ($envContent -match "your_gemini_api_key_here") {
        Write-Host "  ⚠ .env file exists but API key not configured" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "  IMPORTANT: Edit .env file and add your Gemini API key:" -ForegroundColor Cyan
        Write-Host "  1. Get key from: https://makersuite.google.com/app/apikey" -ForegroundColor White
        Write-Host "  2. Open .env in a text editor" -ForegroundColor White
        Write-Host "  3. Replace 'your_gemini_api_key_here' with your actual key" -ForegroundColor White
        Write-Host ""
    } else {
        Write-Host "  ✓ .env file configured" -ForegroundColor Green
    }
} else {
    Write-Host "  ✗ .env file not found" -ForegroundColor Red
}

# Step 4: Generate sample data
Write-Host ""
Write-Host "Step 4: Generating sample data..." -ForegroundColor Yellow
if (Test-Path "forecasted_data.parquet") {
    Write-Host "  ℹ Data file already exists, skipping generation" -ForegroundColor Cyan
} else {
    python generate_sample_data.py
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ Sample data generated successfully" -ForegroundColor Green
    } else {
        Write-Host "  ✗ Failed to generate sample data" -ForegroundColor Red
    }
}

# Step 5: Run setup check
Write-Host ""
Write-Host "Step 5: Running setup verification..." -ForegroundColor Yellow
python setup_check.py

# Final instructions
Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  Setup Complete!" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host ""
Write-Host "1. Configure your API key (if not done):" -ForegroundColor White
Write-Host "   - Edit .env file" -ForegroundColor Gray
Write-Host "   - Add your Gemini API key" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Start the API server:" -ForegroundColor White
Write-Host "   python api.py" -ForegroundColor Green
Write-Host ""
Write-Host "3. In a new terminal, test the API:" -ForegroundColor White
Write-Host "   python test_client.py" -ForegroundColor Green
Write-Host ""
Write-Host "4. Or run examples:" -ForegroundColor White
Write-Host "   python examples.py" -ForegroundColor Green
Write-Host ""
Write-Host "5. Or visit the interactive docs:" -ForegroundColor White
Write-Host "   http://localhost:8000/docs" -ForegroundColor Green
Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""
