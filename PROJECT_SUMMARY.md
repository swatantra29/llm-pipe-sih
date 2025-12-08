# 🌍 Atmospheric Data Analysis API - Complete Implementation

## ✅ What I've Built For You

A **production-ready tool-augmented LLM pipeline** for atmospheric science data analysis using:
- **FastAPI** for the REST API
- **Google Gemini** for natural language understanding
- **Polars** for high-performance data analysis
- **Zero LLM Math** - all calculations done externally

---

## 📁 Project Structure

```
c:\Users\User\OneDrive\Desktop\llm'\
│
├── api.py                      # Main FastAPI application
├── analyzer.py                 # Data analysis functions (Polars)
├── llm_client.py              # Gemini LLM integration
│
├── generate_sample_data.py    # Generate test data
├── test_client.py             # API test client
├── examples.py                # Complete walkthrough examples
├── setup_check.py             # Verify environment setup
│
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables (API key)
├── .env.example              # Template for .env
├── .gitignore                # Git ignore rules
│
├── README.md                 # Full documentation
└── QUICKSTART.txt            # Quick start guide
```

---

## 🚀 Getting Started (3 Steps)

### **Step 1: Configure Your Gemini API Key**

1. Get a free API key: https://makersuite.google.com/app/apikey
2. Edit `.env` file:
   ```
   GEMINI_API_KEY=your_actual_api_key_here
   DATA_PATH=forecasted_data.parquet
   ```

### **Step 2: Generate Sample Data**

```powershell
python generate_sample_data.py
```

This creates `forecasted_data.parquet` with 30 days of synthetic air quality data.

### **Step 3: Start the API**

```powershell
python api.py
```

Visit: http://localhost:8000/docs for interactive API documentation

---

## 🧪 Testing the System

### **Option 1: Run Tests**
```powershell
python test_client.py
```

### **Option 2: Run Examples**
```powershell
python examples.py
```

### **Option 3: Use the API Directly**

**Health Check:**
```powershell
curl http://localhost:8000/
```

**Natural Language Query:**
```powershell
curl -X POST http://localhost:8000/query/simple?query="What is the average PM2.5 concentration?"
```

---

## 🎯 Key Features

### **1. Tool-Augmented Architecture**
```
User Query → LLM (Intent) → Function Calls → Polars (Math) → LLM (Synthesis) → Answer
```

### **2. No LLM Calculations**
- ✅ LLM only generates function calls and synthesizes results
- ✅ All math done by Python/Polars/NumPy/SciPy
- ✅ Built-in validation to prevent LLM arithmetic

### **3. Available Functions**

| Function | Purpose |
|----------|---------|
| `extract_feature` | Extract time-series data |
| `compute_statistics` | Mean, std, max, min, quantiles |
| `find_exceedance_events` | Find threshold violations |
| `compute_correlation` | Pearson correlation (with lag) |
| `compute_crossover_lag` | Time lag between precursor/product |
| `identify_peaks` | Find local maxima/minima |
| `filter_by_conditions` | Multi-condition filtering |

### **4. Domain Knowledge Built-In**

- **EPA NAAQS thresholds** (PM2.5: 35 μg/m³, O3: 0.070 ppm)
- **Atmospheric chemistry** (SO2 → PM2.5 lag: 6-24h, NO2 → O3: 2-6h)
- **Meteorological relationships** (Temp-O3 correlation, wind dilution)

---

## 💡 Example Queries

### Statistics
- "What's the average PM2.5 concentration?"
- "Show me O3 levels by hour of day"

### Exceedances
- "When did PM2.5 exceed 35 μg/m³?"
- "How many O3 violations occurred?"

### Correlations
- "What's the correlation between temperature and O3?"
- "Does wind speed affect PM2.5?"

### Lag Analysis
- "What's the lag between SO2 spikes and PM2.5 increases?"
- "How long after NO2 increases does O3 peak?"

### Conditional
- "What's the average O3 on hot days (>30°C)?"
- "Show PM2.5 when wind speed is low (<2 m/s)"

---

## 📊 Data Schema

The system expects a Polars DataFrame with these columns:

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `timestamp` | DateTime | UTC | Hourly timestamps |
| `SO2_ppm` | Float64 | ppm | Sulfur dioxide |
| `NO2_ppm` | Float64 | ppm | Nitrogen dioxide |
| `O3_ppm` | Float64 | ppm | Ozone |
| `PM25_ugm3` | Float64 | μg/m³ | PM2.5 |
| `PM10_ugm3` | Float64 | μg/m³ | PM10 |
| `CO_ppm` | Float64 | ppm | Carbon monoxide |
| `temperature_C` | Float64 | °C | Temperature |
| `humidity_pct` | Float64 | % | Humidity |
| `wind_speed_ms` | Float64 | m/s | Wind speed |
| `pressure_hPa` | Float64 | hPa | Pressure |

---

## 🔧 Using Your Own Data

Replace `forecasted_data.parquet` with your own data:

```python
import polars as pl

df = pl.read_csv("your_data.csv")
# Ensure you have at minimum: timestamp + pollutant columns
df.write_parquet("forecasted_data.parquet")
```

Then restart the API.

---

## 🛠️ Extending the System

### **Add a New Function**

1. **Add method to `analyzer.py`:**
```python
def compute_aqi(self, column: str) -> Dict[str, Any]:
    """Calculate Air Quality Index"""
    # Your logic here
    return {"aqi": aqi_value}
```

2. **Update system prompt in `llm_client.py`:**
```python
## 8. compute_aqi
**Purpose**: Calculate Air Quality Index from pollutant concentration
```

3. **Test directly:**
```python
executor.execute([{"function": "compute_aqi", "arguments": {"column": "PM25_ugm3"}}])
```

---

## 📝 API Endpoints

### `GET /`
Health check

### `GET /data/summary`
Dataset metadata (date range, columns, record count)

### `GET /data/columns`
Available data columns

### `POST /execute/functions`
Execute function calls directly (bypass LLM)

### `POST /query`
Natural language query (full pipeline with all details)

### `POST /query/simple`
Natural language query (returns answer only)

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| "Data not loaded" | Run `python generate_sample_data.py` |
| "GEMINI_API_KEY not found" | Edit `.env` and add your API key |
| Port 8000 in use | Change port in `api.py`: `uvicorn.run(app, port=8001)` |
| Import errors | Run `pip install -r requirements.txt` |
| LLM returns invalid JSON | Check `llm_client.py:parse_function_calls()` |

---

## 🎓 How It Works

### **Step 1: User Query**
```
"When did PM2.5 exceed 35 μg/m³?"
```

### **Step 2: LLM Generates Function Calls**
```json
{
  "reasoning": "User asks for exceedance events. Use find_exceedance_events with EPA threshold.",
  "function_calls": [{
    "function": "find_exceedance_events",
    "arguments": {
      "column": "PM25_ugm3",
      "threshold": 35.0,
      "operator": ">"
    }
  }]
}
```

### **Step 3: Execute Functions**
Python/Polars performs all calculations:
```python
df.filter(pl.col("PM25_ugm3") > 35.0)
# Find consecutive events, calculate durations, peak values
```

### **Step 4: LLM Synthesizes Results**
```
PM2.5 exceeded 35 μg/m³ on 3 occasions:
- Nov 8-9: 6-hour event, peak 58.3 μg/m³
- Nov 15: 4-hour event, peak 47.1 μg/m³
- Nov 22: 5-hour event, peak 52.8 μg/m³

These exceedances violate EPA NAAQS 24-hour standards...
```

---

## 📚 Next Steps

1. ✅ Run `python setup_check.py` to verify everything is configured
2. ✅ Run `python generate_sample_data.py` to create test data
3. ✅ Run `python api.py` to start the server
4. ✅ Run `python test_client.py` to test the API
5. ✅ Visit http://localhost:8000/docs for interactive documentation
6. ✅ Run `python examples.py` for detailed walkthroughs

---

## 🎯 Production Deployment

For production use:

```powershell
# Use production ASGI server
pip install gunicorn

# Run with multiple workers
gunicorn api:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

Add authentication, rate limiting, and caching as needed.

---

## 📄 License

MIT License - Feel free to use and modify for your needs.

---

## 🤝 Contributing

This is a lightweight prototype designed to demonstrate tool-augmented LLM architecture. 

**Areas for enhancement:**
- Additional domain-specific functions (AQI calculation, health impact)
- Spatial analysis (multiple monitoring stations)
- Time-series forecasting
- Visualization generation
- Results caching
- Authentication & rate limiting

---

**Built with ❤️ for atmospheric science research**

For questions or issues, review the documentation in README.md and QUICKSTART.txt
