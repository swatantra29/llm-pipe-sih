# Atmospheric Data Analysis API

Tool-augmented LLM pipeline for air quality data analysis using FastAPI + Gemini.

## Architecture

```
User Query → LLM (Gemini) → Function Calls → Data Analysis (Polars) → LLM Synthesis → Natural Language Output
```

**Key Design Principles:**
- ✅ LLM only generates function calls and synthesizes results
- ✅ All calculations done by Python/Polars (zero LLM math)
- ✅ Structured function calling via JSON
- ✅ Domain-specific atmospheric science context

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Environment

Create a `.env` file:

```bash
cp .env.example .env
```

Edit `.env` and add your Gemini API key:

```
GEMINI_API_KEY=your_actual_api_key_here
DATA_PATH=forecasted_data.parquet
```

Get a Gemini API key: https://makersuite.google.com/app/apikey

### 3. Generate Sample Data

```bash
python generate_sample_data.py
```

This creates `forecasted_data.parquet` with 30 days of synthetic atmospheric data.

### 4. Start the API Server

```bash
python api.py
```

Or with uvicorn:

```bash
uvicorn api:app --reload
```

The API will be available at `http://localhost:8000`

### 5. Test the API

```bash
python test_client.py
```

Or visit the interactive docs: `http://localhost:8000/docs`

## API Endpoints

### GET `/`
Health check and status

### GET `/data/summary`
Get dataset metadata (date range, columns, record count)

### GET `/data/columns`
Get available data columns and their types

### POST `/execute/functions`
Execute function calls directly (bypass LLM)

**Example:**
```json
{
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
```

### POST `/query`
Natural language query (full pipeline)

**Example:**
```json
{
  "query": "When did PM2.5 exceed 35 μg/m³?"
}
```

**Response:**
```json
{
  "query": "When did PM2.5 exceed 35 μg/m³?",
  "intent_analysis": "User asks for exceedance events",
  "reasoning": "Use find_exceedance_events with EPA threshold",
  "function_calls": [...],
  "function_results": [...],
  "answer": "PM2.5 exceeded 35 μg/m³ on 3 occasions..."
}
```

### POST `/query/simple`
Natural language query (returns only answer)

```
POST /query/simple?query=What's the average O3 concentration?
```

## Available Functions

### 1. `extract_feature`
Extract time-series data for a specific pollutant/variable

### 2. `compute_crossover_lag`
Calculate time lag between precursor and product (e.g., SO2 → PM2.5)

### 3. `compute_statistics`
Calculate mean, std, max, min, quantiles (with optional grouping by hour/day/month)

### 4. `find_exceedance_events`
Identify when pollutants exceed regulatory thresholds

### 5. `compute_correlation`
Calculate Pearson correlation between variables (with optional lag)

### 6. `identify_peaks`
Find local maxima/minima in time-series

### 7. `filter_by_conditions`
Filter data by multiple meteorological conditions

## Data Schema

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

## Example Queries

**Statistics:**
- "What's the average PM2.5 concentration?"
- "What are the O3 levels by hour of day?"

**Exceedances:**
- "When did PM2.5 exceed 35 μg/m³?"
- "How many times did O3 exceed 0.070 ppm?"

**Correlations:**
- "What's the correlation between temperature and O3?"
- "Is there a relationship between wind speed and PM2.5?"

**Lag Analysis:**
- "What's the lag between SO2 spikes and PM2.5 increases?"
- "How long after NO2 peaks does O3 increase?"

**Conditional Analysis:**
- "What's the average O3 on hot days (>30°C)?"
- "What are PM2.5 levels when wind speed is low?"

## Project Structure

```
.
├── api.py                      # FastAPI application
├── analyzer.py                 # Data analysis functions (Polars)
├── llm_client.py              # Gemini LLM integration
├── generate_sample_data.py    # Sample data generator
├── test_client.py             # Test client
├── requirements.txt           # Dependencies
├── .env.example              # Environment template
└── README.md                 # This file
```

## Using Your Own Data

Replace `forecasted_data.parquet` with your own data. Required columns:
- `timestamp` (datetime)
- At least one pollutant column (SO2_ppm, NO2_ppm, O3_ppm, PM25_ugm3, etc.)

Optional meteorological columns enhance analysis capabilities.

## Development

### Adding New Functions

1. Add method to `AtmosphericDataAnalyzer` class in `analyzer.py`
2. Update system prompt in `llm_client.py` to document the new function
3. Test with direct function execution before testing with LLM

### Customizing the LLM

Change the model in `llm_client.py`:
```python
self.model = genai.GenerativeModel("gemini-1.5-pro")  # More powerful
```

### Error Handling

The system validates that the LLM doesn't perform calculations. If you see calculation errors, the validation regex patterns are in `llm_client.py`.

## Troubleshooting

**"Data not loaded"**: Run `generate_sample_data.py` or provide your own `forecasted_data.parquet`

**"GEMINI_API_KEY not found"**: Set the API key in `.env` file

**LLM returns invalid JSON**: The parser attempts to extract JSON from markdown code blocks. Check `llm_client.py:parse_function_calls()` if issues persist.

**Function execution errors**: Check that column names match your data schema exactly.

## License

MIT

## Contributing

This is a lightweight prototype. Areas for enhancement:
- Add more domain-specific functions (e.g., AQI calculation, health impact analysis)
- Support for spatial data (multiple monitoring stations)
- Time-series forecasting functions
- Export results to visualizations
- Caching of function results
- Rate limiting and authentication
