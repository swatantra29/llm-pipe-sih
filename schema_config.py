"""
Shared schema configuration for atmospheric data
Centralized column information to avoid duplication
"""

# Column metadata: (dtype, unit, description)
COLUMN_METADATA = {
    'timestamp': ('DateTime', 'UTC', 'Hourly timestamps'),
    'SO2_ppm': ('Float64', 'ppm', 'Sulfur dioxide concentration'),
    'NO2_ppm': ('Float64', 'ppm', 'Nitrogen dioxide concentration'),
    'O3_ppm': ('Float64', 'ppm', 'Ozone concentration'),
    'PM25_ugm3': ('Float64', 'μg/m³', 'Particulate matter ≤2.5μm'),
    'PM10_ugm3': ('Float64', 'μg/m³', 'Particulate matter ≤10μm'),
    'CO_ppm': ('Float64', 'ppm', 'Carbon monoxide concentration'),
    'temperature_C': ('Float64', '°C', 'Air temperature'),
    'humidity_pct': ('Float64', '%', 'Relative humidity'),
    'wind_speed_ms': ('Float64', 'm/s', 'Wind speed'),
    'pressure_hPa': ('Float64', 'hPa', 'Atmospheric pressure'),
}


def build_schema_table(columns: list) -> str:
    """Build markdown table of schema from column list"""
    table = "| Column Name | Data Type | Unit | Description |\n"
    table += "|------------|-----------|------|-------------|\n"
    
    for col in columns:
        if col in COLUMN_METADATA:
            dtype, unit, desc = COLUMN_METADATA[col]
            table += f"| `{col}` | {dtype} | {unit} | {desc} |\n"
    
    return table
