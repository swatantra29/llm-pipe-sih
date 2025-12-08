"""
Sample data generator for testing the API
Creates a synthetic forecasted_data.parquet file
"""

import polars as pl
import numpy as np
from datetime import datetime, timedelta


def generate_sample_data(
    start_date: str = "2025-11-01",
    days: int = 30,
    hourly: bool = True
) -> pl.DataFrame:
    """Generate synthetic atmospheric data"""
    
    # Time range
    start = datetime.fromisoformat(start_date)
    hours = days * 24 if hourly else days
    timestamps = [start + timedelta(hours=i) for i in range(hours)]
    
    n = len(timestamps)
    
    # Generate correlated atmospheric data
    np.random.seed(42)
    
    # Base meteorological variables
    temperature = 20 + 10 * np.sin(np.linspace(0, 2*np.pi*days/365, n)) + np.random.normal(0, 3, n)
    humidity = 60 + 20 * np.sin(np.linspace(0, 2*np.pi*days/30, n)) + np.random.normal(0, 10, n)
    wind_speed = np.abs(5 + np.random.normal(0, 2, n))
    pressure = 1013 + np.random.normal(0, 5, n)
    
    # Pollutants with diurnal patterns
    hour_of_day = np.array([ts.hour for ts in timestamps])
    
    # O3: peaks in afternoon (photochemical)
    o3_diurnal = 0.03 + 0.04 * np.sin((hour_of_day - 6) * np.pi / 12)
    o3_diurnal[hour_of_day < 6] = 0.02
    O3 = np.maximum(0, o3_diurnal + temperature * 0.001 + np.random.normal(0, 0.01, n))
    
    # NO2: peaks during rush hours
    no2_diurnal = 0.02 + 0.03 * ((hour_of_day == 8) | (hour_of_day == 18)).astype(float)
    NO2 = np.maximum(0, no2_diurnal + np.random.normal(0, 0.01, n))
    
    # SO2: industrial emissions
    SO2 = np.maximum(0, 0.01 + np.random.exponential(0.005, n))
    
    # PM2.5: correlated with precursors (with lag)
    PM25_base = 15 + 0.3 * np.roll(SO2, 12) * 1000 + 0.2 * np.roll(NO2, 6) * 1000
    PM25 = np.maximum(0, PM25_base - wind_speed * 2 + np.random.normal(0, 5, n))
    
    # PM10: correlated with PM2.5
    PM10 = PM25 * 1.5 + np.random.normal(0, 3, n)
    
    # CO: traffic-related
    CO = 0.3 + 0.2 * ((hour_of_day >= 7) & (hour_of_day <= 19)).astype(float) + np.random.normal(0, 0.05, n)
    
    # Add some exceedance events
    exceedance_days = np.random.choice(days, size=3, replace=False)
    for day in exceedance_days:
        start_hour = day * 24
        PM25[start_hour:start_hour+8] += np.random.uniform(20, 40, 8)
        O3[start_hour+10:start_hour+18] += np.random.uniform(0.03, 0.05, 8)
    
    # Create DataFrame
    df = pl.DataFrame({
        'timestamp': timestamps,
        'SO2_ppm': SO2,
        'NO2_ppm': NO2,
        'O3_ppm': O3,
        'PM25_ugm3': PM25,
        'PM10_ugm3': PM10,
        'CO_ppm': CO,
        'temperature_C': temperature,
        'humidity_pct': humidity,
        'wind_speed_ms': wind_speed,
        'pressure_hPa': pressure
    })
    
    return df


if __name__ == "__main__":
    print("Generating sample atmospheric data...")
    
    df = generate_sample_data(start_date="2025-11-01", days=30)
    
    print(f"Generated {len(df)} records")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print("\nSample data:")
    print(df.head(10))
    
    # Save to parquet
    output_path = "forecasted_data.parquet"
    df.write_parquet(output_path)
    print(f"\n✓ Saved to {output_path}")
    
    # Show some statistics
    print("\nData statistics:")
    print(df.select([
        pl.col('PM25_ugm3').mean().alias('PM2.5_mean'),
        pl.col('PM25_ugm3').max().alias('PM2.5_max'),
        pl.col('O3_ppm').mean().alias('O3_mean'),
        pl.col('O3_ppm').max().alias('O3_max'),
    ]))
