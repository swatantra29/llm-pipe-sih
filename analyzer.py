"""
Atmospheric Data Analyzer - Core Function Executor
Executes function calls on Polars DataFrame for air quality analysis
"""

import polars as pl
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from scipy.signal import correlate, find_peaks
from scipy.stats import pearsonr


class AtmosphericDataAnalyzer:
    """Execute function calls on Polars DataFrame"""
    
    def __init__(self, df: pl.DataFrame):
        self.df = df
        self._validate_schema()
    
    def _validate_schema(self):
        """Ensure required columns exist"""
        required = ['timestamp', 'SO2_ppm', 'NO2_ppm', 'O3_ppm', 'PM25_ugm3', 
                   'temperature_C', 'humidity_pct']
        missing = set(required) - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get dataset metadata"""
        return {
            "columns": self.df.columns,
            "n_records": len(self.df),
            "start_date": str(self.df['timestamp'].min()),
            "end_date": str(self.df['timestamp'].max()),
            "schema": {col: str(dtype) for col, dtype in zip(self.df.columns, self.df.dtypes)}
        }
    
    def extract_feature(
        self, 
        column: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        filters: Optional[Dict[str, List]] = None
    ) -> Dict[str, Any]:
        """Extract time-series with optional filtering"""
        
        df_filtered = self.df
        
        # Temporal filter
        if start_date:
            df_filtered = df_filtered.filter(pl.col('timestamp') >= datetime.fromisoformat(start_date))
        if end_date:
            df_filtered = df_filtered.filter(pl.col('timestamp') <= datetime.fromisoformat(end_date))
        
        # Conditional filters
        if filters:
            for col, [operator, value] in filters.items():
                if operator == ">":
                    df_filtered = df_filtered.filter(pl.col(col) > value)
                elif operator == ">=":
                    df_filtered = df_filtered.filter(pl.col(col) >= value)
                elif operator == "<":
                    df_filtered = df_filtered.filter(pl.col(col) < value)
                elif operator == "<=":
                    df_filtered = df_filtered.filter(pl.col(col) <= value)
                elif operator == "==":
                    df_filtered = df_filtered.filter(pl.col(col) == value)
        
        series = df_filtered.select(['timestamp', column])
        
        return {
            "column": column,
            "n_records": len(series),
            "start": str(series['timestamp'].min()) if len(series) > 0 else None,
            "end": str(series['timestamp'].max()) if len(series) > 0 else None,
            "data_sample": series.head(5).to_dicts()
        }
    
    def compute_crossover_lag(
        self,
        precursor_col: str,
        product_col: str,
        max_lag_hours: int = 48,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """Calculate cross-correlation lag"""
        
        df_subset = self.df
        if start_date:
            df_subset = df_subset.filter(pl.col('timestamp') >= datetime.fromisoformat(start_date))
        if end_date:
            df_subset = df_subset.filter(pl.col('timestamp') <= datetime.fromisoformat(end_date))
        
        x = df_subset[precursor_col].to_numpy()
        y = df_subset[product_col].to_numpy()
        
        # Remove NaNs
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]
        
        if len(x) < 10:
            return {
                "error": "Insufficient data points after filtering",
                "n_samples": len(x)
            }
        
        correlation = correlate(y, x, mode='full')
        lags = np.arange(-len(x) + 1, len(x))
        
        # Only positive lags
        positive_mask = (lags >= 0) & (lags <= max_lag_hours)
        positive_lags = lags[positive_mask]
        positive_corr = correlation[positive_mask]
        
        max_idx = np.argmax(positive_corr)
        optimal_lag = int(positive_lags[max_idx])
        
        # Pearson correlation at optimal lag
        y_shifted = y[optimal_lag:]
        x_truncated = x[:len(y_shifted)]
        corr_coef, p_value = pearsonr(x_truncated, y_shifted)
        
        return {
            "precursor": precursor_col,
            "product": product_col,
            "lag_hours": optimal_lag,
            "correlation_coefficient": float(corr_coef),
            "p_value": float(p_value),
            "n_samples": len(x_truncated)
        }
    
    def compute_statistics(
        self,
        column: str,
        metrics: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        groupby: Optional[str] = None
    ) -> Dict[str, Any]:
        """Calculate statistical metrics"""
        
        df_subset = self.df
        if start_date:
            df_subset = df_subset.filter(pl.col('timestamp') >= datetime.fromisoformat(start_date))
        if end_date:
            df_subset = df_subset.filter(pl.col('timestamp') <= datetime.fromisoformat(end_date))
        
        if groupby:
            if groupby == "hour":
                df_subset = df_subset.with_columns(pl.col('timestamp').dt.hour().alias('hour'))
                group_col = 'hour'
            elif groupby == "day":
                df_subset = df_subset.with_columns(pl.col('timestamp').dt.date().alias('day'))
                group_col = 'day'
            elif groupby == "month":
                df_subset = df_subset.with_columns(pl.col('timestamp').dt.month().alias('month'))
                group_col = 'month'
            
            agg_exprs = []
            for metric in metrics:
                if metric == "mean":
                    agg_exprs.append(pl.col(column).mean().alias(f"{column}_mean"))
                elif metric == "std":
                    agg_exprs.append(pl.col(column).std().alias(f"{column}_std"))
                elif metric == "max":
                    agg_exprs.append(pl.col(column).max().alias(f"{column}_max"))
                elif metric == "min":
                    agg_exprs.append(pl.col(column).min().alias(f"{column}_min"))
            
            result_df = df_subset.group_by(group_col).agg(agg_exprs).sort(group_col)
            return {
                "column": column,
                "groupby": groupby,
                "results": result_df.to_dicts()
            }
        
        else:
            result = {"column": column}
            for metric in metrics:
                if metric == "mean":
                    result["mean"] = float(df_subset[column].mean())
                elif metric == "std":
                    result["std"] = float(df_subset[column].std())
                elif metric == "max":
                    result["max"] = float(df_subset[column].max())
                    max_idx = df_subset[column].arg_max()
                    result["max_timestamp"] = str(df_subset['timestamp'][max_idx])
                elif metric == "min":
                    result["min"] = float(df_subset[column].min())
                    min_idx = df_subset[column].arg_min()
                    result["min_timestamp"] = str(df_subset['timestamp'][min_idx])
                elif metric == "median":
                    result["median"] = float(df_subset[column].median())
                elif metric == "q25":
                    result["q25"] = float(df_subset[column].quantile(0.25))
                elif metric == "q75":
                    result["q75"] = float(df_subset[column].quantile(0.75))
            
            return result
    
    def find_exceedance_events(
        self,
        column: str,
        threshold: float,
        operator: str,
        duration_hours: int = 1
    ) -> Dict[str, Any]:
        """Find threshold exceedance events"""
        
        if operator == ">":
            mask = self.df[column] > threshold
        elif operator == ">=":
            mask = self.df[column] >= threshold
        elif operator == "<":
            mask = self.df[column] < threshold
        elif operator == "<=":
            mask = self.df[column] <= threshold
        else:
            raise ValueError(f"Invalid operator: {operator}")
        
        df_exceed = self.df.with_columns(mask.alias('exceeds'))
        
        # Find consecutive events
        events = []
        in_event = False
        event_start = None
        event_values = []
        
        for row in df_exceed.iter_rows(named=True):
            if row['exceeds'] and not in_event:
                in_event = True
                event_start = row['timestamp']
                event_values = [row[column]]
            elif row['exceeds'] and in_event:
                event_values.append(row[column])
            elif not row['exceeds'] and in_event:
                event_duration = len(event_values)
                if event_duration >= duration_hours:
                    events.append({
                        "start": str(event_start),
                        "end": str(row['timestamp']),
                        "duration_hours": event_duration,
                        "peak_value": float(max(event_values))
                    })
                in_event = False
                event_values = []
        
        return {
            "column": column,
            "threshold": threshold,
            "operator": operator,
            "events": events,
            "total_events": len(events)
        }
    
    def compute_correlation(
        self,
        col1: str,
        col2: str,
        lag_hours: int = 0,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """Compute Pearson correlation"""
        
        df_subset = self.df
        if start_date:
            df_subset = df_subset.filter(pl.col('timestamp') >= datetime.fromisoformat(start_date))
        if end_date:
            df_subset = df_subset.filter(pl.col('timestamp') <= datetime.fromisoformat(end_date))
        
        x = df_subset[col1].to_numpy()
        y = df_subset[col2].to_numpy()
        
        if lag_hours != 0:
            if lag_hours > 0:
                y = y[lag_hours:]
                x = x[:len(y)]
            else:
                x = x[abs(lag_hours):]
                y = y[:len(x)]
        
        # Remove NaNs
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]
        
        if len(x) < 3:
            return {
                "error": "Insufficient data points for correlation",
                "n_samples": len(x)
            }
        
        corr_coef, p_value = pearsonr(x, y)
        
        return {
            "col1": col1,
            "col2": col2,
            "correlation": float(corr_coef),
            "p_value": float(p_value),
            "lag_hours": lag_hours,
            "n_samples": len(x)
        }
    
    def identify_peaks(
        self,
        column: str,
        peak_type: str,
        prominence: float = 0.1,
        distance_hours: int = 6
    ) -> Dict[str, Any]:
        """Find local maxima/minima"""
        
        values = self.df[column].to_numpy()
        
        if peak_type == "max":
            data = values
        elif peak_type == "min":
            data = -values
        else:
            raise ValueError("peak_type must be 'max' or 'min'")
        
        # Prominence as fraction of std
        prom_threshold = prominence * np.nanstd(values)
        
        peak_indices, properties = find_peaks(data, prominence=prom_threshold, distance=distance_hours)
        
        peaks_list = []
        for idx in peak_indices:
            peaks_list.append({
                "timestamp": str(self.df['timestamp'][idx]),
                "value": float(values[idx])
            })
        
        return {
            "column": column,
            "peak_type": peak_type,
            "peaks": peaks_list,
            "count": len(peaks_list)
        }
    
    def filter_by_conditions(
        self,
        conditions: Dict[str, List],
        return_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Filter data by multiple conditions"""
        
        df_filtered = self.df
        
        for col, [operator, value] in conditions.items():
            if operator == ">":
                df_filtered = df_filtered.filter(pl.col(col) > value)
            elif operator == ">=":
                df_filtered = df_filtered.filter(pl.col(col) >= value)
            elif operator == "<":
                df_filtered = df_filtered.filter(pl.col(col) < value)
            elif operator == "<=":
                df_filtered = df_filtered.filter(pl.col(col) <= value)
            elif operator == "==":
                df_filtered = df_filtered.filter(pl.col(col) == value)
        
        n_matching = len(df_filtered)
        percentage = (n_matching / len(self.df)) * 100 if len(self.df) > 0 else 0
        
        sample_timestamps = df_filtered['timestamp'].head(10).to_list()
        
        return {
            "n_matching_records": n_matching,
            "percentage": round(percentage, 2),
            "sample_timestamps": [str(ts) for ts in sample_timestamps],
            "conditions": conditions
        }


class FunctionExecutor:
    """Dispatches function calls to analyzer"""
    
    def __init__(self, analyzer: AtmosphericDataAnalyzer):
        self.analyzer = analyzer
    
    def execute(self, function_call: Dict) -> Dict:
        """Execute a single function call"""
        func_name = function_call['function']
        args = function_call.get('arguments', {})
        
        try:
            method = getattr(self.analyzer, func_name)
            result = method(**args)
            return {
                "function": func_name,
                "status": "success",
                "output": result
            }
        except Exception as e:
            return {
                "function": func_name,
                "status": "failed",
                "error": str(e)
            }
    
    def execute_batch(self, function_calls: List[Dict]) -> List[Dict]:
        """Execute multiple function calls"""
        return [self.execute(fc) for fc in function_calls]
