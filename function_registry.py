"""
Function Registry for Dynamic Discovery
Automatically discovers available functions and generates system prompts
"""

import inspect
from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass, field
import json


@dataclass
class FunctionMetadata:
    """Metadata for a registered function"""
    name: str
    description: str
    signature: Dict[str, Any]
    parameters: Dict[str, Dict[str, Any]]
    examples: List[str] = field(default_factory=list)
    domain_knowledge: Optional[str] = None


class FunctionRegistry:
    """Registry for dynamically discovering and documenting available functions"""
    
    def __init__(self):
        self.functions: Dict[str, FunctionMetadata] = {}
    
    def register(
        self,
        func: Callable,
        description: str,
        examples: Optional[List[str]] = None,
        domain_knowledge: Optional[str] = None
    ):
        """Register a function with metadata"""
        sig = inspect.signature(func)
        
        # Extract parameter information
        parameters = {}
        for param_name, param in sig.parameters.items():
            # Skip self, *args, and **kwargs
            if self._should_skip_parameter(param_name, param):
                continue
                
            param_info = {
                'type': self._get_type_name(param.annotation),
                'required': param.default == inspect.Parameter.empty,
                'default': None if param.default == inspect.Parameter.empty else param.default
            }
            parameters[param_name] = param_info
        
        # Create metadata
        metadata = FunctionMetadata(
            name=func.__name__,
            description=description,
            signature={
                'return_type': self._get_type_name(sig.return_annotation)
            },
            parameters=parameters,
            examples=examples or [],
            domain_knowledge=domain_knowledge
        )
        
        self.functions[func.__name__] = metadata
    
    def _get_type_name(self, annotation) -> str:
        """Convert type annotation to string"""
        if annotation == inspect.Parameter.empty or annotation == inspect.Signature.empty:
            return 'Any'
        
        # Handle typing module types
        type_str = str(annotation)
        if 'typing.' in type_str:
            type_str = type_str.replace('typing.', '')
        
        return type_str
    
    def _should_skip_parameter(self, param_name: str, param: inspect.Parameter) -> bool:
        """Check if parameter should be skipped during registration"""
        # Skip 'self' for methods
        if param_name == 'self':
            return True
        # Skip *args and **kwargs
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            return True
        return False
    
    def get_function(self, name: str) -> Optional[FunctionMetadata]:
        """Get metadata for a function"""
        return self.functions.get(name)
    
    def list_functions(self) -> List[str]:
        """List all registered function names"""
        return list(self.functions.keys())
    
    def generate_system_prompt_section(self) -> str:
        """Generate system prompt section from registered functions"""
        sections = []
        
        for i, (name, metadata) in enumerate(self.functions.items(), 1):
            section = f"## {i}. {name}\n"
            section += f"**Purpose**: {metadata.description}\n\n"
            
            # Add signature
            params_str = ", ".join([
                f"{pname}: {pinfo['type']}" + 
                (f" = {pinfo['default']}" if not pinfo['required'] else "")
                for pname, pinfo in metadata.parameters.items()
            ])
            
            section += f"**Signature**:\n```python\n"
            section += f"def {name}({params_str}) -> {metadata.signature['return_type']}\n```\n\n"
            
            # Add parameter details
            if metadata.parameters:
                section += "**Parameters**:\n"
                for pname, pinfo in metadata.parameters.items():
                    required_str = "required" if pinfo['required'] else "optional"
                    section += f"- `{pname}` ({pinfo['type']}, {required_str})"
                    if not pinfo['required']:
                        section += f" - default: {pinfo['default']}"
                    section += "\n"
                section += "\n"
            
            # Add domain knowledge
            if metadata.domain_knowledge:
                section += f"**Domain Knowledge**: {metadata.domain_knowledge}\n\n"
            
            # Add examples
            if metadata.examples:
                section += "**Examples**:\n"
                for example in metadata.examples:
                    section += f"- {example}\n"
                section += "\n"
            
            sections.append(section)
        
        return "\n".join(sections)
    
    def to_json(self) -> str:
        """Export registry as JSON"""
        data = {
            name: {
                'description': meta.description,
                'parameters': meta.parameters,
                'examples': meta.examples,
                'domain_knowledge': meta.domain_knowledge
            }
            for name, meta in self.functions.items()
        }
        return json.dumps(data, indent=2)


def create_analyzer_registry(analyzer) -> FunctionRegistry:
    """Create a function registry from an AtmosphericDataAnalyzer instance"""
    registry = FunctionRegistry()
    
    # Register extract_feature
    registry.register(
        analyzer.extract_feature,
        description="Extract time-series data for a specific pollutant or meteorological variable with optional filtering",
        examples=[
            "Extract O3 data for November 2025",
            "Get PM2.5 data when temperature > 25°C",
            "Extract all SO2 measurements"
        ],
        domain_knowledge="Used for retrieving raw time-series data. Supports temporal and conditional filtering."
    )
    
    # Register compute_crossover_lag
    registry.register(
        analyzer.compute_crossover_lag,
        description="Calculate time lag where cross-correlation between precursor and product peaks",
        examples=[
            "Find lag between SO2 and PM2.5 (sulfate aerosol formation)",
            "Calculate NO2 to O3 lag (photochemical formation)"
        ],
        domain_knowledge="SO2 → PM2.5 (sulfate aerosol): typical lag 6-24 hours. NO2 → O3: typical lag 2-6 hours. Uses cross-correlation analysis."
    )
    
    # Register compute_statistics
    registry.register(
        analyzer.compute_statistics,
        description="Calculate statistical metrics (mean, std, max, min, median, quantiles) with optional temporal grouping",
        examples=[
            "Average PM2.5 concentration over entire dataset",
            "Hourly mean O3 levels (diurnal pattern)",
            "Monthly maximum temperature"
        ],
        domain_knowledge="Supports grouping by hour, day, or month to identify temporal patterns. Returns timestamps for min/max values."
    )
    
    # Register find_exceedance_events
    registry.register(
        analyzer.find_exceedance_events,
        description="Identify timestamps and events where pollutants exceed regulatory thresholds",
        examples=[
            "Find PM2.5 exceedances above EPA threshold (35 μg/m³)",
            "Detect O3 violations (0.070 ppm 8-hour standard)",
            "Identify when SO2 exceeds 75 ppb"
        ],
        domain_knowledge="EPA NAAQS thresholds: O3: 0.070 ppm (8-hour), PM2.5: 35 μg/m³ (24-hour), SO2: 75 ppb (1-hour), NO2: 100 ppb (1-hour). Returns consecutive events with duration and peak values."
    )
    
    # Register compute_correlation
    registry.register(
        analyzer.compute_correlation,
        description="Calculate Pearson correlation coefficient between two variables with optional time lag",
        examples=[
            "Correlation between temperature and O3 (photochemical relationship)",
            "Relationship between wind speed and PM2.5 (dilution effect)",
            "Humidity and PM2.5 correlation (hygroscopic growth)"
        ],
        domain_knowledge="Expected relationships: Temperature-O3 (positive, photochemical), Wind-PM (negative, dilution), Humidity-PM2.5 (complex, hygroscopic growth)."
    )
    
    # Register identify_peaks
    registry.register(
        analyzer.identify_peaks,
        description="Find local maxima or minima in time-series data",
        examples=[
            "Find O3 peak events (pollution episodes)",
            "Identify PM2.5 local maxima",
            "Detect minimum temperature events"
        ],
        domain_knowledge="Uses scipy.signal.find_peaks with prominence threshold. Useful for identifying pollution episodes or meteorological events."
    )
    
    # Register filter_by_conditions
    registry.register(
        analyzer.filter_by_conditions,
        description="Filter data by multiple meteorological or pollutant conditions simultaneously",
        examples=[
            "Hot days: temperature > 30°C",
            "Stagnant conditions: wind_speed < 2 m/s AND temperature > 25°C",
            "High humidity events: humidity > 80%"
        ],
        domain_knowledge="Useful for conditional analysis. Can combine multiple conditions (AND logic). Returns matching record count and sample timestamps."
    )
    
    return registry
