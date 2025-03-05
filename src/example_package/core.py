"""Core functionality module for the example package.

This module demonstrates how to structure your package's core functionality
with proper documentation and type hints.
"""

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime


@dataclass
class DataPoint:
    """A sample dataclass for structured data handling.

    This class demonstrates the use of dataclasses for creating data structures
    with automatic special method generation.

    Attributes:
        value: The main value of the data point.
        timestamp: When this data point was created.
        metadata: Additional information about this data point.
    """
    value: Union[int, float, str]
    timestamp: datetime = datetime.now()
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """Post-initialization processing."""
        if self.metadata is None:
            self.metadata = {}


class ExampleClass:
    """A sample class demonstrating class structure and documentation.

    This class serves as an example of how to structure a Python class
    with proper documentation, type hints, and common patterns.

    Attributes:
        name: A string representing the name of the instance.
        config: A dictionary containing configuration settings.
        data_points: A list of DataPoint objects.
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the ExampleClass instance.

        Args:
            name: The name of the instance.
            config: Optional configuration dictionary.
        """
        self.name = name
        self.config = config or {}
        self.data_points: List[DataPoint] = []

    def add_data_point(self, value: Union[int, float, str], metadata: Optional[Dict[str, Any]] = None) -> DataPoint:
        """Add a new data point to the collection.

        Args:
            value: The value to store.
            metadata: Optional metadata about the data point.

        Returns:
            The created DataPoint instance.
        """
        data_point = DataPoint(value=value, metadata=metadata)
        self.data_points.append(data_point)
        return data_point

    def process_data(self, data: Union[List[int], List[str]]) -> Dict[str, Any]:
        """Process the input data and return results.

        This method demonstrates handling different types of input data
        and returning structured results.

        Args:
            data: A list of integers or strings to process.

        Returns:
            A dictionary containing the processed results.

        Raises:
            ValueError: If the input data is empty.
        """
        if not data:
            raise ValueError("Input data cannot be empty")

        return {
            "input_type": type(data[0]).__name__,
            "count": len(data),
            "first": data[0],
            "last": data[-1],
            "summary": sum(data) if isinstance(data[0], (int, float)) else "".join(data)
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Calculate statistics about the stored data points.

        Returns:
            A dictionary containing various statistics about the data points.
        """
        if not self.data_points:
            return {"count": 0}

        numeric_values = [
            float(dp.value) for dp in self.data_points
            if isinstance(dp.value, (int, float))
        ]

        stats = {
            "total_points": len(self.data_points),
            "numeric_points": len(numeric_values),
        }

        if numeric_values:
            stats.update({
                "min": min(numeric_values),
                "max": max(numeric_values),
                "average": sum(numeric_values) / len(numeric_values)
            })

        return stats


def utility_function(value: int, factor: float = 1.0) -> float:
    """A utility function demonstrating function documentation and type hints.

    Args:
        value: The base value to process.
        factor: A multiplication factor (default: 1.0).

    Returns:
        The processed value.

    Examples:
        >>> utility_function(5, 2.0)
        10.0
        >>> utility_function(3)
        3.0
    """
    return value * factor