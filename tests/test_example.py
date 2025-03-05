"""Example test file to demonstrate how to write tests for your package.

This file contains comprehensive test examples using pytest.
"""

import pytest
from datetime import datetime
from example_package import __version__
from example_package.core import DataPoint, ExampleClass


def test_version():
    """Test that the version is a string and follows semantic versioning format."""
    assert isinstance(__version__, str)
    # Simple check for semantic versioning format (x.y.z)
    assert len(__version__.split(".")) == 3


class TestDataPoint:
    """Test suite for the DataPoint class."""

    def test_init(self):
        """Test DataPoint initialization."""
        value = 42
        dp = DataPoint(value=value)
        assert dp.value == value
        assert isinstance(dp.timestamp, datetime)
        assert isinstance(dp.metadata, dict)
        assert len(dp.metadata) == 0

    def test_init_with_metadata(self):
        """Test DataPoint initialization with metadata."""
        value = "test"
        metadata = {"key": "value"}
        dp = DataPoint(value=value, metadata=metadata)
        assert dp.value == value
        assert dp.metadata == metadata


class TestExampleClass:
    """Test suite for the ExampleClass."""

    @pytest.fixture
    def example(self):
        """Fixture providing an ExampleClass instance."""
        return ExampleClass("test")

    def test_init(self):
        """Test ExampleClass initialization."""
        name = "test"
        config = {"key": "value"}
        obj = ExampleClass(name, config)
        assert obj.name == name
        assert obj.config == config
        assert len(obj.data_points) == 0

    def test_add_data_point(self, example):
        """Test adding data points."""
        value = 42
        metadata = {"unit": "meters"}
        dp = example.add_data_point(value, metadata)
        assert len(example.data_points) == 1
        assert example.data_points[0] == dp
        assert dp.value == value
        assert dp.metadata == metadata

    def test_process_data_with_numbers(self, example):
        """Test processing numeric data."""
        data = [1, 2, 3, 4, 5]
        result = example.process_data(data)
        assert result["input_type"] == "int"
        assert result["count"] == 5
        assert result["first"] == 1
        assert result["last"] == 5
        assert result["summary"] == 15

    def test_process_data_with_strings(self, example):
        """Test processing string data."""
        data = ["a", "b", "c"]
        result = example.process_data(data)
        assert result["input_type"] == "str"
        assert result["count"] == 3
        assert result["first"] == "a"
        assert result["last"] == "c"
        assert result["summary"] == "abc"

    def test_process_data_empty(self, example):
        """Test processing empty data raises ValueError."""
        with pytest.raises(ValueError):
            example.process_data([])

    def test_get_statistics_empty(self, example):
        """Test statistics with no data points."""
        stats = example.get_statistics()
        assert stats == {"count": 0}

    def test_get_statistics(self, example):
        """Test statistics with mixed data points."""
        example.add_data_point(1)
        example.add_data_point(2)
        example.add_data_point("text")
        example.add_data_point(3)

        stats = example.get_statistics()
        assert stats["total_points"] == 4
        assert stats["numeric_points"] == 3
        assert stats["min"] == 1
        assert stats["max"] == 3
        assert stats["average"] == 2.0