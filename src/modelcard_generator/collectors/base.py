"""
Base classes for data collection.

Provides abstract base classes that define the interface for collecting
data from various ML sources like evaluation results, training logs, etc.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Union
from pathlib import Path
import logging


class DataCollector(ABC):
    """Abstract base class for data collectors."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def collect(self, source: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collect data from the specified source.
        
        Args:
            source: The data source (file path, URL, or data dictionary)
            
        Returns:
            Normalized data dictionary
            
        Raises:
            ValueError: If source is invalid or unsupported
            FileNotFoundError: If source file doesn't exist
        """
        pass
    
    @abstractmethod
    def validate_source(self, source: Union[str, Path, Dict[str, Any]]) -> bool:
        """
        Validate that the source is supported and accessible.
        
        Args:
            source: The data source to validate
            
        Returns:
            True if source is valid, False otherwise
        """
        pass
    
    def _load_json_file(self, file_path: Path) -> Dict[str, Any]:
        """Load data from JSON file."""
        import json
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """Load data from YAML file."""
        import yaml
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _load_csv_file(self, file_path: Path) -> Dict[str, Any]:
        """Load data from CSV file."""
        try:
            import pandas as pd
            df = pd.read_csv(file_path)
            return df.to_dict('records')
        except ImportError:
            # Fallback to basic CSV reading
            import csv
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                return list(reader)
    
    def _normalize_path(self, source: Union[str, Path]) -> Path:
        """Normalize source to Path object."""
        return Path(source)
    
    def _is_file_source(self, source: Union[str, Path, Dict[str, Any]]) -> bool:
        """Check if source is a file path."""
        return isinstance(source, (str, Path))
    
    def _load_file_by_extension(self, file_path: Path) -> Dict[str, Any]:
        """Load file data based on extension."""
        suffix = file_path.suffix.lower()
        
        if suffix == '.json':
            return self._load_json_file(file_path)
        elif suffix in ['.yaml', '.yml']:
            return self._load_yaml_file(file_path)
        elif suffix == '.csv':
            return self._load_csv_file(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")


class CollectorRegistry:
    """Registry for managing data collectors."""
    
    def __init__(self):
        self._collectors: Dict[str, DataCollector] = {}
    
    def register(self, name: str, collector: DataCollector) -> None:
        """Register a collector."""
        self._collectors[name] = collector
    
    def get(self, name: str) -> DataCollector:
        """Get a collector by name."""
        if name not in self._collectors:
            raise ValueError(f"Collector not found: {name}")
        return self._collectors[name]
    
    def list_collectors(self) -> list[str]:
        """List all registered collectors."""
        return list(self._collectors.keys())
    
    def collect_all(self, sources: Dict[str, Any]) -> Dict[str, Any]:
        """Collect data from all sources using appropriate collectors."""
        results = {}
        
        for source_name, source_data in sources.items():
            if source_name in self._collectors:
                try:
                    collector = self._collectors[source_name]
                    results[source_name] = collector.collect(source_data)
                except Exception as e:
                    logging.error(f"Failed to collect from {source_name}: {e}")
                    results[source_name] = None
        
        return results