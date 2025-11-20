"""
Synthetic test data generators for AI module tests.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from SharedConstants import DatasetColumnDataType


class TestDataGenerator:
    """Generate synthetic test data for various test scenarios"""
    
    @staticmethod
    def create_simple_classification_data(n_samples: int = 20) -> pd.DataFrame:
        """Create simple classification dataset"""
        np.random.seed(42)  # For reproducible tests
        data = {
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples),
            'feature3': np.random.choice(['A', 'B', 'C'], n_samples),
            'target': np.random.choice([0, 1], n_samples)
        }
        return pd.DataFrame(data)
    
    @staticmethod
    def create_regression_data(n_samples: int = 20) -> pd.DataFrame:
        """Create regression dataset"""
        np.random.seed(42)
        x1 = np.random.normal(0, 1, n_samples)
        x2 = np.random.normal(0, 1, n_samples)
        y = 2 * x1 + 3 * x2 + np.random.normal(0, 0.1, n_samples)
        
        return pd.DataFrame({
            'input1': x1,
            'input2': x2,
            'output': y
        })
    
    @staticmethod
    def create_mixed_types_data(n_samples: int = 15) -> pd.DataFrame:
        """Create dataset with mixed data types"""
        np.random.seed(42)
        return pd.DataFrame({
            'numeric_col': np.random.normal(0, 1, n_samples),
            'integer_col': np.random.randint(1, 100, n_samples),
            'categorical_col': np.random.choice(['red', 'green', 'blue'], n_samples),
            'boolean_col': np.random.choice([True, False], n_samples),
            'text_col': [f'text_{i}' for i in range(n_samples)],
            'target_col': np.random.normal(0, 1, n_samples)
        })
    
    @staticmethod
    def create_json_column_data(n_samples: int = 10) -> pd.DataFrame:
        """Create dataset with JSON-like columns"""
        import json
        
        data = []
        for i in range(n_samples):
            data.append({
                'simple_col': i,
                'json_col': json.dumps({
                    'nested_value': i * 2,
                    'category': 'A' if i % 2 == 0 else 'B'
                }),
                'target': i * 10
            })
        
        return pd.DataFrame(data)
    
    @staticmethod
    def create_missing_values_data(n_samples: int = 20) -> pd.DataFrame:
        """Create dataset with missing values"""
        np.random.seed(42)
        df = pd.DataFrame({
            'col1': np.random.normal(0, 1, n_samples),
            'col2': np.random.choice(['A', 'B', 'C'], n_samples),
            'col3': np.random.normal(0, 1, n_samples),
            'target': np.random.normal(0, 1, n_samples)
        })
        
        # Introduce missing values
        df.loc[0:2, 'col1'] = np.nan
        df.loc[5:7, 'col2'] = None
        df.loc[10:12, 'col3'] = np.nan
        
        return df
    
    @staticmethod
    def get_standard_datatype_mapping() -> Dict[str, DatasetColumnDataType]:
        """Get standard datatype mapping for test data"""
        return {
            'feature1': DatasetColumnDataType.FLOAT,
            'feature2': DatasetColumnDataType.FLOAT,
            'feature3': DatasetColumnDataType.LABEL,
            'target': DatasetColumnDataType.FLOAT,
            'input1': DatasetColumnDataType.FLOAT,
            'input2': DatasetColumnDataType.FLOAT,
            'output': DatasetColumnDataType.FLOAT,
            'numeric': DatasetColumnDataType.FLOAT,
            'integer': DatasetColumnDataType.FLOAT,
            'categorical': DatasetColumnDataType.LABEL,
            'boolean': DatasetColumnDataType.LABEL,
            'text': DatasetColumnDataType.LABEL,
            'simple_col': DatasetColumnDataType.FLOAT,
            'json_col': DatasetColumnDataType.LABEL,
            'col1': DatasetColumnDataType.FLOAT,
            'col2': DatasetColumnDataType.LABEL,
            'col3': DatasetColumnDataType.FLOAT
        }
    
    @staticmethod
    def get_standard_description_mapping() -> Dict[str, str]:
        """Get standard description mapping for test data"""
        return {
            'feature1': 'First feature',
            'feature2': 'Second feature', 
            'feature3': 'Third categorical feature',
            'target': 'Target variable',
            'input1': 'First input',
            'input2': 'Second input',
            'output': 'Output variable',
            'numeric': 'Numeric column',
            'integer': 'Integer column',
            'categorical': 'Categorical column',
            'boolean': 'Boolean column',
            'text': 'Text column',
            'simple_col': 'Simple column',
            'json_col': 'JSON column',
            'col1': 'Column 1',
            'col2': 'Column 2',
            'col3': 'Column 3'
        }
