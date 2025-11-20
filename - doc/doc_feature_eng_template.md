# ML/FeatureEngineeringTemplate.py - Feature Engineering Template System

## Overview

The `FeatureEngineeringTemplate` (FET) module is the foundational transformation engine of the EasyAutoML system, providing a comprehensive, extensible framework for automated feature engineering. It implements over 50 specialized transformation techniques through a unified abstract architecture, enabling intelligent data preprocessing, feature extraction, and format standardization for machine learning pipelines.

**Location**: `ML/FeatureEngineeringTemplate.py`

## Core Architecture

### Abstract Base Class: FeatureEngineeringTemplate

The `FeatureEngineeringTemplate` class serves as the foundational abstraction for all feature engineering transformations, providing a standardized interface that ensures consistency, reliability, and interoperability across the entire FET ecosystem.

```python
class FeatureEngineeringTemplate(ABC):
    """
    Abstract base class defining the standard interface for all feature engineering
    transformations in the EasyAutoML system.

    This class establishes the contract that all FET implementations must follow,
    ensuring consistent behavior, configuration management, and performance
    characteristics across different transformation types.
    """

    # Core identification attributes
    fet_name: str                           # Unique identifier for this FET
    fet_description: str                    # Human-readable description

    # Transformation capabilities
    fet_is_encoder: bool = True             # Can transform data to NN format
    fet_is_decoder: bool = False            # Can reverse transform from NN output
    fet_encoder_is_lossless: bool = False   # Encoder preserves all information

    # Data type compatibility
    fet_data_type_input: DatasetColumnDataType  # Required input data type
    fet_data_type_output: DatasetColumnDataType # Output data type

    # Configuration state
    _configuration_created: bool = False   # Whether configuration is initialized
    _fet_configuration: Dict = {}          # Internal configuration storage
```

### Key Design Principles

#### Unified Transformation Interface
All FETs implement consistent encode/decode methods with standardized signatures, enabling seamless integration and interchangeability within the feature engineering pipeline.

#### Type-Aware Design
Each FET is explicitly designed for specific data types, ensuring type safety and preventing inappropriate transformations that could lead to data corruption or meaningless results.

#### Computational Cost Transparency
Every transformation provides explicit cost calculations, enabling intelligent resource allocation and performance optimization in the broader EasyAutoML system.

#### Configuration Persistence
All FETs support complete serialization and deserialization, enabling configuration reuse across sessions and facilitating reproducible machine learning workflows.

#### Error Resilience and Validation
Robust error handling and comprehensive validation ensure that transformations gracefully handle edge cases, missing data, and unexpected input characteristics.

### FET Classification System

#### By Transformation Type
- **Encoding FETs**: Transform raw data into neural network compatible formats
- **Decoding FETs**: Reverse transformations from neural network outputs
- **Hybrid FETs**: Support both encoding and decoding operations

#### By Data Preservation
- **Lossless FETs**: Perfect reconstruction possible (`fet_encoder_is_lossless = True`)
- **Lossy FETs**: Information may be lost or approximated (`fet_encoder_is_lossless = False`)

#### By Output Dimensionality
- **1:1 FETs**: Single input column produces single output column
- **1:N FETs**: Single input column produces multiple output columns (feature expansion)
- **N:1 FETs**: Multiple input columns combined into single output (feature aggregation)

## Core Transformation Methods

### Primary Transformation Interface

#### `encode(self, column_data: np.ndarray) -> np.ndarray`

**Core encoding transformation with comprehensive validation and error handling**.

```python
@abstractmethod
def encode(self, column_data: np.ndarray) -> np.ndarray:
    """
    Transform input column data into neural network compatible format.

    This method implements the forward transformation, converting raw data
    into a standardized format suitable for neural network processing.
    The transformation may be 1:1, 1:N, or involve complex feature engineering.

    Args:
        column_data: Input column as numpy array (shape: [n_samples])

    Returns:
        Encoded data as numpy array (shape: [n_samples, n_features])
        May have multiple columns for feature expansion transformations

    Raises:
        FETConfigurationError: If FET is not properly configured
        FETDataTypeError: If input data type doesn't match expected type
        FETTransformationError: If transformation fails
    """
```

**Encoding Process Flow**:
```python
def encode(self, column_data: np.ndarray) -> np.ndarray:
    # Step 1: Validate input data and configuration
    self._validate_encode_input(column_data)

    # Step 2: Apply data type specific preprocessing
    preprocessed_data = self._preprocess_encode_data(column_data)

    # Step 3: Execute core transformation logic
    encoded_data = self._execute_encode_transformation(preprocessed_data)

    # Step 4: Apply post-processing and validation
    final_encoded_data = self._postprocess_encoded_data(encoded_data)

    # Step 5: Validate output characteristics
    self._validate_encode_output(final_encoded_data)

    return final_encoded_data
```

**Key Characteristics and Requirements**:
- **Input Validation**: Ensures data type compatibility and proper formatting
- **Shape Handling**: Maintains consistent number of rows, allows column expansion
- **Range Normalization**: Typically produces 0-1 scaled outputs for neural network compatibility
- **Memory Efficiency**: Processes data in chunks for large datasets
- **Error Recovery**: Graceful handling of edge cases and invalid inputs

#### `decode(self, column_data: np.ndarray) -> np.ndarray`

**Reverse transformation for lossless FETs, converting neural network outputs back to original format**.

```python
def decode(self, column_data: np.ndarray) -> np.ndarray:
    """
    Reverse transformation from encoded format back to original data format.

    Only available for lossless transformations (fet_is_decoder = True).
    Reconstructs original data from neural network outputs, enabling
    prediction interpretation and result validation.

    Args:
        column_data: Encoded data from neural network (shape: [n_samples, n_features])

    Returns:
        Decoded data in original format (shape: [n_samples])

    Raises:
        FETNotDecoderError: If FET doesn't support decoding
        FETConfigurationError: If decoder configuration is invalid
        FETTransformationError: If decoding fails
    """
```

**Decoding Process Flow**:
```python
def decode(self, column_data: np.ndarray) -> np.ndarray:
    # Validate decoder availability and input
    if not self.fet_is_decoder:
        raise FETNotDecoderError(f"FET {self.fet_name} does not support decoding")

    # Validate input data structure
    self._validate_decode_input(column_data)

    # Apply reverse transformation
    decoded_data = self._execute_decode_transformation(column_data)

    # Post-process and validate results
    final_decoded_data = self._postprocess_decoded_data(decoded_data)

    return final_decoded_data
```

### Configuration Management System

#### `_create_configuration(self, column_data: np.ndarray, column_datas_infos: Column_datas_infos)`

**Intelligent configuration initialization based on data characteristics**.

```python
@abstractmethod
def _create_configuration(self, column_data: np.ndarray,
                         column_datas_infos: Column_datas_infos) -> None:
    """
    Analyze input data and create transformation-specific configuration.

    This method examines the data characteristics, statistical properties,
    and metadata to determine optimal transformation parameters. It sets
    up scalers, encoders, mappings, and other configuration parameters.

    Args:
        column_data: Raw column data for analysis
        column_datas_infos: Comprehensive column metadata and statistics

    Raises:
        FETConfigurationError: If configuration cannot be created
    """
```

**Configuration Creation Process**:
```python
def _create_configuration(self, column_data: np.ndarray,
                         column_datas_infos: Column_datas_infos) -> None:

    # Analyze data characteristics
    data_stats = self._analyze_data_characteristics(column_data, column_datas_infos)

    # Determine transformation parameters
    transform_params = self._calculate_transformation_parameters(data_stats)

    # Validate parameter compatibility
    self._validate_configuration_parameters(transform_params)

    # Store configuration
    self._fet_configuration = transform_params
    self._configuration_created = True

    logger.info(f"Configuration created for FET {self.fet_name}")
```

#### `serialize_fet_configuration(self) -> Dict[str, Any]`

**Complete configuration export for persistence and sharing**.

```python
def serialize_fet_configuration(self) -> Dict[str, Any]:
    """
    Export FET configuration as serializable dictionary.

    Creates a complete snapshot of the transformation configuration
    that can be saved to database, shared across systems, or
    used for reproducibility.

    Returns:
        Dictionary containing all configuration parameters and metadata

    Raises:
        FETConfigurationError: If configuration is not properly initialized
    """
```

**Serialization Structure**:
```python
def serialize_fet_configuration(self) -> Dict[str, Any]:
    if not self._configuration_created:
        raise FETConfigurationError("Cannot serialize uninitialized FET configuration")

    return {
        "fet_name": self.fet_name,
        "fet_version": self._get_fet_version(),
        "configuration": self._fet_configuration.copy(),
        "creation_timestamp": datetime.now().isoformat(),
        "data_characteristics_hash": self._calculate_data_hash(),
        "compatibility_info": {
            "input_data_type": self.fet_data_type_input.value,
            "output_data_type": self.fet_data_type_output.value,
            "is_lossless": self.fet_encoder_is_lossless,
            "supports_decoding": self.fet_is_decoder
        }
    }
```

#### `load_serialized_fet_configuration(self, serialized_data: Dict[str, Any]) -> None`

**Complete configuration restoration from serialized data**.

```python
def load_serialized_fet_configuration(self, serialized_data: Dict[str, Any]) -> None:
    """
    Restore FET configuration from serialized data.

    Reconstructs the complete transformation state from saved configuration,
    enabling reuse of optimized parameters across sessions.

    Args:
        serialized_data: Configuration dictionary from serialize_fet_configuration()

    Raises:
        FETConfigurationError: If configuration data is invalid or incompatible
    """
```

### Cost and Compatibility Assessment

#### `cls_get_activation_cost(cls, column_datas_infos: Column_datas_infos) -> int`

**Comprehensive cost calculation considering multiple factors**.

```python
@classmethod
@abstractmethod
def cls_get_activation_cost(cls, column_datas_infos: Column_datas_infos) -> int:
    """
    Calculate the computational cost of applying this transformation.

    The cost represents the relative computational expense and resource
    requirements of the transformation. Higher costs indicate more
    complex processing or greater resource demands.

    Args:
        column_datas_infos: Column metadata and statistics

    Returns:
        Integer cost value (higher = more expensive)
    """
```

**Cost Calculation Methodology**:
```python
@classmethod
def cls_get_activation_cost(cls, column_datas_infos: Column_datas_infos) -> int:
    base_cost = cls._get_base_transformation_cost()

    # Factor 1: Output dimensionality cost
    output_columns = cls._calculate_output_columns(column_datas_infos)
    dimensionality_cost = output_columns * cls.DIMENSIONALITY_COST_FACTOR

    # Factor 2: Data size cost
    data_size = column_datas_infos.unique_value_count
    size_cost = min(data_size / 1000, 10) * cls.DATA_SIZE_COST_FACTOR

    # Factor 3: Computational complexity cost
    complexity_cost = cls._get_computational_complexity_cost(column_datas_infos)

    # Factor 4: Memory usage cost
    memory_cost = cls._calculate_memory_usage_cost(column_datas_infos)

    total_cost = base_cost + dimensionality_cost + size_cost + complexity_cost + memory_cost

    return max(1, int(total_cost))  # Minimum cost of 1
```

#### `cls_is_possible_to_enable_this_fet_with_this_infos(cls, column_datas_infos: Column_datas_infos) -> bool`

**Multi-dimensional compatibility assessment**.

```python
@classmethod
def cls_is_possible_to_enable_this_fet_with_this_infos(cls,
                                                     column_datas_infos: Column_datas_infos) -> bool:
    """
    Determine if this FET is compatible with the given column characteristics.

    Performs comprehensive compatibility checking across data types,
    value ranges, statistical properties, and other constraints.

    Args:
        column_datas_infos: Complete column analysis results

    Returns:
        True if FET can be safely applied, False otherwise
    """
```

**Compatibility Validation Process**:
```python
@classmethod
def cls_is_possible_to_enable_this_fet_with_this_infos(cls,
                                                     column_datas_infos: Column_datas_infos) -> bool:

    # Check 1: Data type compatibility
    if not cls._is_data_type_compatible(column_datas_infos.datatype):
        return False

    # Check 2: Value range validation
    if not cls._are_value_ranges_acceptable(column_datas_infos):
        return False

    # Check 3: Statistical property constraints
    if not cls._validate_statistical_properties(column_datas_infos):
        return False

    # Check 4: Data quality requirements
    if not cls._check_data_quality_requirements(column_datas_infos):
        return False

    # Check 5: Resource availability
    if not cls._validate_resource_requirements(column_datas_infos):
        return False

    return True
```

### Additional Utility Methods

#### `get_capabilities(self) -> Dict[str, Any]`

**Comprehensive capability reporting for system integration**.

```python
def get_capabilities(self) -> Dict[str, Any]:
    """
    Return detailed information about FET capabilities and characteristics.

    Provides metadata for integration with other system components,
    enabling intelligent FET selection and optimization.

    Returns:
        Dictionary containing capability information
    """
    return {
        "name": self.fet_name,
        "description": self.fet_description,
        "input_type": self.fet_data_type_input,
        "output_type": self.fet_data_type_output,
        "is_encoder": self.fet_is_encoder,
        "is_decoder": self.fet_is_decoder,
        "is_lossless": self.fet_encoder_is_lossless,
        "typical_cost": self._get_typical_cost(),
        "supported_data_ranges": self._get_supported_ranges(),
        "performance_characteristics": self._get_performance_characteristics()
    }
```

#### `assess_risk(self, column_datas_infos: Column_datas_infos) -> Dict[str, float]`

**Risk assessment for transformation safety evaluation**.

```python
def assess_risk(self, column_datas_infos: Column_datas_infos) -> Dict[str, float]:
    """
    Assess potential risks and safety concerns for applying this transformation.

    Evaluates data compatibility, potential information loss, numerical stability,
    and other risk factors to guide safe FET application.

    Args:
        column_datas_infos: Column characteristics for risk assessment

    Returns:
        Dictionary with risk scores (0.0 = no risk, 1.0 = high risk)
    """
    return {
        "data_loss_risk": self._calculate_data_loss_risk(column_datas_infos),
        "numerical_stability_risk": self._calculate_numerical_stability_risk(column_datas_infos),
        "performance_impact_risk": self._calculate_performance_impact_risk(column_datas_infos),
        "compatibility_risk": self._calculate_compatibility_risk(column_datas_infos),
        "overall_risk_score": self._calculate_overall_risk_score(column_datas_infos)
    }
```

## Usage Patterns and Examples

### Basic FET Instantiation and Usage

```python
from ML import FeatureEngineeringTemplate
from ML.DataFileReader import Column_datas_infos, DatasetColumnDataType
import numpy as np

# Example: Using FETNumericMinMaxFloat for age normalization
def demonstrate_basic_fet_usage():
    # Sample data
    age_data = np.array([25, 30, 35, 40, 45, 50])

    # Create column metadata
    column_info = Column_datas_infos(
        name="customer_age",
        is_input=True,
        is_output=False,
        datatype=DatasetColumnDataType.FLOAT,
        description_user_df="Customer age in years",
        unique_value_count=6,
        missing_percentage=0.0,
        min=25.0, max=50.0,
        mean=37.5, std_dev=9.35,
        skewness=0.0, kurtosis=-1.2,
        quantile02=26.0, quantile03=33.0,
        quantile07=43.0, quantile08=47.0,
        sem=3.82, median=37.5, mode=25,
        str_percent_uppercase=0.0, str_percent_lowercase=0.0,
        str_percent_digit=100.0, str_percent_punctuation=0.0,
        str_percent_operators=0.0, str_percent_underscore=0.0,
        str_percent_space=0.0,
        str_language_en=0.0, str_language_fr=0.0, str_language_de=0.0,
        str_language_it=0.0, str_language_es=0.0, str_language_pt=0.0,
        str_language_others=0.0, str_language_none=100.0,
        fet_list=[]
    )

    # Instantiate FET
    min_max_fet = FETNumericMinMaxFloat()
    min_max_fet.fet_name = "FETNumericMinMaxFloat"
    min_max_fet.fet_description = "Min-Max scaling to 0-1 range"

    # Create configuration
    min_max_fet._create_configuration(age_data, column_info)

    # Encode data
    encoded_age = min_max_fet.encode(age_data)
    print(f"Original age data: {age_data}")
    print(f"Encoded age data: {encoded_age}")

    # Decode back (lossless transformation)
    decoded_age = min_max_fet.decode(encoded_age)
    print(f"Decoded age data: {decoded_age}")
    print(f"Perfect reconstruction: {np.allclose(age_data, decoded_age)}")

    # Serialize configuration
    config = min_max_fet.serialize_fet_configuration()
    print(f"Serialized config keys: {list(config.keys())}")

# Usage
demonstrate_basic_fet_usage()
```

### Advanced FET Management and Selection

```python
from ML import FeatureEngineeringTemplate
from typing import List, Dict, Any

class FETManager:
    """
    Advanced FET management with compatibility checking and optimization.
    """

    def __init__(self):
        self.available_fets = self._load_all_fet_classes()

    def _load_all_fet_classes(self) -> Dict[str, type]:
        """Load all available FET classes dynamically."""
        fet_classes = {}
        # This would dynamically import and register all FET classes
        # For demonstration, we'll show a few key ones
        fet_classes["FETNumericMinMaxFloat"] = FETNumericMinMaxFloat
        fet_classes["FETNumericStandardFloat"] = FETNumericStandardFloat
        fet_classes["FETNumericRobustScalerFloat"] = FETNumericRobustScalerFloat
        return fet_classes

    def find_compatible_fets(self, column_info: Column_datas_infos) -> List[str]:
        """Find all FETs compatible with the given column."""
        compatible_fets = []

        for fet_name, fet_class in self.available_fets.items():
            if fet_class.cls_is_possible_to_enable_this_fet_with_this_infos(column_info):
                compatible_fets.append(fet_name)

        return compatible_fets

    def select_best_fet(self, column_info: Column_datas_infos,
                       budget_limit: int = None) -> str:
        """Select the best FET for the column based on cost and capabilities."""

        compatible_fets = self.find_compatible_fets(column_info)

        if not compatible_fets:
            raise ValueError(f"No compatible FETs found for column {column_info.name}")

        best_fet = None
        best_score = float('inf')

        for fet_name in compatible_fets:
            fet_class = self.available_fets[fet_name]
            cost = fet_class.cls_get_activation_cost(column_info)

            if budget_limit and cost > budget_limit:
                continue

            # Score based on cost (lower is better)
            score = cost

            if score < best_score:
                best_score = score
                best_fet = fet_name

        return best_fet

    def create_optimized_fet(self, fet_name: str, column_data: np.ndarray,
                           column_info: Column_datas_infos):
        """Create and configure an optimized FET instance."""

        fet_class = self.available_fets[fet_name]
        fet_instance = fet_class()
        fet_instance.fet_name = fet_name

        # Configure FET
        fet_instance._create_configuration(column_data, column_info)

        return fet_instance

# Usage example
def optimize_column_transformation():
    fet_manager = FETManager()

    # Find compatible FETs
    compatible = fet_manager.find_compatible_fets(column_info)
    print(f"Compatible FETs: {compatible}")

    # Select best FET within budget
    best_fet_name = fet_manager.select_best_fet(column_info, budget_limit=10)
    print(f"Selected FET: {best_fet_name}")

    # Create optimized FET
    optimized_fet = fet_manager.create_optimized_fet(best_fet_name, age_data, column_info)

    # Use the FET
    encoded = optimized_fet.encode(age_data)
    print(f"Transformation successful, output shape: {encoded.shape}")

# optimize_column_transformation()
```

### FET Cost Analysis and Budget Management

```python
def analyze_fet_costs():
    """Analyze and compare FET costs across different scenarios."""

    test_scenarios = [
        {"name": "Small dataset", "unique_values": 100, "data_type": DatasetColumnDataType.FLOAT},
        {"name": "Medium dataset", "unique_values": 10000, "data_type": DatasetColumnDataType.FLOAT},
        {"name": "Large dataset", "unique_values": 100000, "data_type": DatasetColumnDataType.FLOAT},
        {"name": "Categorical data", "unique_values": 50, "data_type": DatasetColumnDataType.LABEL},
    ]

    fets_to_test = ["FETNumericMinMaxFloat", "FETNumericStandardFloat", "FETNumericRobustScalerFloat"]

    print("FET Cost Analysis:")
    print("=" * 50)

    for scenario in test_scenarios:
        print(f"\nScenario: {scenario['name']}")

        # Create mock column info
        column_info = Column_datas_infos(
            name=f"test_column_{scenario['name'].lower().replace(' ', '_')}",
            is_input=True, is_output=False,
            datatype=scenario['data_type'],
            description_user_df="Test column",
            unique_value_count=scenario['unique_values'],
            missing_percentage=0.0,
            min=0.0, max=100.0, mean=50.0, std_dev=25.0,
            skewness=0.0, kurtosis=0.0,
            quantile02=10.0, quantile03=25.0, quantile07=75.0, quantile08=90.0,
            sem=5.0, median=50.0, mode=50,
            str_percent_uppercase=0.0, str_percent_lowercase=0.0,
            str_percent_digit=0.0, str_percent_punctuation=0.0,
            str_percent_operators=0.0, str_percent_underscore=0.0,
            str_percent_space=0.0,
            str_language_en=0.0, str_language_fr=0.0, str_language_de=0.0,
            str_language_it=0.0, str_language_es=0.0, str_language_pt=0.0,
            str_language_others=0.0, str_language_none=0.0,
            fet_list=[]
        )

        for fet_name in fets_to_test:
            if fet_name in fet_manager.available_fets:
                fet_class = fet_manager.available_fets[fet_name]
                if fet_class.cls_is_possible_to_enable_this_fet_with_this_infos(column_info):
                    cost = fet_class.cls_get_activation_cost(column_info)
                    print(f"  {fet_name}: Cost = {cost}")
                else:
                    print(f"  {fet_name}: Not compatible")

# analyze_fet_costs()
```

## Major FET Categories

### Numeric Transformations

#### FETNumericMinMaxFloat - Linear Scaling

**Purpose**: Min-Max scaling that transforms numeric data to a fixed 0-1 range using linear interpolation.

**Technical Details**:
```python
class FETNumericMinMaxFloat(FeatureEngineeringTemplate):
    """
    Min-Max scaling: X_scaled = (X - X_min) / (X_max - X_min)

    Characteristics:
    - Preserves relative distances between data points
    - Sensitive to outliers (extreme values affect scaling)
    - Always produces outputs in [0, 1] range
    - Lossless transformation (perfect reconstruction possible)
    """

    fet_name = "FETNumericMinMaxFloat"
    fet_description = "Min-Max scaling to 0-1 range preserving relative distances"
    fet_encoder_is_lossless = True
    fet_is_encoder = True
    fet_is_decoder = True
    fet_data_type_input = DatasetColumnDataType.FLOAT
    fet_data_type_output = DatasetColumnDataType.FLOAT
```

**Usage Example**:
```python
# Input: [10, 20, 30, 40, 50]
# Output: [0.0, 0.25, 0.5, 0.75, 1.0]
min_max_fet = FETNumericMinMaxFloat()
min_max_fet._create_configuration(data, column_info)
scaled_data = min_max_fet.encode(data)
```

**Best For**: Data with known bounds, uniform distributions, when preserving relative distances is important.

#### FETNumericStandardFloat - Z-Score Normalization

**Purpose**: Standard (Z-score) normalization that centers data around mean 0 with standard deviation 1.

**Technical Details**:
```python
class FETNumericStandardFloat(FeatureEngineeringTemplate):
    """
    Z-score normalization: X_scaled = (X - μ) / σ

    Where μ is mean and σ is standard deviation.

    Characteristics:
    - Centers data around mean = 0
    - Scales data to std = 1
    - Sensitive to outliers (affects μ and σ calculation)
    - Lossless transformation
    - Useful for algorithms assuming Gaussian distributions
    """

    fet_name = "FETNumericStandardFloat"
    fet_description = "Z-score normalization (mean=0, std=1)"
    fet_encoder_is_lossless = True
    fet_is_encoder = True
    fet_is_decoder = True
    fet_data_type_input = DatasetColumnDataType.FLOAT
    fet_data_type_output = DatasetColumnDataType.FLOAT
```

**Mathematical Properties**:
- **Mean**: 0 (by construction)
- **Standard Deviation**: 1 (by construction)
- **Preserves shape**: Only location and scale change
- **Outlier sensitivity**: Extreme values distort the transformation

**Best For**: Normally distributed data, gradient-based optimization algorithms, when centering is required.

#### FETNumericRobustScalerFloat - Outlier-Resistant Scaling

**Purpose**: Robust scaling using median and IQR (Interquartile Range) for outlier resistance.

**Technical Details**:
```python
class FETNumericRobustScalerFloat(FeatureEngineeringTemplate):
    """
    Robust scaling: X_scaled = (X - median) / IQR

    Where IQR = Q3 - Q1 (75th percentile - 25th percentile)

    Characteristics:
    - Resistant to outliers
    - Uses median instead of mean
    - Uses IQR instead of standard deviation
    - Lossless transformation
    - More stable than standard scaling with outliers
    """

    fet_name = "FETNumericRobustScalerFloat"
    fet_description = "Robust scaling using median and IQR"
    fet_encoder_is_lossless = True
    fet_is_encoder = True
    fet_is_decoder = True
    fet_data_type_input = DatasetColumnDataType.FLOAT
    fet_data_type_output = DatasetColumnDataType.FLOAT
```

**Robust Statistics**:
- **Median**: 50th percentile (robust location estimate)
- **IQR**: Q3 - Q1 (robust scale estimate)
- **Breakdown point**: 50% (can handle up to 50% outliers)

**Best For**: Data with outliers, heavy-tailed distributions, when robustness is more important than efficiency.

#### FET6PowerFloat - Power Transformations

**Purpose**: Apply power transformations (x^6) to stabilize variance and make data more Gaussian-like.

**Technical Details**:
```python
class FET6PowerFloat(FeatureEngineeringTemplate):
    """
    Power transformation: X_transformed = X^6

    Characteristics:
    - Stabilizes variance for positive data
    - Can make distributions more symmetric
    - Non-linear transformation
    - Lossy (information may be lost)
    - One-way transformation (no decoder)
    """

    fet_name = "FET6PowerFloat"
    fet_description = "Sixth power transformation for variance stabilization"
    fet_encoder_is_lossless = False
    fet_is_encoder = True
    fet_is_decoder = False
    fet_data_type_input = DatasetColumnDataType.FLOAT
    fet_data_type_output = DatasetColumnDataType.FLOAT
```

**Use Cases**:
- **Variance stabilization**: When variance increases with mean
- **Symmetry improvement**: Making right-skewed distributions more symmetric
- **Normalization**: Preparing data for parametric statistical tests

**Important Notes**:
- Only works with positive values (x > 0)
- Can amplify small values and compress large values
- Useful for count data or measurements with multiplicative errors

### Categorical Transformations

#### FETMultiplexerAllLabel - One-Hot Encoding

**Purpose**: Convert categorical variables into binary vectors using one-hot encoding, creating separate binary columns for each category.

**Technical Details**:
```python
class FETMultiplexerAllLabel(FeatureEngineeringTemplate):
    """
    One-hot encoding: Convert categorical to binary vectors

    For N categories, creates N binary columns where each column
    represents one category (1 if sample belongs to category, 0 otherwise).

    Characteristics:
    - Creates N binary columns for N categories
    - No information loss (perfect reconstruction possible)
    - Suitable for nominal categorical variables
    - Can create high-dimensional sparse matrices
    """

    fet_name = "FETMultiplexerAllLabel"
    fet_description = "One-hot encoding for categorical variables"
    fet_encoder_is_lossless = True
    fet_is_encoder = True
    fet_is_decoder = True
    fet_data_type_input = DatasetColumnDataType.LABEL
    fet_data_type_output = DatasetColumnDataType.FLOAT  # Binary floats
```

**Encoding Example**:
```python
# Input: ["red", "blue", "red", "green"]
# Output:
# [[1, 0, 0],  # red
#  [0, 1, 0],  # blue
#  [1, 0, 0],  # red
#  [0, 0, 1]]  # green
```

**Best For**: Nominal categorical variables, when category relationships are not ordinal, small number of categories.

#### FETOrdinalEncoderLabel - Ordinal Encoding

**Purpose**: Convert ordinal categorical variables to numeric codes, preserving the order relationship between categories.

**Technical Details**:
```python
class FETOrdinalEncoderLabel(FeatureEngineeringTemplate):
    """
    Ordinal encoding: Convert ordered categories to numeric codes

    Maps categories to integers based on their ordinal relationship.
    Requires knowledge of the correct category ordering.

    Characteristics:
    - Preserves ordinal relationships
    - Single output column
    - Requires predefined category ordering
    - Lossless if ordering is correct
    """

    fet_name = "FETOrdinalEncoderLabel"
    fet_description = "Ordinal encoding preserving category order"
    fet_encoder_is_lossless = True  # If ordering is semantically correct
    fet_is_encoder = True
    fet_is_decoder = True
    fet_data_type_input = DatasetColumnDataType.LABEL
    fet_data_type_output = DatasetColumnDataType.FLOAT
```

**Encoding Example**:
```python
# Input: ["low", "medium", "high", "low"] with order ["low", "medium", "high"]
# Output: [0, 1, 2, 0]
```

**Best For**: Ordinal categorical variables where order matters (e.g., "low/medium/high", "small/medium/large").

### Text and Language Transformations

#### FETLanguageDetection - Language Identification

**Purpose**: Identify the language of text data and create language indicator features.

**Technical Details**:
```python
class FETLanguageDetection(FeatureEngineeringTemplate):
    """
    Language detection: Identify text language and create features

    Uses language detection algorithms to identify the language of text
    and creates binary or probabilistic language indicator features.

    Characteristics:
    - Creates multiple language indicator columns
    - Useful for multilingual datasets
    - Can improve model performance for language-specific patterns
    - Requires sufficient text length for accurate detection
    """

    fet_name = "FETLanguageDetection"
    fet_description = "Language detection and encoding"
    fet_encoder_is_lossless = False  # Language info is derived, not preserved
    fet_is_encoder = True
    fet_is_decoder = False
    fet_data_type_input = DatasetColumnDataType.LANGUAGE
    fet_data_type_output = DatasetColumnDataType.FLOAT
```

**Output Features**: Binary columns for each supported language (English, French, German, Italian, Spanish, Portuguese, etc.).

### Date and Time Transformations

#### FETDateTimeFeatures - Temporal Feature Extraction

**Purpose**: Extract meaningful features from date/time data such as hour, day, month, year, weekday, etc.

**Technical Details**:
```python
class FETDateTimeFeatures(FeatureEngineeringTemplate):
    """
    Date/time feature extraction: Create temporal features from datetime

    Extracts multiple temporal features:
    - Hour of day (0-23)
    - Day of month (1-31)
    - Month (1-12)
    - Year
    - Day of week (0-6)
    - Weekend flag
    - Quarter
    - Season
    - etc.

    Characteristics:
    - Creates multiple output columns
    - Preserves temporal information
    - Enables time-series and calendar-based patterns
    - Lossless transformation (original datetime can be reconstructed)
    """

    fet_name = "FETDateTimeFeatures"
    fet_description = "Extract temporal features from datetime data"
    fet_encoder_is_lossless = True
    fet_is_encoder = True
    fet_is_decoder = True
    fet_data_type_input = DatasetColumnDataType.DATETIME
    fet_data_type_output = DatasetColumnDataType.FLOAT
```

**Common Extracted Features**:
- `hour`: Hour of day (0-23)
- `day_of_month`: Day of month (1-31)
- `month`: Month (1-12)
- `year`: Year
- `day_of_week`: Day of week (0=Monday, 6=Sunday)
- `is_weekend`: Binary weekend indicator
- `quarter`: Quarter of year (1-4)
- `season`: Season (0=Winter, 1=Spring, 2=Summer, 3=Fall)

### Specialized Transformations

#### FETIsMissing - Missing Value Indicators

**Purpose**: Create binary indicators for missing values while preserving the information about which values were originally missing.

**Technical Details**:
```python
class FETIsMissing(FeatureEngineeringTemplate):
    """
    Missing value indicators: Create binary flags for missing data

    For each column with missing values, creates an indicator column
    that shows which rows originally had missing values. This preserves
    the information about missingness patterns.

    Characteristics:
    - Creates binary indicator columns
    - Preserves missing value information
    - Helps models learn from missing value patterns
    - Can be combined with imputation strategies
    """

    fet_name = "FETIsMissing"
    fet_description = "Create missing value indicators"
    fet_encoder_is_lossless = False
    fet_is_encoder = True
    fet_is_decoder = False
    fet_data_type_input = DatasetColumnDataType.FLOAT  # Can work with any type
    fet_data_type_output = DatasetColumnDataType.FLOAT
```

**Usage Pattern**:
```python
# Original: [1.0, NaN, 3.0, NaN, 5.0]
# Missing indicators: [0, 1, 0, 1, 0]
# (Then apply imputation to fill NaN values)
```

## FET Selection and Optimization Strategy

### Compatibility Matrix

| FET Category | FLOAT | LABEL | LANGUAGE | DATETIME | JSON | IMAGE |
|-------------|-------|-------|----------|----------|------|-------|
| Numeric Scaling | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Categorical Encoding | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Text Processing | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| Temporal Features | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| Missing Value Handling | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

### Cost-Based Selection Algorithm

```python
def select_optimal_fets(column_info: Column_datas_infos, budget: int) -> List[str]:
    """
    Select optimal FET combination based on cost and compatibility.
    """

    # Step 1: Find all compatible FETs
    compatible_fets = find_compatible_fets(column_info)

    # Step 2: Calculate cost-benefit ratios
    fet_scores = []
    for fet_name in compatible_fets:
        fet_class = get_fet_class(fet_name)
        cost = fet_class.cls_get_activation_cost(column_info)
        benefit = estimate_fet_benefit(fet_name, column_info)

        if cost <= budget:
            score = benefit / cost if cost > 0 else float('inf')
            fet_scores.append((fet_name, score, cost))

    # Step 3: Select optimal combination using knapsack-like algorithm
    selected_fets = select_fet_combination(fet_scores, budget)

    return selected_fets

def estimate_fet_benefit(fet_name: str, column_info: Column_datas_infos) -> float:
    """
    Estimate the expected benefit of applying a FET.
    """

    # Base benefit scores by FET type
    base_benefits = {
        "FETNumericMinMaxFloat": 0.7,
        "FETNumericStandardFloat": 0.8,
        "FETNumericRobustScalerFloat": 0.6,
        "FETMultiplexerAllLabel": 0.9,
        "FETDateTimeFeatures": 0.8,
        "FETLanguageDetection": 0.5,
    }

    benefit = base_benefits.get(fet_name, 0.5)

    # Adjust based on data characteristics
    if column_info.missing_percentage > 0.1:
        benefit *= 1.2  # Missing values increase transformation value

    if column_info.unique_value_count > 100:
        benefit *= 0.8  # High cardinality may reduce effectiveness

    return benefit
```

## Integration with EasyAutoML Pipeline

### Automatic FET Selection Workflow

```python
def automated_fet_pipeline(data_path: str, target_column: str):
    """
    Complete automated feature engineering pipeline.
    """

    # Phase 1: Data ingestion and analysis
    machine = Machine.create_from_dataset(data_path, target_column)
    data_config = machine.get_data_configuration()

    # Phase 2: Column analysis
    column_analyses = {}
    for column_name in data_config.get_column_names():
        column_analyses[column_name] = machine.get_column_analysis(column_name)

    # Phase 3: Budget allocation
    total_budget = calculate_machine_budget(machine)
    column_budgets = allocate_budget_by_importance(column_analyses, total_budget)

    # Phase 4: FET optimization per column
    fet_configurations = {}
    for column_name, budget in column_budgets.items():
        column_info = column_analyses[column_name]
        optimal_fets = select_optimal_fets(column_info, budget)
        fet_configurations[column_name] = optimal_fets

    # Phase 5: Configuration persistence
    fec = FeatureEngineeringConfiguration(machine, global_dataset_budget=total_budget)
    # Apply optimized configurations...

    return machine, fec
```

## Performance Characteristics and Benchmarks

### Computational Complexity

| FET Type | Time Complexity | Space Complexity | Typical Cost |
|----------|----------------|------------------|--------------|
| Numeric Scaling | O(n) | O(1) | 1-2 |
| One-hot Encoding | O(n×c) | O(c) | c (categories) |
| DateTime Features | O(n) | O(1) | 8-12 |
| Language Detection | O(n×l) | O(l) | 5-10 |
| Power Transformations | O(n) | O(1) | 1-2 |

Where:
- n = number of samples
- c = number of categories
- l = average text length

### Memory Usage Patterns

- **Streaming FETs**: Process data in chunks (min memory usage)
- **Batch FETs**: Load entire column into memory
- **Expansion FETs**: May significantly increase memory requirements (e.g., one-hot encoding)

### Scalability Considerations

- **Large Datasets**: Use streaming-compatible FETs and chunked processing
- **High Cardinality**: Be cautious with expansion FETs (one-hot encoding)
- **Mixed Types**: Balance computational cost across different data types

## Best Practices and Recommendations

### FET Selection Guidelines

1. **Start Simple**: Begin with lossless transformations (MinMax, Standard scaling)
2. **Handle Missing Values**: Use missing value indicators before imputation
3. **Consider Cardinality**: Avoid one-hot encoding for high-cardinality categorical variables
4. **Balance Cost and Benefit**: Don't overspend budget on marginal improvements
5. **Validate Assumptions**: Ensure ordinal encodings reflect true category relationships

### Performance Optimization Tips

1. **Caching**: Reuse FET configurations across similar datasets
2. **Parallelization**: Process independent columns in parallel
3. **Memory Management**: Use streaming for large datasets
4. **Early Stopping**: Stop optimization when marginal gains are minimal
5. **Profiling**: Monitor resource usage and optimize bottlenecks

### Common Pitfalls to Avoid

1. **Data Leakage**: Don't use target information for feature engineering
2. **Over-engineering**: Too many transformations can hurt performance
3. **Ignoring Costs**: High-cost FETs may not be worth the computational expense
4. **Type Mismatches**: Ensure FETs are compatible with data types
5. **Lossy Transformations**: Be aware when information cannot be recovered

## Future Enhancements

### Planned FET Additions

- **Advanced Text Processing**: BERT embeddings, topic modeling
- **Image Processing**: CNN feature extraction, image embeddings
- **Time Series**: Lag features, rolling statistics, seasonal decomposition
- **Geospatial**: Coordinate transformations, distance calculations
- **Domain-Specific**: Industry-specific feature engineering templates

### Research Directions

- **Automated FET Discovery**: Machine learning to create custom transformations
- **Multi-Column FETs**: Transformations that consider relationships between columns
- **Adaptive FETs**: Transformations that adapt based on model feedback
- **Energy-Aware FETs**: Cost functions considering computational energy usage

This comprehensive FET system provides the foundation for intelligent, automated feature engineering in the EasyAutoML platform, enabling users to extract maximum value from their data with minimal manual intervention.
