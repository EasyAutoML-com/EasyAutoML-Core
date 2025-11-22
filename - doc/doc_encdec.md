# EncDec.py - Encoder/Decoder Configuration and Processing Module

## Overview

The `EncDec` class is a critical component in the EasyAutoML machine learning pipeline that handles the encoding and decoding of data between human-readable formats and neural network-compatible formats. It serves as the bridge between pre-processed data and the ML model, managing complex feature engineering transformations while maintaining data integrity and performance.

**Location**: `ML/EncDec.py`

## Description

### What

The EncDec module transforms data between human-readable formats and neural network-compatible formats, serving as the critical bridge in the ML pipeline. It manages encoding configurations that map pre-encoded columns to their Feature Engineering Template (FET) transformations, ensuring data is properly formatted for neural network consumption and predictions are decoded back to interpretable results.

### How

It applies configured Feature Engineering Templates to each column, transforming data through encode_for_ai() which produces 0-1 normalized values, and decode_from_ai() which intelligently merges multiple FET predictions back to human-readable format. The system uses LRU caching for performance optimization and handles complex data types including datetime splitting.

### Where

Used by NNEngine during training and prediction pipelines to prepare data for neural networks and interpret model outputs. Also utilized by Machine during configuration setup and by prediction APIs for end-to-end data transformation.

### When

Called during model training for encoding input/output data and during inference for encoding inputs and decoding prediction results.

## Core Functionality

### Primary Operations

1. **Configuration Creation**: Analyzes dataset structure and creates encoding/decoding configurations
2. **Data Encoding**: Transforms pre-encoded data into neural network compatible format (0-1 normalized values)
3. **Data Decoding**: Converts AI predictions back to human-readable format with intelligent result merging
4. **Configuration Persistence**: Saves and loads encoding configurations from database with versioning
5. **Performance Optimization**: Implements caching and batch processing for large datasets

### Key Configuration Structure

The EncDec configuration is a hierarchical dictionary that maps each pre-encoded column to its transformation pipeline:

```python
self._enc_dec_configuration = {
    "customer_age": {  # Pre-encoded column name
        "is_input": True,           # Used as neural network input
        "is_output": False,         # Not a prediction target
        "column_datatype_enum": DatasetColumnDataType.FLOAT,
        "column_datatype_name": "FLOAT",
        "fet_list": [               # List of Feature Engineering Templates
            {
                "fet_column_name": "customer_age_normalized",
                "fet_class": FETNumericMinMaxFloat,  # FET instance
                "fet_class_name": "FETNumericMinMaxFloat",
                "list_encoded_columns_name": ["customer_age_normalized"],
                "fet_serialized_config": {
                    "min_value": 18.0,
                    "max_value": 85.0,
                    "feature_range": [0.0, 1.0]
                }
            },
            {
                "fet_column_name": "customer_age_power2",
                "fet_class": FET6PowerFloat,
                "fet_class_name": "FET6PowerFloat",
                "list_encoded_columns_name": [
                    "customer_age_power0_33",
                    "customer_age_power0_5",
                    "customer_age_power0_66",
                    "customer_age_power1_5",
                    "customer_age_power2",
                    "customer_age_power3"
                ],
                "fet_serialized_config": {
                    "powers": [0.33, 0.5, 0.66, 1.5, 2.0, 3.0]
                }
            }
        ]
    },
    "product_category": {
        "is_input": True,
        "is_output": False,
        "column_datatype_enum": DatasetColumnDataType.LABEL,
        "column_datatype_name": "LABEL",
        "fet_list": [
            {
                "fet_column_name": "product_category_encoded",
                "fet_class": FETMultiplexerAllLabel,
                "fet_class_name": "FETMultiplexerAllLabel",
                "list_encoded_columns_name": [
                    "category_electronics",
                    "category_clothing",
                    "category_books",
                    "category_home"
                ],
                "fet_serialized_config": {
                    "categories": ["electronics", "clothing", "books", "home"],
                    "handle_unknown": "error"
                }
            }
        ]
    }
}
```

### Configuration Metadata

Each configuration includes additional metadata for optimization and monitoring:

```python
self._column_counts = {
    "input_encoded_columns_count": 45,    # Total input features after encoding
    "output_encoded_columns_count": 3,    # Total output features after encoding
    "total_encoded_columns_count": 48     # Grand total of encoded features
}

self._configuration_timestamp = timezone.now()  # When configuration was created
self._configuration_version = "2.1.0"          # EncDec version for compatibility
```

## Main Functions Deep Analysis

### `__init__(self, machine: Machine, dataframe_pre_encoded: Optional[pd.DataFrame] = None)`

**Purpose**: Initialize EncDec instance with either loading existing configuration or creating new one.

**Deep Workflow**:
1. **Configuration Loading Mode** (`dataframe_pre_encoded = None`):
   - Calls `_init_load_configuration()` to retrieve stored configuration
   - Loads column counts and deserializes FET configurations
   - Validates configuration integrity

2. **Configuration Creation Mode** (`dataframe_pre_encoded` provided):
   - Calls `_init_create_configuration()` to build new configuration
   - Performs comprehensive data analysis and FET instantiation
   - Handles datetime column splitting (date/time separation)
   - Manages error recovery and warning systems

**Critical Interactions**:
- **Machine**: Provides access to stored configurations and data
- **MachineDataConfiguration (MDC)**: Supplies column metadata and data types
- **FeatureEngineeringConfiguration (FEC)**: Defines which FETs are enabled per column
- **FeatureEngineeringTemplate (FET)**: Provides actual encoding/decoding implementations

### `encode_for_ai(self, pre_encoded_dataframe: pd.DataFrame) -> pd.DataFrame`

**Purpose**: Transform pre-encoded data into neural network compatible format (0-1 normalized values).

**Deep Processing Flow**:

#### 1. Input Validation
```python
def encode_for_ai(self, pre_encoded_dataframe: pd.DataFrame) -> pd.DataFrame:
    # Validate input parameters
    if not isinstance(pre_encoded_dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    if pre_encoded_dataframe.empty:
        raise ValueError("Input DataFrame cannot be empty")

    if len(pre_encoded_dataframe) < 5:
        logger.warning("Dataset has fewer than 5 rows - limited statistical reliability")

    # Verify configuration exists
    if not self._enc_dec_configuration:
        raise RuntimeError("EncDec configuration not initialized")
```

#### 2. Column-by-Column Processing Pipeline
```python
encoded_columns_data = []

for column_name in pre_encoded_dataframe.columns:
    # Skip columns not in configuration (e.g., IGNORE columns)
    if column_name not in self._enc_dec_configuration:
        continue

    column_config = self._enc_dec_configuration[column_name]
    column_data = pre_encoded_dataframe[column_name].values

    # Special handling for DATETIME columns
    if column_config["column_datatype_enum"] == DatasetColumnDataType.DATETIME:
        # Split into separate date and time processing
        date_values = pd.to_datetime(column_data).date
        time_values = pd.to_datetime(column_data).time

        # Process date component
        encoded_date = self._encode_datetime_component(date_values, "date")
        encoded_columns_data.extend(encoded_date)

        # Process time component
        encoded_time = self._encode_datetime_component(time_values, "time")
        encoded_columns_data.extend(encoded_time)
        continue
```

#### 3. FET-Specific Encoding with Caching
```python
# Process each FET for the column
for fet_config in column_config["fet_list"]:
    fet_instance = fet_config["fet_class"]
    fet_name = fet_config["fet_class_name"]

    try:
        # Use caching system for performance
        encoded_data = fet_encoder_caching_system_do_encode_with_cache(
            machine_id=self._machine.id,
            fet_instance=fet_instance,
            column_name=f"{column_name}_{fet_name}",
            values_to_encode_nparray=column_data
        )

        # Store encoded column data
        for i, encoded_column in enumerate(encoded_data.T):
            encoded_columns_data.append(encoded_column)
            encoded_column_names.append(
                fet_config["list_encoded_columns_name"][i]
            )

    except Exception as e:
        # Handle encoding failures gracefully
        error_msg = f"FET {fet_name} failed for column {column_name}: {str(e)}"
        self._store_column_error_in_machine(column_name, error_msg)
        logger.error(error_msg)

        # Use fallback encoding (all zeros for this FET's columns)
        fallback_data = np.zeros((len(column_data), len(fet_config["list_encoded_columns_name"])))
        encoded_columns_data.extend(fallback_data.T)
```

#### 4. Intelligent Caching System
```python
@lru_cache(maxsize=999, typed=True)
def fet_encoder_caching_system_do_real_caching(machine_id, column_name, fet_pickled, values_pickled):
    """
    Cached encoding with hash-based key generation.
    Avoids recomputation for identical inputs.
    """
    # Unpickle FET instance and data
    fet_instance = pickle.loads(fet_pickled)
    values = pickle.loads(values_pickled)

    # Skip caching for small datasets
    if len(values) < 250:
        return fet_instance.encode(values)

    # Perform cached encoding
    return fet_instance.encode(values)
```

#### 5. Result Assembly and Validation
```python
# Combine all encoded columns into final DataFrame
final_encoded_data = np.column_stack(encoded_columns_data)
encoded_df = pd.DataFrame(
    final_encoded_data,
    columns=encoded_column_names,
    index=pre_encoded_dataframe.index
)

# Validate output dimensions
expected_columns = self._column_counts["total_encoded_columns_count"]
if encoded_df.shape[1] != expected_columns:
    raise ValueError(
        f"Encoding produced {encoded_df.shape[1]} columns, "
        f"expected {expected_columns}"
    )

# Ensure all values are in neural network compatible range [0, 1]
if not ((encoded_df >= 0) & (encoded_df <= 1)).all().all():
    logger.warning("Some encoded values outside [0,1] range")

return encoded_df
```

#### Performance Optimizations
- **LRU Caching**: `@lru_cache(maxsize=999, typed=True)` for repeated encodings
- **Memory Management**: Avoids caching for small datasets (< 250 elements)
- **Batch Processing**: Handles multiple FET outputs per column efficiently
- **Parallel Processing**: Column encodings can be parallelized
- **Lazy Evaluation**: FETs are only instantiated when needed

### `decode_from_ai(self, data_encoded_from_ai: pd.DataFrame) -> pd.DataFrame`

**Purpose**: Convert neural network predictions back to human-readable format with intelligent result merging.

**Deep Processing Flow**:

#### 1. Output Column Identification
```python
def decode_from_ai(self, data_encoded_from_ai: pd.DataFrame) -> pd.DataFrame:
    # Filter to only output columns (prediction targets)
    output_columns_config = {
        col_name: config for col_name, config in self._enc_dec_configuration.items()
        if config["is_output"]  # Only process columns that are prediction targets
    }

    if not output_columns_config:
        raise ValueError("No output columns found in configuration")

    # Validate input dimensions
    expected_input_cols = self._column_counts["total_encoded_columns_count"]
    if data_encoded_from_ai.shape[1] != expected_input_cols:
        raise ValueError(
            f"Input has {data_encoded_from_ai.shape[1]} columns, "
            f"expected {expected_input_cols}"
        )
```

#### 2. Multi-FET Result Merging Intelligence
```python
decoded_results = {}

for output_col_name, output_config in output_columns_config.items():
    # Extract encoded columns for this output
    encoded_column_indices = self._get_encoded_column_indices_for_output(output_col_name)
    output_predictions = data_encoded_from_ai.iloc[:, encoded_column_indices]

    # Group by FET type for intelligent merging
    fet_groups = self._group_fet_predictions_by_type(output_config, output_predictions)

    # Apply type-specific merging strategies
    for fet_type, predictions in fet_groups.items():
        if fet_type == "float":
            merged_result = self._merge_float_predictions(predictions)
        elif fet_type == "label":
            merged_result = self._merge_label_predictions(predictions)
        elif fet_type == "date":
            merged_result = self._merge_date_predictions(predictions)
        elif fet_type == "time":
            merged_result = self._merge_time_predictions(predictions)
        else:
            merged_result = self._merge_generic_predictions(predictions)

        decoded_results[output_col_name] = merged_result
```

#### 3. Advanced Merging Algorithms

##### Float Value Merging with Similarity Detection
```python
def _merge_float_predictions(self, predictions_2d_array: np.ndarray) -> np.ndarray:
    """
    Intelligently merge multiple float predictions from different FETs.

    Strategy: Average similar values, handle outliers, preserve distributions.
    """
    # Calculate similarity matrix between predictions
    similarity_matrix = self._calculate_prediction_similarities(predictions_2d_array)

    # Group similar predictions (within 10% relative difference)
    similarity_threshold = 0.10
    similar_groups = self._group_similar_predictions(
        predictions_2d_array, similarity_matrix, similarity_threshold
    )

    # Average within groups, keep best representative from each group
    merged_predictions = np.zeros(len(predictions_2d_array))

    for group in similar_groups:
        group_predictions = predictions_2d_array[group]
        # Use median to reduce outlier influence
        merged_predictions[group] = np.median(group_predictions, axis=0)

    return merged_predictions
```

##### Categorical Label Conflict Resolution
```python
def _merge_label_predictions(self, predictions_2d_array: np.ndarray) -> np.ndarray:
    """
    Resolve conflicts when multiple FETs predict different labels.

    Strategy: Consensus-based resolution with confidence weighting.
    """
    # Convert predictions to label indices
    label_predictions = np.argmax(predictions_2d_array, axis=1)

    # Calculate confidence scores (max probability for each prediction)
    confidences = np.max(predictions_2d_array, axis=1)

    # Group predictions by label
    unique_labels = np.unique(label_predictions)

    merged_predictions = np.zeros(len(label_predictions), dtype=int)

    for i in range(len(label_predictions)):
        # Find all FET predictions for this sample
        sample_predictions = label_predictions[:, i]
        sample_confidences = confidences[:, i]

        # Use confidence-weighted voting
        label_votes = {}
        for label, confidence in zip(sample_predictions, sample_confidences):
            label_votes[label] = label_votes.get(label, 0) + confidence

        # Select label with highest confidence-weighted votes
        merged_predictions[i] = max(label_votes.keys(), key=lambda x: label_votes[x])

    return merged_predictions
```

##### Temporal Data Reconstruction
```python
def _merge_temporal_predictions(self, date_predictions: np.ndarray,
                              time_predictions: np.ndarray) -> np.ndarray:
    """
    Reconstruct datetime from separate date and time predictions.

    Strategy: Combine date and time components intelligently.
    """
    # Convert date predictions back to date objects
    date_objects = self._convert_encoded_dates_to_objects(date_predictions)

    # Convert time predictions back to time objects
    time_objects = self._convert_encoded_times_to_objects(time_predictions)

    # Combine into datetime objects
    datetime_objects = []
    for date_obj, time_obj in zip(date_objects, time_objects):
        # Handle cases where date/time components come from different FETs
        if self._are_temporal_components_consistent(date_obj, time_obj):
            combined_dt = datetime.combine(date_obj, time_obj)
        else:
            # Use fallback strategy (e.g., most confident prediction)
            combined_dt = self._resolve_temporal_conflict(date_obj, time_obj)

        datetime_objects.append(combined_dt)

    return np.array(datetime_objects)
```

#### 4. Data Type Validation and Safety
```python
def _validate_fet_type_compatibility(self, fet_predictions: dict) -> bool:
    """
    Ensure FET types are compatible for merging.
    Prevents mixing incompatible data types.
    """
    fet_types = set(fet_predictions.keys())

    # Define compatible type combinations
    compatible_groups = [
        {"float", "int"},  # Numeric types can be merged
        {"label", "categorical"},  # Categorical types compatible
        {"date"}, {"time"}, {"datetime"}  # Temporal types (separate)
    ]

    # Check if all types belong to the same compatible group
    for group in compatible_groups:
        if fet_types.issubset(group):
            return True

    return False
```

#### 5. Result Formatting and Validation
```python
# Format results according to original data types
final_results = {}
for col_name, predictions in decoded_results.items():
    original_datatype = self._enc_dec_configuration[col_name]["column_datatype_enum"]

    if original_datatype == DatasetColumnDataType.FLOAT:
        # Round to appropriate precision
        final_results[col_name] = np.round(predictions, decimals=6)
    elif original_datatype == DatasetColumnDataType.LABEL:
        # Convert indices back to original labels
        final_results[col_name] = self._convert_indices_to_labels(predictions, col_name)
    elif original_datatype in [DatasetColumnDataType.DATE, DatasetColumnDataType.DATETIME]:
        # Format as readable date/time strings
        final_results[col_name] = self._format_temporal_predictions(predictions)
    else:
        final_results[col_name] = predictions

# Create final DataFrame with proper column names and types
decoded_df = pd.DataFrame(final_results, index=data_encoded_from_ai.index)

# Validate output matches expected schema
self._validate_decoding_output(decoded_df)

return decoded_df
```

## Module Interactions Deep Analysis

### With Machine Class

**Data Flow**:
```
Machine.db_machine.enc_dec_configuration_extfield
    ↓ (Serialization/Deserialization)
EncDec._enc_dec_configuration
    ↓ (Runtime Operations)
Neural Network Compatible Data
```

**Configuration Persistence**:
- `save_configuration_in_machine()`: Serializes FET instances to database
- `_load_config_from_machine_and_deserialize_it()`: Recreates FET instances from stored config
- Handles complex object serialization/deserialization cycles

### With FeatureEngineeringTemplate (FET)

**Runtime Integration**:
```python
# Dynamic FET instantiation
fet_instance = getattr(FeatureEngineeringTemplate, fet_name)(column_data, column_info)

# Encoding execution with error handling
try:
    encoded_data = fet_instance.encode(values_to_encode_nparray)
except Exception as e:
    # Graceful error handling and recovery
    self._store_column_error_in_machine(column_name, error_details)
```

**Supported FET Types**:
- **Float-based**: Normalization, scaling, mathematical transformations
- **Label-based**: Categorical encoding, one-hot encoding
- **Time-based**: Temporal feature extraction
- **Date-based**: Calendar feature generation

### With MachineDataConfiguration (MDC)

**Metadata Utilization**:
```python
columns_name_input = mdc_.columns_name_input
columns_name_output = mdc_.columns_name_output
columns_datatype = mdc_.columns_data_type
```

**Data Type Handling**:
- **DATETIME Splitting**: Automatically separates date and time components
- **IGNORE Columns**: Skips encoding for non-relevant columns
- **Type Validation**: Ensures FET compatibility with data types

### With FeatureEngineeringConfiguration (FEC)

**Configuration Synchronization**:
```python
fe_ = FeatureEngineeringConfiguration(self._machine)
fet_names_list = fe_._activated_fet_list_per_column[column_name]
```

**Dynamic Configuration**:
- Adapts to changing FET selections per column
- Handles configuration updates during machine retraining
- Manages budget constraints for feature engineering

## Error Handling and Recovery

### Warning System

**Column-Level Warnings**:
```python
self._store_column_warning_in_machine(
    column_name,
    user_friendly_message,
    detailed_technical_message
)
```

**Automatic Recovery Triggers**:
- `machine_is_re_run_enc_dec = True`: Triggers EncDec reconfiguration
- `machine_is_re_run_fe = True`: Triggers FEC updates
- `machine_is_re_run_mdc = True`: Triggers MDC reconfiguration

### Error Classification

1. **Fatal Errors**: Stop processing, require manual intervention
2. **Recoverable Errors**: Continue with degraded functionality
3. **Warnings**: Log issues but maintain operation

## Performance Characteristics

### Memory Management

**Large Dataset Handling**:
- Efficient numpy array operations
- Streaming processing for memory-intensive operations
- Automatic garbage collection triggers

**Caching Strategy**:
```python
@lru_cache(maxsize=999, typed=True)
def fet_encoder_caching_system_do_real_caching(machine_id, column_name, fet_pickled, values_pickled):
    # Hash-based caching for performance
    fet_instance = pickle.loads(fet_pickled)
    values = pickle.loads(values_pickled)
    return fet_instance.encode(values)
```

### Computational Complexity

**Encoding Complexity**: O(n × f × c)
- n: Number of data rows
- f: Number of FETs per column
- c: Number of columns

**Optimization Strategies**:
- Parallel FET processing within columns
- Selective caching based on data size
- Memory-efficient data structures

## Configuration Persistence

### Serialization Strategy

**Complex Object Handling**:
```python
# Remove non-serializable objects before storage
for fet_idx in range(len(fet_list)):
    fet_list[fet_idx]["fet_class"] = None  # Will be recreated from fet_class_name
```

**Database Integration**:
- Stores in `machine.enc_dec_configuration_extfield`
- Maintains referential integrity
- Supports version migration

## Testing and Validation

### Data Integrity Checks

**Encoding Validation**:
- Column count verification
- Data type consistency
- Value range validation (0-1 for neural network inputs)

**Decoding Validation**:
- Output column verification
- Multi-FET result consistency
- Data type preservation

## Usage Patterns

### Complete ML Pipeline Integration

```python
from ML import Machine, NNEngine, EncDec
import pandas as pd

# 1. Data Preparation and Machine Creation
machine = Machine(
    machine_identifier_or_name="customer_predictor",
    user_dataset_unformatted=pd.read_csv("customer_data.csv"),
    machine_create_user_id=current_user_id,
    decimal_separator=".",
    date_format="DMY"
)

# 2. Initialize ML Components
nn_engine = NNEngine(machine=machine)
enc_dec = nn_engine._enc_dec  # Access EncDec through NNEngine

# 3. Training Data Encoding
training_data = machine.get_random_user_dataframe_for_training_trial(is_for_learning=True)
encoded_training = enc_dec.encode_for_ai(training_data)

validation_data = machine.get_random_user_dataframe_for_training_trial(is_for_evaluation=True)
encoded_validation = enc_dec.encode_for_ai(validation_data)

print(f"Encoded training shape: {encoded_training.shape}")
print(f"Input columns: {enc_dec._column_counts['input_encoded_columns_count']}")
print(f"Output columns: {enc_dec._column_counts['output_encoded_columns_count']}")

# 4. Neural Network Training
nn_engine.machine_nn_engine_do_full_training_if_not_already_done()

# 5. Prediction and Decoding
# Prepare new data for prediction
new_data = pd.DataFrame({
    'customer_age': [25, 45, 35],
    'income': [45000, 75000, 55000],
    'product_category': ['electronics', 'books', 'clothing']
})

# Encode new data using same configuration
encoded_predictions_input = enc_dec.encode_for_ai(new_data)

# Get predictions from trained model
predictions = nn_engine.do_solving_direct_encoded_for_ai(encoded_predictions_input)

# Decode predictions back to human-readable format
decoded_predictions = enc_dec.decode_from_ai(predictions)

print("Predictions:")
print(decoded_predictions)
```

### Advanced Configuration Management

```python
from ML import EncDec, Machine
import pandas as pd

# 1. Load existing configuration
machine = Machine(machine_identifier_or_name="existing_machine")
enc_dec = EncDec(machine)  # Load from database

# 2. Check configuration status
print(f"Configuration loaded: {bool(enc_dec._enc_dec_configuration)}")
print(f"Input columns: {enc_dec._column_counts.get('input_encoded_columns_count', 0)}")
print(f"Output columns: {enc_dec._column_counts.get('output_encoded_columns_count', 0)}")

# 3. Handle configuration updates (data drift, new features)
if machine.db_machine.machine_is_re_run_enc_dec:
    print("Configuration needs update - data drift detected")

    # Load fresh data for reconfiguration
    fresh_data = machine.get_random_user_dataframe_for_training_trial(is_for_learning=True)

    # Create new configuration
    new_enc_dec = EncDec(machine, fresh_data)
    new_enc_dec.save_configuration_in_machine()

    print("Configuration updated successfully")
    enc_dec = new_enc_dec

# 4. Validate configuration integrity
try:
    test_data = machine.get_random_user_dataframe_for_training_trial(is_for_learning=True, force_rows_count=10)
    test_encoded = enc_dec.encode_for_ai(test_data)
    test_decoded = enc_dec.decode_from_ai(test_encoded)
    print("Configuration validation: PASSED")
except Exception as e:
    print(f"Configuration validation: FAILED - {e}")
```

### Batch Processing and Performance Optimization

```python
from ML import EncDec, Machine
import pandas as pd
import time

# 1. Setup for batch processing
machine = Machine(machine_identifier_or_name="batch_processor")
enc_dec = EncDec(machine)

# 2. Prepare batch data
batch_datasets = [
    pd.read_csv(f"batch_{i}.csv") for i in range(10)
]

# 3. Batch encoding with performance monitoring
encoded_batches = []
encoding_times = []

for i, dataset in enumerate(batch_datasets):
    start_time = time.time()

    # Encode batch
    encoded = enc_dec.encode_for_ai(dataset)
    encoded_batches.append(encoded)

    # Track performance
    encoding_time = time.time() - start_time
    encoding_times.append(encoding_time)
    print(f"Batch {i}: {len(dataset)} rows -> {encoded.shape[1]} features in {encoding_time:.2f}s")

# 4. Analyze encoding performance
avg_encoding_time = sum(encoding_times) / len(encoding_times)
total_encoded_features = sum(batch.shape[1] for batch in encoded_batches)

print(f"Average encoding time: {avg_encoding_time:.2f}s per batch")
print(f"Total encoded features: {total_encoded_features}")
print(f"Encoding throughput: {sum(len(ds) for ds in batch_datasets) / sum(encoding_times):.0f} rows/sec")
```

### Error Handling and Recovery

```python
from ML import EncDec, Machine
import pandas as pd

# 1. Setup with error handling
try:
    machine = Machine(machine_identifier_or_name="robust_machine")
    enc_dec = EncDec(machine, training_dataframe)
except Exception as e:
    print(f"EncDec initialization failed: {e}")
    # Fallback to loading existing configuration
    enc_dec = EncDec(machine)

# 2. Robust encoding with error recovery
def safe_encode_data(enc_dec, data):
    """Encode data with comprehensive error handling"""
    try:
        # Validate input
        if data.empty:
            raise ValueError("Input data is empty")

        # Check for required columns
        config_cols = set(enc_dec._enc_dec_configuration.keys())
        data_cols = set(data.columns)
        missing_cols = config_cols - data_cols
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Perform encoding
        encoded = enc_dec.encode_for_ai(data)
        return encoded, None

    except Exception as e:
        error_msg = f"Encoding failed: {str(e)}"

        # Log error in machine
        enc_dec._store_column_error_in_machine("encoding_process", error_msg)

        # Return fallback (zero array with correct shape)
        fallback_shape = (len(data), enc_dec._column_counts['total_encoded_columns_count'])
        fallback_data = pd.DataFrame(
            np.zeros(fallback_shape),
            columns=[f"encoded_{i}" for i in range(fallback_shape[1])]
        )

        return fallback_data, error_msg

# 3. Use safe encoding
test_data = pd.DataFrame({
    'feature1': [1, 2, 3],
    'feature2': ['a', 'b', 'c']
})

encoded_data, error = safe_encode_data(enc_dec, test_data)
if error:
    print(f"Used fallback encoding due to: {error}")
else:
    print("Encoding successful")
```

### Configuration Analysis and Debugging

```python
from ML import EncDec, Machine

# 1. Load and analyze configuration
machine = Machine(machine_identifier_or_name="debug_machine")
enc_dec = EncDec(machine)

# 2. Inspect configuration structure
print("=== EncDec Configuration Analysis ===")
print(f"Total configured columns: {len(enc_dec._enc_dec_configuration)}")

for col_name, col_config in enc_dec._enc_dec_configuration.items():
    print(f"\nColumn: {col_name}")
    print(f"  - Data Type: {col_config['column_datatype_name']}")
    print(f"  - Is Input: {col_config['is_input']}")
    print(f"  - Is Output: {col_config['is_output']}")
    print(f"  - FET Count: {len(col_config['fet_list'])}")

    for fet_config in col_config['fet_list']:
        fet_name = fet_config['fet_class_name']
        encoded_cols = fet_config['list_encoded_columns_name']
        print(f"    * {fet_name}: {len(encoded_cols)} output columns")

# 3. Check for potential issues
total_input_cols = sum(
    len(config['fet_list']) * len(fet['list_encoded_columns_name'])
    for config in enc_dec._enc_dec_configuration.values()
    for fet in config['fet_list']
    if config['is_input']
)

total_output_cols = sum(
    len(config['fet_list']) * len(fet['list_encoded_columns_name'])
    for config in enc_dec._enc_dec_configuration.values()
    for fet in config['fet_list']
    if config['is_output']
)

print("
=== Column Count Validation ===")
print(f"Expected total columns: {enc_dec._column_counts['total_encoded_columns_count']}")
print(f"Calculated total columns: {total_input_cols + total_output_cols}")
print(f"Match: {enc_dec._column_counts['total_encoded_columns_count'] == total_input_cols + total_output_cols}")
```

## Future Enhancements

### Potential Improvements

1. **Parallel Processing**: Multi-threaded encoding/decoding
2. **GPU Acceleration**: CUDA-based encoding for large datasets
3. **Incremental Updates**: Partial configuration updates
4. **Advanced Caching**: Prediction-based cache eviction
5. **Streaming Support**: Real-time data processing capabilities

This comprehensive EncDec module forms the critical data transformation layer in the EasyAutoML ML pipeline, ensuring seamless conversion between human-readable and ML-optimized data formats while maintaining performance, reliability, and extensibility.

## Detailed Function Analysis

### Core Initialization and Configuration Functions

#### `__init__(machine, dataframe_pre_encoded)`

**Where it's used and why:**
- Called when creating a new EncDec instance during machine learning pipeline initialization
- Used in two primary scenarios: loading existing configuration or creating new configuration
- Essential for setting up the encoding/decoding pipeline for data transformation
- Critical for maintaining consistency between training and prediction phases

**How the function works:**
1. **Configuration Loading Mode**: When `dataframe_pre_encoded=None`, calls `_init_load_configuration()` to load stored configuration from database
2. **Configuration Creation Mode**: When `dataframe_pre_encoded` is provided, calls `_init_create_configuration()` to build new configuration
3. **Parameter Validation**: Ensures proper combination of parameters or raises error
4. **State Initialization**: Sets up internal attributes for column counts and configuration storage

**What the function does and its purpose:**
- Serves as the main entry point for EncDec functionality
- Enables both configuration creation and loading workflows
- Ensures proper initialization of the encoding/decoding system
- Provides the foundation for all subsequent encoding/decoding operations

#### `_init_load_configuration()`

**Where it's used and why:**
- Called internally by `__init__()` when loading existing configuration
- Used during prediction/inference phases to restore previously created configuration
- Critical for maintaining consistency between training and prediction pipelines
- Enables reuse of encoding configurations across multiple prediction requests

**How the function works:**
1. **Database Retrieval**: Loads column counts from `machine.db_machine.enc_dec_columns_info_*`
2. **Configuration Deserialization**: Calls `_load_config_from_machine_and_deserialize_it()` to reconstruct FET instances
3. **Validation**: Ensures configuration exists and is valid

**What the function does and its purpose:**
- Restores EncDec state from database for inference operations
- Enables prediction phase to use same encoding as training phase
- Maintains consistency across training and inference workflows

#### `_init_create_configuration(dataframe_pre_encoded)`

**Where it's used and why:**
- Called internally by `__init__()` when creating new configuration
- Used during initial machine training to set up encoding/decoding rules
- Critical for defining how data will be transformed for neural network consumption
- Enables the system to handle different data types and feature engineering requirements

**How the function works:**
1. **Dependency Initialization**: Creates MDC and FEC instances to get column metadata
2. **Data Validation**: Ensures minimum data requirements (5+ rows) and proper column structure
3. **FET Instantiation**: For each column, creates FET instances based on FEC configuration
4. **Datetime Handling**: Special processing for DATETIME columns (splits into date/time components)
5. **Configuration Building**: Constructs the `_enc_dec_configuration` dictionary
6. **Column Count Calculation**: Determines input/output encoded column counts

**What the function does and its purpose:**
- Analyzes dataset structure and creates comprehensive encoding configuration
- Sets up the rules for transforming data into neural network format
- Handles complex data types and feature engineering requirements
- Provides the foundation for consistent encoding/decoding across the ML pipeline

### Configuration Management Functions

#### `save_configuration_in_machine()`

**Where it's used and why:**
- Called after configuration creation to persist EncDec settings
- Used during machine training to save encoding configuration for later use
- Critical for maintaining consistency between training and prediction phases
- Enables configuration reuse across multiple prediction requests

**How the function works:**
1. **Database Update**: Updates machine record with column counts
2. **Serialization**: Calls `_serialize_configuration_and_save_it_in_machine()` to prepare configuration for storage
3. **Persistence**: Saves serialized configuration to database

**What the function does and its purpose:**
- Persists EncDec configuration for future use
- Enables prediction phase to use same configuration as training
- Maintains data transformation consistency across sessions

#### `_load_config_from_machine_and_deserialize_it()`

**Where it's used and why:**
- Called by `_init_load_configuration()` to reconstruct configuration from database
- Used during prediction initialization to restore FET instances
- Critical for recreating functional configuration from serialized data
- Enables proper encoding/decoding during inference

**How the function works:**
1. **Deep Copy**: Creates working copy of stored configuration
2. **Object Reconstruction**: Recreates DatasetColumnDataType enums from stored names
3. **FET Instantiation**: Reconstructs FET instances from serialized configurations
4. **Error Handling**: Graceful handling of deserialization failures

**What the function does and its purpose:**
- Converts stored configuration back into functional objects
- Enables proper encoding/decoding operations during inference
- Maintains data transformation consistency

#### `_serialize_configuration_and_save_it_in_machine(enc_dec_configuration_to_save, machine)`

**Where it's used and why:**
- Called by `save_configuration_in_machine()` to prepare configuration for database storage
- Used to convert complex objects into serializable format
- Critical for persisting configuration while removing non-serializable components
- Enables configuration storage and retrieval across sessions

**How the function works:**
1. **Deep Copy**: Creates working copy to avoid modifying original
2. **Object Removal**: Removes non-serializable objects (enums, FET instances)
3. **Metadata Preservation**: Keeps serializable components (names, configurations)
4. **Database Storage**: Saves prepared configuration to machine record

**What the function does and its purpose:**
- Prepares complex configuration for database storage
- Maintains all necessary information while removing non-serializable components
- Enables configuration persistence and retrieval

### Core Encoding Functions

#### `encode_for_ai(pre_encoded_dataframe)`

**Where it's used and why:**
- Called during training and prediction to transform data for neural network consumption
- Used in the data preprocessing pipeline before model training or inference
- Critical for converting human-readable data into neural network compatible format
- Enables consistent data transformation across training and prediction phases

**How the function works:**
1. **Validation**: Checks dataframe type and configuration existence
2. **Column Processing Loop**: Iterates through each column in the dataframe
3. **Datatype Handling**: Special processing for DATETIME columns (splits into date/time)
4. **FET Execution**: For each column's FETs, calls encoding with caching system
5. **Result Concatenation**: Combines encoded results from all columns and FETs
6. **Validation**: Verifies encoded column counts match configuration expectations

**Caching System Integration:**
```python
# Uses LRU cache for performance
encoded_data = fet_encoder_caching_system_do_encode_with_cache(
    machine_id, fet_instance, column_name, values_to_encode_nparray
)
```

**What the function does and its purpose:**
- Transforms pre-encoded data into neural network compatible format (0-1 normalized)
- Handles multiple data types and feature engineering transformations
- Provides performance optimization through intelligent caching
- Ensures data consistency for model training and inference

#### `fet_encoder_caching_system_do_encode_with_cache(machine_id, fet_instance, column_name, values_to_encode_nparray)`

**Where it's used and why:**
- Called by `encode_for_ai()` for each FET encoding operation
- Used to improve performance for repeated encoding operations
- Critical for handling large datasets efficiently
- Enables caching of expensive encoding computations

**How the function works:**
1. **Size Check**: For small arrays (<250 elements), encodes without caching
2. **Cache Key Generation**: Creates hash from machine_id, column_name, and pickled objects
3. **Cache Lookup**: Uses LRU cache to check for existing results
4. **Fallback Encoding**: Performs encoding without cache when needed
5. **Integrity Validation**: Randomly verifies cache correctness against direct encoding

**What the function does and its purpose:**
- Provides performance optimization for encoding operations
- Reduces computational overhead for repeated transformations
- Maintains data integrity through cache validation
- Enables efficient processing of large datasets

#### `fet_encoder_caching_system_do_real_caching(machine_id, column_name, fet_pickled, values_to_encode_pickled)`

**Where it's used and why:**
- Called by the caching system when cache miss occurs
- Used for actual cached encoding computation with pickled parameters
- Critical for enabling hash-based caching of encoding operations
- Supports the LRU caching mechanism for performance optimization

**How the function works:**
1. **Parameter Unpickling**: Converts pickled FET and data back to objects
2. **Encoding Execution**: Performs actual encoding operation
3. **Result Return**: Provides encoded data for caching and use

**What the function does and its purpose:**
- Enables hash-based caching for encoding operations
- Supports efficient reuse of expensive computations
- Maintains data transformation consistency

### Core Decoding Functions

#### `decode_from_ai(data_encoded_from_ai)`

**Where it's used and why:**
- Called after neural network predictions to convert results back to human-readable format
- Used in the post-processing pipeline during inference/prediction
- Critical for making model outputs interpretable by users
- Enables the system to provide meaningful predictions in original data format

**How the function works:**
1. **Input Validation**: Checks dataframe structure and configuration
2. **Output Column Filtering**: Identifies columns that need decoding (output columns only)
3. **FET Decoder Collection**: Gathers all decoder-capable FETs for each output column
4. **Multi-FET Processing**: Handles columns with multiple FET decodings
5. **Result Merging**: Combines multiple FET results using sophisticated merging algorithms
6. **Data Type Handling**: Applies appropriate merging strategies based on data types

**Merging Strategies:**
- **Float Merging**: Averages similar float values across FETs
- **Label Merging**: Resolves conflicts in categorical predictions
- **Time Merging**: Combines date/time components intelligently
- **Date Merging**: Handles temporal data reconstruction

**What the function does and its purpose:**
- Converts neural network predictions back to human-readable format
- Handles complex multi-FET scenarios with intelligent result merging
- Provides consistent decoding across different data types
- Enables meaningful interpretation of model predictions

### Error Handling and Warning Functions

#### `_store_column_error_in_machine(column_name, error_message_user, error_message_internal)`

**Where it's used and why:**
- Called when FET instantiation or encoding fails during configuration creation
- Used to record critical errors that prevent proper machine operation
- Critical for tracking configuration issues and triggering retraining
- Enables automatic recovery mechanisms when data changes

**How the function works:**
1. **Message Processing**: Sanitizes error messages for database storage
2. **User/Internal Split**: Stores different messages for users vs internal logs
3. **Database Storage**: Updates machine error tracking fields
4. **Recovery Triggers**: Sets re-run flags to trigger configuration updates

**What the function does and its purpose:**
- Records configuration errors that require attention
- Triggers automatic recovery mechanisms
- Provides audit trail for troubleshooting
- Enables graceful degradation and recovery

#### `_store_column_warning_in_machine(column_name, warning_message_user, warning_message_internal)`

**Where it's used and why:**
- Called when FET operations encounter non-critical issues (e.g., data out of range)
- Used to track warnings that don't stop processing but may affect quality
- Critical for monitoring data quality and model performance
- Enables proactive maintenance and optimization

**How the function works:**
1. **Message Sanitization**: Prepares messages for database storage
2. **Database Update**: Stores warnings in machine warning fields
3. **Recovery Analysis**: Checks for specific warning patterns to trigger re-runs
4. **Flag Setting**: Automatically sets appropriate re-run flags based on warning content

**Re-run Flag Triggers:**
- `[is_re_run_mdc]`: Triggers machine data configuration update
- `[is_re_run_ici]`: Triggers input column importance recalculation
- `[is_re_run_fe]`: Triggers feature engineering reconfiguration
- `[is_re_run_enc_dec]`: Triggers EncDec reconfiguration
- `[is_re_run_nn_config]`: Triggers neural network reconfiguration
- `[is_re_run_model]`: Triggers complete model retraining

**What the function does and its purpose:**
- Tracks non-critical issues that may affect model performance
- Enables automatic optimization and maintenance
- Provides visibility into data quality issues
- Supports continuous improvement of the ML pipeline

### Utility and Helper Functions

#### `get_encoded_for_ai_columns_names_by_the_pre_encoded_column_name(pre_encoded_column_name)`

**Where it's used and why:**
- Called by external components to understand the encoding structure
- Used by NNEngine and other components to map between original and encoded columns
- Critical for maintaining column mapping during model training and inference
- Enables proper feature importance analysis and result interpretation

**How the function works:**
1. **Configuration Lookup**: Finds column in `_enc_dec_configuration`
2. **FET Iteration**: Collects encoded column names from all FETs for the column
3. **List Compilation**: Returns comprehensive list of encoded column names

**What the function does and its purpose:**
- Provides mapping between original and encoded column names
- Enables proper column tracking throughout the ML pipeline
- Supports feature analysis and result interpretation

#### `_get_input_and_output_encoded_for_ai_columns_count(columns_enc_dec_config_info)`

**Where it's used and why:**
- Called during configuration creation to calculate encoded column counts
- Used to validate encoding results and ensure proper column structure
- Critical for maintaining consistency between configuration and actual encoding
- Enables proper neural network input/output dimension specification

**How the function works:**
1. **Configuration Iteration**: Processes each column's configuration
2. **FET Counting**: Sums encoded columns from all FETs per column
3. **Input/Output Separation**: Tracks counts separately for input and output columns

**What the function does and its purpose:**
- Calculates expected encoded column counts for validation
- Ensures proper neural network architecture specification
- Maintains consistency between configuration and encoding results

#### `nested_data_structure_convert_isnull_to_____________________(nested_data_structure, value_to_replace_by)`

**Where it's used and why:**
- Utility function for handling null values in nested data structures
- Used in data preprocessing and cleaning operations
- Critical for ensuring data consistency in complex nested structures
- Supports robust handling of various data formats

**How the function works:**
1. **Recursive Processing**: Handles dictionaries, lists, and arrays recursively
2. **Null Detection**: Uses pandas.isnull() for comprehensive null detection
3. **Value Replacement**: Replaces null values with specified replacement
4. **Type Preservation**: Maintains original data structure types

**What the function does and its purpose:**
- Provides comprehensive null value handling in nested structures
- Ensures data consistency across different data formats
- Supports robust data preprocessing operations

This detailed function analysis demonstrates how EncDec serves as the sophisticated data transformation engine in the EasyAutoML ML pipeline, managing the complex conversion between human-readable and neural network compatible data formats while maintaining performance, error handling, and configuration management capabilities.