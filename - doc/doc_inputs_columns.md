# ML/InputsColumnsImportance.py - Input Column Importance Evaluation System

## Overview

The `InputsColumnsImportance` (ICI) module implements a sophisticated, model-aware feature importance evaluation system that quantifies the predictive significance of each input column in a machine learning model. Unlike traditional statistical methods, ICI uses actual model predictions to measure how much each feature contributes to the model's decision-making process.

**Location**: `ML/InputsColumnsImportance.py`

## Description

### What

The InputsColumnsImportance module evaluates the predictive significance of each input column through model-aware perturbation analysis, quantifying how much each feature contributes to model predictions. It provides importance scores that guide computational resource allocation for feature engineering and optimization decisions.

### How

It systematically perturbs each input column by setting values to minimum and maximum extremes, measures the resulting degradation in model prediction performance, and calculates relative importance scores that sum to 1.0. The system uses trained neural networks to evaluate actual model behavior rather than statistical correlations.

### Where

Used by FeatureEngineeringConfiguration to allocate feature engineering budgets intelligently based on column importance. Integrates with NNEngine for model-based importance evaluation.

### When

Called after initial model training to guide feature engineering resource allocation and optimization priorities.

## Core Architecture

### Primary Responsibilities

- **Model-Aware Importance Evaluation**: Measures feature importance through actual model prediction analysis
- **Perturbation-Based Assessment**: Evaluates impact by systematically perturbing input features
- **Computational Resource Optimization**: Guides budget allocation for feature engineering
- **Real-time Performance Monitoring**: Provides ongoing importance assessment during training
- **Caching and Persistence**: Efficiently stores and retrieves importance calculations

### Importance Evaluation Methodology

#### Perturbation-Based Importance Assessment

The ICI system employs a rigorous, model-driven approach to quantify feature importance:

1. **Baseline Performance Establishment**: Evaluates model prediction quality on unperturbed validation data
2. **Systematic Feature Perturbation**: For each input column, systematically tests extreme value scenarios:
   - **Minimum Value Perturbation**: Sets all values in the column to the observed minimum
   - **Maximum Value Perturbation**: Sets all values in the column to the observed maximum
3. **Impact Measurement**: Quantifies how much each perturbation degrades model performance
4. **Importance Quantification**: Calculates relative importance based on prediction loss degradation
5. **Normalization and Scaling**: Produces interpretable importance scores that sum to 1.0

#### Mathematical Foundation

The importance calculation is based on the principle that important features, when perturbed, will cause significant degradation in model performance:

```
Importance_i = (Loss_min_i - Loss_baseline + Loss_max_i - Loss_baseline) / 2
```

Where:
- `Importance_i`: Importance score for feature i
- `Loss_min_i`: Model loss when feature i is set to minimum values
- `Loss_max_i`: Model loss when feature i is set to maximum values
- `Loss_baseline`: Model loss on original unperturbed data

#### Key Advantages

- **Model-Specific**: Importance reflects actual model behavior, not just statistical correlations
- **Contextual**: Considers feature interactions and non-linear relationships
- **Robust**: Works across different data types and model architectures
- **Actionable**: Directly guides feature engineering and resource allocation decisions

### System Architecture Components

#### Core Classes and Attributes

   ```python
class InputsColumnsImportance:
    # Core references
    self._machine: Machine                    # Parent machine instance
    self._nnengine: NNEngine                  # Neural network engine for evaluation

    # Configuration state
    self._column_importance_evaluation: Dict[str, float]  # Importance scores per column
    self._fe_columns_inputs_importance_find_delay_sec: float  # Computation time tracking

    # Metadata
    self._input_columns_names: List[str]      # Input column names
    self._columns_data_type: Dict[str, DatasetColumnDataType]  # Column data types
    self._configuration_reliability_percentage: float  # Assessment quality score

    # Caching system
    self._importance_cache: Dict[str, float]  # Cached importance scores
    self._cache_timestamp: datetime          # Cache validity timestamp
```

#### Configuration Modes

1. **Load Mode**: Retrieves pre-computed importance scores from database storage
2. **Minimum Mode**: Assigns equal importance to all input columns (fast but imprecise)
3. **Best Mode**: Performs comprehensive importance evaluation using trained neural network
4. **Cached Mode**: Uses cached results when available and valid

## Core Methods and Implementation

### Initialization and Configuration Management

#### `__init__(self, machine: Machine, nnengine_for_best_config: NNEngine = None, create_configuration_best: bool = False, create_configuration_simple_minimum: bool = False, load_configuration: bool = False)`

**Intelligent Initialization with Multiple Configuration Strategies**:

   ```python
def __init__(self, machine: Machine, nnengine_for_best_config: NNEngine = None,
             create_configuration_best: bool = False,
             create_configuration_simple_minimum: bool = False,
             load_configuration: bool = False):
    """
    Initialize InputsColumnsImportance with flexible configuration options.

    Args:
        machine: Parent machine instance containing data and model
        nnengine_for_best_config: Trained NN engine for importance evaluation
        create_configuration_best: Perform comprehensive importance evaluation
        create_configuration_simple_minimum: Use equal weighting (fast setup)
        load_configuration: Load pre-computed importance from database

    Raises:
        ValueError: If incompatible configuration options are specified
    """

    # Validate configuration parameters
    config_count = sum([create_configuration_best, create_configuration_simple_minimum, load_configuration])
    if config_count > 1:
        raise ValueError("Only one configuration mode can be specified")

    if config_count == 0:
        raise ValueError("At least one configuration mode must be specified")

    # Store core references
    self._machine = machine
    self._nnengine = nnengine_for_best_config

    # Initialize metadata
    self._initialize_column_metadata()

    # Determine and execute configuration strategy
    if load_configuration:
        self._init_load_configuration()
    elif create_configuration_simple_minimum:
        self._init_minimum_configuration()
    elif create_configuration_best:
        self._init_best_configuration()
```

#### Configuration Mode Implementation

##### 1. Load Configuration Mode
```python
def _init_load_configuration(self):
    """
    Load pre-computed importance scores from machine database.

    This mode provides instant access to previously calculated importance
    scores, enabling fast initialization for existing machines.
    """

    try:
        # Retrieve stored importance evaluation
        stored_importance = self._machine.db_machine.fe_columns_inputs_importance_evaluation
        stored_timing = self._machine.db_machine.fe_columns_inputs_importance_find_delay_sec

        if stored_importance:
            # Validate and deserialize stored configuration
            self._column_importance_evaluation = self._deserialize_importance_scores(stored_importance)
            self._fe_columns_inputs_importance_find_delay_sec = stored_timing or 0.0

            # Validate configuration integrity
            self._validate_loaded_configuration()

            logger.info(f"Loaded ICI configuration for machine {self._machine.id}")
        else:
            # Fallback to minimum configuration if no stored data
            logger.warning("No stored ICI configuration found, using minimum configuration")
            self._init_minimum_configuration()

    except Exception as e:
        logger.error(f"Failed to load ICI configuration: {e}")
        self._init_minimum_configuration()
```

##### 2. Minimum Configuration Mode
```python
def _init_minimum_configuration(self):
    """
    Create equal importance assignment for all input columns.

    This mode provides fast initialization by assuming all input columns
    are equally important. Useful for initial setup or when computational
    resources are limited.
    """

    self._fe_columns_inputs_importance_find_delay_sec = 0.0

    # Calculate equal importance for all input columns
    num_input_columns = len(self._input_columns_names)
    if num_input_columns == 0:
        logger.warning("No input columns found for ICI evaluation")
        self._column_importance_evaluation = {}
        self._configuration_reliability_percentage = 0.0
        return

    equal_importance = 1.0 / num_input_columns

    # Assign equal importance to all columns
   self._column_importance_evaluation = {
        column_name: equal_importance
        for column_name in self._input_columns_names
   }

    # Minimum configuration has low but guaranteed reliability
    self._configuration_reliability_percentage = 25.0

    logger.info(f"Created minimum ICI configuration with {num_input_columns} columns")
   ```

##### 3. Best Configuration Mode
   ```python
def _init_best_configuration(self):
    """
    Perform comprehensive importance evaluation using trained neural network.

    This mode executes the full perturbation-based importance evaluation,
    providing accurate, model-aware importance scores. Requires a trained
    neural network engine.
    """

    # Validate prerequisites
    if not self._nnengine or not self._nnengine.is_nn_trained_and_ready():
        raise ValueError("Trained NNEngine required for best ICI configuration")

    if not self._input_columns_names:
        logger.warning("No input columns available for ICI evaluation")
        self._init_minimum_configuration()
        return

    # Execute comprehensive evaluation
    start_time = time.time()
    try:
   self._generate_configuration_best()
        self._fe_columns_inputs_importance_find_delay_sec = time.time() - start_time

        logger.info(f"Completed best ICI configuration in {self._fe_columns_inputs_importance_find_delay_sec:.2f}s")

    except Exception as e:
        logger.error(f"Best ICI configuration failed: {e}")
        # Fallback to minimum configuration
        self._init_minimum_configuration()
```

### Comprehensive Importance Evaluation Algorithm

#### `_generate_configuration_best() -> None`

**Complete Perturbation-Based Importance Assessment**:

```python
def _generate_configuration_best(self) -> None:
    """
    Execute comprehensive importance evaluation using systematic perturbation.

    This method implements the core ICI algorithm, evaluating each input column's
    importance by measuring its impact on model prediction performance when
    subjected to extreme value perturbations.
    """

    # Step 1: Prepare evaluation dataset
    evaluation_dataframe = self._prepare_evaluation_dataset()

    # Step 2: Establish baseline performance
    baseline_loss = self._calculate_baseline_loss(evaluation_dataframe)

    # Step 3: Initialize importance tracking
    column_loss_impacts = {}
    negative_loss_impacts = []  # Track columns that improve performance when perturbed

    # Step 4: Evaluate each input column
    for column_name in self._input_columns_names:
        try:
            # Calculate perturbation impact for this column
            loss_impact = self._evaluate_column_perturbation(
                column_name, evaluation_dataframe, baseline_loss
            )

            column_loss_impacts[column_name] = loss_impact

            # Track columns with negative impact (potential model issues)
            if loss_impact < 0:
                negative_loss_impacts.append((column_name, loss_impact))

        except Exception as e:
            logger.warning(f"Failed to evaluate column {column_name}: {e}")
            # Assign zero importance for failed evaluations
            column_loss_impacts[column_name] = 0.0

    # Step 5: Process negative impacts (columns that "help" when broken)
    processed_impacts = self._process_negative_impacts(column_loss_impacts, negative_loss_impacts)

    # Step 6: Normalize importance scores
    self._column_importance_evaluation = self._normalize_importance_scores(processed_impacts)

    # Step 7: Calculate configuration reliability
    self._configuration_reliability_percentage = self._calculate_configuration_reliability(
        processed_impacts, baseline_loss
    )

    logger.info(f"ICI evaluation completed. Reliability: {self._configuration_reliability_percentage:.1f}%")
```

#### Individual Column Perturbation Evaluation

```python
def _evaluate_column_perturbation(self, column_name: str,
                                evaluation_dataframe: pd.DataFrame,
                                baseline_loss: float) -> float:
    """
    Evaluate single column importance through perturbation testing.

    Args:
        column_name: Name of column to evaluate
        evaluation_dataframe: Pre-encoded evaluation dataset
        baseline_loss: Baseline prediction loss

    Returns:
        Average loss impact across min/max perturbations
    """

    # Step 1: Determine perturbation values based on data type
    min_value, max_value = self._get_perturbation_values(column_name, evaluation_dataframe)

    # Step 2: Create perturbation datasets
    min_perturbation_df = self._create_perturbation_dataset(
        evaluation_dataframe, column_name, min_value
    )
    max_perturbation_df = self._create_perturbation_dataset(
        evaluation_dataframe, column_name, max_value
    )

    # Step 3: Measure prediction loss for each perturbation
    min_loss = self._calculate_perturbation_loss(min_perturbation_df, evaluation_dataframe)
    max_loss = self._calculate_perturbation_loss(max_perturbation_df, evaluation_dataframe)

    # Step 4: Calculate average loss impact
    avg_loss_impact = (min_loss - baseline_loss + max_loss - baseline_loss) / 2

    return avg_loss_impact
```

#### Perturbation Value Determination

```python
def _get_perturbation_values(self, column_name: str, dataframe: pd.DataFrame):
    """
    Determine appropriate perturbation values based on column data type and characteristics.
    """

    column_dtype = self._columns_data_type[column_name]

    if column_dtype == DatasetColumnDataType.FLOAT:
        # For numeric columns, use stored min/max from machine data configuration
        min_value = self._machine.db_machine.mdc_columns_values_min[column_name]
        max_value = self._machine.db_machine.mdc_columns_values_max[column_name]

    elif column_dtype in [DatasetColumnDataType.LABEL, DatasetColumnDataType.LANGUAGE]:
        # For categorical columns, find actual min/max in the evaluation data
        column_values = dataframe[column_name].dropna()
        min_value = column_values.min()
        max_value = column_values.max()

    else:
        # For other data types, use first and last sorted unique values
        unique_values = sorted(dataframe[column_name].dropna().unique())
        min_value = unique_values[0] if unique_values else 0
        max_value = unique_values[-1] if len(unique_values) > 1 else unique_values[0]

    return min_value, max_value
```

### Loss Calculation and Prediction Quality Assessment

#### `_calculate_prediction_loss_between_df_inputs_solved_and_df_output_ref(...)`

**Multi-Format Prediction Loss Calculation**:

```python
def _calculate_prediction_loss_between_df_inputs_solved_and_df_output_ref(
    self,
    dataframe_input_to_compare_user: pd.DataFrame = None,
    dataframe_input_to_compare_pre_encoded: pd.DataFrame = None,
    dataframe_input_to_compare_encoded_for_ai: pd.DataFrame = None,
    output_dataframe_reference_to_compare_with_encoded_for_ai: pd.DataFrame = None
) -> float:
    """
    Calculate prediction loss between input data and reference outputs.

    This method supports multiple input formats for maximum flexibility,
    automatically handling data preprocessing and encoding as needed.

    Args:
        dataframe_input_to_compare_user: Raw user-format input data
        dataframe_input_to_compare_pre_encoded: Pre-encoded input data
        dataframe_input_to_compare_encoded_for_ai: Fully encoded input data for NN
        output_dataframe_reference_to_compare_with_encoded_for_ai: Reference output data

    Returns:
        Average prediction loss across all output dimensions
    """

    # Step 1: Prepare input data (handle multiple input formats)
    if dataframe_input_to_compare_encoded_for_ai is not None:
        encoded_input = dataframe_input_to_compare_encoded_for_ai
    elif dataframe_input_to_compare_pre_encoded is not None:
        encoded_input = self._nnengine._enc_dec.encode_for_ai(dataframe_input_to_compare_pre_encoded)
    elif dataframe_input_to_compare_user is not None:
        pre_encoded = self._nnengine._mdc.dataframe_pre_encode(dataframe_input_to_compare_user)
        encoded_input = self._nnengine._enc_dec.encode_for_ai(pre_encoded)
    else:
        raise ValueError("No valid input data provided")

    # Step 2: Generate predictions using neural network
    predicted_output = self._nnengine.do_solving_direct_encoded_for_ai(encoded_input)

    # Step 3: Calculate loss using configured loss function
    loss_function = keras.losses.get(self._machine.db_machine.parameter_nn_loss)

    # Compute element-wise losses
    losses = loss_function(
        output_dataframe_reference_to_compare_with_encoded_for_ai.values,
        predicted_output.values
    )

    # Calculate mean loss across all samples and output dimensions
    mean_loss = float(np.nanmean(losses.numpy()))

    return mean_loss
```

### Configuration Persistence and Database Integration

#### `save_configuration_in_machine() -> "InputsColumnsImportance"`

**Comprehensive Configuration Persistence**:

```python
def save_configuration_in_machine(self) -> "InputsColumnsImportance":
    """
    Persist ICI configuration to machine database with comprehensive metadata.

    Saves importance scores, timing information, and reliability metrics
    for future retrieval and analysis.

    Returns:
        Self for method chaining
    """

    # Step 1: Serialize importance evaluation data
    serialized_importance = self._serialize_importance_scores()

    # Step 2: Store in machine database
    self._machine.db_machine.fe_columns_inputs_importance_evaluation = serialized_importance
    self._machine.db_machine.fe_columns_inputs_importance_find_delay_sec = self._fe_columns_inputs_importance_find_delay_sec

    # Step 3: Store additional metadata
    self._machine.db_machine.fe_columns_inputs_importance_reliability = self._configuration_reliability_percentage

    # Step 4: Update machine re-run flags
    self._machine.db_machine.machine_is_re_run_fe_columns_inputs_importance = False

    # Step 5: Persist to database
    self._machine.save_machine_to_db()

    logger.info(f"ICI configuration saved for machine {self._machine.id}")

    return self
```

## Usage Patterns and Practical Examples

### Basic ICI Usage Scenarios

#### 1. Load Existing Importance Configuration
```python
from ML import Machine, InputsColumnsImportance

# Load machine with existing ICI configuration
machine = Machine.load_from_id(machine_id=123)

# Load stored importance scores
ici = InputsColumnsImportance(machine, load_configuration=True)

# Access importance scores
importance_scores = ici.get_column_importance_evaluation()
print(f"Column importance for machine {machine.id}:")
for column, importance in importance_scores.items():
    print(f"  {column}: {importance:.3f}")
```

#### 2. Quick Setup with Equal Importance
```python
from ML import Machine, InputsColumnsImportance

# Create new machine
machine = Machine.create_from_dataset("customer_data.csv", target_column="churn")

# Create ICI with equal importance (fast setup)
ici = InputsColumnsImportance(machine, create_configuration_simple_minimum=True)

print(f"All {len(ici.get_column_importance_evaluation())} input columns have equal importance")
```

#### 3. Comprehensive Importance Evaluation
```python
from ML import Machine, InputsColumnsImportance, NNEngine

# Create and train machine
machine = Machine.create_from_dataset("sales_data.csv", target_column="revenue")
nn_engine = NNEngine(machine)

# Train the neural network (required for ICI evaluation)
nn_engine.machine_nn_engine_do_full_training_if_not_already_done()

# Perform comprehensive importance evaluation
ici = InputsColumnsImportance(
    machine=machine,
    nnengine_for_best_config=nn_engine,
    create_configuration_best=True
)

# Save results for future use
ici.save_configuration_in_machine()

# Display results
importance_scores = ici.get_column_importance_evaluation()
reliability = ici.get_configuration_reliability_percentage()

print(f"ICI evaluation completed with {reliability:.1f}% reliability")
print("Column importance ranking:")
for column, score in sorted(importance_scores.items(), key=lambda x: x[1], reverse=True):
    print(f"  {column}: {score:.4f}")
```

### Advanced Usage Patterns

#### Importance-Based Feature Engineering Budget Allocation
```python
def allocate_fe_budget_by_importance(machine, total_fe_budget: int) -> Dict[str, int]:
    """
    Allocate feature engineering budget based on column importance.
    """

    # Get importance evaluation
    ici = InputsColumnsImportance(machine, load_configuration=True)
    importance_scores = ici.get_column_importance_evaluation()

    # Allocate budget proportionally to importance
    total_importance = sum(importance_scores.values())

    budget_allocation = {}
    for column, importance in importance_scores.items():
        # Allocate budget proportional to importance, with minimum of 1
        column_budget = max(1, int((importance / total_importance) * total_fe_budget))
        budget_allocation[column] = column_budget

    return budget_allocation

# Usage
budget_per_column = allocate_fe_budget_by_importance(machine, total_fe_budget=100)
print(f"FE budget allocation: {budget_per_column}")
```

#### Model Interpretability and Feature Selection
```python
def identify_critical_features(ici: InputsColumnsImportance, threshold: float = 0.05) -> List[str]:
    """
    Identify features that significantly impact model performance.
    """

    importance_scores = ici.get_column_importance_evaluation()

    # Filter features above importance threshold
    critical_features = [
        column for column, score in importance_scores.items()
        if score >= threshold
    ]

    return critical_features

def suggest_feature_removal(ici: InputsColumnsImportance, min_importance: float = 0.01) -> List[str]:
    """
    Suggest features that could potentially be removed due to low importance.
    """

    importance_scores = ici.get_column_importance_evaluation()

    # Find features with very low importance
    low_importance_features = [
        column for column, score in importance_scores.items()
        if score < min_importance
    ]

    return low_importance_features

# Usage
critical = identify_critical_features(ici, threshold=0.1)
removable = suggest_feature_removal(ici, min_importance=0.01)

print(f"Critical features ({len(critical)}): {critical}")
print(f"Potentially removable features ({len(removable)}): {removable}")
```

#### Performance Monitoring and Validation
```python
def validate_ici_configuration(ici: InputsColumnsImportance) -> Dict[str, Any]:
    """
    Comprehensive validation of ICI configuration quality.
    """

    validation_results = {
        "is_valid": True,
        "warnings": [],
        "issues": [],
        "recommendations": []
    }

    # Check configuration reliability
    reliability = ici.get_configuration_reliability_percentage()
    if reliability < 50.0:
        validation_results["issues"].append(f"Low configuration reliability: {reliability:.1f}%")
        validation_results["recommendations"].append("Consider re-evaluating with a better trained model")

    # Check importance score distribution
    importance_scores = ici.get_column_importance_evaluation()
    scores = list(importance_scores.values())

    if len(scores) == 0:
        validation_results["issues"].append("No importance scores available")
        validation_results["is_valid"] = False
        return validation_results

    # Check for uniform distribution (might indicate minimum configuration)
    score_std = np.std(scores)
    if score_std < 0.01:
        validation_results["warnings"].append("Importance scores are nearly uniform")
        validation_results["recommendations"].append("Consider running full importance evaluation")

    # Check for negative importance scores
    negative_scores = [col for col, score in importance_scores.items() if score < 0]
    if negative_scores:
        validation_results["warnings"].append(f"Negative importance scores for: {negative_scores}")
        validation_results["recommendations"].append("Investigate columns with negative importance")

    # Check evaluation time
    eval_time = ici.get_evaluation_time_seconds()
    if eval_time > 300:  # 5 minutes
        validation_results["warnings"].append(f"Long evaluation time: {eval_time:.1f}s")

    validation_results["is_valid"] = len(validation_results["issues"]) == 0

    return validation_results

# Usage
validation = validate_ici_configuration(ici)
if not validation["is_valid"]:
    print("ICI Configuration Issues:", validation["issues"])

if validation["recommendations"]:
    print("Recommendations:", validation["recommendations"])
```

### Integration with Machine Learning Pipeline

#### Complete ML Workflow with ICI
```python
def complete_ml_pipeline_with_ici(data_path: str, target_column: str):
    """
    Complete machine learning pipeline incorporating ICI for optimization.
    """

    # Phase 1: Data ingestion and machine creation
    machine = Machine.create_from_dataset(data_path, target_column)

    # Phase 2: Initial model training
    nn_engine = NNEngine(machine)
    nn_engine.machine_nn_engine_do_full_training_if_not_already_done()

    # Phase 3: Importance evaluation
    ici = InputsColumnsImportance(
        machine=machine,
        nnengine_for_best_config=nn_engine,
        create_configuration_best=True
    )

    # Phase 4: Feature engineering budget allocation
    total_fe_budget = machine.get_machine_level() * 20
    fe_budget_allocation = allocate_fe_budget_by_importance(machine, total_fe_budget)

    # Phase 5: Optimized feature engineering
    fec = FeatureEngineeringConfiguration(
        machine=machine,
        global_dataset_budget=total_fe_budget,
        nn_engine_for_searching_best_config=nn_engine
    )

    # Phase 6: Retrain with optimized features
    nn_engine.machine_nn_engine_do_full_training_if_not_already_done()

    # Phase 7: Final evaluation and persistence
    ici.save_configuration_in_machine()
    fec.save_configuration_in_machine()

    return {
        "machine": machine,
        "ici": ici,
        "fec": fec,
        "nn_engine": nn_engine,
        "importance_scores": ici.get_column_importance_evaluation(),
        "fe_budget_used": sum(fec._cost_per_columns.values())
    }

# Usage
pipeline_result = complete_ml_pipeline_with_ici("marketing_data.csv", "conversion_rate")
print(f"Pipeline completed. Total FE budget used: {pipeline_result['fe_budget_used']}")
```

## Performance Characteristics and Optimization

### Computational Complexity Analysis

| Operation | Time Complexity | Typical Duration | Memory Usage |
|-----------|----------------|------------------|--------------|
| Load Configuration | O(1) | < 1 second | Minimal |
| Minimum Configuration | O(n) | < 1 second | Minimal |
| Importance Evaluation | O(c × n × m) | 30-300 seconds | High |
| Configuration Save | O(1) | < 1 second | Minimal |

Where:
- n = number of samples in evaluation dataset
- c = number of input columns
- m = neural network prediction time per sample

### Caching Strategies for Performance

#### Intelligent Result Caching
```python
class ICICache:
    """
    Intelligent caching system for ICI results.
    """

    def __init__(self, cache_validity_hours: int = 24):
        self.cache_validity_hours = cache_validity_hours
        self._cache = {}

    def get_cached_ici(self, machine_id: int, data_hash: str) -> Optional[Dict]:
        """Retrieve cached ICI results if valid."""

        cache_key = f"{machine_id}_{data_hash}"

        if cache_key in self._cache:
            cached_result, timestamp = self._cache[cache_key]

            # Check cache validity
            if (datetime.now() - timestamp).total_seconds() < self.cache_validity_hours * 3600:
                return cached_result

            # Remove expired cache entry
            del self._cache[cache_key]

        return None

    def cache_ici_result(self, machine_id: int, data_hash: str, ici_result: Dict):
        """Cache ICI evaluation results."""

        cache_key = f"{machine_id}_{data_hash}"
        self._cache[cache_key] = (ici_result, datetime.now())

        # Implement LRU cache eviction if needed
        if len(self._cache) > 100:  # Maximum cache size
            oldest_key = min(self._cache.keys(),
                           key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]
```

### Scalability Considerations

#### Handling Large Datasets
```python
def optimize_ici_for_large_datasets(machine, sample_size: int = 10000):
    """
    Optimize ICI evaluation for large datasets using sampling.
    """

    # Get total dataset size
    total_samples = machine.get_total_sample_count()

    if total_samples > sample_size:
        # Use stratified sampling to maintain data distribution
        evaluation_sample = machine.get_stratified_sample(sample_size)

        # Create ICI with sampled data
        ici = InputsColumnsImportance(
            machine=machine,
            nnengine_for_best_config=machine.get_nn_engine(),
            create_configuration_best=True,
            evaluation_sample=evaluation_sample  # Hypothetical parameter
        )

        # Adjust reliability score for sampling
        ici._configuration_reliability_percentage *= (sample_size / total_samples)

        logger.info(f"ICI evaluation used {sample_size}/{total_samples} samples")
    else:
        # Use full dataset for smaller data
        ici = InputsColumnsImportance(
            machine=machine,
            nnengine_for_best_config=machine.get_nn_engine(),
            create_configuration_best=True
        )

    return ici
```

### Troubleshooting Common Issues

#### Low Configuration Reliability
```python
def diagnose_low_reliability(ici: InputsColumnsImportance) -> List[str]:
    """
    Diagnose causes of low ICI configuration reliability.
    """

    issues = []
    reliability = ici.get_configuration_reliability_percentage()

    if reliability < 30.0:
        issues.append("Very low reliability - model may not be well trained")

    if reliability < 50.0:
        issues.append("Low reliability - consider:")
        issues.append("  - Training model longer or with more data")
        issues.append("  - Using different model architecture")
        issues.append("  - Checking data quality and preprocessing")

    # Check for negative importance scores
    importance_scores = ici.get_column_importance_evaluation()
    negative_count = sum(1 for score in importance_scores.values() if score < 0)

    if negative_count > len(importance_scores) * 0.1:
        issues.append(f"Many negative importance scores ({negative_count} columns)")
        issues.append("  This suggests model issues or data problems")

    return issues

# Usage
if ici.get_configuration_reliability_percentage() < 60.0:
    issues = diagnose_low_reliability(ici)
    for issue in issues:
        print(f"Diagnosis: {issue}")
```

#### Model Inconsistency Detection
```python
def detect_model_inconsistencies(ici: InputsColumnsImportance) -> List[str]:
    """
    Detect potential model training or data quality issues.
    """

    issues = []

    importance_scores = ici.get_column_importance_evaluation()

    # Check for columns that improve performance when set to extremes
    inconsistent_columns = [
        column for column, score in importance_scores.items()
        if score < -0.05  # Significantly negative impact
    ]

    if inconsistent_columns:
        issues.append(f"Model inconsistency detected in columns: {inconsistent_columns}")
        issues.append("  These columns perform better when set to extreme values")
        issues.append("  Possible causes:")
        issues.append("  - Model overfitting to specific value ranges")
        issues.append("  - Data quality issues in these columns")
        issues.append("  - Insufficient model capacity or training")

    # Check for overly dominant features
    max_importance = max(importance_scores.values())
    if max_importance > 0.5:
        dominant_columns = [
            column for column, score in importance_scores.items()
            if score > 0.5
        ]
        issues.append(f"Dominant features detected: {dominant_columns}")
        issues.append("  Consider feature engineering or model regularization")

    return issues

# Usage
inconsistencies = detect_model_inconsistencies(ici)
for issue in inconsistencies:
    print(f"Model Issue: {issue}")
```

## Best Practices and Recommendations

### ICI Configuration Guidelines

1. **Always Train Model First**: Ensure neural network is well-trained before ICI evaluation
2. **Use Appropriate Dataset Size**: Balance evaluation accuracy with computational cost
3. **Monitor Reliability Scores**: Re-evaluate if reliability drops below 60%
4. **Cache Results Wisely**: Leverage caching for stable configurations
5. **Validate Consistency**: Check for model inconsistencies and data quality issues

### Performance Optimization Tips

1. **Batch Processing**: Evaluate multiple columns simultaneously when possible
2. **Early Stopping**: Stop evaluation if reliability becomes clearly insufficient
3. **Parallel Evaluation**: Use multiple model instances for different columns
4. **Incremental Updates**: Update importance scores as new data arrives
5. **Resource Monitoring**: Track memory and CPU usage during evaluation

### Integration Best Practices

1. **Pipeline Integration**: Incorporate ICI into automated ML pipelines
2. **Feedback Loop**: Use ICI results to improve feature engineering decisions
3. **Monitoring**: Continuously monitor importance score stability
4. **Versioning**: Track ICI configuration versions with model versions
5. **Documentation**: Maintain clear documentation of importance evaluation methodology

This comprehensive InputsColumnsImportance system provides the foundation for intelligent, model-aware feature importance evaluation in the EasyAutoML platform, enabling data scientists to make informed decisions about feature engineering, resource allocation, and model optimization.
