# ML/FeatureEngineeringConfiguration.py - Feature Engineering Management System

## Overview

The `FeatureEngineeringConfiguration` (FEC) module is the intelligent orchestration engine for automated feature engineering in the EasyAutoML system. It manages the selection, configuration, and optimization of Feature Engineering Templates (FETs) across dataset columns, implementing sophisticated algorithms that balance predictive performance gains against computational costs.

**Location**: `ML/FeatureEngineeringConfiguration.py`

## Description

### What

The FeatureEngineeringConfiguration module manages intelligent selection and configuration of Feature Engineering Templates (FETs) across all dataset columns, implementing budget-aware optimization that balances predictive performance gains against computational costs. It orchestrates the automated feature engineering process by allocating resources based on column importance and optimizing FET combinations for maximum model performance.

### How

It allocates computational budget proportionally to column importance scores from InputsColumnsImportance, then uses SolutionFinder with ExperimenterColumnFETSelector to discover optimal FET combinations for each column. The system supports three modes: load existing configuration, apply minimum transformations, or perform full optimization with budget constraints.

### Where

Used by Machine during initialization to set up feature engineering, and by NNEngine when configuring the ML pipeline. Integrates with InputsColumnsImportance for budget allocation and SolutionFinder for optimization.

### When

Called after MachineDataConfiguration is complete and before EncDec setup in the ML pipeline initialization sequence.

## Core Architecture

### Primary Responsibilities

- **Intelligent FET Selection**: Automated selection of optimal feature transformations based on data characteristics and performance impact
- **Budget-Aware Optimization**: Respects computational constraints while maximizing performance improvements
- **Multi-Objective Optimization**: Balances accuracy gains, computational costs, and data integrity requirements
- **Configuration Persistence**: Saves and loads optimized FET configurations across sessions
- **Dynamic Reconfiguration**: Adapts to changing data characteristics and performance requirements

### Main Classes

1. **FeatureEngineeringConfiguration**: Global orchestration and budget management across all dataset columns
2. **FeatureEngineeringColumn**: Single-column optimization engine for individual feature engineering decisions

### Key Concepts

#### Feature Engineering Templates (FETs)
FETs are specialized transformation classes that implement specific feature engineering techniques:

- **Numeric FETs**: Scaling, normalization, power transformations
- **Categorical FETs**: One-hot encoding, frequency-based encoding
- **Text FETs**: Language detection, semantic embeddings
- **Temporal FETs**: Date/time feature extraction
- **Hybrid FETs**: Combined transformations for complex data types

#### Budget System
The budget system allocates computational resources based on:
- **Machine Level**: Higher levels get more computational resources
- **Column Importance**: Important columns get larger budgets
- **Input/Output Ratios**: Different allocation strategies for input vs output columns
- **Performance Requirements**: More complex problems get larger budgets

#### Optimization Objectives
Multi-objective optimization considers:
- **Performance Improvement**: Neural network accuracy and loss reduction
- **Computational Cost**: CPU, memory, and time requirements
- **Data Integrity**: Preservation of data relationships and statistical properties
- **Scalability**: Ability to handle large datasets efficiently

## FeatureEngineeringConfiguration Class

### Primary Responsibilities

- **Global Budget Management**: Allocates computational resources across all columns based on importance and machine level
- **Column Importance Integration**: Uses input column importance scores (ICI) for intelligent budget distribution
- **Configuration Persistence**: Saves and loads optimized FET configurations with versioning
- **Validation**: Ensures configuration integrity, data compatibility, and system consistency
- **Performance Monitoring**: Tracks optimization time, resource usage, and configuration effectiveness

### Key Attributes

```python
class FeatureEngineeringConfiguration:
    # Core references
    self._machine: Machine                    # Parent machine instance
    self._ici: InputsColumnsImportance       # Column importance evaluator
    self._nn_engine: NNEngine                # Neural network engine for evaluation

    # Configuration state
    self._activated_fet_list_per_column: Dict[str, List[str]]  # FET selections per column
    self._cost_per_columns: Dict[str, float]  # Cost tracking per column
    self._fe_find_delay_sec: float           # Optimization time tracking
    self._configuration_reliability: float   # Configuration quality score

    # Budget management
    self._global_dataset_budget: int         # Total available budget
    self._budget_allocation_strategy: str    # Budget distribution method
```

### Initialization and Configuration Modes

#### `__init__(self, machine: Machine, global_dataset_budget: Optional[int] = None, nn_engine_for_searching_best_config: Optional[NNEngine] = None, force_configuration_simple_minimum: Optional[bool] = False)`

**Flexible Initialization with Multiple Modes**:

```python
def __init__(self, machine, global_dataset_budget=None, nn_engine_for_searching_best_config=None, force_configuration_simple_minimum=False):
    """
    Initialize FEC with intelligent mode selection based on parameters.

    Args:
        machine: Parent machine instance
        global_dataset_budget: Budget for optimization (None = load mode)
        nn_engine_for_searching_best_config: NN engine for evaluation
        force_configuration_simple_minimum: Force minimal configuration
    """

    # Store core references
    self._machine = machine
    self._nn_engine = nn_engine_for_searching_best_config

    # Determine initialization mode
    if global_dataset_budget is None:
        # Load Mode: Load existing configuration from database
        self._init_load_configuration()
    elif force_configuration_simple_minimum:
        # Minimum Mode: Apply basic transformations only
        self._init_create_configuration_minimum()
    else:
        # Optimization Mode: Intelligent FET selection with budget
        self._init_create_configuration_optimized(global_dataset_budget)
```

#### 1. Load Mode - Configuration Retrieval
```python
def _init_load_configuration(self):
    """Load existing FET configuration from machine database."""

    # Retrieve stored configuration
    stored_config = self._machine.db_machine.fe_columns_fet
    timing_data = self._machine.db_machine.fe_find_delay_sec

    if stored_config:
        # Deserialize and validate configuration
        self._activated_fet_list_per_column = self._deserialize_fet_config(stored_config)
        self._fe_find_delay_sec = timing_data or 0.0

        # Validate configuration integrity
        self._validate_loaded_configuration()
    else:
        # No stored configuration - fallback to minimum
        logger.warning("No stored FEC configuration found, using minimum configuration")
        self._init_create_configuration_minimum()
```

#### 2. Minimum Mode - Basic Compatibility
```python
def _init_create_configuration_minimum(self):
    """Create minimal FET configuration for basic compatibility."""

    self._activated_fet_list_per_column = {}
    self._fe_find_delay_sec = 0.0

    # Get all columns from machine data configuration
    all_columns = self._get_all_machine_columns()

    for column_name in all_columns:
        # Apply minimal lossless transformations
        column_info = self.get_all_column_datas_infos(column_name)
        minimal_fets = self._select_minimum_fets_for_column(column_info)

        self._activated_fet_list_per_column[column_name] = minimal_fets
        self._cost_per_columns[column_name] = self._calculate_column_fet_cost(minimal_fets, column_info)

    self._configuration_reliability = 0.5  # Minimum configuration reliability
```

#### 3. Optimization Mode - Intelligent FET Selection
```python
def _init_create_configuration_optimized(self, global_budget: int):
    """Create optimized FET configuration with budget-aware selection."""

    start_time = time.time()

    # Step 1: Initialize optimization infrastructure
    self._global_dataset_budget = global_budget
    self._activated_fet_list_per_column = {}
    self._cost_per_columns = {}

    # Step 2: Analyze column importance for budget allocation
    column_importance_scores = self._get_column_importance_scores()

    # Step 3: Allocate budget across columns
    budget_allocation = self._allocate_budget_across_columns(
        global_budget, column_importance_scores
    )

    # Step 4: Optimize FET selection for each column
    total_cost_used = 0

    for column_name, column_budget in budget_allocation.items():
        try:
            # Create column-specific optimizer
            column_optimizer = FeatureEngineeringColumn(
                nn_engine_to_use_in_trial=self._nn_engine,
                this_column_datas_infos=self.get_all_column_datas_infos(column_name),
                this_fec_budget_maximum=column_budget
            )

            # Store optimized configuration
            fet_selection = column_optimizer.get_selected_fets()
            fet_cost = column_optimizer.get_total_cost()

            self._activated_fet_list_per_column[column_name] = fet_selection
            self._cost_per_columns[column_name] = fet_cost
            total_cost_used += fet_cost

        except Exception as e:
            logger.warning(f"FET optimization failed for column {column_name}: {e}")
            # Fallback to minimum configuration
            self._activated_fet_list_per_column[column_name] = self._select_minimum_fets_for_column(
                self.get_all_column_datas_infos(column_name)
            )

    # Step 5: Calculate configuration reliability
    self._configuration_reliability = self._assess_configuration_reliability()
    self._fe_find_delay_sec = time.time() - start_time

    logger.info(f"FEC optimization completed in {self._fe_find_delay_sec:.2f}s, "
               f"budget used: {total_cost_used}/{global_budget}")
```

### Advanced Budget Allocation Strategy

#### Intelligent Budget Distribution
```python
def _allocate_budget_across_columns(self, global_budget: int,
                                  importance_scores: Dict[str, float]) -> Dict[str, float]:
    """
    Allocate budget across columns based on importance and data characteristics.
    """

    # Get column metadata
    all_columns = self._get_all_machine_columns()
    input_columns = [col for col in all_columns if self._is_input_column(col)]
    output_columns = [col for col in all_columns if self._is_output_column(col)]

    # Calculate base budget ratios
    input_ratio = len(input_columns) / len(all_columns)
    output_ratio = len(output_columns) / len(all_columns)

    # Allocate base budgets
    input_budget_total = global_budget * input_ratio
    output_budget_total = global_budget * output_ratio

    # Distribute within input columns based on importance
    input_budget_allocation = self._distribute_budget_by_importance(
        input_columns, importance_scores, input_budget_total
    )

    # Distribute output columns equally (typically simpler transformations)
    output_budget_allocation = {
        col: output_budget_total / len(output_columns) for col in output_columns
    }

    # Combine allocations
    final_allocation = {**input_budget_allocation, **output_budget_allocation}

    return final_allocation
```

#### Importance-Based Budget Distribution
```python
def _distribute_budget_by_importance(self, columns: List[str],
                                   importance_scores: Dict[str, float],
                                   total_budget: float) -> Dict[str, float]:
    """
    Distribute budget among columns proportional to their importance scores.
    """

    allocation = {}

    # Normalize importance scores to ensure they sum to 1
    total_importance = sum(importance_scores.get(col, 0.1) for col in columns)

    if total_importance == 0:
        # Equal distribution if no importance scores available
        base_budget = total_budget / len(columns)
        return {col: base_budget for col in columns}

    # Allocate budget proportional to importance
    for column in columns:
        importance = importance_scores.get(column, 0.1)
        column_budget = (importance / total_importance) * total_budget

        # Ensure minimum budget per column
        column_budget = max(column_budget, 1.0)

        allocation[column] = column_budget

    return allocation
```

#### `save_configuration_in_machine() -> FeatureEngineeringConfiguration`

**Comprehensive Persistence Strategy**:

```python
def save_configuration_in_machine(self) -> "FeatureEngineeringConfiguration":
    """
    Persist FEC configuration to machine database with comprehensive metadata.

    Returns:
        Self for method chaining
    """

    # Step 1: Serialize FET configuration
    serialized_config = self._serialize_fet_configuration()

    # Step 2: Calculate and store FET usage statistics
    fet_usage_stats = self._calculate_fet_usage_statistics()
    self._machine.db_machine.fe_fet_usage_statistics = fet_usage_stats

    # Step 3: Store core configuration data
    self._machine.db_machine.fe_columns_fet = serialized_config
    self._machine.db_machine.fe_find_delay_sec = self._fe_find_delay_sec

    # Step 4: Store budget and performance metadata
    self._machine.db_machine.fe_global_budget_used = sum(self._cost_per_columns.values())
    self._machine.db_machine.fe_configuration_reliability = self._configuration_reliability

    # Step 5: Store column-level cost breakdown
    self._machine.db_machine.fe_column_costs = self._cost_per_columns

    # Step 6: Update machine re-run flags
    self._machine.db_machine.machine_is_re_run_fe = False  # Configuration is now up-to-date

    # Step 7: Persist to database
    self._machine.save_machine_to_db()

    logger.info(f"FEC configuration saved for machine {self._machine.id}")

    return self
```

**Data Structure Storage**:
```python
# Serialized configuration structure stored in database
{
    "fet_configurations": {
        "column_name": {
            "fet_list": ["FETNumericMinMaxFloat", "FET6PowerFloat"],
            "total_cost": 7,
            "estimated_performance_gain": 0.15
        },
        ...
    },
    "global_metadata": {
        "total_budget_used": 45,
        "optimization_time": 120.5,
        "configuration_reliability": 0.87,
        "fet_usage_statistics": {
            "FETNumericMinMaxFloat": 8,
            "FETMultiplexerAllLabel": 5,
            "FET6PowerFloat": 12,
            ...
        }
    },
    "version": "2.1.0",
    "created_at": "2024-01-15T10:30:00Z"
}
```

#### `get_all_column_datas_infos(column_name: str) -> Column_datas_infos`

**Comprehensive Metadata Collection and Analysis**:

```python
def get_all_column_datas_infos(self, column_name: str) -> Column_datas_infos:
    """
    Gather comprehensive metadata for column to inform FET selection decisions.

    Args:
        column_name: Name of column to analyze

    Returns:
        Column_datas_infos namedtuple with complete column analysis
    """

    # Get basic column information from machine data configuration
    mdc = self._machine._mdc
    column_datatype = mdc.get_column_data_type(column_name)
    is_input = mdc.is_column_input(column_name)
    is_output = mdc.is_column_output(column_name)

    # Gather statistical properties
    column_data = self._get_column_data_sample(column_name)
    stats = self._calculate_column_statistics(column_data)

    # Perform text analysis if applicable
    text_analysis = self._analyze_text_properties(column_data) if column_datatype == DatasetColumnDataType.LANGUAGE else {}

    # Get current FET configuration (if any)
    current_fets = self._activated_fet_list_per_column.get(column_name, [])

    # Assemble comprehensive metadata
    column_info = Column_datas_infos(
        name=column_name,
        is_input=is_input,
        is_output=is_output,
        datatype=column_datatype,
        description_user_df=self._get_column_description(column_name),

        # Statistical properties
        unique_value_count=stats['unique_count'],
        missing_percentage=stats['missing_percentage'],
        min=stats['min'],
        max=stats['max'],
        mean=stats['mean'],
        std_dev=stats['std_dev'],
        skewness=stats['skewness'],
        kurtosis=stats['kurtosis'],
        quantile02=stats['quantiles'][0.2],
        quantile03=stats['quantiles'][0.3],
        quantile07=stats['quantiles'][0.7],
        quantile08=stats['quantiles'][0.8],
        sem=stats['sem'],
        median=stats['median'],
        mode=stats['mode'],

        # Text analysis features
        str_percent_uppercase=text_analysis.get('uppercase_percentage', 0.0),
        str_percent_lowercase=text_analysis.get('lowercase_percentage', 0.0),
        str_percent_digit=text_analysis.get('digit_percentage', 0.0),
        str_percent_punctuation=text_analysis.get('punctuation_percentage', 0.0),
        str_percent_operators=text_analysis.get('operator_percentage', 0.0),
        str_percent_underscore=text_analysis.get('underscore_percentage', 0.0),
        str_percent_space=text_analysis.get('space_percentage', 0.0),

        # Language detection scores
        str_language_en=text_analysis.get('language_scores', {}).get('en', 0.0),
        str_language_fr=text_analysis.get('language_scores', {}).get('fr', 0.0),
        str_language_de=text_analysis.get('language_scores', {}).get('de', 0.0),
        str_language_it=text_analysis.get('language_scores', {}).get('it', 0.0),
        str_language_es=text_analysis.get('language_scores', {}).get('es', 0.0),
        str_language_pt=text_analysis.get('language_scores', {}).get('pt', 0.0),
        str_language_others=text_analysis.get('language_scores', {}).get('others', 0.0),
        str_language_none=text_analysis.get('language_scores', {}).get('none', 0.0),

        # Current FET configuration
        fet_list=current_fets
    )

    return column_info
```

**Metadata Utilization Strategies**:
- **Statistical Analysis**: Guides selection of scaling/normalization FETs based on distribution characteristics
- **Text Properties**: Informs language detection and text processing FET selection
- **Data Quality Assessment**: Influences robustness requirements and missing value handling
- **Type-Specific Optimization**: Enables data type appropriate FET selection
- **Performance Prediction**: Helps estimate transformation impact on model performance

## Usage Patterns and Examples

### Basic Usage - Loading Existing Configuration

```python
from ML import Machine, FeatureEngineeringConfiguration

# Load existing machine with stored FET configuration
machine = Machine.load_from_id(machine_id=123)

# Load existing FET configuration (no optimization)
fec = FeatureEngineeringConfiguration(machine)

# Access current configuration
current_fets = fec.get_activated_fet_list_per_column()
print(f"Machine {machine.id} has FET configuration for {len(current_fets)} columns")
```

### Advanced Usage - Optimizing Feature Engineering

```python
from ML import Machine, FeatureEngineeringConfiguration, NNEngine

# Create machine and initialize NN engine
machine = Machine.create_from_dataset(
    data_source="customer_data.csv",
    target_column="churn_risk"
)

# Initialize NN engine for evaluation
nn_engine = NNEngine(machine)

# Create optimized FET configuration with budget
global_budget = 100  # Based on machine level and dataset size
fec = FeatureEngineeringConfiguration(
    machine=machine,
    global_dataset_budget=global_budget,
    nn_engine_for_searching_best_config=nn_engine
)

# Save optimized configuration
fec.save_configuration_in_machine()

print(f"FEC optimization completed in {fec._fe_find_delay_sec:.2f}s")
print(f"Configuration reliability: {fec._configuration_reliability:.2f}")
```

### Minimum Configuration - Fast Setup

```python
from ML import Machine, FeatureEngineeringConfiguration

# Create machine for quick prototyping
machine = Machine.create_from_dataframe(df_customer_data, target_column="purchase_amount")

# Create minimum FET configuration (no optimization, just compatibility)
fec = FeatureEngineeringConfiguration(
    machine=machine,
    force_configuration_simple_minimum=True
)

# Configuration is automatically saved
fec.save_configuration_in_machine()

print("Minimum FET configuration applied for fast prototyping")
```

### Integration with Machine Lifecycle

```python
from ML import Machine, FeatureEngineeringConfiguration, NNEngine

def setup_machine_with_optimized_features(data_path: str, target_column: str, budget: int = None):
    """
    Complete machine setup with intelligent feature engineering.
    """

    # Step 1: Create machine
    machine = Machine.create_from_dataset(
        data_source=data_path,
        target_column=target_column
    )

    # Step 2: Determine appropriate budget based on machine level
    if budget is None:
        machine_level = machine.get_machine_level()
        budget = machine_level * 20  # Scale budget with machine level

    # Step 3: Create NN engine for FET evaluation
    nn_engine = NNEngine(machine)

    # Step 4: Optimize feature engineering configuration
    fec = FeatureEngineeringConfiguration(
        machine=machine,
        global_dataset_budget=budget,
        nn_engine_for_searching_best_config=nn_engine
    )

    # Step 5: Persist configuration
    fec.save_configuration_in_machine()

    # Step 6: Return complete setup
    return machine, fec, nn_engine

# Usage
machine, fec, nn_engine = setup_machine_with_optimized_features(
    "sales_data.csv",
    target_column="revenue",
    budget=150
)
```

### Configuration Analysis and Monitoring

```python
def analyze_fet_configuration(fec: FeatureEngineeringConfiguration):
    """
    Analyze and report on FET configuration effectiveness.
    """

    print(f"=== FET Configuration Analysis ===")
    print(f"Optimization Time: {fec._fe_find_delay_sec:.2f}s")
    print(f"Configuration Reliability: {fec._configuration_reliability:.2f}")
    print(f"Total Budget Used: {sum(fec._cost_per_columns.values())}")

    # Analyze FET usage by column
    fet_usage = {}
    for col, fets in fec._activated_fet_list_per_column.items():
        for fet in fets:
            fet_usage[fet] = fet_usage.get(fet, 0) + 1

    print(f"\nFET Usage Statistics:")
    for fet_name, count in sorted(fet_usage.items(), key=lambda x: x[1], reverse=True):
        print(f"  {fet_name}: {count} columns")

    # Cost analysis
    print(f"\nColumn Cost Breakdown:")
    for col, cost in sorted(fec._cost_per_columns.items(), key=lambda x: x[1], reverse=True):
        print(f"  {col}: {cost}")

# Usage
analyze_fet_configuration(fec)
```

## FeatureEngineeringColumn Class

### Single Column Optimization Engine

The `FeatureEngineeringColumn` class handles the complex task of selecting optimal FET combinations for individual dataset columns. It implements sophisticated optimization algorithms while respecting budget constraints and performance requirements.

### Key Attributes

```python
class FeatureEngineeringColumn:
    # Core references
    self._nn_engine: Optional[NNEngine]              # Neural network evaluator
    self._column_info: Column_datas_infos           # Column metadata
    self._budget_max: float                         # Maximum budget for this column

    # Optimization state
    self._selected_fets: List[str]                  # Chosen FET combination
    self._total_cost: float                         # Total computational cost
    self._estimated_performance_gain: float         # Predicted improvement
    self._optimization_confidence: float            # Confidence in selection

    # Search space
    self._compatible_fets: List[str]                # FETs suitable for this column
    self._solution_space: Dict                      # Search space definition
```

### Initialization and Optimization Modes

#### `__init__(self, nn_engine_to_use_in_trial: Optional[NNEngine] = None, this_column_datas_infos: Optional[Column_datas_infos] = None, this_fec_budget_maximum: Optional[float] = None, force_load_this_fet_names: Optional[Dict[str, bool]] = None)`

**Intelligent Mode Selection**:

```python
def __init__(self, nn_engine_to_use_in_trial=None, this_column_datas_infos=None,
             this_fec_budget_maximum=None, force_load_this_fet_names=None):
    """
    Initialize column optimizer with intelligent mode selection.

    Args:
        nn_engine_to_use_in_trial: NN engine for performance evaluation
        this_column_datas_infos: Complete column metadata
        this_fec_budget_maximum: Maximum budget for this column
        force_load_this_fet_names: Pre-determined FET selections (bypass optimization)
    """

    # Store core parameters
    self._nn_engine = nn_engine_to_use_in_trial
    self._column_info = this_column_datas_infos
    self._budget_max = this_fec_budget_maximum

    # Determine optimization mode
    if force_load_this_fet_names is not None:
        # Forced Configuration Mode
        self._init_forced_configuration(force_load_this_fet_names)
    elif self._budget_max <= 1.0:
        # Minimum Configuration Mode
        self._init_minimum_configuration()
    else:
        # Experimental Optimization Mode
        self._init_experimental_optimization()
```

#### 1. Experimental Optimization Mode

```python
def _init_experimental_optimization(self):
    """Perform intelligent FET selection using experimental evaluation."""

    # Step 1: Generate compatible FET candidates
    self._compatible_fets = self._identify_compatible_fets()

    # Step 2: Build solution space for optimization
    self._solution_space = self._generate_fet_solution_space()

    # Step 3: Create experimenter for FET evaluation
    experimenter = ExperimenterColumnFETSelector(
        nn_engine_trial=self._nn_engine,
        column_info=self._column_info,
        budget_max=self._budget_max
    )

    # Step 4: Execute optimization using SolutionFinder
    solution_finder = SolutionFinder(
        experimenter=experimenter,
        solution_space=self._solution_space,
        budget_max=self._budget_max,
        optimization_criteria=["performance_gain", "cost_efficiency", "data_integrity"]
    )

    # Step 5: Find optimal FET combination
    optimal_solution = solution_finder.find_optimal_solution()

    # Step 6: Store results
    self._selected_fets = optimal_solution.get("fet_combination", [])
    self._total_cost = optimal_solution.get("total_cost", 0.0)
    self._estimated_performance_gain = optimal_solution.get("performance_gain", 0.0)
    self._optimization_confidence = optimal_solution.get("confidence", 0.5)
```

#### 2. Forced Configuration Mode

```python
def _init_forced_configuration(self, forced_fets: Dict[str, bool]):
    """Apply predetermined FET selections without optimization."""

    self._selected_fets = []
    self._total_cost = 0.0

    # Apply forced FET selections
    for fet_name, enabled in forced_fets.items():
        if enabled and self._is_fet_compatible(fet_name):
            self._selected_fets.append(fet_name)
            self._total_cost += self._calculate_fet_cost(fet_name)

    # Validate forced configuration
    if not self._validate_fet_combination(self._selected_fets):
        logger.warning(f"Forced FET combination may have conflicts: {self._selected_fets}")

    self._estimated_performance_gain = 0.0  # Unknown without evaluation
    self._optimization_confidence = 0.8  # High confidence in explicit selection
```

#### 3. Minimum Configuration Mode

```python
def _init_minimum_configuration(self):
    """Apply minimal lossless transformations for basic compatibility."""

    # Select only essential, lossless FETs
    self._selected_fets = self._select_minimum_lossless_fets()
    self._total_cost = self._calculate_fet_cost(self._selected_fets)

    # Minimum configuration has predictable but limited performance impact
    self._estimated_performance_gain = 0.05  # Conservative estimate
    self._optimization_confidence = 0.9  # High confidence in stability
```

### Advanced Optimization Algorithm

#### Solution Space Generation

```python
def _generate_fet_solution_space(self) -> Dict:
    """
    Generate comprehensive solution space for FET optimization.
    """

    solution_space = {
        "fet_candidates": {},
        "constraints": {},
        "cost_model": {},
        "compatibility_rules": {}
    }

    # Generate individual FET candidates
    for fet_name in self._compatible_fets:
        fet_info = self._analyze_fet_candidate(fet_name)

        solution_space["fet_candidates"][fet_name] = {
            "cost": fet_info["cost"],
            "expected_gain": fet_info["expected_performance_gain"],
            "data_requirements": fet_info["data_requirements"],
            "output_columns": fet_info["output_columns"],
            "risk_level": fet_info["risk_level"]
        }

    # Define compatibility constraints
    solution_space["constraints"] = self._generate_fet_constraints()

    # Build cost model
    solution_space["cost_model"] = self._build_cost_model()

    return solution_space
```

#### Multi-Objective Optimization

```python
def _evaluate_fet_combination(self, fet_combination: List[str]) -> Dict[str, float]:
    """
    Evaluate FET combination across multiple objectives.
    """

    evaluation_results = {}

    # Objective 1: Performance Improvement
    performance_gain = self._measure_performance_improvement(fet_combination)
    evaluation_results["performance_gain"] = performance_gain

    # Objective 2: Computational Cost
    total_cost = self._calculate_fet_cost(fet_combination)
    cost_efficiency = performance_gain / max(total_cost, 0.1)  # Avoid division by zero
    evaluation_results["cost_efficiency"] = cost_efficiency

    # Objective 3: Data Integrity
    data_integrity_score = self._assess_data_integrity(fet_combination)
    evaluation_results["data_integrity"] = data_integrity_score

    # Objective 4: Scalability
    scalability_score = self._assess_scalability(fet_combination)
    evaluation_results["scalability"] = scalability_score

    # Calculate composite score (weighted average)
    weights = {
        "performance_gain": 0.4,
        "cost_efficiency": 0.3,
        "data_integrity": 0.2,
        "scalability": 0.1
    }

    composite_score = sum(
        evaluation_results[obj] * weight
        for obj, weight in weights.items()
    )

    evaluation_results["composite_score"] = composite_score

    return evaluation_results
```

## Module Interactions and Integration

### With SolutionFinder - Optimization Engine

**Advanced Optimization Integration**:

```python
from ML import SolutionFinder, ExperimenterColumnFETSelector

def optimize_fet_configuration_advanced(machine, column_info, budget):
    """
    Advanced FET optimization using SolutionFinder with custom criteria.
    """

    # Create experimenter for FET evaluation
    experimenter = ExperimenterColumnFETSelector(
        nn_engine_trial=machine.get_nn_engine(),
        column_info=column_info,
        budget_max=budget,
        evaluation_metrics=["accuracy", "f1_score", "computational_cost"]
    )

    # Configure solution finder with multiple optimization criteria
    solution_finder = SolutionFinder(
        experimenter=experimenter,
        solution_space=generate_fet_solution_space(column_info),
        budget_max=budget,
        optimization_criteria={
            "primary": "performance_improvement",
            "secondary": ["cost_efficiency", "data_integrity"],
            "constraints": ["max_cost", "min_performance_gain"]
        },
        convergence_criteria={
            "max_iterations": 100,
            "improvement_threshold": 0.001,
            "time_limit_seconds": 300
        }
    )

    # Execute optimization
    optimal_solution = solution_finder.find_optimal_solution()

    return {
        "best_fet_combination": optimal_solution["fet_combination"],
        "expected_performance_gain": optimal_solution["performance_gain"],
        "total_cost": optimal_solution["total_cost"],
        "optimization_metadata": optimal_solution["metadata"]
    }

# Usage
optimization_result = optimize_fet_configuration_advanced(machine, column_info, 50)
```

**SolutionFinder Integration Benefits**:
- **Multi-objective optimization** across performance, cost, and data integrity
- **Constraint satisfaction** ensuring budget limits and compatibility requirements
- **Intelligent search strategies** including genetic algorithms and Bayesian optimization
- **Convergence guarantees** with configurable stopping criteria
- **Robust error handling** with fallback mechanisms

### With FeatureEngineeringTemplate - FET Management

**Dynamic FET Instantiation and Management**:

```python
from ML import FeatureEngineeringTemplate

class FETManager:
    """
    Advanced FET management with dynamic instantiation and validation.
    """

    def __init__(self, column_data, column_info):
        self.column_data = column_data
        self.column_info = column_info
        self.fet_cache = {}  # Cache instantiated FETs

    def get_fet_instance(self, fet_name: str):
        """Get or create FET instance with caching."""

        if fet_name not in self.fet_cache:
            # Dynamic FET instantiation
            fet_class = getattr(FeatureEngineeringTemplate, fet_name, None)
            if fet_class is None:
                raise ValueError(f"Unknown FET: {fet_name}")

            # Instantiate with column-specific configuration
            fet_instance = fet_class(
                column_data=self.column_data,
                column_datas_infos=self.column_info
            )

            self.fet_cache[fet_name] = fet_instance

        return self.fet_cache[fet_name]

    def evaluate_fet_compatibility(self, fet_name: str) -> Dict:
        """Comprehensive FET compatibility and capability assessment."""

        fet_instance = self.get_fet_instance(fet_name)

        return {
            "is_compatible": fet_instance.is_compatible(),
            "cost": fet_instance.calculate_cost(),
            "capabilities": fet_instance.get_capabilities(),
            "risk_assessment": fet_instance.assess_risk(),
            "performance_estimate": fet_instance.estimate_performance_impact()
        }

    def apply_fet_transformations(self, fet_names: List[str]) -> pd.DataFrame:
        """Apply multiple FET transformations in optimal order."""

        transformed_data = self.column_data.copy()

        # Sort FETs by dependency order
        ordered_fets = self._resolve_fet_dependencies(fet_names)

        for fet_name in ordered_fets:
            fet_instance = self.get_fet_instance(fet_name)
            transformed_data = fet_instance.transform(transformed_data)

        return transformed_data

# Usage
fet_manager = FETManager(column_data, column_info)

# Evaluate FET compatibility
compatibility = fet_manager.evaluate_fet_compatibility("FETNumericMinMaxFloat")
print(f"Cost: {compatibility['cost']}, Compatible: {compatibility['is_compatible']}")

# Apply transformations
transformed = fet_manager.apply_fet_transformations(["FETNumericMinMaxFloat", "FET6PowerFloat"])
```

**FeatureEngineeringTemplate Integration Features**:
- **Dynamic instantiation** of FET classes by name
- **Capability assessment** for compatibility checking
- **Cost calculation** for budget management
- **Risk assessment** for safety evaluation
- **Performance estimation** for optimization guidance

### With Machine - Core Integration

**Seamless Machine Lifecycle Integration**:

```python
from ML import Machine, FeatureEngineeringConfiguration, EncDec

def complete_machine_setup_with_fec(data_path: str, target_column: str):
    """
    Complete machine setup integrating FEC with other components.
    """

    # Phase 1: Create machine and basic data configuration
    machine = Machine.create_from_dataset(data_path, target_column)

    # Phase 2: Initialize NN engine for evaluation
    nn_engine = machine.get_nn_engine()

    # Phase 3: Optimize feature engineering configuration
    budget = machine.get_machine_level() * 25  # Scale budget with machine level

    fec = FeatureEngineeringConfiguration(
        machine=machine,
        global_dataset_budget=budget,
        nn_engine_for_searching_best_config=nn_engine
    )

    # Phase 4: Save FEC configuration
    fec.save_configuration_in_machine()

    # Phase 5: Initialize EncDec with optimized features
    encdec = EncDec(machine)

    # Phase 6: Train and validate complete pipeline
    nn_engine.machine_nn_engine_do_full_training_if_not_already_done()

    return {
        "machine": machine,
        "fec": fec,
        "encdec": encdec,
        "nn_engine": nn_engine,
        "performance_metrics": nn_engine.get_training_metrics()
    }

# Usage
setup_result = complete_machine_setup_with_fec("customer_data.csv", "churn_probability")
```

### With InputsColumnsImportance - Intelligent Budget Allocation

**Importance-Driven Resource Allocation**:

```python
from ML import InputsColumnsImportance

def allocate_budget_by_importance(machine, total_budget: int) -> Dict[str, float]:
    """
    Allocate FEC budget based on column importance scores.
    """

    # Get column importance evaluator
    ici = InputsColumnsImportance(machine)

    # Calculate importance scores for all input columns
    importance_scores = {}
    input_columns = machine._mdc.get_input_columns()

    for column in input_columns:
        importance = ici.calculate_column_importance(column)
        importance_scores[column] = importance

    # Allocate budget proportionally to importance
    total_importance = sum(importance_scores.values())

    if total_importance == 0:
        # Equal allocation if no importance data
        base_budget = total_budget / len(input_columns)
        return {col: base_budget for col in input_columns}

    # Proportional allocation with minimum guarantees
    allocation = {}
    for column, importance in importance_scores.items():
        column_budget = (importance / total_importance) * total_budget
        column_budget = max(column_budget, 1.0)  # Minimum budget
        allocation[column] = column_budget

    return allocation

# Usage
budget_allocation = allocate_budget_by_importance(machine, 100)
print(f"Budget allocation: {budget_allocation}")
```

## Best Practices and Performance Optimization

### Budget Management Strategies

#### 1. Dynamic Budget Scaling
```python
def calculate_optimal_budget(machine) -> int:
    """
    Calculate optimal budget based on multiple factors.
    """

    base_budget = machine.get_machine_level() * 15

    # Scale by dataset size
    data_size_factor = min(machine.get_dataset_size() / 10000, 3.0)
    budget = base_budget * data_size_factor

    # Scale by data complexity
    complexity_factor = machine.assess_data_complexity()
    budget = budget * complexity_factor

    # Scale by available computational resources
    resource_factor = machine.get_available_resources() / machine.get_required_resources()
    budget = budget * min(resource_factor, 2.0)

    return int(max(budget, 10))  # Minimum budget
```

#### 2. Progressive Budget Allocation
```python
def progressive_budget_optimization(machine, max_budget: int):
    """
    Apply FEC optimization with progressive budget increases.
    """

    budgets_to_try = [10, 25, 50, 100, max_budget]
    best_fec = None
    best_performance = 0

    for budget in budgets_to_try:
        if budget > max_budget:
            break

        # Create FEC with current budget
        fec = FeatureEngineeringConfiguration(
            machine=machine,
            global_dataset_budget=budget,
            nn_engine_for_searching_best_config=machine.get_nn_engine()
        )

        # Evaluate performance
        performance = evaluate_fec_performance(fec)

        if performance > best_performance:
            best_performance = performance
            best_fec = fec

        # Early stopping if performance gain is minimal
        if best_performance > 0.8:  # Good enough performance
            break

    return best_fec
```

### Configuration Validation and Monitoring

#### Comprehensive Configuration Validation
```python
def validate_fec_configuration(fec: FeatureEngineeringConfiguration) -> Dict:
    """
    Comprehensive validation of FEC configuration.
    """

    validation_results = {
        "is_valid": True,
        "warnings": [],
        "errors": [],
        "recommendations": []
    }

    # Check budget compliance
    total_cost = sum(fec._cost_per_columns.values())
    if total_cost > fec._global_dataset_budget:
        validation_results["errors"].append(f"Budget exceeded: {total_cost} > {fec._global_dataset_budget}")

    # Check FET compatibility
    for column, fets in fec._activated_fet_list_per_column.items():
        column_info = fec.get_all_column_datas_infos(column)
        for fet in fets:
            if not fec._is_fet_compatible_with_column(fet, column_info):
                validation_results["errors"].append(f"Incompatible FET {fet} for column {column}")

    # Check for redundant transformations
    redundant_fets = fec._detect_redundant_transformations()
    if redundant_fets:
        validation_results["warnings"].extend([f"Redundant FETs: {redundant}" for redundant in redundant_fets])

    # Performance recommendations
    if fec._configuration_reliability < 0.7:
        validation_results["recommendations"].append("Consider increasing budget for better optimization")

    validation_results["is_valid"] = len(validation_results["errors"]) == 0

    return validation_results

# Usage
validation = validate_fec_configuration(fec)
if not validation["is_valid"]:
    print("Configuration errors:", validation["errors"])
```

### Performance Monitoring and Optimization

#### Real-time Performance Tracking
```python
def monitor_fec_performance(fec: FeatureEngineeringConfiguration):
    """
    Monitor FEC performance and resource usage in real-time.
    """

    import time
    import psutil

    start_time = time.time()
    start_memory = psutil.virtual_memory().used

    # Track key metrics
    metrics = {
        "optimization_start": start_time,
        "initial_memory": start_memory,
        "columns_processed": 0,
        "fets_evaluated": 0,
        "budget_utilization": 0.0
    }

    # Monitor during optimization
    while not fec._optimization_complete:
        time.sleep(1)  # Monitoring interval

        current_memory = psutil.virtual_memory().used
        memory_delta = current_memory - start_memory

        metrics.update({
            "current_memory": current_memory,
            "memory_delta_mb": memory_delta / (1024 * 1024),
            "cpu_usage": psutil.cpu_percent(),
            "elapsed_time": time.time() - start_time
        })

        # Log progress
        print(f"FEC Progress: {metrics['columns_processed']} columns, "
              f"Memory: {metrics['memory_delta_mb']:.1f}MB, "
              f"Time: {metrics['elapsed_time']:.1f}s")

    # Final metrics
    final_metrics = {
        "total_time": time.time() - start_time,
        "total_memory_used": (psutil.virtual_memory().used - start_memory) / (1024 * 1024),
        "final_budget_utilization": sum(fec._cost_per_columns.values()) / fec._global_dataset_budget,
        "configuration_reliability": fec._configuration_reliability
    }

    return final_metrics

# Usage
performance_metrics = monitor_fec_performance(fec)
print(f"Optimization completed in {performance_metrics['total_time']:.2f}s")
```

### Troubleshooting Common Issues

#### Budget-Related Issues
```python
def troubleshoot_budget_issues(fec: FeatureEngineeringConfiguration):
    """
    Diagnose and resolve budget-related configuration issues.
    """

    issues = []

    # Check budget utilization
    budget_used = sum(fec._cost_per_columns.values())
    budget_available = fec._global_dataset_budget
    utilization_rate = budget_used / budget_available

    if utilization_rate > 0.95:
        issues.append("Budget nearly exhausted - consider increasing budget for better optimization")
    elif utilization_rate < 0.3:
        issues.append("Budget underutilized - consider reducing budget or increasing complexity")

    # Check column budget distribution
    max_column_budget = max(fec._cost_per_columns.values())
    min_column_budget = min(fec._cost_per_columns.values())

    if max_column_budget > budget_available * 0.5:
        issues.append(f"Column budget imbalance - one column uses {max_column_budget:.1f} of {budget_available} total budget")

    # Check minimum configurations
    min_config_columns = [col for col, cost in fec._cost_per_columns.items() if cost <= 1.0]
    if len(min_config_columns) > len(fec._cost_per_columns) * 0.5:
        issues.append("Many columns have minimum configuration - budget may be too low")

    return issues

# Usage
budget_issues = troubleshoot_budget_issues(fec)
for issue in budget_issues:
    print(f"Budget Issue: {issue}")
```

#### Performance Optimization Issues
```python
def optimize_fec_performance(fec: FeatureEngineeringConfiguration):
    """
    Apply performance optimizations to FEC configuration.
    """

    optimizations = []

    # Caching optimization
    if not hasattr(fec, '_fet_cache'):
        fec._fet_cache = {}
        optimizations.append("Added FET caching to reduce instantiation overhead")

    # Parallel processing for independent columns
    independent_columns = fec._identify_independent_columns()
    if len(independent_columns) > 1:
        fec._enable_parallel_processing(independent_columns)
        optimizations.append(f"Enabled parallel processing for {len(independent_columns)} independent columns")

    # Memory optimization
    if fec._should_enable_memory_optimization():
        fec._enable_memory_optimization()
        optimizations.append("Enabled memory optimization for large datasets")

    # Early stopping optimization
    if fec._configuration_reliability > 0.8:
        fec._enable_early_stopping()
        optimizations.append("Enabled early stopping for high-confidence configurations")

    return optimizations

# Usage
performance_opts = optimize_fec_performance(fec)
print("Applied optimizations:", performance_opts)
```

This comprehensive documentation provides developers with the knowledge and tools needed to effectively utilize the `FeatureEngineeringConfiguration` system for intelligent, budget-aware feature engineering optimization in the EasyAutoML platform.

