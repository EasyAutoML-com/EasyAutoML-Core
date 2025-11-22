# ML/Machine.py - Core Machine Learning Machine Management System

## Overview

The `Machine` class is the central orchestrator in the EasyAutoML.com system, managing the complete lifecycle of machine learning models from creation to deployment. It serves as the primary interface between users, data, and the underlying ML infrastructure, providing comprehensive machine learning pipeline management.

**Location**: `ML/Machine.py`

## Description

### What

The Machine class orchestrates the complete machine learning lifecycle from creation to deployment, serving as the central coordinator for all ML components and the primary interface between users, data, and the underlying infrastructure. It manages data storage, configuration persistence, access control, and resource allocation across the entire ML pipeline.

### How

It creates dynamic database tables for machine-specific data storage, coordinates initialization of all ML components (MDC, ICI, FEC, EncDec, NNEngine) in proper dependency order, and manages configuration state through re-run flags. The system handles data ingestion with intelligent partitioning and provides flexible data access patterns.

### Where

Used as the primary entry point by all applications and higher-level interfaces (MachineEasyAutoML, APIs), and referenced by all ML components for data access and configuration management. Serves as the foundation for the entire AutoML system.

### When

Created at the start of any ML workflow and persists throughout the model lifecycle from initial data ingestion through training to production deployment.

## Core Architecture

### Primary Responsibilities

- **Machine Lifecycle Management**: Creation, loading, updating, and deletion of ML models
- **Data Pipeline Orchestration**: Coordinates data preprocessing, feature engineering, and model training
- **Configuration Management**: Maintains all machine learning configurations and parameters
- **Access Control**: Implements user and team-based permissions
- **Resource Management**: Handles computational resources and billing integration

### Key Attributes

```python
class Machine:
    # Database model instances
    self.db_machine: machine_model  # Main machine configuration
    self.db_data_input_lines        # Input data storage
    self.db_data_output_lines       # Output data storage

    # Core identification
    self.id: int                    # Machine unique identifier
```

## Initialization Methods

### Machine Creation Patterns

#### 1. Create from Dataset - Basic Approach

```python
from ML import Machine
import pandas as pd

# Load data from CSV file
user_dataframe = pd.read_csv("customer_churn_data.csv")

# Create new machine with automatic data processing
machine = Machine(
    machine_identifier_or_name="customer_churn_predictor_v2",
    user_dataset_unformatted=user_dataframe,
    machine_level=3,  # Higher level for more resources
    machine_create_user_id=current_user_id,
    decimal_separator=".",  # Decimal separator for number parsing
    date_format="DMY"       # Date format (DMY or MDY)
)

print(f"Machine created with ID: {machine.id}")
print(f"Machine level: {machine.db_machine.machine_level}")
```

**Complete Creation Workflow**:
1. **Input Validation**: Verifies dataset structure and minimum requirements (5+ rows, columns)
2. **DataFileReader Integration**: Automatically processes data with type detection and cleaning
3. **MachineDataConfiguration**: Analyzes column types, missing values, and statistical properties
4. **Database Setup**: Creates machine record and dynamic data tables (Machine_NNN_DataInputLines, Machine_NNN_DataOutputLines)
5. **Initial Configuration**: Sets up default feature engineering and encoding configurations
6. **Access Control**: Assigns ownership and team permissions

#### 2. Create from DataFileReader - Advanced Configuration

```python
from ML import Machine, DataFileReader
import pandas as pd

# Step 1: Pre-process data with DataFileReader for full control
dfr = DataFileReader(
    data_source="sales_data.csv",
    decimal_separator=".",     # European decimal format
    date_format="DMY",         # Day-Month-Year format
    force_create_with_this_datatypes={
        "customer_id": "LABEL",    # Categorical identifier
        "revenue": "FLOAT",        # Sales amount
        "order_date": "DATE",      # Order date
        "category": "LABEL",       # Product category
        "is_returned": "LABEL"     # Binary outcome
    },
    force_create_with_this_descriptions={
        "customer_id": "Unique customer identifier",
        "revenue": "Total order revenue in euros",
        "order_date": "Date when order was placed",
        "category": "Product category classification",
        "is_returned": "Whether the order was returned"
    }
)

# Step 2: Create machine with explicit column specifications
machine = Machine(
    machine_identifier_or_name="sales_prediction_model",
    dfr=dfr,
    force_create_with_this_inputs={
        "revenue": True,
        "order_date": True,
        "category": True
    },
    force_create_with_this_outputs={
        "is_returned": True
    },
    machine_level=4,  # Enterprise level with maximum resources
    machine_create_user_id=user_id,
    machine_create_team_id=team_id,  # Optional team ownership
    decimal_separator=".",
    date_format="DMY"
)

print(f"Processed {len(dfr.get_formatted_user_dataframe())} rows")
print(f"Detected column types: {dfr.get_user_columns_datatype}")
```

#### 3. Create from DataFrame - Programmatic Approach

```python
from ML import Machine
import pandas as pd

# Prepare data programmatically
data = {
    'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'feature2': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
    'feature3': [10.5, 20.3, 15.7, 8.9, 12.1, 25.4, 18.6, 9.8, 14.2, 22.9],
    'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
}

user_dataframe = pd.DataFrame(data)

# Create machine with programmatic data
machine = Machine(
    machine_identifier_or_name="programmatic_model",
    user_dataset_unformatted=user_dataframe,
    machine_level=2,
    machine_create_user_id=user_id,
    decimal_separator=".",
    date_format="DMY"
)
```

#### 4. Load Existing Machine - Multiple Access Patterns

```python
from ML import Machine

# Pattern 1: Load by ID (most efficient)
machine_by_id = Machine(
    machine_identifier_or_name=123,  # Machine ID
    machine_access_check_with_user_id=current_user_id
)

# Pattern 2: Load by name (user-friendly)
machine_by_name = Machine(
    machine_identifier_or_name="customer_churn_predictor_v2",
    machine_access_check_with_user_id=current_user_id
)

# Pattern 3: Load for team access
machine_team = Machine(
    machine_identifier_or_name="team_project_model",
    machine_access_check_with_user_id=team_member_id
)

# Pattern 4: Load for administrative access
machine_admin = Machine(
    machine_identifier_or_name="any_machine_name",
    machine_access_check_with_user_id=admin_user_id  # Admin bypasses ownership checks
)

# Verify successful loading
print(f"Loaded machine: {machine_by_id.db_machine.machine_name}")
print(f"Owner: {machine_by_id.db_machine.machine_owner_user.email}")
print(f"Level: {machine_by_id.db_machine.machine_level}")
print(f"Created: {machine_by_id.db_machine.created_at}")
```

#### 5. Advanced Creation with Custom Configuration

```python
from ML import Machine
import pandas as pd

# Load and prepare data
raw_data = pd.read_csv("complex_dataset.csv")

# Advanced machine creation with full control
machine = Machine(
    machine_identifier_or_name="advanced_ml_model",
    user_dataset_unformatted=raw_data,
    machine_level=5,  # Maximum level
    machine_create_user_id=user_id,
    machine_create_team_id=team_id,

    # Data processing parameters
    decimal_separator=",",    # European format
    date_format="DMY",        # European date format

    # Force specific column configurations
    force_create_with_this_inputs={
        "numerical_feature": True,
        "categorical_feature": True,
        "date_feature": True,
        "text_feature": True
    },
    force_create_with_this_outputs={
        "prediction_target": True
    },

    # Additional metadata
    force_create_with_this_descriptions={
        "numerical_feature": "Key numerical predictor variable",
        "categorical_feature": "Categorical classification feature",
        "prediction_target": "Primary prediction outcome"
    }
)

# Post-creation configuration
print(f"Machine created: {machine.db_machine.machine_name}")
print(f"Data tables created: Machine_{machine.id}_DataInputLines, Machine_{machine.id}_DataOutputLines")
print(f"Initial status: Ready for training")
```

## Data Management System

### Dynamic Table Creation

The Machine class creates machine-specific database tables to store training and prediction data. This approach provides isolation, scalability, and optimized access patterns for machine learning operations.

#### Table Creation Process

```python
# Automatic table creation during machine initialization
def data_lines_create_both_tables(self):
    """Create dynamic input and output data tables for this machine"""

    # Create input data table (Machine_NNN_DataInputLines)
    input_table_name = f"Machine_{self.id}_DataInputLines"
    input_model = self._create_dynamic_model(
        input_table_name,
        self._get_input_column_definitions(),
        is_input_table=True
    )

    # Create output data table (Machine_NNN_DataOutputLines)
    output_table_name = f"Machine_{self.id}_DataOutputLines"
    output_model = self._create_dynamic_model(
        output_table_name,
        self._get_output_column_definitions(),
        is_input_table=False
    )

    # Store model references for future operations
    self.db_data_input_lines = input_model
    self.db_data_output_lines = output_model
```

#### Table Structure Details

**Input Data Table (Machine_NNN_DataInputLines)**:
```sql
CREATE TABLE Machine_123_DataInputLines (
    Line_ID INT AUTO_INCREMENT PRIMARY KEY,
    is_for_learning BOOLEAN DEFAULT TRUE,        -- Training data flag
    is_for_evaluation BOOLEAN DEFAULT FALSE,     -- Validation data flag
    feature1 FLOAT,                              -- Numerical features
    feature2 VARCHAR(255),                       -- Categorical features
    feature3 DATE,                               -- Date features
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    -- Optimized indexes for ML training
    INDEX idx_learning (is_for_learning),
    INDEX idx_evaluation (is_for_evaluation),
    INDEX idx_feature1 (feature1),               -- Index on important features
    INDEX idx_created (created_at)
) ENGINE=InnoDB;
```

**Output Data Table (Machine_NNN_DataOutputLines)**:
```sql
CREATE TABLE Machine_123_DataOutputLines (
    Line_ID INT AUTO_INCREMENT PRIMARY KEY,
    target_variable FLOAT,                       -- Prediction targets
    is_for_learning BOOLEAN DEFAULT TRUE,
    is_for_evaluation BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Foreign key relationship to input table
    FOREIGN KEY (Line_ID) REFERENCES Machine_123_DataInputLines(Line_ID)
) ENGINE=InnoDB;
```

**Key Design Features**:
- **Primary Key**: Auto-incrementing Line_ID for data alignment
- **Training Flags**: `is_for_learning` and `is_for_evaluation` for data partitioning
- **Dynamic Columns**: One column per dataset feature (automatically created)
- **Indexing Strategy**: Optimized for ML training access patterns
- **Foreign Key Constraints**: Maintains input-output data alignment

### Data Operations

#### Data Ingestion with Advanced Partitioning

```python
def data_lines_append(
    self,
    user_dataframe_to_append: pd.DataFrame,
    split_lines_in_learning_and_evaluation: bool = True,
    evaluation_percentage: float = 0.2,
    stratify_by_output: bool = True,
    random_seed: int = 42,
    **kwargs
) -> dict:
    """
    Advanced data ingestion with intelligent partitioning.

    Args:
        user_dataframe_to_append: Data to add to machine
        split_lines_in_learning_and_evaluation: Whether to split data
        evaluation_percentage: Percentage for evaluation (default 20%)
        stratify_by_output: Maintain class distribution in splits
        random_seed: Random seed for reproducible splits

    Returns:
        Dict with ingestion statistics and partition information
    """

    # Validate input data structure
    self._validate_input_dataframe(user_dataframe_to_append)

    # Handle data partitioning
    if split_lines_in_learning_and_evaluation:
        train_data, eval_data = self._partition_data(
            user_dataframe_to_append,
            evaluation_percentage,
            stratify_by_output,
            random_seed
        )
    else:
        train_data = user_dataframe_to_append
        eval_data = pd.DataFrame()

    # Insert data with transaction safety
    with transaction.atomic():
        train_count = self._insert_data_batch(train_data, is_learning=True)
        eval_count = self._insert_data_batch(eval_data, is_learning=False)

    return {
        "total_rows_added": len(user_dataframe_to_append),
        "training_rows": train_count,
        "evaluation_rows": eval_count,
        "input_columns": len(user_dataframe_to_append.columns) - len(self._output_columns),
        "output_columns": len(self._output_columns)
    }
```

**Advanced Partitioning Strategies**:

```python
def _partition_data(self, dataframe, eval_pct, stratify, seed):
    """Intelligent data partitioning with stratification"""

    if stratify and self._output_columns:
        # Stratified split maintains class distribution
        from sklearn.model_selection import train_test_split

        # Get output column for stratification
        output_col = self._output_columns[0]  # Primary output
        stratify_labels = dataframe[output_col]

        train_data, eval_data = train_test_split(
            dataframe,
            test_size=eval_pct,
            stratify=stratify_labels,
            random_state=seed
        )
    else:
        # Random split without stratification
        split_idx = int(len(dataframe) * (1 - eval_pct))
        train_data = dataframe[:split_idx].copy()
        eval_data = dataframe[split_idx:].copy()

    return train_data, eval_data
```

#### Data Retrieval with Flexible Access Patterns

```python
def data_lines_read(
    self,
    read_mode: str = "all",
    limit: int = None,
    random_sample: bool = False,
    random_seed: int = 42,
    include_metadata: bool = False,
    **filters
) -> pd.DataFrame:
    """
    Flexible data retrieval with multiple access patterns.

    Args:
        read_mode: "all", "training", "evaluation", "random_sample"
        limit: Maximum number of rows to return
        random_sample: Whether to return random subset
        random_seed: Random seed for reproducibility
        include_metadata: Include training flags and timestamps
        **filters: Additional column-based filters

    Returns:
        DataFrame with requested data
    """

    # Build query based on read mode
    query_filters = {}

    if read_mode == "training":
        query_filters['is_for_learning'] = True
    elif read_mode == "evaluation":
        query_filters['is_for_evaluation'] = True
    elif read_mode == "all":
        pass  # No additional filters
    else:
        raise ValueError(f"Unknown read_mode: {read_mode}")

    # Apply additional filters
    query_filters.update(filters)

    # Execute query with optimizations
    if random_sample and limit:
        # Random sampling query
        queryset = self.db_data_input_lines.objects.filter(
            **query_filters
        ).order_by('?')[:limit]
    elif limit:
        # Limited ordered query
        queryset = self.db_data_input_lines.objects.filter(
            **query_filters
        )[:limit]
    else:
        # Full dataset query
        queryset = self.db_data_input_lines.objects.filter(**query_filters)

    # Convert to DataFrame
    data = pd.DataFrame(list(queryset.values()))

    if not include_metadata:
        # Remove metadata columns
        metadata_cols = ['Line_ID', 'is_for_learning', 'is_for_evaluation',
                        'created_at', 'updated_at']
        data = data.drop(columns=[col for col in metadata_cols if col in data.columns])

    return data
```

**Read Mode Options**:
- `"all"`: Complete dataset (default)
- `"training"`: Only training partition (`is_for_learning=True`)
- `"evaluation"`: Only evaluation partition (`is_for_evaluation=True`)
- `"random_sample"`: Random subset for quick analysis or testing

#### Data Update Operations

```python
def data_lines_update(
    self,
    update_conditions: dict,
    update_values: dict,
    limit: int = None
) -> int:
    """
    Update existing data rows based on conditions.

    Args:
        update_conditions: Dictionary of column-value conditions
        update_values: Dictionary of column-value updates
        limit: Maximum number of rows to update

    Returns:
        Number of rows updated
    """

    # Build update query
    queryset = self.db_data_input_lines.objects.filter(**update_conditions)

    if limit:
        queryset = queryset[:limit]

    # Execute update
    return queryset.update(**update_values)
```

#### Data Deletion Operations

```python
def data_lines_delete(
    self,
    delete_conditions: dict,
    limit: int = None,
    cascade: bool = True
) -> int:
    """
    Delete data rows based on conditions.

    Args:
        delete_conditions: Dictionary of column-value conditions
        limit: Maximum number of rows to delete
        cascade: Whether to delete corresponding output data

    Returns:
        Number of rows deleted
    """

    # Delete from input table
    queryset = self.db_data_input_lines.objects.filter(**delete_conditions)

    if limit:
        queryset = queryset[:limit]

    deleted_count = queryset.count()

    if cascade and self.db_data_output_lines is not None:
        # Delete corresponding output data
        line_ids = list(queryset.values_list('Line_ID', flat=True))
        self.db_data_output_lines.objects.filter(Line_ID__in=line_ids).delete()

    # Delete input data
    queryset.delete()

    return deleted_count
```

### Data Quality and Integrity

#### Data Validation Methods

```python
def validate_data_integrity(self) -> dict:
    """
    Comprehensive data integrity check.

    Returns:
        Dictionary with validation results and issues found
    """

    issues = []

    # Check input-output alignment
    input_count = self.db_data_input_lines.objects.count()
    output_count = self.db_data_output_lines.objects.count()

    if input_count != output_count:
        issues.append({
            'type': 'alignment_mismatch',
            'message': f'Input rows ({input_count}) != Output rows ({output_count})',
            'severity': 'critical'
        })

    # Check for missing values
    for column in self._get_input_columns():
        null_count = self.db_data_input_lines.objects.filter(
            **{f"{column}__isnull": True}
        ).count()

        if null_count > 0:
            percentage = (null_count / input_count) * 100
            issues.append({
                'type': 'missing_values',
                'column': column,
                'count': null_count,
                'percentage': percentage,
                'severity': 'warning' if percentage < 5 else 'error'
            })

    # Check data type consistency
    type_issues = self._validate_column_types()
    issues.extend(type_issues)

    return {
        'total_rows': input_count,
        'issues_found': len(issues),
        'issues': issues,
        'data_integrity': 'good' if not issues else 'compromised'
    }
```

### Performance Optimization

#### Batch Operations for Large Datasets

```python
def bulk_data_operations(self, operations: list) -> dict:
    """
    Execute multiple data operations in optimized batches.

    Args:
        operations: List of operation dictionaries

    Returns:
        Results summary for all operations
    """

    results = []

    with transaction.atomic():
        for operation in operations:
            op_type = operation.get('type')

            if op_type == 'insert':
                result = self._bulk_insert(operation['data'])
            elif op_type == 'update':
                result = self._bulk_update(operation['conditions'], operation['updates'])
            elif op_type == 'delete':
                result = self._bulk_delete(operation['conditions'])

            results.append(result)

    return {
        'operations_completed': len(results),
        'results': results,
        'transaction_successful': True
    }
```

#### Query Optimization Techniques

```python
def get_optimized_queryset(self, filters: dict, ordering: list = None) -> QuerySet:
    """
    Create optimized queryset with proper indexing utilization.
    """

    queryset = self.db_data_input_lines.objects.filter(**filters)

    # Use select_related for foreign keys (if any)
    queryset = queryset.select_related()

    # Use only() to limit fields if not all are needed
    if 'fields' in filters:
        queryset = queryset.only(*filters['fields'])

    # Apply ordering for consistent pagination
    if ordering:
        queryset = queryset.order_by(*ordering)

    # Use iterator() for memory-efficient processing of large datasets
    if filters.get('large_dataset', False):
        return queryset.iterator()

    return queryset
```

## Configuration Management

### Machine Level System

**Hierarchical Resource Allocation**:
```python
class MachineLevel:
    def __init__(self, level: int):
        # Determines computational limits and capabilities
        self.level = level

    def feature_engineering_budget(self) -> tuple[int, int]:
        # Returns (min_budget, max_budget) for feature engineering
        return self._level_configs[level]["fe_budget"]

    def nn_shape_count_of_neurons_max(self) -> tuple[int, int]:
        # Returns (min_neurons, max_neurons) limits
        return self._level_configs[level]["max_neurons"]
```

**Level Progression**:
- **Level 1**: Basic capabilities, minimal resources
- **Level 2**: Standard features, moderate resources
- **Level 3**: Advanced features, substantial resources
- **Level 4+**: Enterprise capabilities, maximum resources

### Configuration State Tracking

**Re-run Flags**:
```python
# Flags indicating which configurations need updating
self.db_machine.machine_is_re_run_mdc = True      # Data configuration
self.db_machine.machine_is_re_run_ici = True      # Input importance
self.db_machine.machine_is_re_run_fe = True       # Feature engineering
self.db_machine.machine_is_re_run_enc_dec = True  # Encoding/decoding
self.db_machine.machine_is_re_run_nn_config = True # Neural network
self.db_machine.machine_is_re_run_model = True    # Model retraining
```

## Access Control and Security

### User Authorization

```python
@staticmethod
def is_this_machine_exist_and_authorized(
    machine_identifier_or_name,
    machine_check_access_user_id: int
) -> bool:
    """
    Validates user access to specified machine.
    Checks ownership and team membership permissions.
    """
```

**Authorization Rules**:
1. **Direct Ownership**: User owns the machine
2. **Team Access**: User belongs to team that owns the machine
3. **Public Access**: Special cases for public/shared machines
4. **Super Admin**: Administrative override capabilities

### Data Privacy

**Access Logging**:
```python
# All data access operations are logged
logger.info(f"User {user_id} accessed machine {machine_id}")
```

**Audit Trail**: Complete record of all machine operations and data access

## Error and Warning Management

### Column-Level Error Tracking

```python
def store_error(self, column_name: str, error_message: str):
    """Store column-specific errors for user notification"""
    if column_name not in self.db_machine.machine_columns_errors:
        self.db_machine.machine_columns_errors[column_name] = error_message
    else:
        # Append to existing errors with size limits
        if len(str(self.db_machine.machine_columns_errors)) < 50000:
            self.db_machine.machine_columns_errors[column_name] += "\n\n" + error_message
```

### Warning System

```python
def store_warning(self, column_name: str, warning_message: str):
    """Store non-critical issues for user awareness"""
    # Similar to error storage but with different size limits
    # Warnings don't stop processing but inform users of issues
```

**Error Categories**:
- **Data Quality Issues**: Missing values, outliers, type mismatches
- **Configuration Problems**: Invalid parameters, compatibility issues
- **Performance Warnings**: Training convergence issues, resource constraints
- **System Limitations**: Platform-specific constraints and limitations

## Machine Operations

### Lifecycle Management

#### Copy Operation

```python
def copy(self, new_user: user_model) -> "Machine":
    """Create complete copy of machine for different user"""
    # Clones configuration but resets trained models
    # Requires retraining for the new user
    # Maintains data integrity and access controls
```

**Copy Behavior**:
- **Configuration Preservation**: All settings and parameters copied
- **Data Duplication**: Complete dataset duplication
- **Model Reset**: Trained models not copied (requires retraining)
- **Ownership Transfer**: New user becomes owner

#### Deletion

```python
def delete(self) -> NoReturn:
    """Complete machine removal with cleanup"""
    # Removes main machine record
    # Drops dynamic data tables
    # Cleans up related configurations
    # Logs deletion for audit purposes
```

### State Persistence

```python
def save_machine_to_db(self) -> NoReturn:
    """Persist all machine state to database"""
    # Handles type conversions (np.float32 -> str for Django compatibility)
    # Updates timestamps
    # Ensures data consistency
```

## Integration with AI Pipeline

### NNEngine Integration

```python
# Machine provides foundation for NNEngine operations
nn_engine = NNEngine(machine=self, allow_re_run_configuration=True)

# NNEngine accesses machine data and configurations
training_data = machine.get_random_user_dataframe_for_training_trial(is_for_learning=True)
```

### Feature Engineering Coordination

```python
# Machine coordinates feature engineering processes
from AI.FeatureEngineeringConfiguration import FeatureEngineeringConfiguration

fec = FeatureEngineeringConfiguration(
    machine=self,
    global_dataset_budget=budget,
    nn_engine_for_searching_best_config=nn_engine
)
```

### Encoding/Decoding Management

```python
# Machine manages encoding/decoding configurations
from AI.EncDec import EncDec

enc_dec = EncDec(machine=self, dataframe_pre_encoded=data)
encoded_data = enc_dec.encode_for_ai(data)
```

## Performance and Scaling

### Resource Optimization

**Data Access Patterns**:
- **Batch Processing**: Efficient bulk data operations
- **Memory Management**: Streaming for large datasets
- **Caching Strategy**: Intelligent data caching and prefetching

**Computational Limits**:
- **Training Data Limits**: Configurable maximum training samples
- **Feature Count Limits**: Maximum features per machine level
- **Processing Timeouts**: Prevents runaway computations

### Monitoring and Analytics

```python
def get_machine_overview_information(self, **flags) -> dict:
    """
    Comprehensive machine analytics and metadata.
    Used for optimization decisions and user reporting.
    """
```

**Analytics Categories**:
- **Base Information**: Name, level, column counts, data types
- **Feature Engineering**: FET usage statistics, budget utilization
- **Neural Network**: Architecture details, performance metrics
- **Training History**: Epoch progress, convergence tracking
- **Resource Usage**: Computational costs, time expenditures

## Module Interactions

### With EasyAutoMLDBModels

**Database Abstraction**:
```python
# Access to all Django models through centralized interface
eaml_db = EasyAutoMLDBModels()
machine_model = eaml_db.Machine
user_model = eaml_db.User
```

**Connection Management**:
- **Transaction Support**: Django ORM transaction handling
- **Connection Pooling**: Efficient database connection reuse
- **Error Recovery**: Automatic reconnection on failures

### With MachineDataConfiguration

**Data Analysis Pipeline**:
```python
# MDC analyzes raw data and creates structured configuration
mdc = MachineDataConfiguration(
    machine=self,
    user_dataframe_for_create_cfg=data,
    force_create_with_this_inputs=input_spec,
    force_create_with_this_outputs=output_spec
)
```

**Metadata Management**:
- **Column Type Detection**: Automatic data type classification
- **Statistical Analysis**: Distribution analysis and outlier detection
- **Relationship Discovery**: Feature correlation and dependency analysis

### With Billing System

**Resource Tracking**:
```python
# Integration with billing for resource usage tracking
from eaml_db import Billing

billing_record = Billing.objects.create(
    machine=self.db_machine,
    operation_type="training",
    computational_cost=cost,
    user=self.db_machine.machine_owner_user
)
```

## Future Enhancements

### Potential Improvements

1. **Distributed Processing**: Multi-machine parallel training
2. **Model Versioning**: Complete model lifecycle management
3. **Auto-Scaling**: Dynamic resource allocation based on workload
4. **Advanced Security**: End-to-end encryption and access controls
5. **Real-time Inference**: Streaming prediction capabilities
6. **Model Marketplace**: Share and reuse trained models
7. **Automated Optimization**: Self-tuning based on usage patterns

## Usage Patterns

### Complete Machine Learning Workflow

```python
from ML import Machine, NNEngine
import pandas as pd

# 1. Create machine from dataset
machine = Machine(
    machine_identifier_or_name="predictive_model",
    user_dataset_unformatted=user_data,
    machine_create_user_id=user_id,
    decimal_separator=".",
    date_format="DMY"
)

# 2. Configure and train
nn_engine = NNEngine(machine=machine)
nn_engine.machine_nn_engine_do_full_training_if_not_already_done()

# 3. Deploy for predictions
predictions = nn_engine.do_solving(user_input_data)

# 4. Monitor and maintain
machine.save_machine_to_db()  # Persist any updates
```

### Batch Processing

```python
from ML import EasyAutoMLDBModels, Machine

# Process multiple machines efficiently
eaml_db = EasyAutoMLDBModels()
machines = eaml_db.Machine.objects.filter(machine_owner_user_id=user_id)
for machine_db in machines:
    # Load machine instance
    machine = Machine(machine_identifier_or_name=machine_db.id)
    # Access machine properties and methods
    # machine.save_machine_to_db()
```

### Administrative Operations

```python
from ML import EasyAutoMLDBModels, Machine

# Machine management for administrators
eaml_db = EasyAutoMLDBModels()
all_machines = eaml_db.Machine.objects.all()
for machine_db in all_machines:
    # Load machine instance
    machine = Machine(machine_identifier_or_name=machine_db.id)
    # Access machine properties
    # machine.save_machine_to_db()
```

The Machine class serves as the comprehensive foundation for the EasyAutoML.com system, providing robust machine learning model management with enterprise-grade features for security, scalability, and performance.

## Detailed Function Analysis

### Core Initialization and Machine Creation Functions

#### `__init__(machine_identifier_or_name, user_dataset_unformatted, dfr, machine_level, machine_access_check_with_user_id, machine_create_user_id, machine_create_team_id, force_create_with_this_inputs, force_create_with_this_outputs, force_create_with_this_descriptions, decimal_separator, date_format, **kwargs)`

**Where it's used and why:**
- Called when creating new machines or loading existing ones
- Used throughout the system to instantiate Machine objects for all ML operations
- Critical for establishing the machine learning workflow from data ingestion to model deployment
- Enables both creation and loading workflows depending on provided parameters

**How the function works:**
1. **Parameter Analysis**: Evaluates input parameters to determine operation type (create vs load)
2. **Creation Mode**: When `machine_identifier_or_name` is string and `user_dataset_unformatted` is provided
3. **DFR Mode**: When `machine_identifier_or_name` is string and `dfr` (DataFileReader) is provided
4. **Load by ID Mode**: When `machine_identifier_or_name` is integer for existing machine lookup
5. **Load by Name Mode**: When `machine_identifier_or_name` is string without dataframe for name-based lookup
6. **Delegation**: Routes to appropriate private initialization method based on parameters

**What the function does and its purpose:**
- Serves as the unified entry point for all Machine object creation and loading
- Provides flexible initialization supporting multiple creation patterns
- Ensures proper parameter validation and error handling
- Establishes the foundation for machine lifecycle management

#### `_init_create_new_machine_by_machine_name_and_dataset(machine_name, user_dataset_unformatted, force_create_with_this_inputs, force_create_with_this_outputs, force_create_with_this_descriptions, machine_level, machine_owner_user_id, machine_owner_team_id, decimal_separator, date_format, **kwargs)`

**Where it's used and why:**
- Called internally by `__init__()` when creating machines from raw datasets
- Used when users upload data files and want to create new ML models
- Critical for the machine creation workflow in the user interface
- Enables automated data processing and machine setup from user-provided datasets

**How the function works:**
1. **DataFileReader Creation**: Instantiates DFR to process the raw dataset
2. **Delegation**: Calls `_init_create_new_machine_by_machine_name_and_dfr()` with processed data
3. **Parameter Passing**: Forwards all configuration parameters to the DFR-based creation method

**What the function does and its purpose:**
- Provides high-level machine creation from user datasets
- Handles data preprocessing through DataFileReader integration
- Enables user-friendly machine creation without requiring technical data processing knowledge

#### `_init_create_new_machine_by_machine_name_and_dfr(machine_name, dfr, force_create_with_this_inputs, force_create_with_this_outputs, force_create_with_this_descriptions, machine_level, machine_owner_user_id, machine_owner_team_id, decimal_separator, date_format, **kwargs)`

**Where it's used and why:**
- Called by dataset-based creation and DFR-based creation methods
- Used when creating machines with pre-processed DataFileReader objects
- Critical for the core machine creation workflow in the system
- Enables programmatic machine creation with full configuration control

**How the function works:**
1. **MachineDataConfiguration Creation**: Instantiates MDC to analyze column types and relationships
2. **Validation**: Ensures at least one input and one output column exist
3. **Machine Level Determination**: Sets up resource allocation based on machine level
4. **Database Record Creation**: Creates the main machine record in the database
5. **Table Creation**: Calls `data_lines_create_both_tables()` to create dynamic data tables
6. **Data Loading**: Calls `data_lines_append()` to load initial data with training/evaluation split

**What the function does and its purpose:**
- Implements the complete machine creation workflow
- Coordinates multiple system components (MDC, DFR, database)
- Establishes the machine's data structure and initial state
- Provides the foundation for subsequent ML operations

#### `_init_load_machine_by_id(machine_id, machine_access_check_with_user_id)`

**Where it's used and why:**
- Called by `__init__()` when loading existing machines by ID
- Used throughout the system when specific machines need to be accessed by their unique identifier
- Critical for machine retrieval in API endpoints and background processing
- Enables direct machine access for operations like training, prediction, and management

**How the function works:**
1. **Database Query**: Attempts to load machine record by ID
2. **Error Handling**: Graceful handling of non-existent machines
3. **Dynamic Model Loading**: Retrieves input and output data table models
4. **State Initialization**: Sets up all machine attributes for operation

**What the function does and its purpose:**
- Provides secure, ID-based machine retrieval
- Enables programmatic access to existing machines
- Supports the machine lifecycle management operations

#### `_init_load_machine_by_name(machine_name_to_load, machine_access_user_id)`

**Where it's used and why:**
- Called by `__init__()` when loading machines by name
- Used in user interfaces and API endpoints where users reference machines by name
- Critical for user-friendly machine access patterns
- Supports machine discovery and selection workflows

**How the function works:**
1. **Database Query**: Searches for machine by name and owner
2. **Security Check**: Ensures user has access to the machine
3. **Model Loading**: Retrieves associated data table models
4. **State Setup**: Initializes machine object for operations

**What the function does and its purpose:**
- Enables name-based machine retrieval with security controls
- Supports user-centric machine management workflows
- Provides human-readable machine identification

### Data Management and Table Operations Functions

#### `data_lines_create_both_tables()`

**Where it's used and why:**
- Called during machine creation to set up dynamic data storage
- Used to create machine-specific database tables for input and output data
- Critical for establishing the data persistence layer for each machine
- Enables scalable data storage with machine-specific table structures

**How the function works:**
1. **Validation**: Ensures machine ID exists before table creation
2. **Schema Editor**: Uses Django's schema editor for table creation
3. **Model Instantiation**: Creates input and output data table models
4. **Reference Storage**: Stores model references for subsequent operations

**What the function does and its purpose:**
- Creates the database infrastructure for machine data storage
- Enables dynamic table creation for each machine instance
- Supports scalable data management across multiple machines

#### `data_lines_append(user_dataframe_to_append, split_lines_in_learning_and_evaluation, **kwargs)`

**Where it's used and why:**
- Called when adding new data to existing machines
- Used during machine creation and data updates throughout the ML lifecycle
- Critical for data ingestion and machine learning data management
- Enables continuous data updates and model retraining workflows

**How the function works:**
1. **Data Validation**: Checks for required input and output columns
2. **Type Validation**: Ensures output columns are either all present or all absent
3. **Locking**: Uses `DoMachineLockTables` for thread-safe operations
4. **Index Generation**: Creates consecutive Line_ID indices for new data
5. **Data Distribution**: Splits data between training and evaluation sets when requested

**What the function does and its purpose:**
- Provides thread-safe data ingestion for machine learning datasets
- Supports both training data updates and prediction data handling
- Enables automated data partitioning for model validation

#### `data_lines_read(sort_by, rows_count_limit, **kwargs)`

**Where it's used and why:**
- Called throughout the ML pipeline to retrieve training and evaluation data
- Used by NNEngine for model training and validation
- Critical for providing data to machine learning algorithms
- Enables flexible data access patterns for different ML operations

**How the function works:**
1. **Delegation**: Calls the database model's read method
2. **Parameter Passing**: Forwards sorting, limiting, and filtering parameters
3. **Result Processing**: Returns formatted pandas DataFrame

**What the function does and its purpose:**
- Provides unified interface for data retrieval operations
- Supports various data access patterns for ML workflows
- Enables efficient data loading for training and inference

### Machine State and Configuration Management Functions

#### `save_machine_to_db()`

**Where it's used and why:**
- Called after any machine state changes to persist to database
- Used throughout the system when machine configuration is updated
- Critical for maintaining machine state consistency across sessions
- Enables durable storage of machine learning configurations

**How the function works:**
1. **Type Conversion**: Handles numpy float32 to string conversion for Django compatibility
2. **Database Persistence**: Saves machine record with all current state
3. **Error Handling**: Manages type conversion edge cases

**What the function does and its purpose:**
- Ensures machine state persistence across system restarts
- Maintains configuration consistency for long-term operations
- Provides data durability for machine learning workflows

#### `store_error(column_name, error_message)`

**Where it's used and why:**
- Called when processing errors occur during machine operations
- Used by various components (EncDec, NNEngine, etc.) to record issues
- Critical for error tracking and user notification
- Enables debugging and troubleshooting of machine learning workflows

**How the function works:**
1. **Error Storage**: Adds errors to machine's error tracking dictionary
2. **Size Management**: Implements size limits to prevent memory issues
3. **Database Update**: Persists changes immediately for reliability

**What the function does and its purpose:**
- Provides error tracking for machine operations
- Enables user notification of processing issues
- Supports debugging and system monitoring

#### `store_warning(column_name, warning_message)`

**Where it's used and why:**
- Called when non-critical issues occur during processing
- Used to track warnings that don't stop processing but may affect quality
- Critical for monitoring data quality and system health
- Enables proactive maintenance and optimization

**How the function works:**
1. **Warning Storage**: Adds warnings to machine's warning tracking
2. **Size Management**: Implements limits to prevent excessive storage
3. **Database Persistence**: Saves warnings for user visibility

**What the function does and its purpose:**
- Tracks non-critical issues for user awareness
- Enables monitoring of data quality and processing health
- Supports continuous improvement of ML workflows

### Machine Lifecycle Management Functions

#### `copy(new_user)`

**Where it's used and why:**
- Called when users want to duplicate existing machines
- Used in collaborative workflows and template creation
- Critical for enabling machine reuse and sharing
- Supports machine learning workflow templating

**How the function works:**
1. **Model Duplication**: Creates copy of machine database record
2. **Configuration Reset**: Clears trained models (requires retraining)
3. **Data Duplication**: Copies all training data to new machine
4. **Ownership Transfer**: Assigns new owner to copied machine

**What the function does and its purpose:**
- Enables machine duplication for different users
- Supports collaborative ML development workflows
- Provides template functionality for similar use cases

#### `delete()`

**Where it's used and why:**
- Called when machines need to be completely removed
- Used for cleanup and resource management
- Critical for maintaining system health and resource efficiency
- Enables proper machine lifecycle termination

**How the function works:**
1. **Record Deletion**: Removes main machine database record
2. **Table Cleanup**: Drops associated dynamic data tables
3. **Error Handling**: Graceful handling of deletion failures

**What the function does and its purpose:**
- Provides complete machine removal functionality
- Ensures proper cleanup of all associated resources
- Maintains system health through resource reclamation

### Access Control and Security Functions

#### `is_this_machine_exist_and_authorized(machine_identifier_or_name, machine_check_access_user_id)`

**Where it's used and why:**
- Called before any machine operations to verify access permissions
- Used throughout the API and user interface for security validation
- Critical for maintaining data privacy and access control
- Enables user-specific machine management

**How the function works:**
1. **Machine Lookup**: Searches for machine by ID or name
2. **Ownership Verification**: Confirms user has access to the machine
3. **Result Return**: Boolean indicating authorization status

**What the function does and its purpose:**
- Provides security validation for machine access
- Enables user-specific machine management
- Supports data privacy and access control requirements

### Configuration Readiness Functions

#### `is_nn_solving_ready()`

**Where it's used and why:**
- Called to determine if a machine is ready for prediction operations
- Used by WorkDispatcher to decide if solving work should be assigned
- Critical for ensuring prediction operations only run on properly configured machines
- Enables validation of machine readiness before inference

**How the function works:**
1. **Configuration Checks**: Verifies all required configurations are present
2. **Boolean Logic**: Returns true only if all components are ready
3. **Comprehensive Validation**: Checks MDC, ICI, FE, EncDec, NN config, and model

**What the function does and its purpose:**
- Validates complete machine readiness for prediction
- Prevents prediction on incompletely configured machines
- Ensures reliable inference operations

#### `is_nn_training_pending()`

**Where it's used and why:**
- Called to determine if a machine needs retraining
- Used by WorkDispatcher to prioritize training work
- Critical for identifying machines that require updates
- Enables automated retraining workflows

**How the function works:**
1. **Flag Aggregation**: Checks all re-run flags
2. **OR Logic**: Returns true if any retraining is needed

**What the function does and its purpose:**
- Identifies machines requiring retraining
- Supports automated model maintenance
- Enables continuous model improvement

### Data Access and Utility Functions

#### `get_random_user_dataframe_for_training_trial(is_for_learning, is_for_evaluation, force_rows_count, force_row_count_same_as_for_evaluation, only_column_direction_type)`

**Where it's used and why:**
- Called by NNEngine during training to get data samples
- Used for training trials and validation data preparation
- Critical for providing appropriately sized data samples for ML algorithms
- Enables efficient training with representative data subsets

**How the function works:**
1. **Machine Level Integration**: Uses MachineLevel for size limits
2. **Data Filtering**: Applies appropriate filters for learning vs evaluation
3. **Size Management**: Respects configured limits and debug constraints
4. **Random Sampling**: Provides random data samples when requested

**What the function does and its purpose:**
- Provides optimized data samples for training operations
- Supports various data access patterns for ML workflows
- Enables efficient resource usage during training

#### `get_machine_overview_information(with_base_info, with_fec_encdec_info, with_nn_model_info, with_training_infos, with_training_cycle_result, with_training_eval_result)`

**Where it's used and why:**
- Called by various components to get comprehensive machine information
- Used by MachineEasyAutoML for performance prediction
- Critical for providing context for machine learning optimization decisions
- Enables data-driven decision making across the system

**How the function works:**
1. **Conditional Data Gathering**: Collects information based on requested flags
2. **Multi-Source Integration**: Aggregates data from multiple machine attributes
3. **Structured Output**: Returns comprehensive dictionary with all requested information

**What the function does and its purpose:**
- Provides comprehensive machine metadata for optimization
- Supports performance prediction and resource allocation
- Enables system-wide machine learning optimization

#### `get_list_of_columns_name(column_mode, dataframe_status, dataframe_status)`

**Where it's used and why:**
- Called throughout the system to get column lists for different contexts
- Used by data processing components to understand data structure
- Critical for maintaining consistency between different data representations
- Enables proper column mapping across the ML pipeline

**How the function works:**
1. **Type Validation**: Ensures proper enum types are used
2. **Context-Aware Selection**: Returns appropriate column lists based on parameters
3. **Dynamic Resolution**: Handles different dataframe status types (USER, PRE_ENCODED, ENCODED_FOR_AI)

**What the function does and its purpose:**
- Provides context-aware column information
- Supports different data processing stages
- Enables consistent column handling across components

### Machine Level and Resource Management Functions

#### `scale_loss_to_user_loss(loss_to_scale)`

**Where it's used and why:**
- Called when displaying loss values to users
- Used in user interfaces and reporting systems
- Critical for providing intuitive loss metrics to non-technical users
- Enables consistent loss representation across the platform

**How the function works:**
1. **Scaler Application**: Uses stored loss scaler to normalize values
2. **Range Limiting**: Ensures loss values stay within 0-1 range
3. **Fallback Handling**: Manages cases where scaler is not available

**What the function does and its purpose:**
- Provides user-friendly loss representations
- Enables consistent loss interpretation across different models
- Supports intuitive user experience in ML applications

### Data Line Management Functions

#### `data_input_lines_append(dataframe_to_append, split_lines_in_learning_and_evaluation, already_locked_skip_lock, **kwargs)`

**Where it's used and why:**
- Called by `data_lines_append()` for input data handling
- Used to manage the complex input data insertion workflow
- Critical for maintaining data integrity during concurrent operations
- Enables thread-safe data operations in multi-user environments

**How the function works:**
1. **Data Preparation**: Sets up required flag columns
2. **Training/Evaluation Split**: Randomly assigns data to learning or evaluation sets
3. **Locking Strategy**: Uses database locks for thread safety
4. **Index Management**: Generates proper Line_ID sequences
5. **Operation Logging**: Creates DataLinesOperation records for dispatcher

**What the function does and its purpose:**
- Manages complex input data insertion with proper partitioning
- Ensures thread-safe operations in concurrent environments
- Supports the training/evaluation data split workflow

#### `data_output_lines_append(dataframe, already_locked_skip_lock)`

**Where it's used and why:**
- Called by `data_lines_append()` for output data handling
- Used when inserting prediction targets or output data
- Critical for maintaining input-output data alignment
- Supports both training data (with targets) and prediction data (without targets)

**How the function works:**
1. **Index Validation**: Ensures proper Line_ID indexing
2. **Column Validation**: Verifies output column structure
3. **Locking**: Uses appropriate locking strategy
4. **Data Insertion**: Performs bulk data insertion

**What the function does and its purpose:**
- Handles output data insertion with proper synchronization
- Maintains data integrity between input and output tables
- Supports various data insertion scenarios

### Configuration State Management Functions

#### `clear_config_mdc()`, `clear_config_ici()`, `clear_config_fe()`, `clear_config_enc_dec()`, `clear_config_nn_configuration()`, `clear_config_nn_model()`

**Where it's used and why:**
- Called when configurations need to be reset for retraining
- Used by the system when data changes require complete reconfiguration
- Critical for maintaining consistency when machine state changes
- Enables proper handling of configuration invalidation scenarios

**How the function works:**
1. **Selective Clearing**: Each function clears specific configuration components
2. **Database Updates**: Persists configuration clearing immediately
3. **State Management**: Updates machine re-run flags as appropriate

**What the function does and its purpose:**
- Provides granular configuration reset capabilities
- Supports incremental reconfiguration when needed
- Maintains system consistency during state changes

### Utility and Helper Functions

#### `get_count_of_rows_per_isforflags()`

**Where it's used and why:**
- Called to get statistics about data distribution
- Used for monitoring and reporting data usage patterns
- Critical for understanding machine learning data composition
- Enables data quality assessment and resource planning

**How the function works:**
1. **Query Execution**: Counts rows for each flag combination
2. **Result Aggregation**: Returns comprehensive statistics dictionary

**What the function does and its purpose:**
- Provides data distribution insights
- Supports monitoring and optimization decisions
- Enables understanding of training data composition

### Integration Points and Dependencies

#### With EasyAutoMLDBModels
- **Model Access**: Provides database model references
- **User Management**: Handles user and team relationships
- **Data Operations**: Manages DataLinesOperation records

#### With MachineDataConfiguration (MDC)
- **Column Analysis**: Provides column type and relationship information
- **Data Structure**: Defines input/output column mappings
- **Configuration State**: Tracks MDC readiness and validity

#### With FeatureEngineeringConfiguration (FEC)
- **FET Management**: Coordinates feature engineering transformations
- **Budget Tracking**: Manages feature engineering resource allocation

#### With EncDec
- **Data Transformation**: Manages encoding/decoding configurations
- **Configuration State**: Tracks EncDec readiness and validity

#### With NNEngine
- **Model Training**: Provides data and configuration for training
- **Prediction Support**: Enables inference operations
- **Performance Optimization**: Supplies machine context for optimization

#### With WorkDispatcher/WorkProcessor
- **Work Assignment**: Provides machine context for distributed processing
- **State Management**: Tracks machine readiness for work assignment

This detailed function analysis demonstrates how the Machine class serves as the central orchestrator for the EasyAutoML.com system, managing the complete machine learning lifecycle from data ingestion through model deployment while maintaining data integrity, security, and performance optimization.