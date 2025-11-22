# ML/Experimenter.py - Experimental Evaluation Framework

## Overview

The `Experimenter` module implements a sophisticated experimental framework for evaluating machine learning configurations in the EasyAutoML.com system. It provides abstract base classes and concrete implementations for testing different neural network architectures, feature engineering techniques, and optimization parameters through systematic experimentation.

**Location**: `ML/Experimenter.py`

## Description

### What

The Experimenter module evaluates machine learning configurations through controlled experiments, measuring the performance impact of different neural network architectures and feature engineering selections. It provides a standardized framework for testing parameter combinations and returning quantitative performance metrics.

### How

It executes trial training with candidate configurations through the _do_single() method, capturing loss, accuracy, and resource utilization metrics. The system supports batch evaluation of multiple configurations and implements robust error handling with graceful degradation for failed experiments.

### Where

Used by SolutionFinder during optimization to evaluate candidate configurations, and by FeatureEngineeringConfiguration to test different FET combinations. Integrates with NNEngine for trial training infrastructure.

### When

Called during neural network architecture search and feature engineering optimization phases to quantitatively assess configuration quality.

## Core Architecture

### Abstract Base Class: Experimenter

```python
from abc import ABC, abstractmethod
from typing import Optional, Union, List, Dict, Any
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Experimenter(ABC):
    """
    Abstract base class for all experimental evaluators in the EasyAutoML system.

    Provides standardized interface for running experiments with consistent
    input/output handling, error management, and performance monitoring.

    Key Features:
    - Unified experimental interface across different optimization domains
    - Robust error handling with graceful degradation
    - Performance monitoring and resource tracking
    - Configurable experimental parameters and constraints
    """

    def __init__(self, experiment_name: str = "base_experiment"):
        """
        Initialize experimenter with basic configuration.

        Args:
            experiment_name: Identifier for this experimental setup
        """
        self.experiment_name = experiment_name
        self.experiment_history = []
        self.performance_metrics = {}

    @abstractmethod
    def _do_single(self, configuration: Union[pd.Series, Dict]) -> Optional[pd.Series]:
        """
        Execute single experimental trial with given configuration.

        Args:
            configuration: Experimental parameters to evaluate

        Returns:
            Performance metrics as pandas Series, or None if failed
        """
        pass

    def do(self, user_inputs: Union[pd.DataFrame, Dict, List]) -> Optional[pd.DataFrame]:
        """
        Main experimental interface supporting batch and single evaluations.

        Args:
            user_inputs: Experimental configurations to evaluate
                      - pd.DataFrame: Batch evaluation (each row is a config)
                      - Dict: Single configuration evaluation
                      - List: Multiple single configurations

        Returns:
            DataFrame with results for each configuration, or None if all failed
        """
        try:
            if isinstance(user_inputs, pd.DataFrame):
                return self._do_batch(user_inputs)
            elif isinstance(user_inputs, dict):
                return self._do_single_config(user_inputs)
            elif isinstance(user_inputs, list):
                return self._do_multiple_configs(user_inputs)
            else:
                raise ValueError(f"Unsupported input type: {type(user_inputs)}")

        except Exception as e:
            logger.error(f"Experiment failed: {str(e)}")
            return None

    def _do_batch(self, configurations_df: pd.DataFrame) -> pd.DataFrame:
        """Execute batch evaluation of multiple configurations."""
        results = []

        for idx, config_row in configurations_df.iterrows():
            try:
                result = self._do_single(config_row)
                if result is not None:
                    results.append(result)
                else:
                    # Create failure result
                    failure_result = pd.Series({
                        'experiment_status': 'failed',
                        'error_message': 'Experiment returned None'
                    })
                    results.append(failure_result)

            except Exception as e:
                logger.warning(f"Configuration {idx} failed: {str(e)}")
                failure_result = pd.Series({
                    'experiment_status': 'error',
                    'error_message': str(e)
                })
                results.append(failure_result)

        if not results:
            return None

        # Compile results into DataFrame
        results_df = pd.DataFrame(results)

        # Add metadata
        results_df['experiment_timestamp'] = pd.Timestamp.now()
        results_df['experiment_name'] = self.experiment_name

        # Store in history
        self.experiment_history.append({
            'timestamp': pd.Timestamp.now(),
            'configurations_tested': len(configurations_df),
            'successful_experiments': len(results_df[results_df['experiment_status'] != 'failed']),
            'results': results_df
        })

        return results_df

    def _do_single_config(self, config_dict: Dict) -> pd.DataFrame:
        """Execute single configuration evaluation."""
        config_series = pd.Series(config_dict)
        result = self._do_single(config_series)

        if result is None:
            return pd.DataFrame([{
                'experiment_status': 'failed',
                'error_message': 'Single experiment returned None'
            }])

        return pd.DataFrame([result])

    def _do_multiple_configs(self, config_list: List) -> pd.DataFrame:
        """Execute multiple single configurations."""
        all_results = []

        for config in config_list:
            result_df = self._do_single_config(config)
            all_results.append(result_df)

        if not all_results:
            return None

        return pd.concat(all_results, ignore_index=True)

    def get_experiment_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about experimental runs."""
        if not self.experiment_history:
            return {'total_experiments': 0}

        total_experiments = sum(run['configurations_tested'] for run in self.experiment_history)
        successful_experiments = sum(run['successful_experiments'] for run in self.experiment_history)

        return {
            'total_experiments': total_experiments,
            'successful_experiments': successful_experiments,
            'success_rate': successful_experiments / total_experiments if total_experiments > 0 else 0,
            'total_runs': len(self.experiment_history),
            'average_experiments_per_run': total_experiments / len(self.experiment_history),
            'experiment_name': self.experiment_name
        }

    def clear_history(self):
        """Clear experimental history for fresh start."""
        self.experiment_history = []
        self.performance_metrics = {}
```

## ExperimenterNNConfiguration - Neural Network Architecture Evaluation

### Class Overview

The `ExperimenterNNConfiguration` class evaluates neural network architectures by performing controlled training trials. It systematically tests different network topologies, optimizers, learning rates, and hyperparameters to identify optimal configurations for specific datasets and machine learning tasks.

**Key Capabilities**:
- Automated neural architecture search within computational budgets
- Multi-objective optimization balancing accuracy and computational cost
- Robust error handling for failed training experiments
- Standardized performance metrics for comparative analysis
- Integration with machine level resource constraints

### Architecture Search Space

The experimenter explores multiple neural network dimensions:

#### Network Topology Parameters
```python
# Layer configurations
layer_options = {
    'dense_layers': [1, 2, 3, 4, 5],  # Number of hidden layers
    'neurons_per_layer': [16, 32, 64, 128, 256, 512],  # Neurons per layer
    'layer_connectivity': ['dense', 'residual', 'highway'],  # Connection patterns
}

# Activation functions
activation_options = ['relu', 'elu', 'selu', 'tanh', 'sigmoid', 'swish']
```

#### Optimization Parameters
```python
# Optimizers and learning strategies
optimizer_options = {
    'adam': {'beta_1': [0.9, 0.95], 'beta_2': [0.999, 0.9999]},
    'rmsprop': {'rho': [0.9, 0.95], 'momentum': [0.0, 0.9]},
    'sgd': {'momentum': [0.0, 0.9], 'nesterov': [True, False]}
}

# Learning rate schedules
lr_schedule_options = ['constant', 'exponential_decay', 'cosine_annealing', 'step_decay']
```

#### Regularization Techniques
```python
regularization_options = {
    'dropout': [0.0, 0.1, 0.2, 0.3, 0.5],
    'l1_lambda': [0.0, 1e-5, 1e-4, 1e-3],
    'l2_lambda': [0.0, 1e-5, 1e-4, 1e-3],
    'batch_norm': [True, False]
}
```

### Key Functions

#### `__init__(self, nnengine_trial: NNEngine, experiment_name: str = "nn_config_experiment")`

**Comprehensive Initialization Workflow**:

```python
def __init__(self, nnengine_trial: NNEngine, experiment_name: str = "nn_config_experiment"):
    """
    Initialize NN configuration experimenter with full evaluation pipeline.

    Args:
        nnengine_trial: NNEngine instance for training trials
        experiment_name: Identifier for this experimental session
    """

    super().__init__(experiment_name)

    # Core dependencies
    self.nnengine = nnengine_trial
    self.machine = nnengine_trial._machine

    # Resource constraints from machine level
    self.max_neurons = self.machine.db_machine.machine_level * 100  # Level-based scaling
    self.max_layers = min(5, self.machine.db_machine.machine_level)

    # Data preparation and caching
    self._prepare_evaluation_datasets()

    # Performance baseline establishment
    self.baseline_loss = self._establish_performance_baseline()

    # Cost calculation parameters
    self.neuron_cost_weight = 0.6
    self.layer_cost_weight = 0.4

    logger.info(f"NN Configuration Experimenter initialized for machine level {self.machine.db_machine.machine_level}")
```

**Data Preparation Strategy**:
```python
def _prepare_evaluation_datasets(self):
    """Prepare and cache datasets for consistent experimental evaluation."""

    # Get representative training sample
    training_df = self.machine.get_random_user_dataframe_for_training_trial(
        is_for_learning=True,
        force_rows_count=min(1000, self.machine.db_machine.machine_level * 100)
    )

    # Get validation sample
    validation_df = self.machine.get_random_user_dataframe_for_training_trial(
        is_for_evaluation=True,
        force_rows_count=min(500, self.machine.db_machine.machine_level * 50)
    )

    # Pre-encode data for consistent evaluation
    self.training_encoded = self.nnengine._enc_dec.encode_for_ai(
        self.nnengine._mdc.dataframe_pre_encode(training_df)
    )

    self.validation_encoded = self.nnengine._enc_dec.encode_for_ai(
        self.nnengine._mdc.dataframe_pre_encode(validation_df)
    )

    # Cache for performance
    self._data_cache = {
        'training': self.training_encoded,
        'validation': self.validation_encoded,
        'prepared_at': pd.Timestamp.now()
    }
```

#### `_do_single(self, nn_config_params: pd.Series) -> Optional[pd.Series]`

**Experimental Execution Flow with Comprehensive Monitoring**:

```python
def _do_single(self, nn_config_params: pd.Series) -> Optional[pd.Series]:
    """
    Execute single neural network configuration experiment.

    Args:
        nn_config_params: Neural network configuration parameters

    Returns:
        Performance metrics or None if experiment failed
    """

    experiment_start = time.time()

    try:
        # Step 1: Parse and validate configuration parameters
        nn_config = self._parse_nn_configuration(nn_config_params)

        # Step 2: Cost evaluation (early exit for over-budget configs)
        cost_metrics = self._calculate_configuration_cost(nn_config)
        if cost_metrics['total_cost_percent'] > 100:
            return pd.Series({
                'experiment_status': 'over_budget',
                'Result_cost_neurons_percent_budget': cost_metrics['neuron_cost_percent'],
                'Result_cost_layers_percent_budget': cost_metrics['layer_cost_percent'],
                'Result_loss_scaled': float('inf'),
                'Result_epoch_done_percent': 0.0
            })

        # Step 3: Build neural network model
        model = self._build_nn_model_from_config(nn_config)

        # Step 4: Execute training trial
        training_result = self._execute_training_trial(model, nn_config)

        # Step 5: Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(
            training_result, cost_metrics
        )

        # Step 6: Package results
        result = pd.Series({
            'experiment_status': 'success',
            'Result_loss_scaled': performance_metrics['scaled_loss'],
            'Result_epoch_done_percent': performance_metrics['epoch_completion'],
            'Result_cost_neurons_percent_budget': cost_metrics['neuron_cost_percent'],
            'Result_cost_layers_percent_budget': cost_metrics['layer_cost_percent'],
            'training_duration_seconds': time.time() - experiment_start,
            'final_loss': training_result.get('final_loss', float('inf')),
            'final_accuracy': training_result.get('final_accuracy', 0.0),
            'model_parameters': sum([layer.count_params() for layer in model.layers])
        })

        return result

    except Exception as e:
        logger.warning(f"NN configuration experiment failed: {str(e)}")

        # Return failure result with available information
        return pd.Series({
            'experiment_status': 'error',
            'error_message': str(e),
            'Result_loss_scaled': float('inf'),
            'Result_epoch_done_percent': 0.0,
            'Result_cost_neurons_percent_budget': cost_metrics.get('neuron_cost_percent', 100),
            'Result_cost_layers_percent_budget': cost_metrics.get('layer_cost_percent', 100)
        })
```

### Cost Calculation and Budget Management

#### Configuration Cost Analysis
```python
def _calculate_configuration_cost(self, nn_config: Dict) -> Dict[str, float]:
    """
    Calculate computational cost of neural network configuration.

    Returns cost as percentage of machine level budget.
    """

    # Neuron cost calculation
    total_neurons = sum(layer['neurons'] for layer in nn_config['layers'])
    neuron_cost_percent = (total_neurons / self.max_neurons) * 100

    # Layer cost calculation (with complexity weighting)
    num_layers = len(nn_config['layers'])
    layer_complexity = sum(
        layer.get('complexity_weight', 1.0) for layer in nn_config['layers']
    )
    layer_cost_percent = (layer_complexity / self.max_layers) * 100

    # Total cost with weighted combination
    total_cost_percent = (
        self.neuron_cost_weight * neuron_cost_percent +
        self.layer_cost_weight * layer_cost_percent
    )

    return {
        'neuron_cost_percent': neuron_cost_percent,
        'layer_cost_percent': layer_cost_percent,
        'total_cost_percent': total_cost_percent,
        'budget_exceeded': total_cost_percent > 100
    }
```

### Output Metrics and Performance Evaluation

#### Comprehensive Performance Metrics
- **`Result_loss_scaled`**: Loss normalized by baseline performance (lower is better)
- **`Result_epoch_done_percent`**: Training completion percentage (100% = full training)
- **`Result_cost_neurons_percent_budget`**: Neuron utilization as % of budget
- **`Result_cost_layers_percent_budget`**: Layer complexity as % of budget
- **`training_duration_seconds`**: Actual training time for performance monitoring
- **`model_parameters`**: Total trainable parameters for complexity assessment
- **`final_loss`** & **`final_accuracy`**: Raw performance metrics

#### Performance Scaling Strategy
```python
def _calculate_performance_metrics(self, training_result: Dict, cost_metrics: Dict) -> Dict[str, float]:
    """Calculate scaled performance metrics for optimization."""

    raw_loss = training_result.get('final_loss', float('inf'))
    epoch_completion = training_result.get('epoch_completion_percent', 0.0)

    # Scale loss relative to baseline (handle division by zero)
    if self.baseline_loss > 0:
        scaled_loss = raw_loss / self.baseline_loss
    else:
        scaled_loss = raw_loss  # Fallback for zero baseline

    # Apply cost penalty for over-budget configurations
    cost_penalty = max(0, cost_metrics['total_cost_percent'] - 100) / 100
    scaled_loss *= (1 + cost_penalty * 0.5)  # 50% penalty per 100% budget excess

    # Normalize epoch completion to 0-1 range
    epoch_completion = min(epoch_completion / 100.0, 1.0)

    return {
        'scaled_loss': scaled_loss,
        'epoch_completion': epoch_completion,
        'cost_penalty_applied': cost_penalty > 0
    }
```

### Advanced Features

#### Configuration Sampling Strategies
```python
def generate_nn_configuration_space(self, num_samples: int = 100) -> pd.DataFrame:
    """
    Generate diverse neural network configurations for experimentation.

    Uses intelligent sampling to explore promising regions of configuration space.
    """

    configurations = []

    for _ in range(num_samples):
        config = {
            'num_layers': np.random.choice([1, 2, 3, 4]),
            'neurons_per_layer': np.random.choice([32, 64, 128, 256]),
            'activation': np.random.choice(['relu', 'elu', 'selu']),
            'optimizer': np.random.choice(['adam', 'rmsprop']),
            'learning_rate': 10 ** np.random.uniform(-4, -2),  # 1e-4 to 1e-2
            'dropout_rate': np.random.uniform(0.0, 0.5),
            'batch_norm': np.random.choice([True, False])
        }

        # Calculate estimated cost
        config['estimated_cost'] = self._estimate_configuration_cost(config)

        configurations.append(config)

    return pd.DataFrame(configurations)
```

#### Intelligent Early Stopping
```python
def _configure_adaptive_training(self, nn_config: Dict) -> Dict:
    """
    Configure adaptive training parameters based on configuration complexity.
    """

    complexity_score = self._calculate_configuration_complexity(nn_config)

    # Adjust training parameters based on complexity
    if complexity_score > 0.8:  # High complexity
        training_params = {
            'max_epochs': 50,
            'early_stopping_patience': 10,
            'batch_size': 16,
            'validation_freq': 5
        }
    elif complexity_score > 0.5:  # Medium complexity
        training_params = {
            'max_epochs': 100,
            'early_stopping_patience': 15,
            'batch_size': 32,
            'validation_freq': 3
        }
    else:  # Low complexity
        training_params = {
            'max_epochs': 200,
            'early_stopping_patience': 20,
            'batch_size': 64,
            'validation_freq': 2
        }

    return training_params
```

## ExperimenterColumnFETSelector - Feature Engineering Evaluation

### Class Overview

Evaluates Feature Engineering Template (FET) selections for individual columns by measuring their impact on neural network performance through controlled experiments.

### Key Functions

#### `__init__(self, nn_engine_to_use_in_trial: NNEngine, column_datas_infos: Column_datas_infos, fec_budget_max: int)`

**Initialization Strategy**:
1. **Dataset Caching Setup**: Pre-encodes datasets for consistent experimental conditions
2. **Minimum Configuration Baseline**: Establishes performance baseline with minimum feature engineering
3. **Loss Scaler Initialization**: Computes scaling factor for relative performance measurement

#### `_do_single(self, df_user_data_input: pd.Series) -> Optional[pd.Series]`

**Feature Engineering Evaluation Process**:
1. **FET Configuration Extraction**: Parses experimental parameters to identify FET selections
2. **Dynamic Configuration Application**: Creates and applies new feature engineering configuration
3. **Configuration Synchronization**: Updates all dependent machine configurations
4. **Performance Evaluation**: Executes trial training with new FET setup
5. **Cost-Benefit Analysis**: Calculates feature engineering cost relative to budget

## Module Interactions

### With SolutionFinder

**Integration Pattern**: SolutionFinder uses Experimenter instances for configuration evaluation, calling `experimenter.do()` with candidate configurations and using returned metrics for optimization decisions.

### With NNEngine

**Training Infrastructure**: Provides trial training capabilities, data pipelines, and performance monitoring. Experimenter leverages NNEngine's infrastructure for consistent and efficient experimental execution.

### With FeatureEngineeringTemplate

**Dynamic FET Management**: Dynamically loads and tests different FETs, evaluates their individual and combined impact on model performance.

### With MachineDataConfiguration

**Data Consistency**: Ensures consistent data preprocessing across experiments, utilizes column metadata and data distribution statistics to guide experimental design.

## Performance Optimization

### Caching Mechanisms
- **Dataset Caching**: Avoids repeated preprocessing by caching encoded datasets
- **Configuration Reuse**: Reuses baseline configurations for comparative analysis

### Computational Efficiency
- **Batch Processing**: Handles multiple configurations efficiently
- **Resource Management**: Optimizes memory usage and CPU utilization

## Error Handling

### Experimental Failure Management
- **Graceful Degradation**: Returns None for failed experiments
- **Result Validation**: Checks for valid training outcomes
- **Configuration Recovery**: Handles invalid configurations with fallbacks

## Usage Patterns

### Neural Network Architecture Search

#### Basic NN Configuration Experimentation
```python
from ML import Machine, NNEngine, ExperimenterNNConfiguration
import pandas as pd

# 1. Setup machine and NN engine
machine = Machine(machine_identifier_or_name="nn_optimization_test")
nn_engine = NNEngine(machine)

# 2. Create NN configuration experimenter
nn_experimenter = ExperimenterNNConfiguration(
    nnengine_trial=nn_engine,
    experiment_name="architecture_search_v1"
)

# 3. Generate configuration space to explore
config_space = nn_experimenter.generate_nn_configuration_space(num_samples=50)
print(f"Generated {len(config_space)} neural network configurations")

# 4. Execute batch experimentation
results = nn_experimenter.do(config_space)
print(f"Completed {len(results)} experiments")

# 5. Analyze results
successful_experiments = results[results['experiment_status'] == 'success']
if not successful_experiments.empty:
    best_config = successful_experiments.loc[successful_experiments['Result_loss_scaled'].idxmin()]
    print(f"Best configuration found:")
    print(f"  - Scaled Loss: {best_config['Result_loss_scaled']:.4f}")
    print(f"  - Training Time: {best_config['training_duration_seconds']:.1f}s")
    print(f"  - Model Parameters: {best_config['model_parameters']}")

# 6. Get experiment statistics
stats = nn_experimenter.get_experiment_statistics()
print(f"Success Rate: {stats['success_rate']:.1%}")
```

#### Advanced Architecture Optimization with Custom Constraints
```python
from ML import Machine, NNEngine, ExperimenterNNConfiguration
import pandas as pd

# Custom configuration generator with domain constraints
def generate_domain_specific_configs(num_samples=25):
    """Generate NN configs suitable for time series forecasting"""

    configs = []
    for _ in range(num_samples):
        # Time series specific architecture choices
        config = {
            'num_layers': np.random.choice([2, 3, 4]),  # Deeper networks for temporal patterns
            'neurons_per_layer': np.random.choice([64, 128, 256]),  # Larger capacity
            'activation': 'relu',  # Standard for regression tasks
            'optimizer': 'adam',   # Good default for most tasks
            'learning_rate': 10 ** np.random.uniform(-3, -2),  # 1e-3 to 1e-2
            'dropout_rate': np.random.uniform(0.1, 0.3),  # Moderate regularization
            'batch_norm': np.random.choice([True, False]),
            'recurrent_layers': np.random.choice([0, 1]),  # Optional LSTM/GRU layers
        }

        # Add time series specific features
        if config['recurrent_layers'] > 0:
            config['recurrent_type'] = np.random.choice(['LSTM', 'GRU'])
            config['recurrent_units'] = np.random.choice([32, 64, 128])

        configs.append(config)

    return pd.DataFrame(configs)

# Execute domain-specific optimization
machine = Machine(machine_identifier_or_name="time_series_model")
nn_engine = NNEngine(machine)

experimenter = ExperimenterNNConfiguration(nn_engine, "time_series_optimization")
custom_configs = generate_domain_specific_configs(30)

results = experimenter.do(custom_configs)

# Find best time series configuration
best_result = results[results['experiment_status'] == 'success'].nsmallest(1, 'Result_loss_scaled').iloc[0]
print(f"Optimal time series configuration found with loss: {best_result['Result_loss_scaled']:.4f}")
```

### Feature Engineering Optimization

#### Basic FET Selection Experimentation
```python
from ML import Machine, NNEngine, ExperimenterColumnFETSelector
from SharedConstants import DatasetColumnDataType

# 1. Setup machine and identify target column
machine = Machine(machine_identifier_or_name="fet_optimization_test")
nn_engine = NNEngine(machine)

# 2. Get column information for experimentation
column_name = "customer_age"  # Target column for FET optimization
column_info = machine._fe.get_all_column_datas_infos(column_name)

# 3. Create FET experimenter with budget constraints
budget_max = machine.db_machine.machine_level * 10  # Level-based budget
fet_experimenter = ExperimenterColumnFETSelector(
    nn_engine_to_use_in_trial=nn_engine,
    column_datas_infos=column_info,
    fec_budget_max=budget_max,
    experiment_name="age_feature_optimization"
)

# 4. Generate FET configuration space
fet_configs = fet_experimenter.generate_fet_configuration_space(num_samples=20)
print(f"Testing {len(fet_configs)} FET combinations for column '{column_name}'")

# 5. Execute experimentation
results = fet_experimenter.do(fet_configs)

# 6. Analyze FET performance
if not results.empty:
    successful_tests = results[results['experiment_status'] == 'success']
    best_fet_config = successful_tests.nsmallest(1, 'Result_loss_scaled').iloc[0]

    print(f"Best FET configuration:")
    print(f"  - Loss Improvement: {best_fet_config['Result_loss_scaled']:.4f}")
    print(f"  - Cost: {best_fet_config['Result_cost_percent_budget']:.1f}% of budget")
    print(f"  - Selected FETs: {best_fet_config.get('selected_fets', 'N/A')}")
```

#### Multi-Column FET Optimization
```python
from ML import Machine, NNEngine, ExperimenterColumnFETSelector
import pandas as pd

def optimize_all_columns_fets(machine, nn_engine, optimization_budget=50):
    """
    Perform FET optimization across all input columns.
    """

    results_summary = []

    # Get all input columns
    input_columns = machine.get_list_of_columns_name(
        column_mode=DatasetColumnDataType.FLOAT,  # Can be adapted for other types
        dataframe_status="USER"
    )

    for column_name in input_columns:
        print(f"Optimizing FETs for column: {column_name}")

        # Get column metadata
        column_info = machine._fe.get_all_column_datas_infos(column_name)

        # Calculate column-specific budget
        column_importance = machine._ici._column_importance_evaluation.get(column_name, 0.5)
        column_budget = int(optimization_budget * column_importance)

        # Create and run experimenter
        experimenter = ExperimenterColumnFETSelector(
            nn_engine, column_info, column_budget,
            experiment_name=f"fet_opt_{column_name}"
        )

        # Generate and test configurations
        configs = experimenter.generate_fet_configuration_space(num_samples=15)
        results = experimenter.do(configs)

        # Extract best result
        if not results.empty:
            best_result = results[results['experiment_status'] == 'success'].nsmallest(1, 'Result_loss_scaled')
            if not best_result.empty:
                best_config = best_result.iloc[0]
                results_summary.append({
                    'column': column_name,
                    'best_loss': best_config['Result_loss_scaled'],
                    'cost_percent': best_config['Result_cost_percent_budget'],
                    'selected_fets': best_config.get('selected_fets', [])
                })

    return pd.DataFrame(results_summary)

# Execute multi-column optimization
machine = Machine(machine_identifier_or_name="multi_column_fet_opt")
nn_engine = NNEngine(machine)

optimization_results = optimize_all_columns_fets(machine, nn_engine)
print("Multi-column FET optimization completed:")
print(optimization_results)
```

### Advanced Experimental Workflows

#### Bayesian Optimization Integration
```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.optimize import minimize
from scipy.stats import norm

class BayesianNNOptimizer:
    """
    Advanced NN optimization using Bayesian optimization.
    """

    def __init__(self, experimenter):
        self.experimenter = experimenter
        self.gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True
        )
        self.X_observed = []
        self.y_observed = []

    def optimize(self, num_iterations=20):
        """Execute Bayesian optimization of NN configurations."""

        bounds = {
            'learning_rate': (-4, -1),    # 1e-4 to 1e-1 (log scale)
            'num_layers': (1, 5),         # 1 to 5 layers
            'neurons_per_layer': (32, 512), # 32 to 512 neurons
            'dropout_rate': (0.0, 0.5)    # 0% to 50% dropout
        }

        for iteration in range(num_iterations):
            # Generate next configuration using acquisition function
            next_config = self._acquisition_function(bounds)

            # Evaluate configuration
            result = self.experimenter.do(next_config)

            if result is not None and not result.empty:
                loss = result.iloc[0]['Result_loss_scaled']
                if not np.isinf(loss):
                    # Update GP model
                    self.X_observed.append([
                        next_config['learning_rate'],
                        next_config['num_layers'],
                        next_config['neurons_per_layer'],
                        next_config['dropout_rate']
                    ])
                    self.y_observed.append(loss)

                    if len(self.X_observed) > 2:
                        self.gp.fit(self.X_observed, self.y_observed)

        # Return best configuration found
        best_idx = np.argmin(self.y_observed)
        return self.X_observed[best_idx], self.y_observed[best_idx]

    def _acquisition_function(self, bounds):
        """Expected Improvement acquisition function."""

        def ei(x):
            if len(self.X_observed) < 2:
                # Random sampling for initial points
                return np.random.random()

            x = x.reshape(1, -1)
            mu, sigma = self.gp.predict(x, return_std=True)

            # Expected improvement calculation
            if sigma == 0:
                return 0

            current_best = min(self.y_observed)
            z = (current_best - mu) / sigma
            ei_value = (current_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

            return -ei_value  # Negative for minimization

        # Optimize acquisition function
        x0 = np.array([
            np.random.uniform(bounds['learning_rate'][0], bounds['learning_rate'][1]),
            np.random.uniform(bounds['num_layers'][0], bounds['num_layers'][1]),
            np.random.uniform(bounds['neurons_per_layer'][0], bounds['neurons_per_layer'][1]),
            np.random.uniform(bounds['dropout_rate'][0], bounds['dropout_rate'][1])
        ])

        result = minimize(ei, x0, bounds=[
            bounds['learning_rate'], bounds['num_layers'],
            bounds['neurons_per_layer'], bounds['dropout_rate']
        ])

        return {
            'learning_rate': 10 ** result.x[0],  # Convert from log scale
            'num_layers': int(round(result.x[1])),
            'neurons_per_layer': int(round(result.x[2])),
            'dropout_rate': result.x[3]
        }

# Usage example
machine = Machine(machine_identifier_or_name="bayesian_opt_test")
nn_engine = NNEngine(machine)
experimenter = ExperimenterNNConfiguration(nn_engine)

optimizer = BayesianNNOptimizer(experimenter)
best_config, best_loss = optimizer.optimize(num_iterations=30)

print(f"Bayesian optimization found configuration with loss: {best_loss}")
```

#### Experimental Result Analysis and Visualization
```python
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_experiment_results(experimenter, results_df):
    """
    Comprehensive analysis of experimental results.
    """

    # Filter successful experiments
    successful = results_df[results_df['experiment_status'] == 'success']

    if successful.empty:
        print("No successful experiments to analyze")
        return

    # Performance distribution
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Loss distribution
    axes[0,0].hist(successful['Result_loss_scaled'], bins=20, alpha=0.7)
    axes[0,0].set_title('Loss Distribution')
    axes[0,0].set_xlabel('Scaled Loss')
    axes[0,0].set_ylabel('Frequency')

    # Cost vs Performance scatter
    axes[0,1].scatter(successful['Result_cost_neurons_percent_budget'],
                      successful['Result_loss_scaled'], alpha=0.6)
    axes[0,1].set_title('Cost vs Performance')
    axes[0,1].set_xlabel('Neuron Cost (% of budget)')
    axes[0,1].set_ylabel('Scaled Loss')

    # Training time analysis
    if 'training_duration_seconds' in successful.columns:
        axes[1,0].scatter(successful['training_duration_seconds'],
                          successful['Result_loss_scaled'], alpha=0.6)
        axes[1,0].set_title('Training Time vs Performance')
        axes[1,0].set_xlabel('Training Duration (seconds)')
        axes[1,0].set_ylabel('Scaled Loss')

    # Success rate over time
    experiment_stats = experimenter.get_experiment_statistics()
    axes[1,1].bar(['Successful', 'Failed'],
                  [experiment_stats['successful_experiments'],
                   experiment_stats['total_experiments'] - experiment_stats['successful_experiments']])
    axes[1,1].set_title('Experiment Success Rate')
    axes[1,1].set_ylabel('Number of Experiments')

    plt.tight_layout()
    plt.show()

    # Statistical summary
    print("
=== Experiment Analysis Summary ===")
    print(f"Total Experiments: {experiment_stats['total_experiments']}")
    print(f"Success Rate: {experiment_stats['success_rate']:.1%}")
    print(".4f")
    print(".1f")
    print(f"Best Configuration Cost: {successful['Result_cost_neurons_percent_budget'].min():.1f}%")

# Usage
machine = Machine(machine_identifier_or_name="analysis_test")
nn_engine = NNEngine(machine)
experimenter = ExperimenterNNConfiguration(nn_engine)

# Assuming results_df contains experimental results
analyze_experiment_results(experimenter, results_df)
```

## Key Features

- **Standardized Interface**: Consistent API across different experimental types
- **Performance Metrics**: Comprehensive evaluation metrics including accuracy, loss, and computational cost
- **Error Resilience**: Robust error handling and recovery mechanisms
- **Scalable Design**: Adapts to different computational resources and experimental scales
- **Modular Architecture**: Easily extensible for new experimental methodologies

The Experimenter framework provides a robust foundation for systematic evaluation and optimization of machine learning configurations in the EasyAutoML.com system.

## Detailed Function Analysis

### Core Abstract Base Class Functions

#### `do(user_inputs)`

**Where it's used and why:**
- Called by SolutionFinder and other optimization components to evaluate configurations
- Used as the primary interface for experimental evaluation throughout the system
- Critical for maintaining consistent experimental evaluation across different optimization algorithms
- Enables batch processing of multiple configuration candidates

**How the function works:**
1. **Input Type Detection**: Determines if input is single dict or multiple DataFrame
2. **Delegation**: Routes to appropriate processing method based on input type
3. **Batch Processing**: For DataFrames, applies `_do_single` to each row
4. **Result Aggregation**: Collects results from individual experiments

**What the function does and its purpose:**
- Provides unified interface for experimental evaluation
- Handles both single and batch configuration testing
- Enables scalable experimental workflows
- Maintains consistent input/output contract across all experimenter types

#### `_do_single(df_user_data_input)` (Abstract Method)

**Where it's used and why:**
- Implemented by concrete experimenter classes to define specific evaluation logic
- Used by the `do()` method to process individual configuration candidates
- Critical for defining the actual experimental methodology for each use case
- Enables polymorphic experimental evaluation

**How the function works:**
1. **Abstract Implementation**: Must be overridden by concrete classes
2. **Single Row Processing**: Always processes exactly one configuration
3. **Result Standardization**: Returns consistent output format across implementations
4. **Error Handling**: Provides graceful failure handling with None returns

**What the function does and its purpose:**
- Defines the core experimental logic for configuration evaluation
- Enables extensible experimental methodologies
- Maintains consistent experimental interface
- Supports error resilience through standardized failure handling

### ExperimenterNNConfiguration Functions

#### `__init__(nnengine_trial)`

**Where it's used and why:**
- Called when creating neural network configuration experiments
- Used by SolutionFinder during NN architecture optimization
- Critical for setting up efficient NN configuration evaluation
- Enables systematic testing of different neural network architectures

**How the function works:**
1. **Data Preparation**: Pre-encodes training and validation datasets for consistent conditions
2. **Resource Budget Setup**: Calculates maximum neuron and layer limits for cost evaluation
3. **Validation Checks**: Ensures loss scaler parameter is available
4. **Caching Strategy**: Prepares encoded data to avoid repeated preprocessing

**What the function does and its purpose:**
- Initializes the NN configuration experimental environment
- Prepares optimized data pipelines for efficient experimentation
- Establishes evaluation criteria and resource constraints
- Enables rapid iteration over NN architecture candidates

#### `_do_single(nnshape_type_machine_to_evaluate)`

**Where it's used and why:**
- Called by the `do()` method for each NN configuration to evaluate
- Used during SolutionFinder's NN architecture search process
- Critical for evaluating the performance of different neural network designs
- Enables data-driven selection of optimal NN configurations

**How the function works:**
1. **Configuration Creation**: Builds NNConfiguration from experimental parameters
2. **Trial Training**: Executes fast training with specified NN architecture
3. **Performance Metrics**: Captures loss, accuracy, and training progress
4. **Cost Calculation**: Computes resource utilization costs
5. **Result Scaling**: Normalizes metrics for comparative analysis

**What the function does and its purpose:**
- Evaluates NN configuration performance through controlled experimentation
- Provides quantitative metrics for architecture comparison
- Enables cost-benefit analysis of different NN designs
- Supports automated NN architecture optimization

### ExperimenterColumnFETSelector Functions

#### `__init__(nn_engine_to_use_in_trial, column_datas_infos, fec_budget_max)`

**Where it's used and why:**
- Called when setting up feature engineering experiments for specific columns
- Used by SolutionFinder during feature engineering optimization
- Critical for evaluating the impact of different FET combinations on model performance
- Enables systematic feature engineering optimization

**How the function works:**
1. **Dataset Preparation**: Caches pre-encoded training and validation data
2. **Baseline Configuration**: Creates minimum FEC configuration for comparison
3. **EncDec Setup**: Initializes encoding/decoding configuration
4. **Loss Scaler Calculation**: Computes scaling factor for relative performance measurement
5. **Resource Constraints**: Sets up feature engineering budget limits

**What the function does and its purpose:**
- Initializes the feature engineering experimental environment
- Establishes baseline performance metrics for comparison
- Prepares efficient evaluation pipeline for FET combinations
- Enables quantitative evaluation of feature engineering strategies

#### `_do_single(df_user_data_input)`

**Where it's used and why:**
- Called for each FET combination to evaluate during feature engineering optimization
- Used by SolutionFinder to test different feature engineering approaches
- Critical for determining optimal FET selections for individual columns
- Enables data-driven feature engineering decisions

**How the function works:**
1. **FET Configuration Extraction**: Parses experimental parameters to identify FET selections
2. **Dynamic Configuration**: Creates new FEC configuration with selected FETs
3. **Configuration Application**: Updates machine with new feature engineering setup
4. **EncDec Recreation**: Regenerates encoding configuration for new FETs
5. **Trial Training**: Executes training with new feature engineering configuration
6. **Performance Evaluation**: Measures impact on model performance
7. **Cost Analysis**: Calculates feature engineering resource utilization

**What the function does and its purpose:**
- Evaluates the effectiveness of different FET combinations
- Provides quantitative metrics for feature engineering optimization
- Enables cost-benefit analysis of feature engineering strategies
- Supports automated feature selection and engineering

### Integration Points and Dependencies

#### With SolutionFinder
- **Configuration Evaluation**: SolutionFinder uses Experimenter instances to evaluate candidate configurations
- **Optimization Loop**: Provides feedback loop for iterative optimization algorithms
- **Performance Metrics**: Supplies standardized metrics for optimization decisions

#### With NNEngine
- **Trial Training Infrastructure**: Leverages NNEngine's training capabilities for experimental evaluation
- **Data Pipeline**: Uses NNEngine's data preprocessing and encoding pipelines
- **Performance Monitoring**: Utilizes NNEngine's performance tracking mechanisms

#### With FeatureEngineeringConfiguration (FEC)
- **Dynamic Configuration**: Creates and applies different FEC configurations during experiments
- **Budget Management**: Enforces feature engineering resource constraints
- **Configuration Persistence**: Manages FEC configuration updates in machine

#### With EncDec
- **Data Transformation**: Manages encoding/decoding configurations for experimental setups
- **Configuration Synchronization**: Ensures EncDec configuration matches current FEC setup
- **Performance Consistency**: Maintains consistent data transformation across experiments

#### With MachineDataConfiguration (MDC)
- **Data Preparation**: Provides consistent data preprocessing for experimental conditions
- **Column Metadata**: Supplies column information for feature engineering decisions
- **Data Integrity**: Ensures consistent data representation across experiments

#### With FeatureEngineeringTemplate (FET)
- **Dynamic Loading**: Loads and tests different FET implementations during experiments
- **Performance Evaluation**: Measures individual and combined FET effectiveness
- **Configuration Management**: Handles FET parameter configuration and validation

### Performance Optimization Strategies

#### Caching Mechanisms
- **Dataset Caching**: Pre-encoded datasets avoid repeated preprocessing overhead
- **Configuration Reuse**: Baseline configurations reused for comparative analysis
- **Result Caching**: Avoids redundant evaluations of identical configurations

#### Computational Efficiency
- **Batch Processing**: Handles multiple configurations efficiently
- **Resource Management**: Optimizes memory usage and computational resources
- **Parallel Evaluation**: Supports concurrent experimental evaluation

#### Memory Optimization
- **Dataset Size Management**: Limits dataset sizes for experimental efficiency
- **Configuration Cloning**: Uses shallow copying where appropriate
- **Garbage Collection**: Explicit cleanup of temporary configurations

### Error Handling and Recovery

#### Experimental Failure Management
- **Graceful Degradation**: Returns None for failed experiments rather than crashing
- **Result Validation**: Comprehensive validation of training outcomes
- **Configuration Recovery**: Handles invalid configurations with fallback mechanisms
- **Logging Integration**: Detailed error logging for debugging and monitoring

#### Robustness Features
- **Exception Handling**: Comprehensive try-catch blocks around experimental operations
- **Parameter Validation**: Validates input parameters and configuration validity
- **Resource Limits**: Prevents runaway resource consumption during experiments

### Usage Patterns and Examples

#### Neural Network Architecture Optimization
```python
from ML import NNEngine, ExperimenterNNConfiguration, Machine
import pandas as pd

# Initialize NN configuration experimenter
machine = Machine(machine_identifier_or_name="my_machine")
nnengine = NNEngine(machine)
experimenter = ExperimenterNNConfiguration(nnengine)

# Evaluate multiple NN configurations
configurations = pd.DataFrame([...])  # NN parameter combinations
results = experimenter.do(configurations)

# Analyze results for optimal configuration
optimal_config = results.loc[results['Result_loss_scaled'].idxmin()]
```

#### Feature Engineering Optimization
```python
from ML import ExperimenterColumnFETSelector, NNEngine, Machine
import pandas as pd

# Initialize FET selection experimenter
machine = Machine(machine_identifier_or_name="my_machine")
nnengine = NNEngine(machine)
experimenter = ExperimenterColumnFETSelector(
    nnengine, 
    column_info, 
    fec_budget_max
)

# Evaluate different FET combinations
fet_configurations = pd.DataFrame([...])  # FET parameter combinations
results = experimenter.do(fet_configurations)

# Select optimal FET combination
optimal_fet = results.loc[results['Result_loss_scaled'].idxmin()]
```

#### Integration with SolutionFinder
```python
from ML import SolutionFinder, Machine

# SolutionFinder uses Experimenter for optimization
machine = Machine(machine_identifier_or_name="my_machine")
solution_finder = SolutionFinder(
    machine=machine,
    experimenter=experimenter,
    optimization_algorithm=algorithm
)

# Run optimization process
optimal_solution = solution_finder.find_optimal_configuration()
```

This detailed analysis demonstrates how the Experimenter framework serves as the experimental evaluation engine in the EasyAutoML.com system, providing systematic, quantitative evaluation of machine learning configurations while maintaining performance, reliability, and extensibility across different experimental methodologies.