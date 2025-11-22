# ML/NNConfiguration.py - Neural Network Architecture Configuration

## Overview

The `NNConfiguration` module manages neural network architecture specifications and hyperparameters for the EasyAutoML.com system. It provides a structured approach to defining, validating, and adapting neural network configurations across different problem types and computational constraints.

**Location**: `ML/NNConfiguration.py`

## Description

### What

The NNConfiguration module manages neural network architecture specifications and hyperparameters, defining layer structures, neuron counts, optimizers, and training parameters. It provides structured configuration management with validation, serialization, and dynamic adaptation to changing input/output dimensions.

### How

It uses NNShape class to represent network topology with percentage-based neuron allocation, validates configurations against resource constraints, and builds Keras-compatible model specifications. The system supports both manual configuration and automated discovery through SolutionFinder integration.

### Where

Used by NNEngine to build and configure neural network models, and by ExperimenterNNConfiguration during architecture optimization. Integrates with Machine for configuration persistence.

### When

Created during model initialization and updated during architecture optimization phases before neural network training begins.

## Core Functionality

### Primary Responsibilities

- **Architecture Definition**: Specify neural network layer structures and neuron counts
- **Hyperparameter Management**: Configure optimizers, learning rates, and training parameters
- **Dynamic Adaptation**: Adjust configurations based on input/output dimensions
- **Validation**: Ensure configuration compatibility and resource requirements
- **Serialization**: Save and restore network configurations

### Key Components

```python
class NNConfiguration:
    """
    Comprehensive neural network configuration management.
    Handles architecture specification, hyperparameter tuning, and validation.
    """
```

## Configuration Structure

### Core Parameters

```python
# Neural network architecture
nn_optimizer: str              # Optimization algorithm ("adam", "sgd", "rmsprop")
nn_shape: NNShape             # Layer structure and neuron counts
num_of_input_neurons: int     # Input layer size
num_of_output_neurons: int    # Output layer size

# Training parameters
nn_epochs: int                # Maximum training epochs
nn_batch_size: int           # Batch size for training
nn_learning_rate: float      # Learning rate
nn_loss: str                 # Loss function

# Regularization
nn_dropout_rate: float       # Dropout probability
nn_l2_regularization: float  # L2 regularization factor
```

### NNShape Class

**Layer Architecture Specification**:
```python
class NNShape:
    """
    Defines neural network topology and layer configuration.
    """
    def __init__(self, shape_data):
        # Parse layer specifications
        self._hidden_layers = shape_data.get("parameter_nn_shape_layers", [])
        self._neurons_per_layer = shape_data.get("parameter_nn_shape_neurons", [])

    def get_hidden_layers_count(self) -> int:
        """Return number of hidden layers"""
        return len(self._hidden_layers)

    def neurons_total_count(self, input_count: int, output_count: int) -> int:
        """Calculate total neuron count across all layers"""
        total = input_count + output_count
        for neurons in self._neurons_per_layer:
            total += neurons
        return total
```

## Configuration Creation

### Standard Initialization

```python
def __init__(self, configuration_dict: dict):
    """
    Create NNConfiguration from parameter dictionary.

    :param configuration_dict: Dictionary containing NN parameters
    """
    # Extract core parameters
    self.nn_optimizer = configuration_dict.get("nn_optimizer", "adam")
    self.nn_shape = NNShape(configuration_dict)
    self.num_of_input_neurons = configuration_dict.get("num_of_input_neurons")
    self.num_of_output_neurons = configuration_dict.get("num_of_output_neurons")

    # Set default training parameters
    self._set_default_training_parameters()
```

### Dynamic Adaptation

```python
def adapt_config_to_new_enc_dec(self, enc_dec):
    """
    Adjust configuration when encoding/decoding changes.

    :param enc_dec: EncDec object with updated dimensions
    """
    # Update input/output neuron counts
    self.num_of_input_neurons = enc_dec._columns_input_encoded_count
    self.num_of_output_neurons = enc_dec._columns_output_encoded_count

    # Validate configuration compatibility
    self._validate_configuration()
```

## Architecture Optimization

### Resource-Aware Configuration

```python
def _validate_configuration(self):
    """
    Ensure configuration meets resource and architectural constraints.
    """
    # Check neuron count limits
    total_neurons = self.nn_shape.neurons_total_count(
        self.num_of_input_neurons,
        self.num_of_output_neurons
    )

    # Validate against machine level limits
    max_neurons = self._machine_level_constraints["max_neurons"]
    if total_neurons > max_neurons:
        raise ValueError(f"Configuration exceeds neuron limit: {total_neurons} > {max_neurons}")
```

### Layer Architecture Management

```python
def get_keras_layers_configuration(self):
    """
    Generate Keras-compatible layer specifications.

    :return: List of layer configurations for model building
    """
    layers_config = []

    # Input layer
    layers_config.append({
        "type": "input",
        "neurons": self.num_of_input_neurons,
        "activation": "relu"
    })

    # Hidden layers
    for i, neurons in enumerate(self.nn_shape._neurons_per_layer):
        layers_config.append({
            "type": "dense",
            "neurons": neurons,
            "activation": "relu",
            "dropout": self.nn_dropout_rate if i > 0 else 0
        })

    # Output layer
    layers_config.append({
        "type": "dense",
        "neurons": self.num_of_output_neurons,
        "activation": self._get_output_activation()
    })

    return layers_config
```

## Hyperparameter Management

### Optimizer Configuration

```python
def get_optimizer_configuration(self):
    """
    Generate optimizer configuration for Keras.

    :return: Configured Keras optimizer
    """
    optimizer_config = {
        "adam": lambda: keras.optimizers.Adam(learning_rate=self.nn_learning_rate),
        "sgd": lambda: keras.optimizers.SGD(learning_rate=self.nn_learning_rate),
        "rmsprop": lambda: keras.optimizers.RMSprop(learning_rate=self.nn_learning_rate)
    }

    if self.nn_optimizer not in optimizer_config:
        raise ValueError(f"Unsupported optimizer: {self.nn_optimizer}")

    return optimizer_config[self.nn_optimizer]()
```

### Loss Function Selection

```python
def _get_loss_function(self):
    """
    Determine appropriate loss function based on problem type.

    :return: Keras loss function
    """
    # Regression problems
    if self._is_regression_problem():
        return "mse"  # Mean Squared Error

    # Binary classification
    elif self.num_of_output_neurons == 1:
        return "binary_crossentropy"

    # Multi-class classification
    else:
        return "categorical_crossentropy"
```

## Configuration Persistence

### Serialization

```python
def serialize_configuration(self) -> dict:
    """
    Export configuration for storage.

    :return: Dictionary containing all configuration parameters
    """
    return {
        "nn_optimizer": self.nn_optimizer,
        "nn_shape_layers": self.nn_shape._hidden_layers,
        "nn_shape_neurons": self.nn_shape._neurons_per_layer,
        "num_of_input_neurons": self.num_of_input_neurons,
        "num_of_output_neurons": self.num_of_output_neurons,
        "nn_epochs": self.nn_epochs,
        "nn_batch_size": self.nn_batch_size,
        "nn_learning_rate": self.nn_learning_rate,
        "nn_loss": self.nn_loss,
        "nn_dropout_rate": self.nn_dropout_rate,
        "nn_l2_regularization": self.nn_l2_regularization
    }
```

### Deserialization

```python
@classmethod
def load_from_serialized(cls, serialized_config: dict):
    """
    Reconstruct configuration from stored data.

    :param serialized_config: Previously serialized configuration
    :return: NNConfiguration instance
    """
    return cls(serialized_config)
```

## Integration with Core Systems

### NNEngine Integration

```python
# NNEngine uses NNConfiguration for model building
nn_config = NNConfiguration({
    "nn_optimizer": "adam",
    "nn_shape": NNShape({"parameter_nn_shape_layers": [64, 32]}),
    "num_of_input_neurons": 100,
    "num_of_output_neurons": 10
})

# NNEngine builds model using configuration
model = nn_engine._build_keras_model(nn_config)
```

### Experimenter Integration

```python
# Experimenter evaluates different configurations
experimenter = ExperimenterNNConfiguration(nn_engine)

# Test configuration performance
result = experimenter._do_single(nn_config_parameters)
```

## Performance Optimization

### Resource Management

```python
def calculate_resource_requirements(self) -> dict:
    """
    Estimate computational resources needed for training.

    :return: Resource requirements dictionary
    """
    total_parameters = self._calculate_parameter_count()
    memory_requirement = self._estimate_memory_usage()
    training_time_estimate = self._estimate_training_time()

    return {
        "parameters": total_parameters,
        "memory_gb": memory_requirement,
        "estimated_training_hours": training_time_estimate
    }
```

### Architecture Validation

```python
def validate_architecture(self) -> bool:
    """
    Comprehensive architecture validation.

    :return: True if architecture is valid
    """
    checks = [
        self._validate_layer_sizes(),
        self._validate_activation_functions(),
        self._validate_gradient_flow(),
        self._validate_resource_limits()
    ]

    return all(checks)
```

## Usage Patterns

### Basic Configuration

```python
# Simple neural network configuration
config = NNConfiguration({
    "nn_optimizer": "adam",
    "nn_shape": NNShape({
        "parameter_nn_shape_layers": [128, 64],
        "parameter_nn_shape_neurons": [128, 64]
    }),
    "num_of_input_neurons": 784,  # MNIST input
    "num_of_output_neurons": 10   # MNIST classes
})
```

### Advanced Configuration

```python
# Complex configuration with custom parameters
advanced_config = NNConfiguration({
    "nn_optimizer": "adam",
    "nn_shape": NNShape({
        "parameter_nn_shape_layers": [512, 256, 128, 64],
        "parameter_nn_shape_neurons": [512, 256, 128, 64]
    }),
    "num_of_input_neurons": 1000,
    "num_of_output_neurons": 50,
    "nn_epochs": 200,
    "nn_batch_size": 32,
    "nn_learning_rate": 0.001,
    "nn_dropout_rate": 0.3,
    "nn_l2_regularization": 0.0001
})
```

### Configuration Adaptation

```python
# Dynamic adaptation to new data dimensions
original_config = NNConfiguration(base_params)
original_config.adapt_config_to_new_enc_dec(updated_enc_dec)

# Configuration automatically adjusts to new input/output sizes
print(f"Adapted input neurons: {original_config.num_of_input_neurons}")
print(f"Adapted output neurons: {original_config.num_of_output_neurons}")
```

## Error Handling

### Configuration Validation

```python
def _validate_configuration(self):
    """Comprehensive configuration validation"""
    errors = []

    if self.num_of_input_neurons <= 0:
        errors.append("Input neuron count must be positive")

    if self.num_of_output_neurons <= 0:
        errors.append("Output neuron count must be positive")

    if self.nn_learning_rate <= 0 or self.nn_learning_rate > 1:
        errors.append("Learning rate must be between 0 and 1")

    if errors:
        raise ValueError(f"Configuration validation failed: {', '.join(errors)}")
```

### Resource Limit Enforcement

```python
def _check_resource_limits(self):
    """Ensure configuration stays within resource bounds"""
    if self._calculate_parameter_count() > MAX_PARAMETERS:
        raise ResourceError("Configuration exceeds parameter limit")

    if self._estimate_memory_usage() > MAX_MEMORY_GB:
        raise ResourceError("Configuration exceeds memory limit")
```

The NNConfiguration module provides a robust, flexible framework for managing neural network architectures, ensuring optimal performance while maintaining resource efficiency and architectural integrity.

## Detailed Function Analysis

### Core Initialization and Configuration Functions

#### `__init__(machine_or_nnconfiguration, machine_nnengine_for_searching_best_nnconfig, force_find_new_best_configuration)`

**Where it's used and why:**
- Called when creating a new NNConfiguration instance for neural network architecture management
- Used throughout the AutoML pipeline when neural network training is required
- Critical for establishing the neural network architecture and hyperparameters
- Enables both automatic configuration discovery and manual configuration specification

**How the function works:**
1. **Parameter Validation**: Determines whether input is a Machine object or configuration dictionary
2. **Configuration Strategy**: Decides between loading existing configuration or finding new optimal configuration
3. **Machine Integration**: Establishes connection with Machine object for data access and constraints
4. **Auto-Discovery Logic**: Triggers automatic configuration optimization when no valid configuration exists

**Configuration Loading Strategy:**
```python
if isinstance(machine_or_nnconfiguration, dict):
    # Load from configuration dictionary
    self._init_set_configuration_from_dict(machine_or_nnconfiguration)

elif not isinstance(machine_or_nnconfiguration, Machine):
    logger.error("machine_or_nnconfiguration must be dict or Machine!")

else:
    # Machine-based initialization
    self.machine = machine_or_nnconfiguration
    if force_find_new_best_configuration or not self.machine.is_config_ready_nn_configuration():
        # Find optimal configuration
        self._init_find_configuration_best(self.machine, machine_nnengine_for_searching_best_nnconfig)
    else:
        # Load existing configuration
        self._init_load_configuration_from_machine(self.machine)
```

**What the function does and its purpose:**
- Provides unified interface for neural network configuration management
- Supports both automatic optimization and manual configuration
- Ensures proper integration with Machine lifecycle and data pipeline
- Maintains configuration consistency across the AutoML system

#### `_init_set_configuration_from_dict(dict_NNConfiguration)`

**Where it's used and why:**
- Called internally when loading configuration from a dictionary representation
- Used when deserializing configurations from storage or when configurations are passed programmatically
- Critical for configuration persistence and transfer between system components
- Enables configuration sharing and backup/restore functionality

**How the function works:**
1. **Dictionary Parsing**: Extracts neural network parameters from structured dictionary
2. **Validation**: Ensures all required configuration parameters are present
3. **State Initialization**: Sets up internal NNConfiguration state from dictionary values

**Dictionary Structure Mapping:**
```python
# Required dictionary structure
dict_NNConfiguration = {
    "nn_optimizer": "adam",           # Optimizer type
    "nn_shape": NNShape_object,       # Neural network architecture
    "num_of_input_neurons": 100,      # Input dimension
    "num_of_output_neurons": 10       # Output dimension
}

# Internal state mapping
self.nn_shape_instance = dict_NNConfiguration["nn_shape"]
self.nn_optimizer = dict_NNConfiguration["nn_optimizer"]
self.num_of_input_neurons = dict_NNConfiguration["num_of_input_neurons"]
self.num_of_output_neurons = dict_NNConfiguration["num_of_output_neurons"]
```

**What the function does and its purpose:**
- Enables configuration serialization and deserialization
- Supports configuration exchange between system components
- Maintains configuration integrity during storage and retrieval

#### `_init_load_configuration_from_machine(machine)`

**Where it's used and why:**
- Called internally when loading existing neural network configuration from a Machine object
- Used when Machine already has a trained neural network configuration stored
- Critical for resuming work with previously configured and trained models
- Enables efficient reuse of validated configurations

**How the function works:**
1. **Configuration Validation**: Checks that Machine has valid neural network configuration
2. **Parameter Extraction**: Retrieves stored configuration parameters from Machine database
3. **State Reconstruction**: Rebuilds NNConfiguration state from stored values

**Machine Integration Process:**
```python
# Validate configuration exists
if not machine.is_config_ready_nn_configuration():
    logger.error("Unable to load NNConfiguration because there is no configuration saved in machine")

# Extract stored configuration
self.num_of_input_neurons = machine.db_machine.enc_dec_columns_info_input_encode_count
self.num_of_output_neurons = machine.db_machine.enc_dec_columns_info_output_encode_count
self.nn_optimizer = machine.db_machine.parameter_nn_optimizer

# Reconstruct neural network shape
self.nn_shape_instance = NNShape(nnshape_type_machine=machine.db_machine.parameter_nn_shape)
```

**What the function does and its purpose:**
- Enables seamless resumption of work with existing neural network configurations
- Maintains consistency between Machine state and NNConfiguration
- Supports efficient loading of previously validated architectures

#### `_init_find_configuration_best(machine, machine_nnengine)`

**Where it's used and why:**
- Called internally when no valid neural network configuration exists and optimization is needed
- Used during initial Machine setup and when forcing configuration rediscovery
- Critical for automated neural network architecture optimization
- Enables data-driven configuration discovery through systematic evaluation

**How the function works:**
1. **Prerequisites Check**: Ensures EncDec configuration is complete before NN optimization
2. **Parameter Extraction**: Gets input/output dimensions from Machine configuration
3. **Configuration Discovery**: Uses SolutionFinder to explore optimal architectures
4. **Performance Tracking**: Records timing and optimization results

**Optimization Workflow:**
```python
# Validate prerequisites
if not machine.is_config_ready_enc_dec():
    logger.error("You need to run EncDec before NNConfiguration")

# Extract dimensions
self.num_of_input_neurons = machine.db_machine.enc_dec_columns_info_input_encode_count
self.num_of_output_neurons = machine.db_machine.enc_dec_columns_info_output_encode_count

# Find optimal configuration using SolutionFinder
best_nn_configuration_found = self._find_machine_best_nn_configuration(machine, machine_nnengine)

if not best_nn_configuration_found:
    logger.error("Configuration optimization failed to find suitable solution")

# Set optimal configuration
self.nn_shape_instance = NNShape(nnshape_type_machine=best_nn_configuration_found)
self.nn_optimizer = best_nn_configuration_found["parameter_nn_optimizer"]

# Record optimization duration
machine.db_machine.parameter_nn_find_delay_sec = default_timer() - delay_total_started_at
```

**What the function does and its purpose:**
- Provides automated neural network architecture optimization
- Uses systematic evaluation to find optimal configurations
- Integrates with SolutionFinder for comprehensive parameter exploration
- Ensures optimal starting configurations for neural network training

### Configuration Management Functions

#### `save_configuration_in_machine(save_config_in_machine)`

**Where it's used and why:**
- Called when persisting neural network configuration to Machine database
- Used after configuration optimization or manual configuration updates
- Critical for maintaining configuration persistence across system restarts
- Enables configuration sharing and backup across different system instances

**How the function works:**
1. **Target Validation**: Determines which Machine object to save configuration to
2. **Parameter Serialization**: Converts configuration to database-compatible format
3. **Database Update**: Stores configuration parameters in Machine database
4. **Logging**: Records successful configuration persistence

**Configuration Persistence Process:**
```python
# Determine target Machine
if save_config_in_machine:
    db_model_machine = save_config_in_machine.db_machine
elif self.machine:
    db_model_machine = self.machine.db_machine
else:
    logger.error("No target Machine available for configuration saving")

# Serialize and save configuration
db_model_machine.parameter_nn_shape = self.nn_shape_instance.get_machine_nn_shape().to_dict()
db_model_machine.parameter_nn_optimizer = self.nn_optimizer
db_model_machine.parameter_nn_find_delay_sec = self.nn_find_delay_sec

logger.debug("NNConfiguration attributes saved in database")
```

**What the function does and its purpose:**
- Ensures configuration persistence for long-term availability
- Maintains configuration integrity across system lifecycle
- Enables configuration recovery and system resilience

#### `adapt_config_to_new_enc_dec(new_enc_dec)`

**Where it's used and why:**
- Called when data encoding/decoding configuration changes (e.g., after feature engineering updates)
- Used to maintain configuration compatibility when input/output dimensions change
- Critical for ensuring neural network architecture remains valid after data pipeline modifications
- Enables dynamic adaptation to evolving data processing requirements

**How the function works:**
1. **Machine Validation**: Ensures adaptation is happening within valid Machine context
2. **Dimension Update**: Adjusts input/output neuron counts to match new EncDec configuration
3. **Architecture Preservation**: Maintains neural network shape while updating dimensions

**Adaptation Process:**
```python
# Validate context
if not self.machine:
    logger.error("Configuration adaptation requires valid Machine context")

if not self.nn_shape_instance:
    logger.error("Configuration adaptation requires existing neural network shape")

# Update dimensions (neuron percentages remain unchanged)
self.num_of_input_neurons = new_enc_dec._columns_input_encoded_count
self.num_of_output_neurons = new_enc_dec._columns_output_encoded_count

# Neural network shape automatically adapts through percentage-based calculations
```

**What the function does and its purpose:**
- Maintains configuration validity during data pipeline evolution
- Enables seamless adaptation to changing data processing requirements
- Preserves architectural decisions while accommodating dimension changes

### Configuration Discovery Functions

#### `_evaluate_all_initial_shapes(machine, nnengine)`

**Where it's used and why:**
- Called during configuration optimization to establish baseline performance
- Used as initial evaluation step in the SolutionFinder optimization process
- Critical for providing reference points and early termination criteria
- Enables efficient exploration by pre-evaluating common architectures

**How the function works:**
1. **Sample Data Preparation**: Creates training/evaluation datasets for performance assessment
2. **Configuration Iteration**: Evaluates each predefined initial configuration
3. **Model Training**: Performs rapid training trials for each configuration
4. **Performance Recording**: Captures loss and training progress metrics

**Evaluation Process:**
```python
# Prepare evaluation datasets
random_user_dataframe_for_training_trial_evaluation = machine.get_random_user_dataframe_for_training_trial(is_for_evaluation=True)
random_user_dataframe_for_training_trial_learning = machine.get_random_user_dataframe_for_training_trial(is_for_learning=True)

# Evaluate each initial configuration
for configuration_number, configuration in enumerate(NNCONFIGURATION_FINDER_INITIAL_EVALUATIONS_SHAPE):
    # Convert configuration to DataFrame format
    pandas_shape = pd.DataFrame([list(val.values()) for val in configuration.values()],
                               columns=[column_name for column_name in configuration["output"].keys()],
                               index=list(configuration.keys()))

    # Create NNConfiguration for evaluation
    nnengine._nn_configuration = NNConfiguration({
        "nn_optimizer": NNCONFIGURATION_FINDER_INITIAL_EVALUATIONS_OPTIMIZER[configuration_number],
        "nn_shape": NNShape(nnshape_type_user=pandas_shape),
        "num_of_input_neurons": self.machine.db_machine.enc_dec_columns_info_input_encode_count,
        "num_of_output_neurons": self.machine.db_machine.enc_dec_columns_info_output_encode_count,
    })

    # Prepare encoded data
    encoded_for_ai_dataframe = nnengine._enc_dec.encode_for_ai(
        nnengine._mdc.dataframe_pre_encode(random_user_dataframe_for_training_trial_learning))
    encoded_for_ai_validation_dataframe = nnengine._enc_dec.encode_for_ai(
        nnengine._mdc.dataframe_pre_encode(random_user_dataframe_for_training_trial_evaluation))

    # Perform rapid training trial
    result_test_loss, result_test_accuracy, epoch_done_percent = nnengine.do_training_trial(
        encoded_for_ai_dataframe=encoded_for_ai_dataframe,
        encoded_for_ai_validation_dataframe=encoded_for_ai_validation_dataframe)

    # Record results for SolutionFinder
    result_loss_scaled_dict.update({
        f"NNCONFIGURATION_FINDER_INITIAL_EVALUATIONS_LOSS_{configuration_number}":
        machine.scale_loss_to_user_loss(result_test_loss) if result_test_loss else None
    })
    result_epoch_done_percent_dict.update({
        f"NNCONFIGURATION_FINDER_INITIAL_EVALUATIONS_EPOCH_{configuration_number}": epoch_done_percent
    })
```

**What the function does and its purpose:**
- Establishes performance baselines for common neural network architectures
- Provides data for informed configuration optimization decisions
- Enables early pruning of suboptimal configuration spaces
- Supports efficient exploration through pre-computed reference points

#### `_find_machine_best_nn_configuration(machine, nnengine)`

**Where it's used and why:**
- Called as the core optimization function for discovering optimal neural network configurations
- Used when comprehensive architecture search is required
- Critical for automated neural network design and hyperparameter optimization
- Enables systematic exploration of configuration space for optimal performance

**How the function works:**
1. **Baseline Evaluation**: Runs initial shape evaluations to establish reference performance
2. **Machine Context Gathering**: Collects relevant machine and data characteristics
3. **SolutionFinder Integration**: Sets up comprehensive parameter space exploration
4. **Optimization Execution**: Performs systematic configuration evaluation
5. **Result Recording**: Stores optimization results for future reference

**Comprehensive Optimization Process:**
```python
# Establish performance baselines
initial_shapes_result_loss_scaled_dict, initial_shapes_result_epoch_done_percent_dict = self._evaluate_all_initial_shapes(machine, nnengine)

# Gather machine context information
dict_possible_values_constant_machine_overview = machine.get_machine_overview_information(
    with_base_info=True,
    with_fec_encdec_info=True)

# Include baseline results in optimization context
dict_possible_values_constant_machine_overview.update(initial_shapes_result_loss_scaled_dict)
dict_possible_values_constant_machine_overview.update(initial_shapes_result_epoch_done_percent_dict)

# Define parameter search space
dict_possible_values_varying_nn_config = {
    "parameter_nn_optimizer": NNCONFIGURATION_ALL_POSSIBLE_OPTIMIZER_FUNCTION,
    "LayerTypeActivationOutput": NNCONFIGURATION_ALL_POSSIBLE_LAYER_TYPE_ACTIVATION,
    "HiddenLayerCount": [i for i in range(1, MachineLevel(machine).nn_shape_count_of_layer_max()[1] + 1)],
}

# Add neuron and regularization parameters for each layer
for i in range(1, NNCONFIGURATION_MAX_POSSIBLE_LAYER_COUNT):
    dict_possible_values_varying_nn_config.update({
        f"NeuronPercentage{i}": NNCONFIGURATION_FINDER_POSSIBLE_NEURON_PERCENTAGE,
        f"LayerTypeActivation{i}": NNCONFIGURATION_ALL_POSSIBLE_LAYER_TYPE_ACTIVATION,
        f"DropOut{i}": NNCONFIGURATION_FINDER_POSSIBLE_DROPOUT_VALUE,
        f"BatchNormalization{i}": NNCONFIGURATION_FINDER_POSSIBLE_BATCHNORMALIZATION_VALUE
    })

# Define optimization objectives
dict_solution_score_score_evaluation = {
    "Result_loss_scaled": "---(70%)",      # Minimize scaled loss (70% weight)
    "Result_epoch_done_percent": "+++(20%)", # Maximize training completion (20% weight)
    "Result_cost_neurons_percent_budget": "---(5%)",    # Minimize neuron cost (5% weight)
    "Result_cost_layers_percent_budget": "---(5%)",     # Minimize layer cost (5% weight)
}

# Create machine-specific SolutionFinder name
if machine.db_machine.enc_dec_columns_info_input_encode_count > 1000:
    approximation_input_size = ">1000"
elif machine.db_machine.enc_dec_columns_info_input_encode_count > 100:
    approximation_input_size = "100-1000"
else:
    approximation_input_size = "<100"

if machine.db_machine.enc_dec_columns_info_output_encode_count > 100:
    approximation_output_size = ">100"
else:
    approximation_output_size = "<100"

solution_finder_name = f"NNConfiguration--Level={machine.db_machine.machine_level}--Inputs={approximation_input_size}--Outputs={approximation_output_size}"

# Execute optimization
solution_finder = SolutionFinder(solution_finder_name)
solution_found = solution_finder.find_solution(
    dict_possible_values_constant_machine_overview,
    dict_possible_values_varying_nn_config,
    SolutionScore(dict_solution_score_score_evaluation),
    ExperimenterNNConfiguration(nnengine)
)

# Record optimization results
MachineEasyAutoML_NNConfig = MachineEasyAutoML("__Results_Find_Best_NN_Configuration__")
MachineEasyAutoML_NNConfig.learn_this_inputs_outputs(
    inputsOnly_or_Both_inputsOutputs=dict_possible_values_constant_machine_overview,
    outputs_optional={
        "Result_NNConfig": solution_found,
        "Result_delay sec": solution_finder.result_delay_sec,
        "Result_evaluate_count_better_score": solution_finder.result_evaluate_count_better_score,
        "Result_best_solution_final_score": solution_finder.result_best_solution_final_score,
        "Result_evaluate_count_run": solution_finder.result_evaluate_count_run,
        "Result_shorter_cycles_enabled": solution_finder.result_shorter_cycles_enabled
    })

return solution_found
```

**What the function does and its purpose:**
- Performs comprehensive neural network architecture optimization
- Uses multi-objective optimization balancing performance and resource costs
- Integrates with SolutionFinder for systematic parameter exploration
- Provides data-driven configuration discovery for optimal neural network design

### Model Building Functions

#### `build_keras_nn_model(nn_loss, force_weight_initializer)`

**Where it's used and why:**
- Called when constructing the actual Keras neural network model for training or inference
- Used by NNEngine when creating models for training or prediction
- Critical for translating configuration specifications into executable neural networks
- Enables seamless integration with TensorFlow/Keras framework

**How the function works:**
1. **Validation**: Ensures input/output neuron counts are valid
2. **Model Creation**: Initializes empty Keras Sequential model
3. **Layer Construction**: Builds network layers according to configuration
4. **Regularization Application**: Adds dropout and batch normalization as specified
5. **Model Compilation**: Configures optimizer, loss function, and metrics

**Keras Model Construction Process:**
```python
# Validate neuron counts
if self.num_of_input_neurons == 0 or self.num_of_output_neurons == 0:
    logger.error("Unable to generate model with zero neurons")

# Create Keras model
neural_network_model = keras.Sequential()

# Add input layer
neural_network_model.add(
    keras.layers.InputLayer(input_shape=(self.num_of_input_neurons,))
)

# Build hidden layers according to configuration
nn_shape = self.nn_shape_instance
user_nn_shape = self.nn_shape_instance.get_user_nn_shape()

for layer in user_nn_shape.index:
    if nn_shape.layer_get_type(layer) == "dense":
        # Calculate neuron count for this layer
        layer_neurons_count = nn_shape.layer_neuron_count(
            layer,
            self.num_of_input_neurons,
            self.num_of_output_neurons
        )

        if layer_neurons_count == 0:
            logger.error("Cannot create layer with zero neurons")

        # Add dense layer
        neural_network_model.add(
            keras.layers.Dense(
                layer_neurons_count,
                name=layer,
                activation=nn_shape.layer_get_activation_function(layer),
                kernel_initializer=force_weight_initializer
            )
        )

        # Add regularization (skip for output layer)
        if layer == "output":
            break

        if nn_shape.layer_get_dropout_level(layer):
            neural_network_model.add(
                keras.layers.Dropout(nn_shape.layer_get_dropout_level(layer))
            )

        if nn_shape.layer_have_batch_normalization(layer):
            neural_network_model.add(keras.layers.BatchNormalization())

# Compile model
neural_network_model.compile(
    optimizer=self.nn_optimizer,
    loss=nn_loss,
    metrics=['accuracy']
)

# Log model summary if detailed logging enabled
if ENABLE_LOGGER_DEBUG_NNEngine_DETAILED:
    with io.StringIO() as stream:
        neural_network_model.summary(print_fn=lambda x: stream.write(x + "\n"))
        logger.debug(stream.getvalue())
```

**What the function does and its purpose:**
- Translates configuration specifications into executable neural network models
- Handles complex layer construction including regularization and activation functions
- Provides comprehensive model compilation with appropriate loss functions and metrics
- Enables seamless integration with TensorFlow/Keras training and inference pipelines

### NNShape Class Functions

#### `NNShape.__init__(nnshape_type_machine, nnshape_type_user)`

**Where it's used and why:**
- Called when creating NNShape instances to represent neural network architectures
- Used throughout the system for managing neural network topology specifications
- Critical for maintaining consistent representation of network architectures
- Enables conversion between different shape representation formats

**How the function works:**
1. **Format Detection**: Determines whether input is machine format or user format
2. **Storage Assignment**: Stores appropriate format and marks other as None for lazy conversion
3. **Validation**: Ensures exactly one format is provided

**Shape Format Handling:**
```python
if not nnshape_type_user is None and not nnshape_type_machine is None:
    logger.error("Please do not specify both nnshape_type_user and nnshape_type_machine simultaneously")

if isinstance(nnshape_type_user, pd.DataFrame):
    # User format: variable layers with detailed specifications
    self._user_nn_shape = nnshape_type_user.copy()
    self._machine_nn_shape = None

elif isinstance(nnshape_type_machine, pd.Series):
    # Machine format: fixed structure for database storage
    self._machine_nn_shape = nnshape_type_machine.copy()
    self._user_nn_shape = None

elif isinstance(nnshape_type_machine, dict):
    # Dictionary conversion to machine format
    self._machine_nn_shape = pd.Series(nnshape_type_machine)
    self._user_nn_shape = None

else:
    logger.error("Input must be DataFrame (user format) or Series/dict (machine format)")
```

**What the function does and its purpose:**
- Provides unified interface for neural network shape representations
- Enables conversion between user-friendly and storage-efficient formats
- Maintains shape integrity across different system components

#### `get_user_nn_shape()`

**Where it's used and why:**
- Called when user-friendly neural network shape representation is needed
- Used by UI components, logging, and user-facing operations
- Critical for providing human-readable network architecture information
- Enables visualization and debugging of network structures

**How the function works:**
1. **Cache Check**: Returns cached user shape if already converted
2. **Lazy Conversion**: Converts from machine format if needed
3. **Caching**: Stores converted shape for future use

**What the function does and its purpose:**
- Provides user-friendly access to neural network architecture
- Enables visualization and interpretation of network structure
- Supports debugging and monitoring of neural network configurations

#### `get_machine_nn_shape()`

**Where it's used and why:**
- Called when machine-efficient neural network shape representation is needed
- Used for database storage and system-internal operations
- Critical for maintaining compact representation for persistence
- Enables efficient storage and retrieval of network configurations

**How the function works:**
1. **Cache Check**: Returns cached machine shape if already converted
2. **Lazy Conversion**: Converts from user format if needed
3. **Caching**: Stores converted shape for future use

**What the function does and its purpose:**
- Provides efficient storage format for neural network architectures
- Enables compact persistence of configuration data
- Supports system-internal operations requiring standardized format

#### `layer_neuron_count(layer_identifier, input_neuron_count, output_neuron_count)`

**Where it's used and why:**
- Called when calculating actual neuron counts for specific layers
- Used during model construction and resource estimation
- Critical for translating percentage-based configurations to absolute neuron counts
- Enables dynamic adaptation to different input/output dimensions

**How the function works:**
1. **Layer Type Detection**: Determines whether layer is input, output, or hidden
2. **Neuron Calculation**: Applies percentage-based formulas for hidden layers
3. **Boundary Enforcement**: Ensures minimum neuron count of 1

**Neuron Count Calculation Logic:**
```python
if layer_identifier in ["input", 0]:
    return int(input_neuron_count)

elif layer_identifier in ["output", -1]:
    return int(output_neuron_count)

else:
    # Hidden layer calculation using percentage-based approach
    layer_neurons_percentage = self.get_neurons_percentage(layer_identifier)

    if isinstance(layer_identifier, str):
        layer_identifier = self._convert_identifier_layer_name_to_layer_number(layer_identifier)

    if self.get_hidden_layers_count() == 1:
        # Single hidden layer: equal contribution from input and output
        part_input = (layer_neurons_percentage * input_neuron_count / 100 / 2)
        part_output = (layer_neurons_percentage * output_neuron_count / 100 / 2)
    else:
        # Multiple hidden layers: weighted contribution based on position
        part_from_input = self.get_hidden_layers_count() - layer_identifier
        part_input = (layer_neurons_percentage * part_from_input /
                     (self.get_hidden_layers_count() - 1) * input_neuron_count / 100)

        part_from_output = layer_identifier - 1
        part_output = (layer_neurons_percentage * part_from_output /
                      (self.get_hidden_layers_count() - 1) * output_neuron_count / 100)

    return max(1, int(part_input + part_output))
```

**What the function does and its purpose:**
- Translates percentage-based configurations to absolute neuron counts
- Enables dynamic adaptation to different problem sizes
- Maintains architectural proportions across different scales
- Supports resource planning and model complexity estimation

#### `neurons_total_count(input_neuron_count, output_neuron_count)`

**Where it's used and why:**
- Called when estimating total computational resources required
- Used for resource planning, hardware requirements assessment, and cost estimation
- Critical for determining model complexity and training resource needs
- Enables capacity planning and resource allocation decisions

**How the function works:**
1. **Layer Iteration**: Sums neuron counts across all hidden layers
2. **Calculation**: Uses layer_neuron_count for each layer
3. **Aggregation**: Returns total neuron count

**What the function does and its purpose:**
- Provides quantitative measure of model complexity
- Enables resource planning and hardware selection
- Supports performance optimization and capacity planning

#### `weight_total_count(input_neuron_count, output_neuron_count)`

**Where it's used and why:**
- Called when estimating total parameter count (weights and biases)
- Used for memory requirement estimation and computational cost analysis
- Critical for determining training and inference resource requirements
- Enables optimization decisions based on model size constraints

**How the function works:**
1. **Layer Collection**: Gathers neuron counts for all layers including input/output
2. **Weight Calculation**: Computes connections between consecutive layers
3. **Summation**: Aggregates total parameter count across network

**Weight Calculation Process:**
```python
# Get neuron counts for all layers
layers_neuron_count = [
    self.layer_neuron_count(layer_identifier, input_neuron_count, output_neuron_count)
    for layer_identifier in range(0, self.get_hidden_layers_count() + 1)
]
# Add output layer
layers_neuron_count.append(
    self.layer_neuron_count(-1, input_neuron_count, output_neuron_count)
)

# Calculate total weights (connections between layers)
return sum(
    layer_1_neuron_count * layer_2_neuron_count
    for layer_1_neuron_count, layer_2_neuron_count in zip(
        layers_neuron_count[:-1], layers_neuron_count[1:]
    )
)
```

**What the function does and its purpose:**
- Quantifies total learnable parameters in the neural network
- Enables memory usage estimation and computational cost analysis
- Supports model size optimization and resource allocation
- Facilitates hardware selection and deployment planning

### Conversion and Utility Functions

#### `_convert_machine_nn_shape_to_user_nn_shape(machine_nn_shape)`

**Where it's used and why:**
- Called internally when converting machine-format shapes to user-friendly format
- Used for displaying configurations to users and enabling user modifications
- Critical for maintaining user-friendly interface while using efficient storage format
- Enables configuration visualization and editing capabilities

**How the function works:**
1. **Formula Selection**: Chooses appropriate conversion formula based on hidden layer count
2. **Parameter Extraction**: Retrieves layer parameters from machine format
3. **Formula Application**: Applies mathematical formulas to expand fixed format to variable format
4. **DataFrame Construction**: Creates user-friendly DataFrame representation

**Conversion Process:**
```python
# Select conversion formula based on hidden layer count
formula, layer_type_index = MACHINE_NN_SHAPE_TO_USER_NN_SHAPE_CONVERT_INFO[int(machine_nn_shape["HiddenLayerCount"])]

# Extract layer parameters using formula
layer_type_activation = pd.Series([
    machine_nn_shape[f"LayerTypeActivation{i}"] for i in layer_type_index
], dtype=object)

# Apply formulas to expand parameters
neuron_percentage, drop_out, batch_normalization = (
    pd.Series(
        self._calculate_formula_in_all_cells_of_dataset(
            formula,
            machine_nn_shape,
            [f"{column_name}{i}" for i in range(1, 4)]
        ),
        dtype=float if column_name != "LayerTypeActivation" else object
    )
    for column_name in ["NeuronPercentage", "DropOut", "BatchNormalization"]
)

# Construct user DataFrame
user_nn_shape = pd.DataFrame({
    "NeuronPercentage": neuron_percentage,
    "LayerTypeActivation": layer_type_activation,
    "DropOut": drop_out.round(1),
    "BatchNormalization": np.rint(batch_normalization + 0.001).astype(bool),
}, columns=USER_NN_SHAPE_COLUMNS_NAME)

# Add output layer
user_nn_shape.loc["output"] = [
    None,
    machine_nn_shape["LayerTypeActivationOutput"],
    None,
    None
]

return user_nn_shape
```

**What the function does and its purpose:**
- Converts compact machine format to detailed user format
- Enables user-friendly configuration visualization and editing
- Maintains consistency between storage and presentation formats
- Supports configuration modification and customization

#### `_convert_user_nn_shape_to_machine_nn_shape(user_nn_shape)`

**Where it's used and why:**
- Called internally when converting user-format shapes to machine-efficient format
- Used for database storage and system-internal operations
- Critical for maintaining compact representation for persistence and computation
- Enables efficient storage and retrieval of configuration data

**How the function works:**
1. **Formula Selection**: Chooses appropriate compression formula based on layer count
2. **Parameter Aggregation**: Applies mathematical formulas to compress variable format to fixed format
3. **Series Construction**: Creates machine-efficient Series representation

**Compression Process:**
```python
# Select compression formula
formula, layer_type_index = USER_NN_SHAPE_TO_MACHINE_NN_SHAPE_CONVERT_INFO[hidden_layer_count]

# Apply compression formulas
neuron_percentage, drop_out, batch_normalization = (
    np.array(
        self._calculate_formula_in_all_cells_of_dataset(
            formula,
            user_nn_shape,
            [(f"hidden_{i}", column_name) for i in range(1, 1 + hidden_layer_count)]
        ),
        dtype=object
    )
    for column_name in ["NeuronPercentage", "DropOut", "BatchNormalization"]
)

# Extract layer type activations
layer_type_activation = [
    user_nn_shape.loc[f"hidden_{i}", "LayerTypeActivation"]
    for i in layer_type_index
]

# Construct machine Series
machine_nn_shape = pd.Series(
    np.concatenate((
        np.column_stack((
            neuron_percentage,
            layer_type_activation,
            drop_out,
            batch_normalization
        )).ravel(),
        [user_nn_shape.loc["output", "LayerTypeActivation"]],
        [hidden_layer_count]
    )),
    index=NNShape.get_list_of_nn_shape_columns_names(),
    dtype=object
)

return machine_nn_shape
```

**What the function does and its purpose:**
- Compresses detailed user format to compact machine format
- Enables efficient storage and retrieval of configuration data
- Maintains data integrity during format conversion
- Supports system-internal operations requiring standardized format

This detailed analysis demonstrates how NNConfiguration serves as the comprehensive neural network architecture management system in the EasyAutoML framework, providing sophisticated configuration discovery, optimization, and management capabilities that ensure optimal neural network performance while maintaining efficiency and adaptability across diverse machine learning scenarios.