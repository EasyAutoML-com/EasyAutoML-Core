# ML/NNEngine.py - Neural Network Training and Inference Engine

## Overview

The `NNEngine` class serves as the core neural network execution engine in the EasyAutoML.com system. It orchestrates the complete machine learning pipeline from configuration to deployment, managing model training, evaluation, prediction, and resource optimization.

**Location**: `ML/NNEngine.py`

## Core Functionality

### Primary Responsibilities

- **Model Lifecycle Management**: Create, train, evaluate, and deploy neural networks
- **Pipeline Orchestration**: Coordinate data preprocessing, feature engineering, and encoding/decoding
- **Resource Management**: Optimize computational resources and memory usage
- **Performance Monitoring**: Track training progress and model quality metrics
- **Scalability Control**: Adapt to different machine levels and computational constraints

### Architecture Overview

```python
class NNEngine:
    """
    Comprehensive neural network training and inference engine.
    Manages complete ML pipeline from data to predictions.
    """
```

## Core Components

### Component Integration

```python
def __init__(self, machine: Machine, allow_re_run_configuration: bool = False):
    """
    Initialize NNEngine with machine and configuration management.

    :param machine: Machine instance containing data and configuration
    :param allow_re_run_configuration: Whether to rebuild configurations if needed
    """

    # Core component references
    self._machine = machine
    self._mdc = None          # MachineDataConfiguration
    self._ici = None          # InputsColumnsImportance
    self._fe = None           # FeatureEngineeringConfiguration
    self._enc_dec = None      # EncDec (Encoding/Decoding)
    self._nn_configuration = None
    self._nn_model = None
```

## Configuration Management

### Dynamic Configuration Loading

```python
def _init_load_or_create_configuration(self, allow_re_run_configuration: bool = False):
    """
    Load existing or create new configuration pipeline.
    Ensures all components are properly initialized and synchronized.
    """

    # 1. MachineDataConfiguration (MDC)
    if not self._machine.is_config_ready_mdc() or allow_re_run_configuration:
        self._mdc = MachineDataConfiguration(self._machine)
        self._mdc.save_configuration_in_machine()

    # 2. InputsColumnsImportance (ICI)
    if not self._machine.is_config_ready_ici() or allow_re_run_configuration:
        self._ici = InputsColumnsImportance(
            self._machine, self, create_configuration_best=True
        )
        self._ici.save_configuration_in_machine()

    # 3. FeatureEngineeringConfiguration (FEC)
    if not self._machine.is_config_ready_fe() or allow_re_run_configuration:
        budget = self._machine.db_machine.fe_budget_total
        self._fe = FeatureEngineeringConfiguration(
            self._machine, global_dataset_budget=budget, nn_engine_for_searching_best_config=self
        )
        self._fe.save_configuration_in_machine()

    # 4. Encoding/Decoding (EncDec)
    if not self._machine.is_config_ready_enc_dec() or allow_re_run_configuration:
        training_data = self._mdc.dataframe_pre_encode(
            self._machine.get_random_user_dataframe_for_training_trial(is_for_learning=True)
        )
        self._enc_dec = EncDec(self._machine, training_data)
        self._enc_dec.save_configuration_in_machine()

    # 5. Neural Network Configuration
    if not self._machine.is_config_ready_nn_configuration() or allow_re_run_configuration:
        self._nn_configuration = self._create_optimal_nn_configuration()
```

## Model Training

### Training Architecture and Algorithms

The NNEngine implements sophisticated training algorithms optimized for automated machine learning scenarios. It supports multiple training modes, adaptive learning strategies, and comprehensive performance monitoring.

#### Neural Network Architecture Generation

```python
def _create_optimal_nn_configuration(self) -> NNConfiguration:
    """
    Generate optimal neural network architecture based on data characteristics.

    Uses experimental optimization to find best:
    - Number of layers and neurons
    - Activation functions
    - Regularization techniques
    - Learning parameters
    """

    # Analyze data characteristics
    input_features = len(self._enc_dec._column_counts['input_encoded_columns_count'])
    output_features = len(self._enc_dec._column_counts['output_encoded_columns_count'])
    dataset_size = self._get_training_dataset_size()

    # Determine architecture complexity based on data size and complexity
    if dataset_size < 1000:
        # Small dataset - simple architecture to avoid overfitting
        architecture = self._generate_simple_architecture(input_features, output_features)
    elif dataset_size < 10000:
        # Medium dataset - balanced architecture
        architecture = self._generate_balanced_architecture(input_features, output_features)
    else:
        # Large dataset - complex architecture with regularization
        architecture = self._generate_complex_architecture(input_features, output_features)

    # Create NNConfiguration with optimal parameters
    nn_config = NNConfiguration(
        architecture=architecture,
        optimizer=self._select_optimal_optimizer(),
        loss_function=self._select_loss_function(),
        metrics=self._select_evaluation_metrics(),
        callbacks=self._configure_training_callbacks()
    )

    return nn_config
```

#### Training Trial System

```python
def do_training_trial(
    self,
    encoded_for_ai_dataframe=None,
    encoded_for_ai_validation_dataframe=None,
    pre_encoded_dataframe=None,
    pre_encoded_validation_dataframe=None,
    max_epochs: int = None,
    early_stopping_patience: int = 10,
    learning_rate: float = None
) -> tuple[float, float, float, dict]:
    """
    Perform controlled training trial with comprehensive monitoring.

    Args:
        encoded_for_ai_dataframe: Pre-encoded training data
        encoded_for_ai_validation_dataframe: Pre-encoded validation data
        pre_encoded_dataframe: Raw training data (alternative)
        pre_encoded_validation_dataframe: Raw validation data (alternative)
        max_epochs: Maximum training epochs
        early_stopping_patience: Early stopping patience
        learning_rate: Custom learning rate

    Returns:
        Tuple of (final_loss, final_accuracy, epoch_percentage, training_metrics)
    """

    # Data preparation with validation
    if pre_encoded_dataframe is not None:
        if len(pre_encoded_dataframe) < 10:
            raise ValueError("Training dataset too small (< 10 samples)")

        encoded_for_ai_dataframe = self._enc_dec.encode_for_ai(pre_encoded_dataframe)
        encoded_for_ai_validation_dataframe = self._enc_dec.encode_for_ai(pre_encoded_validation_dataframe)

    # Validate data dimensions
    self._validate_training_data_dimensions(encoded_for_ai_dataframe, encoded_for_ai_validation_dataframe)

    # Build or load model with configuration
    if self._nn_model is None:
        self._nn_model = self._build_keras_model(learning_rate=learning_rate)

    # Configure training parameters
    training_config = self._configure_training_parameters(
        max_epochs=max_epochs,
        early_stopping_patience=early_stopping_patience
    )

    # Execute training with comprehensive monitoring
    try:
        training_start_time = time.time()

        history = self._nn_model.fit(
            encoded_for_ai_dataframe,
            validation_data=encoded_for_ai_validation_dataframe,
            **training_config,
            verbose=0  # Silent training for automated execution
        )

        training_duration = time.time() - training_start_time

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        # Return failure indicators
        return float('inf'), 0.0, 0.0, {'error': str(e)}

    # Extract comprehensive metrics
    training_metrics = self._extract_training_metrics(history, training_duration)

    final_loss = history.history['val_loss'][-1]
    final_accuracy = history.history.get('val_accuracy', [0.0])[-1]
    epoch_percentage = len(history.history['val_loss']) / training_config['epochs']

    return final_loss, final_accuracy, epoch_percentage, training_metrics
```

#### Advanced Training Strategies

##### Adaptive Learning Rate Scheduling

```python
def _configure_adaptive_learning_rate(self) -> tf.keras.callbacks.Callback:
    """
    Implement adaptive learning rate scheduling based on training progress.
    """

    def learning_rate_schedule(epoch, lr):
        """Dynamic learning rate adjustment"""
        if epoch < 10:
            return lr  # Warm-up period
        elif epoch < 50:
            return lr * 0.95  # Gradual decay
        else:
            return max(lr * 0.99, 1e-6)  # Slow decay with minimum

    return tf.keras.callbacks.LearningRateScheduler(learning_rate_schedule)
```

##### Early Stopping with Multiple Criteria

```python
def _configure_early_stopping(self, patience: int = 15) -> tf.keras.callbacks.Callback:
    """
    Configure early stopping with multiple stopping criteria.
    """

    return tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        mode='min',
        min_delta=1e-4,  # Minimum change to qualify as improvement
        verbose=0
    )
```

##### Gradient Monitoring and NaN Detection

```python
def _configure_gradient_monitoring(self) -> tf.keras.callbacks.Callback:
    """
    Monitor gradients for numerical stability and learning issues.
    """

    class GradientMonitor(tf.keras.callbacks.Callback):
        def on_batch_end(self, batch, logs=None):
            # Check for NaN/inf gradients
            for layer in self.model.layers:
                weights = layer.get_weights()
                for weight_matrix in weights:
                    if np.any(np.isnan(weight_matrix)) or np.any(np.isinf(weight_matrix)):
                        logger.warning(f"NaN/Inf detected in layer {layer.name}")
                        break

    return GradientMonitor()
```

### Full Training Pipeline

```python
def machine_nn_engine_do_full_training_if_not_already_done(
    self,
    force_retrain: bool = False,
    optimization_budget: int = 100,
    max_training_time_hours: float = 24.0
) -> dict:
    """
    Execute complete automated training pipeline with optimization.

    Args:
        force_retrain: Force retraining even if model exists
        optimization_budget: Budget for hyperparameter optimization
        max_training_time_hours: Maximum training time limit

    Returns:
        Training results and performance metrics
    """

    training_start_time = time.time()
    training_results = {
        'success': False,
        'training_duration': 0,
        'final_loss': float('inf'),
        'final_accuracy': 0.0,
        'optimization_attempts': 0,
        'best_configuration': None
    }

    try:
        # Step 1: Ensure all configurations are ready
        if not self._machine.is_nn_solving_ready() or force_retrain:
            logger.info("Initializing/reinitializing ML pipeline configurations")
            self._init_load_or_create_configuration(allow_re_run_configuration=True)

        # Step 2: Architecture optimization (if budget allows)
        if optimization_budget > 10:
            logger.info(f"Performing architecture optimization (budget: {optimization_budget})")
            best_config = self._optimize_neural_architecture(optimization_budget)
            self._nn_configuration = best_config
            training_results['optimization_attempts'] = len(self._optimization_history)

        # Step 3: Prepare training data with augmentation if needed
        training_data = self._prepare_training_data()
        validation_data = self._prepare_validation_data()

        # Apply data augmentation for small datasets
        if len(training_data[0]) < 1000:
            training_data = self._apply_data_augmentation(training_data)

        # Step 4: Execute full training with monitoring
        logger.info("Starting full model training")

        training_history = self._execute_full_training(
            training_data,
            validation_data,
            max_time_hours=max_training_time_hours
        )

        # Step 5: Model validation and quality assessment
        validation_metrics = self._validate_trained_model(validation_data)

        # Step 6: Save trained model and update machine state
        self._save_trained_model()
        self._update_machine_training_status()

        # Compile results
        training_results.update({
            'success': True,
            'training_duration': time.time() - training_start_time,
            'final_loss': validation_metrics['loss'],
            'final_accuracy': validation_metrics['accuracy'],
            'training_history': training_history,
            'validation_metrics': validation_metrics,
            'model_size_mb': self._calculate_model_size(),
            'training_samples': len(training_data[0])
        })

        logger.info(f"Training completed successfully in {training_results['training_duration']:.1f}s")

    except Exception as e:
        logger.error(f"Full training failed: {str(e)}")
        training_results['error'] = str(e)

    return training_results
```

#### Training Data Preparation with Augmentation

```python
def _prepare_training_data(self) -> tuple:
    """
    Prepare and augment training data for optimal model performance.
    """

    # Get base training data
    training_df = self._machine.get_random_user_dataframe_for_training_trial(
        is_for_learning=True
    )

    # Pre-encode data
    pre_encoded = self._mdc.dataframe_pre_encode(training_df)

    # Encode for neural network
    encoded_training = self._enc_dec.encode_for_ai(pre_encoded)

    # Apply intelligent data augmentation for small datasets
    if len(encoded_training) < 1000:
        encoded_training = self._augment_training_data(encoded_training)

    # Get validation data
    validation_df = self._machine.get_random_user_dataframe_for_training_trial(
        is_for_evaluation=True
    )
    pre_encoded_val = self._mdc.dataframe_pre_encode(validation_df)
    encoded_validation = self._enc_dec.encode_for_ai(pre_encoded_val)

    return encoded_training, encoded_validation
```

#### Intelligent Data Augmentation

```python
def _augment_training_data(self, data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply intelligent data augmentation techniques based on data characteristics.
    """

    augmented_data = data.copy()

    # Analyze data types in configuration
    for col_name, col_config in self._enc_dec._enc_dec_configuration.items():
        if not col_config['is_input']:
            continue

        data_type = col_config['column_datatype_name']

        if data_type == 'FLOAT':
            # Add noise to numerical features
            noise_factor = 0.05  # 5% noise
            noise = np.random.normal(0, noise_factor, len(augmented_data))
            augmented_data[col_name] = augmented_data[col_name] * (1 + noise)

        elif data_type == 'LABEL':
            # Skip categorical features (would require domain knowledge)
            continue

    # Add synthetic samples if augmentation insufficient
    if len(augmented_data) < 500:
        synthetic_samples = self._generate_synthetic_samples(data, target_size=500)
        augmented_data = pd.concat([augmented_data, synthetic_samples], ignore_index=True)

    return augmented_data
```

## Prediction and Inference

### Direct Solving

```python
def do_solving_direct_encoded_for_ai(self, encoded_for_ai_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Generate predictions from encoded data.

    :param encoded_for_ai_dataframe: Encoded input data
    :return: Raw prediction results
    """
    if not self._machine.is_nn_solving_ready():
        raise RuntimeError("Model not ready for predictions")

    # Load model if needed
    if self._nn_model is None:
        self._nn_model = self._load_trained_model()

    # Generate predictions
    predictions = self._nn_model.predict(encoded_for_ai_dataframe)

    return pd.DataFrame(predictions)
```

### Complete Prediction Pipeline

```python
def do_solving(self, user_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    End-to-end prediction from raw user data to final results.

    :param user_dataframe: Raw user input data
    :return: Human-readable prediction results
    """

    # 1. Pre-encode user data
    pre_encoded = self._mdc.dataframe_pre_encode(user_dataframe)

    # 2. Apply feature engineering
    encoded_for_ai = self._enc_dec.encode_for_ai(pre_encoded)

    # 3. Generate predictions
    raw_predictions = self.do_solving_direct_encoded_for_ai(encoded_for_ai)

    # 4. Decode to human-readable format
    decoded_predictions = self._enc_dec.decode_from_ai(raw_predictions)

    # 5. Post-decode to original format
    final_predictions = self._mdc.dataframe_post_decode(decoded_predictions)

    return final_predictions
```

## Model Management

### Keras Model Building

```python
def _build_keras_model(self) -> keras.Model:
    """
    Construct Keras model from NNConfiguration.

    :return: Compiled Keras model
    """

    # Get layer configuration
    layers_config = self._nn_configuration.get_keras_layers_configuration()

    # Build sequential model
    model = keras.Sequential()

    for layer_config in layers_config:
        if layer_config["type"] == "input":
            model.add(keras.layers.Input(shape=(layer_config["neurons"],)))
        elif layer_config["type"] == "dense":
            model.add(keras.layers.Dense(
                layer_config["neurons"],
                activation=layer_config["activation"]
            ))
            if layer_config.get("dropout", 0) > 0:
                model.add(keras.layers.Dropout(layer_config["dropout"]))

    # Configure optimizer and compile
    optimizer = self._nn_configuration.get_optimizer_configuration()
    model.compile(
        optimizer=optimizer,
        loss=self._nn_configuration.nn_loss,
        metrics=['accuracy']
    )

    return model
```

### Model Persistence

```python
def _save_trained_model(self):
    """Save trained model to database"""
    model_data = pickle.dumps(self._nn_model)
    self._machine.db_machine.training_nn_model = model_data
    self._machine.save_machine_to_db()

def _load_trained_model(self) -> keras.Model:
    """Load trained model from database"""
    model_data = self._machine.db_machine.training_nn_model
    return pickle.loads(model_data)
```

## Resource Management

### Memory Optimization

```python
def _optimize_memory_usage(self):
    """
    Implement memory optimization strategies.
    """
    # Clear unnecessary data
    if hasattr(self, '_training_cache'):
        del self._training_cache

    # Force garbage collection
    import gc
    gc.collect()

    # Optimize Keras memory usage
    keras.backend.clear_session()
```

### Computational Resource Control

```python
def _get_resource_limits(self) -> dict:
    """
    Determine computational limits based on machine level.

    :return: Resource constraint dictionary
    """
    machine_level = self._machine.db_machine.machine_level

    limits = {
        1: {"max_memory_gb": 2, "max_training_time_hours": 1},
        2: {"max_memory_gb": 4, "max_training_time_hours": 2},
        3: {"max_memory_gb": 8, "max_training_time_hours": 4},
        4: {"max_memory_gb": 16, "max_training_time_hours": 8}
    }

    return limits.get(machine_level, limits[1])
```

## Performance Monitoring

### Training Callbacks

```python
def _get_training_callbacks(self) -> list:
    """
    Configure training monitoring and control callbacks.

    :return: List of Keras callbacks
    """
    callbacks = [
        # Early stopping
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),

        # Learning rate scheduling
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        ),

        # Model checkpointing
        keras.callbacks.ModelCheckpoint(
            filepath=self._get_checkpoint_path(),
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=True
        )
    ]

    return callbacks
```

### Progress Tracking

```python
def _log_training_progress(self, epoch: int, logs: dict):
    """
    Log training progress and metrics.

    :param epoch: Current epoch number
    :param logs: Training metrics dictionary
    """
    progress_data = {
        "epoch": epoch,
        "loss": logs.get('loss'),
        "val_loss": logs.get('val_loss'),
        "accuracy": logs.get('accuracy'),
        "val_accuracy": logs.get('val_accuracy'),
        "timestamp": datetime.datetime.now()
    }

    # Store in machine for monitoring
    self._machine.db_machine.log_work_status.update({
        str(datetime.datetime.now()): f"Epoch {epoch}: loss={logs.get('loss', 'N/A')}"
    })
```

## Integration with Optimization Systems

### Experimenter Integration

```python
# Support for configuration testing
def get_experimenter_ready_engine(self):
    """
    Prepare NNEngine for experimental evaluation.

    :return: Configured NNEngine for testing
    """
    # Ensure minimal configuration
    if not self._machine.is_config_ready_mdc():
        self._mdc = MachineDataConfiguration(self._machine, force_update_configuration_with_this_dataset=True)
        self._mdc.save_configuration_in_machine()

    # Create basic EncDec
    basic_training_data = self._mdc.dataframe_pre_encode(
        self._machine.get_random_user_dataframe_for_training_trial(is_for_learning=True)
    )
    self._enc_dec = EncDec(self._machine, basic_training_data)
    self._enc_dec.save_configuration_in_machine()

    return self
```

### SolutionFinder Integration

```python
# Support automated configuration optimization
def evaluate_configuration(self, config_params: dict) -> dict:
    """
    Evaluate neural network configuration performance.

    :param config_params: Configuration parameters to test
    :return: Performance metrics dictionary
    """

    # Apply configuration
    self._nn_configuration = NNConfiguration(config_params)

    # Perform trial training
    loss, accuracy, epoch_pct = self.do_training_trial()

    return {
        "loss": loss,
        "accuracy": accuracy,
        "completion_percentage": epoch_pct,
        "config_valid": True
    }
```

## Usage Patterns

### Basic Training Workflow

```python
# 1. Initialize NNEngine
nn_engine = NNEngine(machine=machine)

# 2. Ensure configurations are ready
nn_engine._init_load_or_create_configuration()

# 3. Perform full training
nn_engine.machine_nn_engine_do_full_training_if_not_already_done()

# 4. Make predictions
predictions = nn_engine.do_solving(user_data)
```

### Trial Training for Optimization

```python
# 1. Prepare for experimental evaluation
nn_engine = NNEngine(machine=machine)
nn_engine.get_experimenter_ready_engine()

# 2. Test different configurations
for config in configuration_candidates:
    metrics = nn_engine.evaluate_configuration(config)
    # Analyze performance
    if metrics["accuracy"] > best_accuracy:
        best_config = config
        best_accuracy = metrics["accuracy"]
```

### Prediction Pipeline

```python
# 1. Load trained NNEngine
nn_engine = NNEngine(machine=machine)

# 2. Verify model readiness
if nn_engine._machine.is_nn_solving_ready():
    # 3. Make predictions
    predictions = nn_engine.do_solving(user_input_data)
    print("Predictions:", predictions)
else:
    print("Model not ready - requires training")
```

## Error Handling and Resilience

### Configuration Validation

```python
def _validate_configuration_integrity(self):
    """
    Ensure all configurations are compatible and valid.
    """
    issues = []

    if not self._mdc:
        issues.append("MachineDataConfiguration not initialized")

    if not self._enc_dec:
        issues.append("EncDec not initialized")

    if not self._nn_configuration:
        issues.append("NNConfiguration not initialized")

    if issues:
        raise ConfigurationError(f"Configuration issues: {', '.join(issues)}")
```

### Training Error Recovery

```python
def _handle_training_error(self, error: Exception):
    """
    Handle training failures with appropriate recovery actions.
    """
    error_msg = f"Training failed: {str(error)}"

    # Log error
    self._machine.db_machine.log_work_status.update({
        str(datetime.datetime.now()): f"ERROR: {error_msg}"
    })

    # Attempt recovery based on error type
    if "out of memory" in str(error).lower():
        self._reduce_memory_usage()
        # Retry with smaller batch size
    elif "gradient explosion" in str(error).lower():
        self._stabilize_gradients()
        # Retry with gradient clipping
    else:
        # Mark for configuration rebuild
        self._machine.db_machine.machine_is_re_run_nn_config = True
```

## Performance Optimization

### Caching Strategies

```python
# Cache frequently used data transformations
@lru_cache(maxsize=10)
def _cached_data_preparation(self, data_hash: str):
    """Cache data preprocessing results"""
    return self._mdc.dataframe_pre_encode(data_hash)
```

### Parallel Processing

```python
def _setup_distributed_training(self):
    """Configure distributed training if available"""
    if self._is_distributed_available():
        # Configure multi-GPU training
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            self._nn_model = self._build_keras_model()
    else:
        self._nn_model = self._build_keras_model()
```

The NNEngine module provides a comprehensive, production-ready neural network training and inference engine that seamlessly integrates with the broader EasyAutoML ecosystem.

## Detailed Function Analysis

### Core Initialization and Configuration Functions

#### `__init__(machine, allow_re_run_configuration)`

**Where it's used and why:**
- Called when creating a new NNEngine instance for neural network operations
- Used throughout the AutoML pipeline when neural network training or inference is required
- Critical for establishing the complete ML pipeline with all necessary components
- Enables seamless integration with Machine lifecycle and configuration management

**How the function works:**
1. **Parameter Validation**: Ensures valid Machine object is provided
2. **Component Initialization**: Sets up all internal component references
3. **Error Handling**: Wraps configuration loading in try-catch for production resilience
4. **Configuration Loading**: Triggers the comprehensive configuration setup process

**Initialization Flow:**
```python
# Validate machine parameter
if not isinstance(machine, Machine):
    logger.error("The constructor argument must have an instance of the machine")

# Initialize component references
self._machine = machine
self._mdc = None        # MachineDataConfiguration
self._ici = None        # InputsColumnsImportance
self._fe = None         # FeatureEngineeringConfiguration
self._enc_dec = None    # EncDec
self._nn_configuration = None
self._nn_model = None

# Load or create configurations with error handling
if IS_RUNNING_IN_DEBUG_MODE:
    self._init_load_or_create_configuration(allow_re_run_configuration)
else:
    try:
        self._init_load_or_create_configuration(allow_re_run_configuration)
    except Exception as e:
        # Log error and mark machine as failed
        machine.db_machine.log_work_status.update({str(datetime.now()): False})
        machine.db_machine.log_work_message.update({
            str(datetime.now()): f"Error Unable to load NNEngine for machine {machine} because {e}"
        })
        machine.db_machine.save_machine_to_db()
        logger.error(f"Unable to load NNEngine for machine {machine} because {e}")
```

**What the function does and its purpose:**
- Establishes complete neural network execution environment
- Provides unified interface to all ML pipeline components
- Ensures production-ready error handling and logging
- Maintains component synchronization and state consistency

#### `_init_load_or_create_configuration(allow_re_run_configuration, update_data_infos_stats)`

**Where it's used and why:**
- Called internally during NNEngine initialization to set up all ML pipeline components
- Used whenever the ML pipeline needs to be configured or reconfigured
- Critical for ensuring all components (MDC, ICI, FE, EncDec, NNConfig) are properly initialized
- Enables progressive configuration building and dependency management

**How the function works:**
1. **Configuration State Assessment**: Evaluates current readiness of each component
2. **Dependency Management**: Ensures components are built in correct order (MDC → ICI → FE → EncDec → NNConfig → NNModel)
3. **Conditional Rebuilding**: Only rebuilds components when necessary or explicitly requested
4. **Progressive Enhancement**: Each configuration step can trigger rebuilding of dependent components
5. **Resource Optimization**: Caches data to avoid redundant processing
6. **Recursive Prevention**: Uses global flags to prevent infinite configuration loops

**Configuration Pipeline Execution:**
```python
# Prevent recursive configuration in EasyAutoML machines
self.machine_nn_engine_configuration_set_starting(self._machine)

# Cache data for efficiency
_c_full_df_user = self._machine.data_lines_read()
_c_full_df_pre_encoded = None

# 1. MachineDataConfiguration (MDC) - Foundation component
if not self._machine.is_config_ready_mdc() or allow_re_run_configuration:
    _c_full_df_user = self._machine.data_lines_read()
    self._mdc = MachineDataConfiguration(self._machine, _c_full_df_user,
                                        force_create_with_this_inputs=...,
                                        force_create_with_this_outputs=...)
    self._mdc.save_configuration_in_machine()
    # Force rebuild of dependent components
    self._machine.clear_config_ici()
    self._machine.clear_config_fe()
    self._machine.clear_config_enc_dec()
    self._machine.clear_config_nn_configuration()
    self._machine.clear_config_nn_model()

# 2. InputsColumnsImportance (ICI) - Feature selection
if not self._machine.is_config_ready_ici() or allow_re_run_configuration:
    # Choose between best and minimum configuration based on prerequisites
    if (self._machine.is_config_ready_fe() and
        self._machine.is_config_ready_enc_dec() and
        self._machine.is_config_ready_nn_configuration() and
        self.is_nn_trained_and_ready()):
        self._ici = InputsColumnsImportance(self._machine, create_configuration_best=True,
                                          nnengine_for_best_config=self)
    else:
        self._ici = InputsColumnsImportance(self._machine, create_configuration_simple_minimum=True)
    self._ici.save_configuration_in_machine()

# Continue with FE, EncDec, NNConfiguration, NNModel in sequence...
# Each step checks prerequisites and forces dependent rebuilds

# Prevent recursive configuration
self.machine_nn_engine_configuration_set_finished(self._machine)
```

**What the function does and its purpose:**
- Orchestrates complete ML pipeline configuration in dependency order
- Manages component interdependencies and rebuild triggers
- Optimizes resource usage through intelligent caching
- Prevents configuration loops in complex machine networks
- Ensures consistent and complete ML pipeline setup

### Configuration Control Functions

#### `machine_nn_engine_configuration_set_starting(machine)`

**Where it's used and why:**
- Called before configuration operations to prevent recursive loops
- Used in complex machine networks where machines can reference each other
- Critical for preventing infinite loops in EasyAutoML scenarios
- Enables safe operation in interconnected machine environments

**How the function works:**
1. **Global State Management**: Updates global dictionary tracking active configurations
2. **Machine Identification**: Uses machine ID as unique identifier
3. **Timestamp Recording**: Records when configuration started for monitoring

**What the function does and its purpose:**
- Prevents configuration deadlocks in machine networks
- Enables monitoring of active configuration processes
- Supports safe concurrent machine operations

#### `machine_nn_engine_configuration_is_configurating(machine)`

**Where it's used and why:**
- Called to check if a machine is currently undergoing configuration
- Used throughout the system to avoid conflicts and recursive operations
- Critical for maintaining system stability during configuration operations
- Enables intelligent scheduling and resource allocation

**How the function works:**
1. **Global State Check**: Queries global dictionary for machine status
2. **Boolean Return**: Provides clear indication of configuration state

**What the function does and its purpose:**
- Enables conflict detection and avoidance
- Supports intelligent system coordination
- Maintains operational safety in distributed environments

### Data Processing Functions

#### `dataframe_full_encode(user_dataframe)`

**Where it's used and why:**
- Called when converting raw user data to neural network-ready format
- Used throughout the prediction and training pipelines
- Critical for maintaining data consistency between different processing stages
- Enables seamless integration of user data into ML operations

**How the function works:**
1. **Input Validation**: Ensures DataFrame format and content validity
2. **Two-Stage Encoding**: Pre-encode (MDC) followed by AI encoding (EncDec)
3. **Error Checking**: Validates that required components are initialized
4. **Format Consistency**: Maintains data structure integrity

**Encoding Pipeline:**
```python
# Validate input
if not isinstance(user_dataframe, pd.DataFrame):
    logger.error("user_dataframe must have a pandas DataFrame type")
if len(user_dataframe) == 0:
    logger.error("user_dataframe is empty")
if self._mdc is None:
    logger.error("Unable to pre-encode because mdc is not ready")
if self._enc_dec is None:
    logger.error("Unable to encode because encdec is not ready")

# Execute encoding pipeline
pre_encoded_dataframe = self._mdc.dataframe_pre_encode(user_dataframe)
encoded_dataframe = self._enc_dec.encode_for_ai(pre_encoded_dataframe)

return encoded_dataframe
```

**What the function does and its purpose:**
- Provides complete data transformation pipeline
- Ensures data compatibility with neural network requirements
- Maintains encoding consistency across operations
- Supports both training and inference data flows

#### `dataframe_full_decode(dataframe_encoded_for_ai)`

**Where it's used and why:**
- Called when converting neural network outputs back to user-readable format
- Used in prediction pipelines to return human-interpretable results
- Critical for maintaining user experience and result interpretability
- Enables seamless conversion between ML and user domains

**How the function works:**
1. **Input Validation**: Ensures proper encoded DataFrame format
2. **Two-Stage Decoding**: AI decode (EncDec) followed by post-decode (MDC)
3. **Component Verification**: Validates required decoding components
4. **Result Formatting**: Returns properly formatted user data

**What the function does and its purpose:**
- Completes the data transformation cycle from ML back to user domain
- Ensures prediction results are interpretable and useful
- Maintains data format consistency throughout the pipeline

### Training Functions

#### `do_training_and_save()`

**Where it's used and why:**
- Called when performing complete neural network training pipeline
- Used by WorkProcessor to execute training jobs
- Critical for the core ML training functionality of the system
- Enables automated, production-ready model training

**How the function works:**
1. **Multi-Cycle Training**: Executes training in 4 progressive cycles with increasing sophistication
2. **Configuration Evolution**: Each cycle can trigger component reconfiguration for optimization
3. **Error Recovery**: Handles training failures with appropriate recovery actions
4. **Result Persistence**: Saves trained model and updates machine status
5. **Resource Management**: Manages system resources during long-running training
6. **Progress Tracking**: Provides comprehensive logging and status updates

**Training Cycles Logic:**
```python
# Cycle 1: Basic training with current configurations
if not do_machine_training_single_cycle():
    return  # Exit on failure

# Cycle 2: ICI optimization
self._machine.db_machine.machine_is_re_run_ici = True
self._init_load_or_create_configuration(allow_re_run_configuration=True)
if not do_machine_training_single_cycle():
    return

# Cycle 3: FE optimization
self._machine.db_machine.machine_is_re_run_fe = True
self._init_load_or_create_configuration(allow_re_run_configuration=True)
if not do_machine_training_single_cycle():
    return

# Cycle 4: Final optimization cycle
self._init_load_or_create_configuration(allow_re_run_configuration=True)
if not do_machine_training_single_cycle():
    return

# Handle any remaining ReRun flags
if any rerun flags still set:
    # Additional cycle to clear remaining issues
    self._init_load_or_create_configuration(allow_re_run_configuration=True)
    if not do_machine_training_single_cycle():
        return

# Training completed successfully
self._save_nn_model_to_db()
self._machine.data_input_lines_mark_all_IsForLearning_as_IsLearned()
```

**What the function does and its purpose:**
- Provides complete, production-ready training pipeline
- Implements progressive optimization through multiple cycles
- Ensures robust error handling and recovery
- Maintains comprehensive training progress tracking
- Enables automated model deployment and management

#### `_do_one_training_cycle()`

**Where it's used and why:**
- Called internally by the training pipeline for individual training cycles
- Used to execute single training iterations with full error handling
- Critical for implementing the progressive training strategy
- Enables fine-grained control over training parameters and monitoring

**How the function works:**
1. **Resource Validation**: Checks model size and system constraints
2. **Data Preparation**: Loads and encodes training/validation datasets
3. **Model Building**: Constructs Keras model from configuration
4. **Training Execution**: Performs actual neural network training with callbacks
5. **Error Handling**: Manages various training failure scenarios
6. **Result Persistence**: Saves training metrics and model state
7. **Performance Tracking**: Records detailed training statistics

**Training Execution Process:**
```python
# Validate model constraints
total_weights_count = self._nn_configuration.nn_shape_instance.weight_total_count(
    self._nn_configuration.num_of_input_neurons,
    self._nn_configuration.num_of_output_neurons
)
if total_weights_count < NNENGINE_WEIGHT_COUNT_MIN or total_weights_count > NNENGINE_WEIGHT_COUNT_MAX:
    logger.warning(f"Model size {total_weights_count} outside recommended range")

# Prepare training data
input_data_for_training = self.dataframe_full_encode(
    self._machine.data_lines_read(IsForLearning=True, rows_count_limit=DEBUG_TRAINING_ROWS_COUNT_LIMIT)
)
input_data_for_validation = self.dataframe_full_encode(
    self._machine.data_lines_read(IsForEvaluation=True, rows_count_limit=DEBUG_TRAINING_ROWS_COUNT_LIMIT)
)

# Split into inputs and outputs
(input_train, output_train) = self._split_dataframe_into_input_and_output(
    input_data_for_training, dataframe_status=DataframeEncodingType.ENCODED_FOR_AI)
(input_val, output_val) = self._split_dataframe_into_input_and_output(
    input_data_for_validation, dataframe_status=DataframeEncodingType.ENCODED_FOR_AI)

# Compute optimal training parameters
training_batch_size = self._compute_optimal_batch_size(mode_slow=True)
first_training_epoch_count_max = self._compute_optimal_epoch_count(mode_slow=True)

# Build Keras model with different weight initializers for robustness
for attempt in range(NNENGINE_TRAINING_ATTEMPT_TRAINING_START_CORRECTLY_MAX):
    keras_weights_initializer = self._get_weight_initializer_for_attempt(attempt)

    neural_network_model_to_train = self._nn_configuration.build_keras_nn_model(
        self._machine.db_machine.parameter_nn_loss,
        force_weight_initializer=keras_weights_initializer
    )

    # Execute training with early stopping
    history = neural_network_model_to_train.fit(
        input_train, output_train,
        validation_data=(input_val, output_val),
        batch_size=training_batch_size,
        epochs=first_training_epoch_count_max,
        callbacks=[keras_earlystopping_fit_callback],
        verbose=0
    )

    # Check training success
    if self._is_training_successful(history, keras_earlystopping_fit_callback):
        # Training successful - save results
        self._nn_model = neural_network_model_to_train
        self._record_training_metrics(history, training_batch_size, first_training_epoch_count_max)
        return True
    else:
        # Training failed - try different initialization
        continue

# All attempts failed
self._nn_model = None
self._machine.db_machine.machine_is_re_run_fe = True  # Trigger FE rebuild
return False
```

**What the function does and its purpose:**
- Implements robust single-cycle training with multiple failure recovery strategies
- Provides comprehensive training monitoring and metrics collection
- Enables adaptive parameter selection based on data characteristics
- Supports multiple weight initialization strategies for training stability
- Maintains detailed training progress and error tracking

#### `do_training_trial(pre_encoded_dataframe, pre_encoded_validation_dataframe, encoded_for_ai_dataframe, encoded_for_ai_validation_dataframe)`

**Where it's used and why:**
- Called for rapid configuration testing and optimization
- Used by SolutionFinder and Experimenter for evaluating different neural network architectures
- Critical for automated hyperparameter optimization and configuration discovery
- Enables efficient exploration of configuration space without full training commitment

**How the function works:**
1. **Parameter Flexibility**: Accepts data in multiple encoding formats
2. **Configuration Validation**: Ensures all required components are available
3. **Model Size Checking**: Validates neural network complexity constraints
4. **Data Preparation**: Encodes data to neural network format
5. **Rapid Training**: Performs accelerated training with reduced epochs
6. **Performance Evaluation**: Returns loss, accuracy, and completion metrics

**Trial Training Process:**
```python
# Validate parameters and components
if pre_encoded_dataframe is not None and encoded_for_ai_dataframe is not None:
    logger.error("Cannot specify both encoded and pre-encoded data simultaneously")

if not self._nn_configuration:
    logger.error("NNConfiguration required for trial training")
if not self._mdc or not self._enc_dec:
    logger.error("MDC and EncDec required for data processing")

# Check model complexity
total_weights_count = self._nn_configuration.nn_shape_instance.weight_total_count(
    self._nn_configuration.num_of_input_neurons,
    self._nn_configuration.num_of_output_neurons)
if total_weights_count < NNENGINE_WEIGHT_COUNT_MIN or total_weights_count > NNENGINE_WEIGHT_COUNT_MAX:
    return None, None, None  # Model too complex or too simple

# Prepare data
if pre_encoded_dataframe is not None:
    encoded_for_ai_dataframe = self._enc_dec.encode_for_ai(pre_encoded_dataframe)
    encoded_for_ai_validation_dataframe = self._enc_dec.encode_for_ai(pre_encoded_validation_dataframe)

# Validate dataset sizes
if len(encoded_for_ai_dataframe) < 100:
    logger.warning("Training dataset smaller than recommended minimum")
if len(encoded_for_ai_validation_dataframe) < 25:
    logger.warning("Validation dataset smaller than recommended minimum")

# Split data
(input_training, output_training) = self._split_dataframe_into_input_and_output(
    encoded_for_ai_dataframe, DataframeEncodingType.ENCODED_FOR_AI)
(input_validation, output_validation) = self._split_dataframe_into_input_and_output(
    encoded_for_ai_validation_dataframe, DataframeEncodingType.ENCODED_FOR_AI)

# Compute trial parameters (faster than full training)
batch_size = self._compute_optimal_batch_size(mode_fast=True)
epochs_count_max = self._compute_optimal_epoch_count(mode_fast=True)

# Execute trial training
keras_earlystopping_fit_callback = EarlyStopping(
    monitor="val_loss", mode="min", patience=25, min_delta=0.0001)

neural_network_model = self._nn_configuration.build_keras_nn_model(
    self._machine.db_machine.parameter_nn_loss)

# Multiple attempts with different initializations
for fit_trial_attempt in range(NNENGINE_TRAINING_ATTEMPT_TRAINING_START_CORRECTLY_MAX):
    try:
        neural_network_model.fit(
            input_training, output_training,
            epochs=epochs_count_max,
            batch_size=batch_size,
            verbose=1 if ENABLE_LOGGER_DEBUG_NNEngine_DETAILED else 0,
            callbacks=[keras_earlystopping_fit_callback],
            validation_data=(input_validation, output_validation)
        )
    except (tferrors.InternalError, tferrors.ResourceExhaustedError) as e:
        return None, None, None  # Resource exhaustion
    except Exception as e:
        logger.error(f"Trial training failed: {e}")
        continue

    # Check if training completed sufficiently
    epoch_done = keras_earlystopping_fit_callback.stopped_epoch or epochs_count_max
    if epoch_done > 25:  # Minimum completion threshold
        # Evaluate final performance
        loss, accuracy = neural_network_model.evaluate(
            input_validation, output_validation, verbose=0)

        loss = float(loss) if not np.isnan(loss) else None
        accuracy = float(accuracy) if not np.isnan(accuracy) else None

        if ENABLE_LOGGER_DEBUG_NNEngine:
            scaled_loss = self._machine.scale_loss_to_user_loss(loss) if loss else None
            logger.debug(f"Trial training successful - Loss: {scaled_loss}, Accuracy: {accuracy}")

        return loss, accuracy, (epoch_done / epochs_count_max)

# All attempts failed
return None, None, None
```

**What the function does and its purpose:**
- Provides efficient configuration evaluation for optimization algorithms
- Enables rapid prototyping and testing of neural network architectures
- Supports automated hyperparameter optimization workflows
- Maintains resource efficiency through accelerated training protocols
- Returns standardized performance metrics for comparative analysis

### Prediction and Inference Functions

#### `do_solving_all_data_lines()`

**Where it's used and why:**
- Called when processing all pending prediction requests in the database
- Used by batch processing systems to handle accumulated prediction tasks
- Critical for high-throughput prediction scenarios
- Enables efficient processing of multiple prediction requests

**How the function works:**
1. **Data Retrieval**: Fetches all pending prediction requests from database
2. **Batch Processing**: Processes all requests in a single operation
3. **Result Storage**: Saves predictions back to database
4. **Status Updates**: Marks processed requests as completed

**What the function does and its purpose:**
- Supports batch prediction workflows
- Optimizes database operations through bulk processing
- Maintains prediction request lifecycle management
- Enables scalable prediction services

#### `do_solving_direct_encoded_for_ai(dataframe_to_solve_encoded_for_ai)`

**Where it's used and why:**
- Called when making predictions from pre-encoded data
- Used internally by the prediction pipeline for efficiency
- Critical for maintaining data format consistency in ML operations
- Enables direct neural network inference without re-encoding

**How the function works:**
1. **Input Validation**: Ensures correct data format and dimensions
2. **Model Loading**: Retrieves trained neural network model
3. **Prediction Execution**: Performs neural network inference
4. **Result Formatting**: Returns properly structured prediction results

**Prediction Process:**
```python
# Validate input dimensions
if len(dataframe_to_solve_encoded_for_ai.columns) != self._machine.db_machine.enc_dec_columns_info_input_encode_count:
    logger.error("Input data dimensions don't match model expectations")

# Load trained model
neuron_network_model = self._get_nn_model_from_db()

# Execute prediction
solved_data = neuron_network_model.predict(dataframe_to_solve_encoded_for_ai)

# Format results
solved_dataframe_encoded_for_ai = pd.DataFrame(
    solved_data,
    columns=self._machine.get_list_of_columns_name(
        ColumnDirectionType.OUTPUT,
        dataframe_status=DataframeEncodingType.ENCODED_FOR_AI),
    index=dataframe_to_solve_encoded_for_ai.index
)

return solved_dataframe_encoded_for_ai
```

**What the function does and its purpose:**
- Provides direct neural network prediction capabilities
- Maintains efficiency by avoiding redundant data transformations
- Ensures prediction result consistency and format
- Supports integration with broader prediction pipelines

#### `do_solving_direct_dataframe_user(dataframe_to_solve_user, disable_evaluation_loss_solving)`

**Where it's used and why:**
- Called when making predictions from raw user data
- Used by external applications requiring end-to-end prediction services
- Critical for providing user-friendly prediction interfaces
- Enables seamless integration of ML predictions into applications

**How the function works:**
1. **Data Encoding**: Converts user data through full encoding pipeline
2. **Neural Prediction**: Executes prediction on encoded data
3. **Result Decoding**: Converts predictions back to user-readable format
4. **Optional Loss Evaluation**: Can include prediction confidence metrics

**Complete Prediction Pipeline:**
```python
# Encode user data to neural network format
dataframe_to_solve_encoded_for_ai = self.dataframe_full_encode(dataframe_to_solve_user)

# Execute neural network prediction
solved_dataframe = self.do_solving_direct_encoded_for_ai(dataframe_to_solve_encoded_for_ai)

# Decode predictions to user format
decoded_solved_data_of_machine_source = self.dataframe_full_decode(solved_dataframe)

# Optional: Add loss/confidence evaluation
if not disable_evaluation_loss_solving and self._machine.db_machine.multimachines_Is_confidence_enabled:
    # Calculate prediction confidence using loss prediction machine
    loss_prediction_machine = Machine(self._machine.db_machine.multimachines_Id_loss_predictor_machine)
    nn_engine_loss = NNEngine(loss_prediction_machine, allow_re_run_configuration=False)

    # Prepare data for loss prediction
    loss_prediction_dataframe_to_solve = dataframe_to_solve_user.join(
        decoded_solved_data_of_machine_source, how="inner")

    # Predict losses
    solved_loss = nn_engine_loss.do_solving_direct_dataframe_user(
        loss_prediction_dataframe_to_solve, disable_evaluation_loss_solving=True)

    # Combine predictions with confidence scores
    decoded_solved_data_of_machine_source = decoded_solved_data_of_machine_source.join(
        solved_loss, how="inner")

return decoded_solved_data_of_machine_source
```

**What the function does and its purpose:**
- Provides complete end-to-end prediction pipeline
- Handles all data transformations automatically
- Supports optional confidence evaluation
- Enables seamless application integration

### Model Management Functions

#### `_get_nn_model_from_db()`

**Where it's used and why:**
- Called whenever trained neural network model is needed
- Used throughout prediction and evaluation operations
- Critical for maintaining model persistence and accessibility
- Enables efficient model loading and caching

**How the function works:**
1. **Cache Check**: Returns cached model if already loaded
2. **Database Retrieval**: Loads model weights from database storage
3. **Model Reconstruction**: Builds Keras model and applies stored weights
4. **Error Handling**: Manages loading failures gracefully

**Model Loading Process:**
```python
# Check if model already cached
if self._nn_model is not None:
    return self._nn_model

# Validate stored model exists
if self._machine.db_machine.training_nn_model_extfield is None:
    logger.error("No trained model found in database")

# Load and deserialize model weights
try:
    nn_model_pickles_bytes = self._machine.db_machine.training_nn_model_extfield
    nn_model_np_array = pickle.loads(nn_model_pickles_bytes)
except Exception as e:
    logger.error(f"Failed to deserialize model: {e}")

# Reconstruct Keras model architecture
neural_network_model = self._nn_configuration.build_keras_nn_model(
    self._machine.db_machine.parameter_nn_loss)

# Apply stored weights
try:
    neural_network_model.set_weights(nn_model_np_array)
except Exception as e:
    logger.error(f"Failed to apply model weights: {e}")

return neural_network_model
```

**What the function does and its purpose:**
- Provides efficient access to trained neural network models
- Manages model persistence and loading operations
- Supports model caching for performance optimization
- Ensures model integrity during storage and retrieval

#### `_save_nn_model_to_db()`

**Where it's used and why:**
- Called after successful training to persist the trained model
- Used to ensure model availability across system restarts
- Critical for maintaining trained model accessibility
- Enables model deployment and reuse

**How the function works:**
1. **Weight Extraction**: Gets model weights from Keras model
2. **Serialization**: Converts weights to storable format
3. **Database Storage**: Saves serialized model to machine database
4. **Logging**: Records successful model persistence

**Model Persistence Process:**
```python
# Extract model weights
nn_model_pickle_bytes = pickle.dumps(self._nn_model.get_weights())

# Store in database
self._machine.db_machine.training_nn_model_extfield = nn_model_pickle_bytes

# Log successful operation
if ENABLE_LOGGER_DEBUG_NNEngine:
    logger.debug(f"Model saved to database for {self._machine}")
```

**What the function does and its purpose:**
- Ensures trained models are preserved for future use
- Supports model deployment across system instances
- Maintains model availability for prediction services
- Enables model versioning and backup

#### `_delete_nn_model_from_db()`

**Where it's used and why:**
- Called when removing stored models (cleanup, replacement)
- Used during model rebuilding or system maintenance
- Critical for managing storage resources and model lifecycle
- Enables clean model replacement and updates

**How the function works:**
1. **Memory Cleanup**: Clears cached model reference
2. **Database Removal**: Deletes stored model from database
3. **Reference Management**: Ensures no dangling references

**What the function does and its purpose:**
- Manages model lifecycle and cleanup operations
- Optimizes storage resource usage
- Supports model replacement and updates
- Maintains database cleanliness

### Evaluation Functions

#### `do_evaluation()`

**Where it's used and why:**
- Called after training to assess model performance
- Used to generate comprehensive performance metrics
- Critical for model validation and quality assessment
- Enables performance monitoring and optimization

**How the function works:**
1. **Model Loading**: Retrieves trained neural network model
2. **Data Preparation**: Prepares evaluation datasets
3. **Column-wise Evaluation**: Computes per-column loss and accuracy
4. **Aggregate Metrics**: Calculates overall model performance
5. **Noise Robustness**: Tests model with noisy inputs
6. **Result Storage**: Saves evaluation metrics to database

**Comprehensive Evaluation Process:**
```python
# Load trained model
neural_network_model = self._get_nn_model_from_db()

# Prepare evaluation datasets
dataframe_pre_encoded_training = self._mdc.dataframe_pre_encode(
    self._machine.get_random_user_dataframe_for_training_trial(is_for_learning=True))
dataframe_pre_encoded_evaluation = self._mdc.dataframe_pre_encode(
    self._machine.get_random_user_dataframe_for_training_trial(is_for_evaluation=True))

# Evaluate per-column performance
losses_for_each_output_columns, accuracy_for_each_output_columns = \
    self.compute_loss_accuracy_of_pre_encoded_dataframe_for_each_columns(
        dataframe_pre_encoded_evaluation)

# Store column-wise results
self._machine.db_machine.training_eval_outputs_cols_loss_sample_evaluation = \
    losses_for_each_output_columns
self._machine.db_machine.training_eval_outputs_cols_accuracy_sample_evaluation = \
    accuracy_for_each_output_columns

# Evaluate aggregate performance on training data
encoded_training = self._enc_dec.encode_for_ai(dataframe_pre_encoded_training)
(input_train, output_train) = self._split_dataframe_into_input_and_output(
    encoded_training, DataframeEncodingType.ENCODED_FOR_AI)

loss_training, accuracy_training = neural_network_model.evaluate(
    input_train, output_train, verbose=0)
self._machine.db_machine.training_eval_loss_sample_training = \
    self._machine.scale_loss_to_user_loss(loss_training)
self._machine.db_machine.training_eval_accuracy_sample_training = accuracy_training

# Test robustness to input noise
input_train_noisy = self._add_noise_to_dataframe_encoded_for_ai(input_train)
loss_training_noisy, accuracy_training_noisy = neural_network_model.evaluate(
    input_train_noisy, output_train, verbose=0)
self._machine.db_machine.training_eval_loss_sample_training_noise = \
    self._machine.scale_loss_to_user_loss(loss_training_noisy)
self._machine.db_machine.training_eval_accuracy_sample_training_noise = accuracy_training_noisy

# Evaluate on evaluation data
encoded_evaluation = self._enc_dec.encode_for_ai(dataframe_pre_encoded_evaluation)
(input_eval, output_eval) = self._split_dataframe_into_input_and_output(
    encoded_evaluation, DataframeEncodingType.ENCODED_FOR_AI)

loss_evaluation, accuracy_evaluation = neural_network_model.evaluate(
    input_eval, output_eval, verbose=0)
self._machine.db_machine.training_eval_loss_sample_evaluation = \
    self._machine.scale_loss_to_user_loss(loss_evaluation)
self._machine.db_machine.training_eval_accuracy_sample_evaluation = accuracy_evaluation

# Test evaluation data with noise
input_eval_noisy = self._add_noise_to_dataframe_encoded_for_ai(input_eval)
loss_evaluation_noisy, accuracy_evaluation_noisy = neural_network_model.evaluate(
    input_eval_noisy, output_eval, verbose=0)
self._machine.db_machine.training_eval_loss_sample_evaluation_noise = \
    self._machine.scale_loss_to_user_loss(loss_evaluation_noisy)
self._machine.db_machine.training_eval_accuracy_sample_evaluation_noise = accuracy_evaluation_noisy
```

**What the function does and its purpose:**
- Provides comprehensive model performance assessment
- Evaluates both individual column and aggregate performance
- Tests model robustness to input variations
- Generates detailed metrics for model validation and monitoring
- Supports data-driven model improvement decisions

#### `compute_loss_accuracy_of_pre_encoded_dataframe_for_each_columns(dataframe_pre_encoded)`

**Where it's used and why:**
- Called during evaluation to assess per-column model performance
- Used for detailed performance analysis and debugging
- Critical for understanding model behavior across different outputs
- Enables targeted model improvement and feature engineering

**How the function works:**
1. **Data Encoding**: Prepares evaluation data for neural network
2. **Prediction Generation**: Creates model predictions for evaluation data
3. **Per-Column Analysis**: Computes loss and accuracy for each output column
4. **Result Aggregation**: Returns comprehensive performance metrics

**Per-Column Evaluation Process:**
```python
# Encode data for neural network
dataframe_outputs_encoded_for_ai = self._enc_dec.encode_for_ai(
    dataframe_pre_encoded[output_columns_pre_encoded])

# Generate predictions
predicted_outputs_encoded_for_ai = self.do_solving_direct_encoded_for_ai(
    self._enc_dec.encode_for_ai(dataframe_pre_encoded[input_columns_pre_encoded]))

# Initialize result containers
losses_for_each_output_columns = {}
accuracy_for_each_output_columns = {}

# Evaluate each output column
for current_output_column_pre_encoded in output_columns_pre_encoded:
    # Get encoded column names for this user column
    all_columns_encoded_for_ai = self._enc_dec.get_encoded_for_ai_columns_names_by_the_pre_encoded_column_name(
        current_output_column_pre_encoded)

    if all_columns_encoded_for_ai:
        # Compute loss using configured loss function
        loss_fn = get_loss_function(self._machine.db_machine.parameter_nn_loss)

        current_column_loss = np.nanmean(loss_fn(
            dataframe_outputs_encoded_for_ai[all_columns_encoded_for_ai],
            predicted_outputs_encoded_for_ai[all_columns_encoded_for_ai]
        ).numpy())

        if np.isnan(current_column_loss):
            current_column_loss = 0  # Handle edge cases

        # Scale loss to user-interpretable range
        losses_for_each_output_columns[current_output_column_pre_encoded] = \
            self._machine.scale_loss_to_user_loss(float(current_column_loss))

        # Compute accuracy
        acc = Accuracy()
        acc.update_state(
            dataframe_outputs_encoded_for_ai[all_columns_encoded_for_ai],
            predicted_outputs_encoded_for_ai[all_columns_encoded_for_ai]
        )
        accuracy_for_each_output_columns[current_output_column_pre_encoded] = \
            float(np.nanmean(acc.result().numpy()))

return losses_for_each_output_columns, accuracy_for_each_output_columns
```

**What the function does and its purpose:**
- Provides granular performance analysis by output column
- Enables identification of problematic predictions
- Supports targeted model improvement strategies
- Facilitates detailed performance reporting and monitoring

### Utility Functions

#### `_compute_loss_of_random_dataset(parameter_nn_loss)`

**Where it's used and why:**
- Called during EncDec configuration to establish loss scaling baseline
- Used to normalize loss values for user interpretation
- Critical for providing meaningful performance metrics
- Enables consistent loss reporting across different problem types

**How the function works:**
1. **Random Data Generation**: Creates randomized version of evaluation data
2. **Loss Computation**: Calculates loss between original and randomized data
3. **Maximum Loss Extraction**: Finds worst-case loss scenario
4. **Scaling Factor**: Returns value for loss normalization

**Loss Scaling Process:**
```python
# Get loss function
loss_fn = get_loss_function(parameter_nn_loss)

# Prepare evaluation dataset
dataframe_evaluation_loss_scaler_encoded_for_ai = self._enc_dec.encode_for_ai(
    self._mdc.dataframe_pre_encode(
        self._machine.get_random_user_dataframe_for_training_trial(is_for_evaluation=True)))

# Split into inputs and outputs
(input_loss_scaler, output_loss_scaler) = self._split_dataframe_into_input_and_output(
    dataframe_evaluation_loss_scaler_encoded_for_ai, DataframeEncodingType.ENCODED_FOR_AI)

# Create randomized outputs
output_loss_scaler_randomized = output_loss_scaler.copy()
for row_idx in range(output_loss_scaler_randomized.shape[0]):
    for col_idx in range(output_loss_scaler_randomized.shape[1]):
        output_loss_scaler_randomized.iloc[row_idx, col_idx] = \
            output_loss_scaler.iloc[int(random.uniform(0, output_loss_scaler_randomized.shape[0])), col_idx]

# Compute maximum loss (worst-case scenario)
loss_on_random_dataset = np.nanmax(loss_fn(
    output_loss_scaler, output_loss_scaler_randomized).numpy())

if np.isnan(loss_on_random_dataset) or loss_on_random_dataset == 0:
    logger.error("Unable to compute loss scaling factor")

return float(loss_on_random_dataset)
```

**What the function does and its purpose:**
- Establishes baseline for loss value interpretation
- Enables consistent performance reporting across models
- Supports user-friendly performance metrics
- Normalizes loss values for meaningful comparison

#### `_compute_optimal_batch_size(mode_slow, mode_fast)`

**Where it's used and why:**
- Called to determine optimal batch size for training efficiency
- Used to balance memory usage and training speed
- Critical for maximizing GPU utilization and training performance
- Enables adaptive training parameter selection

**How the function works:**
1. **Mode Selection**: Chooses batch size based on training speed requirements
2. **Fixed Size Selection**: Returns predefined optimal batch sizes

**Batch Size Selection Logic:**
```python
if mode_fast and mode_slow:
    logger.error("Cannot specify both fast and slow modes")
elif mode_fast:
    batch_size = 64  # Smaller batch for faster iteration
elif mode_slow:
    batch_size = 32  # Larger batch for stable convergence
else:
    logger.error("Must specify either fast or slow mode")

return int(batch_size)
```

**What the function does and its purpose:**
- Optimizes training efficiency based on operational requirements
- Balances computational speed with model convergence quality
- Supports different training scenarios (optimization vs. final training)

#### `_compute_optimal_epoch_count(mode_slow, mode_fast)`

**Where it's used and why:**
- Called to determine optimal maximum epoch count for training
- Used to set appropriate training duration limits
- Critical for preventing overfitting and managing training time
- Enables adaptive training duration based on requirements

**How the function works:**
1. **Mode Selection**: Chooses epoch count based on training speed requirements
2. **Duration Setting**: Returns appropriate maximum training epochs

**Epoch Count Selection Logic:**
```python
if mode_fast and mode_slow:
    logger.error("Cannot specify both fast and slow modes")
elif mode_fast:
    optimal_epoch_count = 100  # Limited epochs for quick evaluation
elif mode_slow:
    optimal_epoch_count = 100 * 100  # Extended training for convergence
else:
    logger.error("Must specify either fast or slow mode")

return int(optimal_epoch_count)
```

**What the function does and its purpose:**
- Sets appropriate training duration limits
- Prevents excessive training time in evaluation scenarios
- Allows sufficient training time for model convergence
- Supports different training objectives (speed vs. accuracy)

#### `_split_dataframe_into_input_and_output(dataframe_, dataframe_status)`

**Where it's used and why:**
- Called throughout the system to separate input and output data
- Used for preparing data for neural network training and inference
- Critical for maintaining proper data structure for ML operations
- Enables consistent data handling across different components

**How the function works:**
1. **Column Retrieval**: Gets input and output column names based on status
2. **Validation**: Ensures all required columns exist in dataframe
3. **Data Splitting**: Returns separate input and output dataframes

**Data Splitting Process:**
```python
# Validate input type
if not isinstance(dataframe_, pd.DataFrame):
    logger.error("Input must be pandas DataFrame")

# Get column names based on encoding status
input_columns = self._machine.get_list_of_columns_name(
    ColumnDirectionType.INPUT, dataframe_status)
output_columns = self._machine.get_list_of_columns_name(
    ColumnDirectionType.OUTPUT, dataframe_status)

# Validate column availability
if not all(col in dataframe_.columns for col in input_columns):
    logger.error(f"Missing input columns: {input_columns}")
if not all(col in dataframe_.columns for col in output_columns):
    logger.error(f"Missing output columns: {output_columns}")

# Return split dataframes
return dataframe_[input_columns], dataframe_[output_columns]
```

**What the function does and its purpose:**
- Provides standardized data splitting functionality
- Ensures data integrity for neural network operations
- Supports multiple data encoding formats
- Maintains consistency across the ML pipeline

#### `_add_noise_to_dataframe_encoded_for_ai(dataframe_encoded_for_ai, percentage_of_noise)`

**Where it's used and why:**
- Called during evaluation to test model robustness
- Used to assess model performance under input perturbations
- Critical for understanding model reliability and generalization
- Enables comprehensive model validation and quality assessment

**How the function works:**
1. **Noise Generation**: Creates random noise within specified percentage
2. **Data Modification**: Applies noise to input data while preserving bounds
3. **Format Preservation**: Maintains encoded data format and constraints

**Noise Addition Process:**
```python
# Create copy to avoid modifying original
dataframe_encoded_for_ai_with_noise = dataframe_encoded_for_ai.copy()

# Apply noise: value ± (value × percentage_of_noise / 2)
dataframe_encoded_for_ai_with_noise = np.clip(
    dataframe_encoded_for_ai_with_noise +
    np.random.rand(*dataframe_encoded_for_ai_with_noise.shape) *
    percentage_of_noise / 100 -
    percentage_of_noise / 200,
    0, 1  # Maintain encoded data bounds
)

return dataframe_encoded_for_ai_with_noise
```

**What the function does and its purpose:**
- Tests model robustness to input variations
- Provides insights into model generalization capabilities
- Supports comprehensive model validation
- Enables identification of brittle model behaviors

### Integration Points and Dependencies

#### With Machine Class
- **Data Access**: Retrieves training and evaluation data from machine storage
- **Configuration Management**: Updates machine configuration states and flags
- **Status Tracking**: Records training progress and error conditions
- **Result Storage**: Saves trained models and evaluation metrics

#### With MachineDataConfiguration (MDC)
- **Data Preprocessing**: Handles raw data cleaning and normalization
- **Format Conversion**: Transforms user data to ML-ready format
- **Schema Management**: Maintains column type and structure information
- **Data Validation**: Ensures data quality and consistency

#### With EncDec
- **Data Encoding**: Converts processed data to neural network format
- **Data Decoding**: Transforms neural network outputs to user format
- **Column Mapping**: Manages relationships between user and encoded columns
- **Type Preservation**: Maintains data type information through transformations

#### With NNConfiguration
- **Architecture Definition**: Provides neural network structure specifications
- **Model Building**: Constructs Keras models from configuration
- **Parameter Management**: Handles optimizer and loss function settings
- **Resource Estimation**: Provides model complexity and size information

#### With Training Components (ICI, FE)
- **Feature Selection**: Integrates with InputsColumnsImportance for feature optimization
- **Feature Engineering**: Coordinates with FeatureEngineeringConfiguration
- **Progressive Training**: Supports multi-cycle training with component optimization
- **Configuration Dependencies**: Manages inter-component rebuild triggers

#### With SolutionFinder and Experimenter
- **Configuration Optimization**: Supports automated architecture search
- **Performance Evaluation**: Provides trial training capabilities
- **Parameter Exploration**: Enables systematic hyperparameter optimization
- **Result Integration**: Feeds optimization results back to configuration

#### With WorkProcessor
- **Job Execution**: Serves as core execution engine for training jobs
- **Resource Management**: Handles computational resource allocation
- **Progress Monitoring**: Provides training status and logging
- **Error Recovery**: Implements robust error handling and recovery

### Performance Optimization Strategies

#### Memory Management
- **Lazy Loading**: Loads models and data only when needed
- **Caching**: Maintains frequently used data in memory
- **Resource Cleanup**: Clears unused data structures
- **Batch Processing**: Optimizes memory usage through efficient batching

#### Computational Efficiency
- **GPU Utilization**: Maximizes GPU resource usage during training
- **Parallel Processing**: Supports distributed training when available
- **Early Stopping**: Prevents unnecessary training iterations
- **Adaptive Parameters**: Adjusts batch sizes and learning rates dynamically

#### Data Processing Optimization
- **Format Consistency**: Maintains efficient data representations
- **Pipeline Efficiency**: Minimizes redundant data transformations
- **Caching Strategy**: Reuses computed transformations
- **Memory Mapping**: Handles large datasets efficiently

### Error Handling and Recovery

#### Training Failure Management
- **Multiple Initialization Attempts**: Tries different weight initializers
- **Resource Error Handling**: Manages GPU memory and computational limits
- **Configuration Fallbacks**: Falls back to simpler configurations on failure
- **Progress Preservation**: Saves partial results when possible

#### Data Validation
- **Format Checking**: Validates input data structure and content
- **Dimension Verification**: Ensures data compatibility with model expectations
- **Quality Assessment**: Checks for data integrity issues
- **Error Propagation**: Provides meaningful error messages for debugging

#### System Resilience
- **Graceful Degradation**: Continues operation with reduced functionality
- **Recovery Mechanisms**: Implements automatic retry and recovery strategies
- **State Preservation**: Maintains system state during error conditions
- **Logging and Monitoring**: Provides comprehensive error tracking and reporting

### Usage Patterns and Examples

#### Complete Training Workflow
```python
# Initialize NNEngine
nn_engine = NNEngine(machine=machine, allow_re_run_configuration=True)

# Execute full training pipeline
nn_engine.do_training_and_save()

# Training automatically:
# 1. Loads/configures all components (MDC, ICI, FE, EncDec, NNConfig)
# 2. Performs multi-cycle training with progressive optimization
# 3. Saves trained model and evaluation metrics
# 4. Updates machine status and configuration flags
```

#### Trial Training for Optimization
```python
# Prepare data for trial training
pre_encoded_data = nn_engine._mdc.dataframe_pre_encode(user_dataframe)
pre_encoded_validation = nn_engine._mdc.dataframe_pre_encode(validation_dataframe)

# Execute trial training
loss, accuracy, completion_pct = nn_engine.do_training_trial(
    pre_encoded_dataframe=pre_encoded_data,
    pre_encoded_validation_dataframe=pre_encoded_validation
)

# Use results for configuration optimization
if accuracy > best_accuracy:
    best_config = current_nn_configuration
```

#### Prediction Pipeline
```python
# Load trained NNEngine
nn_engine = NNEngine(machine=machine)

# Make predictions from raw user data
predictions = nn_engine.do_solving_direct_dataframe_user(user_input_data)

# Predictions automatically:
# 1. Encode user data through MDC and EncDec
# 2. Execute neural network inference
# 3. Decode results to user-readable format
# 4. Return formatted predictions
```

#### Batch Processing
```python
# Process all pending prediction requests
nn_engine.do_solving_all_data_lines()

# System automatically:
# 1. Retrieves all pending requests from database
# 2. Processes predictions in batch
# 3. Stores results back to database
# 4. Updates request status
```

#### Evaluation and Monitoring
```python
# Perform comprehensive model evaluation
nn_engine.do_evaluation()

# Generates and stores:
# - Per-column loss and accuracy metrics
# - Aggregate training and evaluation performance
# - Robustness to input noise
# - Comprehensive performance statistics
```

This detailed analysis demonstrates how NNEngine serves as the comprehensive neural network execution engine in the EasyAutoML.com system, orchestrating the complete machine learning lifecycle from data preparation through model deployment while maintaining robust error handling, performance optimization, and seamless integration with the broader AutoML ecosystem.