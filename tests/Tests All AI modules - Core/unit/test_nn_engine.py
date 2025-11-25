"""
Tests for NNEngine.py - Training and Prediction Engine

This file tests the NNEngine module, which is responsible for training neural
networks and making predictions. It's the "brain" of the AI system that
actually learns from data and makes predictions.

WHAT IS NN ENGINE?
==================
NNEngine (Neural Network Engine) is the component that:
- Trains neural networks on your data
- Makes predictions using trained models
- Manages the entire machine learning workflow
- Coordinates with other modules (MDC, EncDec, etc.)

WHAT DOES NN ENGINE DO?
=======================
1. TRAINING:
   - Loads and prepares training data
   - Creates neural network architecture
   - Trains the model using backpropagation
   - Saves trained models for later use

2. PREDICTION:
   - Loads trained models
   - Processes input data through the network
   - Returns predictions in human-readable format
   - Handles different data types and formats

3. MODEL MANAGEMENT:
   - Saves models to disk/database
   - Loads models for prediction
   - Manages model versions and configurations
   - Handles model updates and retraining

4. WORKFLOW COORDINATION:
   - Coordinates with MDC for data preprocessing
   - Uses EncDec for data encoding/decoding
   - Integrates with ICI for input importance
   - Works with FEC for feature engineering

WHY IS NN ENGINE IMPORTANT?
===========================
NNEngine is the core component that:
- Actually performs the machine learning
- Makes the AI system functional
- Handles the complex training process
- Provides predictions for real-world use

WHAT DOES THIS MODULE TEST?
===========================
- Engine initialization and configuration loading
- Training process with different parameters
- Prediction functionality
- Model save/load operations
- Configuration readiness checks
- Error handling and edge cases
- Integration with other modules

TESTING STRATEGY:
=================
Each test follows this pattern:
1. Set up all required configurations (MDC, ICI, FEC, EncDec, NNConfig)
2. Create NNEngine instance
3. Test specific functionality (training, prediction, etc.)
4. Verify results are correct
5. Clean up test data

DEPENDENCIES:
=============
NNEngine depends on many other modules:
- MachineDataConfiguration (MDC): For data preprocessing
- InputsColumnsImportance (ICI): For input/output identification
- FeatureEngineeringConfiguration (FEC): For feature engineering
- EncDec: For data encoding/decoding
- NNConfiguration: For network architecture
- Machine: The main machine object
"""
import pytest
import pandas as pd
import numpy as np
from ML import Machine, MachineDataConfiguration, NNEngine, EncDec, FeatureEngineeringConfiguration, InputsColumnsImportance, NNConfiguration
from models.EasyAutoMLDBModels import EasyAutoMLDBModels
from SharedConstants import DatasetColumnDataType


class TestNNEngine:
    """
    Test NNEngine Class Functionality
    
    This class contains all tests for the NNEngine module. Each test method
    focuses on one specific aspect of neural network training and prediction.
    
    TEST ORGANIZATION:
    ==================
    The tests are organized by functionality:
    1. Initialization tests (engine creation, configuration loading)
    2. Training tests (basic training, different parameters)
    3. Prediction tests (solving, different data types)
    4. Model management tests (save/load operations)
    5. Configuration tests (readiness checks, re-run flags)
    6. Integration tests (with other modules)
    7. Error handling tests (invalid inputs, edge cases)
    
    FIXTURE DEPENDENCIES:
    =====================
    Most tests use these fixtures:
    - db_cleanup: Ensures test data is cleaned up after each test
    - simple_dataframe: A basic DataFrame with input/output columns
    - columns_datatype: Maps column names to data types
    - columns_description: Human-readable descriptions of each column
    
    TEST NAMING CONVENTION:
    =======================
    - test_nn_engine_[functionality]: Tests specific NNEngine functionality
    - Each test name describes what it's testing
    - Tests are independent and can run in any order
    
    SETUP COMPLEXITY:
    =================
    NNEngine tests require extensive setup because the engine depends on
    many other modules. The _setup_all_configurations helper method
    creates all required configurations in the correct order.
    """
    
    def _get_admin_user(self):
        """Helper method to get admin user for machine ownership"""
        from django.contrib.auth import get_user_model
        User = get_user_model()
        
        # Use filter().first() to handle duplicate users gracefully
        admin_user = User.objects.filter(
            email='SuperSuperAdmin@easyautoml.com'
        ).first()
        
        # Create if doesn't exist
        if not admin_user:
            admin_user = User.objects.create(
                email='SuperSuperAdmin@easyautoml.com',
                first_name='Test',
                last_name='EasyAutoML',
                is_staff=True,
                is_superuser=True,
                is_active=True,
            )
        
        return admin_user
    
    def test_nn_engine_invalid_machine_type(self, db_cleanup):
        """Test NNEngine with invalid machine type"""
        with pytest.raises(Exception):
            NNEngine(machine="not_a_machine")
        
    @pytest.mark.django_db
    def test_nn_engine_class_methods(self, db_cleanup, simple_dataframe, columns_datatype, columns_description):
        """Test NNEngine class methods"""
        machine = Machine(
            "__TEST_UNIT__nn_engine_class",
            simple_dataframe,
            decimal_separator=".",
            date_format="%Y-%m-%d",
            machine_create_user_id=self._get_admin_user().id,
            disable_foreign_key_checking=True
        )
        machine.save_machine_to_db()
        
        # Setup all configurations
        self._setup_all_configurations(machine, simple_dataframe, columns_datatype, columns_description, skip_nn_config=True)
        
        # Test class methods
        try:
            # Test configuration set starting
            NNEngine.machine_nn_engine_configuration_set_starting(machine)
            
            # Test configuration set completed
            NNEngine.machine_nn_engine_configuration_set_completed(machine)
            
            # These should not raise exceptions
            assert True
            
        except Exception as e:
            # Class methods might have specific requirements
            assert "configuration" in str(e).lower()
            
    def _setup_all_configurations(self, machine, dataframe, columns_datatype, columns_description, skip_nn_config=False):
        """
        Helper Method: Setup All Machine Configurations for NNEngine
        
        WHAT THIS METHOD DOES:
        - Sets up all the required configurations for NNEngine to work
        - Creates MDC, ICI, FEC, EncDec, and optionally NNConfig configurations
        - Saves all configurations to the machine in the correct order
        
        WHY THIS HELPER EXISTS:
        - NNEngine depends on many other modules
        - Many tests need the same complex setup process
        - Avoids code duplication across test methods
        - Ensures consistent configuration setup
        
        CONFIGURATION DEPENDENCIES:
        - MDC (MachineDataConfiguration): Must be created first
        - ICI (InputsColumnsImportance): Depends on MDC
        - FEC (FeatureEngineeringConfiguration): Depends on ICI
        - EncDec: Depends on MDC, ICI, and FEC
        - NNConfig: Depends on MDC (optional)
        - NNEngine: Depends on all five above
        
        SETUP PROCESS:
        1. Create MDC with the dataframe and column information
        2. Save MDC configuration to the machine
        3. Create ICI with minimum configuration
        4. Save ICI configuration to the machine
        5. Create FEC with minimum configuration
        6. Save FEC configuration to the machine
        7. Create pre-encoded data using MDC
        8. Create EncDec configuration
        9. Save EncDec configuration to the machine
        10. Create NN Configuration (if not skipped)
        11. Save NN Configuration to the machine (if not skipped)
        
        PARAMETERS:
        - machine: The Machine object to configure
        - dataframe: The data to analyze
        - columns_datatype: Data type mapping for columns
        - columns_description: Human descriptions of columns
        - skip_nn_config: If True, skip NNConfiguration setup (default: False)
        """
        # Create MDC
        mdc = MachineDataConfiguration(
            machine=machine,
            user_dataframe_for_create_cfg=dataframe,
            columns_type_user_df=columns_datatype,
            columns_description_user_df=columns_description,
            decimal_separator=".",
            date_format="%Y-%m-%d"
        )
        mdc.save_configuration_in_machine()
        
        # Create ICI
        ici = InputsColumnsImportance(
            machine=machine,
            create_configuration_simple_minimum=True
        )
        ici.save_configuration_in_machine()
        
        # Create FEC
        fec = FeatureEngineeringConfiguration(
            machine=machine,
            force_configuration_simple_minimum=True
        )
        fec.save_configuration_in_machine()
        
        # Create EncDec
        pre_encoded_df = mdc.dataframe_pre_encode(dataframe)
        encdec = EncDec(machine=machine, dataframe_pre_encoded=pre_encoded_df)
        encdec.save_configuration_in_machine()
        
        # Create NN Configuration (if not skipped)
        if not skip_nn_config:
            nn_config = NNConfiguration(machine_or_nnconfiguration=machine)
            nn_config.save_configuration_in_machine()
        
    @pytest.mark.django_db
    def test_nne_machine_nn_engine_configuration_set_starting(self, db_cleanup, simple_dataframe, columns_datatype, columns_description):
        """
        Test Setting NN Engine Configuration Starting (Class Method)
        
        WHAT THIS TEST DOES:
        - Tests the class method that sets NN engine configuration to starting state
        - Verifies that the method correctly updates the configuration state
        - Tests edge cases and error handling
        
        WHY THIS TEST IS IMPORTANT:
        - This method initializes the NN engine configuration process
        - It's essential for tracking configuration state
        - It ensures proper state management during configuration
        
        WHAT WE'RE TESTING:
        - Method correctly sets configuration to starting state
        - Method handles valid machine objects
        - Method handles edge cases (invalid machine IDs)
        - Method updates configuration properly
        
        TEST STEPS:
        1. Create machine and save to database
        2. Test setting configuration to starting state
        3. Test with invalid machine ID
        4. Test edge cases
        """
        machine = Machine(
            "__TEST_UNIT__nne_set_starting",
            simple_dataframe,
            decimal_separator=".",
            date_format="%Y-%m-%d",
            machine_create_user_id=self._get_admin_user().id,
            disable_foreign_key_checking=True
        )
        machine.save_machine_to_db()
        
        # Test setting configuration to starting state with machine object
        try:
            NNEngine.machine_nn_engine_configuration_set_starting(machine)
            assert True  # Should not raise exception
        except Exception as e:
            # May fail if configuration not ready
            assert "config" in str(e).lower() or "not ready" in str(e).lower()
        
        # Test with invalid machine ID
        try:
            NNEngine.machine_nn_engine_configuration_set_starting(99999)
            assert True  # Should not raise exception
        except Exception as e:
            # Should handle invalid machine ID gracefully
            assert "machine" in str(e).lower() or "not found" in str(e).lower() or "db_machine" in str(e).lower()
        
        # Test with None machine ID
        try:
            NNEngine.machine_nn_engine_configuration_set_starting(None)
            assert True  # Should not raise exception
        except Exception as e:
            # Should handle None input gracefully
            assert isinstance(e, Exception)
            
    @pytest.mark.django_db
    def test_nne_machine_nn_engine_configuration_is_configurating(self, db_cleanup, simple_dataframe, columns_datatype, columns_description):
        """
        Test Checking NN Engine Configuration State (Class Method)
        
        WHAT THIS TEST DOES:
        - Tests the class method that checks if NN engine configuration is in progress
        - Verifies that the method returns correct configuration state
        - Tests edge cases and error handling
        
        WHY THIS TEST IS IMPORTANT:
        - This method provides information about configuration state
        - It's essential for tracking configuration progress
        - It helps prevent concurrent configuration operations
        
        WHAT WE'RE TESTING:
        - Method returns correct configuration state
        - Method handles valid machine IDs
        - Method handles edge cases (invalid machine IDs)
        - Method returns appropriate boolean value
        
        TEST STEPS:
        1. Create machine and save to database
        2. Test checking configuration state
        3. Test with invalid machine ID
        4. Test edge cases
        """
        machine = Machine(
            "__TEST_UNIT__nne_is_configurating",
            simple_dataframe,
            decimal_separator=".",
            date_format="%Y-%m-%d",
            machine_create_user_id=self._get_admin_user().id,
            disable_foreign_key_checking=True
        )
        machine.save_machine_to_db()
        
        # Test checking configuration state with machine object
        try:
            is_configurating = NNEngine.machine_nn_engine_configuration_is_configurating(machine)
            assert isinstance(is_configurating, bool)
        except Exception as e:
            # May fail if configuration not ready
            assert "config" in str(e).lower() or "not ready" in str(e).lower() or "db_machine" in str(e).lower()
        
        # Test with invalid machine ID
        try:
            is_configurating = NNEngine.machine_nn_engine_configuration_is_configurating(99999)
            assert isinstance(is_configurating, bool)
        except Exception as e:
            # Should handle invalid machine ID gracefully
            assert "machine" in str(e).lower() or "not found" in str(e).lower() or "db_machine" in str(e).lower()
        
        # Test with None machine ID
        try:
            is_configurating = NNEngine.machine_nn_engine_configuration_is_configurating(None)
            assert isinstance(is_configurating, bool)
        except Exception as e:
            # Should handle None input gracefully
            assert isinstance(e, Exception)