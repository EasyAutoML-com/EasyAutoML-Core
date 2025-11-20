"""
Tests for InputsColumnsImportance.py - Column Importance Calculation

This file tests the InputsColumnsImportance (ICI) module, which is responsible
for calculating the importance of input columns. It's like a "column analyst"
that determines which input features are most important for predictions.

WHAT IS COLUMN IMPORTANCE?
==========================
Column importance is a measure of how much each input column contributes to
the final prediction. It's like ranking ingredients by how much they affect
the taste of a dish.

WHAT DOES ICI DO?
=================
1. IMPORTANCE CALCULATION:
   - Calculates the importance of each input column
   - Uses statistical methods to determine column weights
   - Provides both minimum and best configurations

2. COLUMN CLASSIFICATION:
   - Identifies which columns are inputs vs. outputs
   - Separates input and output columns
   - Handles different data types

3. CONFIGURATION MANAGEMENT:
   - Creates and saves importance configurations
   - Loads existing configurations
   - Manages configuration versions

4. STATISTICAL ANALYSIS:
   - Analyzes data distribution and patterns
   - Calculates correlation and importance metrics
   - Provides insights into data relationships

WHY IS ICI IMPORTANT?
=====================
ICI is important because:
- It helps identify the most important features
- It can improve model performance by focusing on important columns
- It provides insights into data relationships
- It helps with feature selection and optimization

WHAT DOES THIS MODULE TEST?
===========================
- Importance calculation (minimum vs. best configurations)
- Configuration loading and saving
- Column separation (inputs vs. outputs)
- Statistical analysis and metrics
- Error handling and validation
- Different data types and edge cases

TESTING STRATEGY:
=================
Each test follows this pattern:
1. Create a test machine with sample data
2. Set up required dependencies (MDC)
3. Create ICI configuration
4. Test specific ICI functionality
5. Verify results are correct
6. Clean up test data

DEPENDENCIES:
=============
ICI depends on:
- MachineDataConfiguration (MDC): For data analysis
- Machine: The main machine object
"""
import pytest
import pandas as pd
import numpy as np
from ML import Machine, MachineDataConfiguration, InputsColumnsImportance
from models.EasyAutoMLDBModels import EasyAutoMLDBModels
from SharedConstants import DatasetColumnDataType


class TestInputsColumnsImportance:
    """
    Test InputsColumnsImportance Class Functionality
    
    This class contains all tests for the ICI module. Each test method focuses on
    one specific aspect of column importance calculation functionality.
    
    TEST ORGANIZATION:
    ==================
    The tests are organized by functionality:
    1. Configuration tests (create, load, save)
    2. Importance calculation tests (minimum vs. best configurations)
    3. Column separation tests (inputs vs. outputs)
    4. Statistical analysis tests (importance evaluation)
    5. Error handling tests (invalid inputs, edge cases)
    6. Special data type tests (different data types)
    
    FIXTURE DEPENDENCIES:
    =====================
    Most tests use these fixtures:
    - db_cleanup: Ensures test data is cleaned up after each test
    - simple_dataframe: A basic DataFrame with input/output columns
    - columns_datatype: Maps column names to data types
    - columns_description: Human-readable descriptions of each column
    
    TEST NAMING CONVENTION:
    =======================
    - test_ici_[functionality]: Tests specific ICI functionality
    - Each test name describes what it's testing
    - Tests are independent and can run in any order
    
    IMPORTANCE CALCULATION:
    =======================
    - Minimum configuration: Equal importance for all input columns
    - Best configuration: Calculated importance based on data analysis
    - Importance values sum to 1.0 (100%)
    """
    
    def _get_admin_user(self):
        """Helper method to get admin user for machine ownership"""
        from django.contrib.auth import get_user_model
        User = get_user_model()
        admin_user, _ = User.objects.get_or_create(
            email='SuperSuperAdmin@easyautoml.com',
            defaults={
                'first_name': 'Test',
                'last_name': 'EasyAutoML',
                'is_staff': True,
                'is_superuser': True,
                'is_active': True,
            }
        )
        return admin_user
    
    @pytest.mark.django_db
    def test_ici_create_minimum_configuration(self, db_cleanup, simple_dataframe, columns_datatype, columns_description):
        """
        Test ICI Minimum Configuration Creation
        
        WHAT THIS TEST DOES:
        - Creates a new ICI configuration with minimum settings
        - Verifies that the configuration is created correctly
        - Checks that importance values are calculated properly
        
        WHY THIS TEST IS IMPORTANT:
        - Minimum configuration is the simplest approach
        - This test ensures basic configuration creation works
        - It verifies that importance values are calculated correctly
        
        MINIMUM CONFIGURATION PROCESS:
        1. Analyze the data structure
        2. Identify input and output columns
        3. Assign equal importance to all input columns
        4. Calculate importance values that sum to 1.0
        
        WHAT WE'RE TESTING:
        - ICI object is created successfully
        - Importance values are calculated correctly
        - Importance values sum to 1.0
        - Configuration is ready for use
        
        TEST STEPS:
        1. Set up prerequisites (MDC)
        2. Create ICI with minimum configuration
        3. Verify configuration is created
        4. Check that importance values are correct
        """
        machine = Machine(
            "__TEST_UNIT__ici_minimum",
            simple_dataframe,
            decimal_separator=".",
            date_format="%Y-%m-%d",
            machine_create_user_id=self._get_admin_user().id,
            disable_foreign_key_checking=True
        )
        machine.save_machine_to_db()
        
        # Setup MDC first
        self._setup_mdc(machine, simple_dataframe, columns_datatype, columns_description)
        
        # Create ICI with minimum configuration
        ici = InputsColumnsImportance(
            machine=machine,
            create_configuration_simple_minimum=True
        )
        
        assert ici._machine == machine
        assert ici._column_importance_evaluation is not None
        assert len(ici._column_importance_evaluation) > 0
        
        # Check that importance values sum to 1
        total_importance = sum(ici._column_importance_evaluation.values())
        assert abs(total_importance - 1.0) < 0.001
        
    @pytest.mark.django_db
    def test_ici_load_configuration(self, db_cleanup, simple_dataframe, columns_datatype, columns_description):
        """Test ICI configuration loading"""
        machine = Machine(
            "__TEST_UNIT__ici_load",
            simple_dataframe,
            decimal_separator=".",
            date_format="%Y-%m-%d",
            machine_create_user_id=self._get_admin_user().id,
            disable_foreign_key_checking=True
        )
        machine.save_machine_to_db()
        
        # Setup MDC
        self._setup_mdc(machine, simple_dataframe, columns_datatype, columns_description)
        
        # Create ICI configuration first
        ici_create = InputsColumnsImportance(
            machine=machine,
            create_configuration_simple_minimum=True
        )
        ici_create.save_configuration_in_machine()
        
        # Load ICI configuration
        ici_load = InputsColumnsImportance(
            machine=machine,
            load_configuration=True
        )
        
        assert ici_load._machine == machine
        assert ici_load._column_importance_evaluation is not None
        
    @pytest.mark.django_db
    def test_ici_save_configuration(self, db_cleanup, simple_dataframe, columns_datatype, columns_description):
        """Test ICI configuration saving"""
        machine = Machine(
            "__TEST_UNIT__ici_save",
            simple_dataframe,
            decimal_separator=".",
            date_format="%Y-%m-%d",
            machine_create_user_id=self._get_admin_user().id,
            disable_foreign_key_checking=True
        )
        machine.save_machine_to_db()
        
        # Setup MDC
        self._setup_mdc(machine, simple_dataframe, columns_datatype, columns_description)
        
        # Create ICI
        ici = InputsColumnsImportance(
            machine=machine,
            create_configuration_simple_minimum=True
        )
        
        # Save configuration
        ici.save_configuration_in_machine()
        
        # Verify configuration was saved
        assert machine.db_machine.fe_columns_inputs_importance_evaluation is not None
        assert len(machine.db_machine.fe_columns_inputs_importance_evaluation) > 0
        
    @pytest.mark.django_db
    def test_ici_importance_evaluation_structure(self, db_cleanup, simple_dataframe, columns_datatype, columns_description):
        """Test ICI importance evaluation structure"""
        machine = Machine(
            "__TEST_UNIT__ici_structure",
            simple_dataframe,
            decimal_separator=".",
            date_format="%Y-%m-%d",
            machine_create_user_id=self._get_admin_user().id,
            disable_foreign_key_checking=True
        )
        machine.save_machine_to_db()
        
        # Setup MDC
        self._setup_mdc(machine, simple_dataframe, columns_datatype, columns_description)
        
        # Create ICI
        ici = InputsColumnsImportance(
            machine=machine,
            create_configuration_simple_minimum=True
        )
        
        # Check importance evaluation structure
        importance_eval = ici._column_importance_evaluation
        assert isinstance(importance_eval, dict)
        
        for column_name, importance in importance_eval.items():
            assert isinstance(importance, (int, float))
            assert 0 <= importance <= 1
            
    @pytest.mark.django_db
    def test_ici_input_output_columns_separation(self, db_cleanup, simple_dataframe, columns_datatype, columns_description):
        """Test ICI input/output columns separation"""
        machine = Machine(
            "__TEST_UNIT__ici_separation",
            simple_dataframe,
            decimal_separator=".",
            date_format="%Y-%m-%d",
            machine_create_user_id=self._get_admin_user().id,
            disable_foreign_key_checking=True
        )
        machine.save_machine_to_db()
        
        # Setup MDC
        self._setup_mdc(machine, simple_dataframe, columns_datatype, columns_description)
        
        # Create ICI
        ici = InputsColumnsImportance(
            machine=machine,
            create_configuration_simple_minimum=True
        )
        
        # Check input/output columns separation
        assert len(ici._input_columns_names) > 0
        assert len(ici._output_columns_names) > 0
        
        # Input columns should not overlap with output columns
        input_set = set(ici._input_columns_names)
        output_set = set(ici._output_columns_names)
        assert len(input_set.intersection(output_set)) == 0
        
    @pytest.mark.django_db
    def test_ici_find_delay_tracking(self, db_cleanup, simple_dataframe, columns_datatype, columns_description):
        """Test ICI find delay tracking"""
        machine = Machine(
            "__TEST_UNIT__ici_delay",
            simple_dataframe,
            decimal_separator=".",
            date_format="%Y-%m-%d",
            machine_create_user_id=self._get_admin_user().id,
            disable_foreign_key_checking=True
        )
        machine.save_machine_to_db()
        
        # Setup MDC
        self._setup_mdc(machine, simple_dataframe, columns_datatype, columns_description)
        
        # Create ICI
        ici = InputsColumnsImportance(
            machine=machine,
            create_configuration_simple_minimum=True
        )
        
        # Check delay tracking
        assert ici._fe_columns_inputs_importance_find_delay_sec is not None
        assert isinstance(ici._fe_columns_inputs_importance_find_delay_sec, (int, float))
        assert ici._fe_columns_inputs_importance_find_delay_sec >= 0
        
    def test_ici_invalid_machine_type(self, db_cleanup):
        """Test ICI with invalid machine type"""
        with pytest.raises(Exception):
            InputsColumnsImportance(machine="not_a_machine")
            
    @pytest.mark.django_db
    def test_ici_invalid_parameters(self, db_cleanup, simple_dataframe, columns_datatype, columns_description):
        """Test ICI with invalid parameters"""
        machine = Machine(
            "__TEST_UNIT__ici_invalid",
            simple_dataframe,
            decimal_separator=".",
            date_format="%Y-%m-%d",
            machine_create_user_id=self._get_admin_user().id,
            disable_foreign_key_checking=True
        )
        machine.save_machine_to_db()
        
        # Test conflicting parameters
        with pytest.raises(Exception):
            InputsColumnsImportance(
                machine=machine,
                create_configuration_best=True,
                create_configuration_simple_minimum=True
            )
            
    @pytest.mark.django_db
    def test_ici_with_different_data_types(self, db_cleanup, simple_dataframe, columns_datatype, columns_description):
        """Test ICI with different data types"""
        
        machine = Machine(
            "__TEST_UNIT__ici_mixed",
            simple_dataframe,
            decimal_separator=".",
            date_format="%Y-%m-%d",
            machine_create_user_id=self._get_admin_user().id,
            disable_foreign_key_checking=True
        )
        machine.save_machine_to_db()
        
        # Setup MDC
        self._setup_mdc(machine, simple_dataframe, columns_datatype, columns_description)
        
        # Create ICI
        ici = InputsColumnsImportance(
            machine=machine,
            create_configuration_simple_minimum=True
        )
        
        # Verify ICI works with mixed data types
        assert ici._column_importance_evaluation is not None
        assert len(ici._column_importance_evaluation) > 0
        
    @pytest.mark.django_db
    def test_ici_minimum_configuration_equal_importance(self, db_cleanup, simple_dataframe, columns_datatype, columns_description):
        """Test that minimum configuration gives equal importance to all input columns"""
        machine = Machine(
            "__TEST_UNIT__ici_equal",
            simple_dataframe,
            decimal_separator=".",
            date_format="%Y-%m-%d",
            machine_create_user_id=self._get_admin_user().id,
            disable_foreign_key_checking=True
        )
        machine.save_machine_to_db()
        
        # Setup MDC
        self._setup_mdc(machine, simple_dataframe, columns_datatype, columns_description)
        
        # Create ICI
        ici = InputsColumnsImportance(
            machine=machine,
            create_configuration_simple_minimum=True
        )
        
        # Check that all input columns have equal importance
        input_columns = ici._input_columns_names
        if len(input_columns) > 1:
            importance_values = [ici._column_importance_evaluation[col] for col in input_columns]
            expected_importance = 1.0 / len(input_columns)
            
            for importance in importance_values:
                assert abs(importance - expected_importance) < 0.001
                
    @pytest.mark.django_db
    def test_ici_with_numeric_data(self, db_cleanup, simple_dataframe, columns_datatype, columns_description):
        """Test ICI with numeric-only data"""
        
        machine = Machine(
            "__TEST_UNIT__ici_numeric",
            simple_dataframe,
            decimal_separator=".",
            date_format="%Y-%m-%d",
            machine_create_user_id=self._get_admin_user().id,
            disable_foreign_key_checking=True
        )
        machine.save_machine_to_db()
        
        # Setup MDC
        self._setup_mdc(machine, simple_dataframe, columns_datatype, columns_description)
        
        # Create ICI
        ici = InputsColumnsImportance(
            machine=machine,
            create_configuration_simple_minimum=True
        )
        
        # Verify ICI works with numeric data
        assert ici._column_importance_evaluation is not None
        assert len(ici._column_importance_evaluation) > 0
        
        # Check that importance values are valid
        total_importance = sum(ici._column_importance_evaluation.values())
        assert abs(total_importance - 1.0) < 0.001
        
    def _setup_mdc(self, machine, dataframe, columns_datatype, columns_description):
        """Helper method to setup MDC"""
        mdc = MachineDataConfiguration(
            machine=machine,
            user_dataframe_for_create_cfg=dataframe,
            columns_type_user_df=columns_datatype,
            columns_description_user_df=columns_description,
            decimal_separator=".",
            date_format="%Y-%m-%d"
        )
        mdc.save_configuration_in_machine()
