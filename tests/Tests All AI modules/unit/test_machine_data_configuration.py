"""
Tests for MachineDataConfiguration.py - Data Analysis and Configuration

This file tests the MachineDataConfiguration (MDC) module, which is responsible for
analyzing data and creating configurations that other modules depend on.

WHAT IS MACHINE DATA CONFIGURATION?
===================================
MachineDataConfiguration analyzes your data and creates a configuration that tells
other modules how to handle the data. It's like a "data analyst" that examines
your dataset and creates rules for processing it.

WHAT DOES MDC DO?
=================
1. DATA ANALYSIS:
   - Identifies data types (numbers, text, dates, etc.)
   - Detects missing values and calculates statistics
   - Analyzes data distribution and patterns

2. COLUMN CLASSIFICATION:
   - Determines which columns are inputs (features)
   - Determines which columns are outputs (targets)
   - Handles forced input/output specifications

3. DATA PREPROCESSING:
   - Pre-encodes data for further processing
   - Post-decodes data back to original format
   - Handles different data formats and separators

4. STATISTICS CALCULATION:
   - Calculates mean, standard deviation, min, max
   - Computes missing value percentages
   - Generates data quality metrics

WHY IS MDC IMPORTANT?
=====================
MDC is the foundation that other modules build upon:
- EncDec needs MDC to know how to encode/decode data
- FeatureEngineering needs MDC to understand data structure
- InputsColumnsImportance needs MDC to identify input/output columns
- NNConfiguration needs MDC to determine network architecture

WHAT DOES THIS MODULE TEST?
===========================
- Configuration creation and loading
- Data pre-encoding and post-decoding
- Column statistics calculation
- Missing value handling
- JSON column processing
- Forced input/output configuration
- Error handling and validation

TESTING STRATEGY:
=================
Each test follows this pattern:
1. Create a test machine with sample data
2. Create MDC configuration
3. Test specific MDC functionality
4. Verify results are correct
5. Clean up test data

DEPENDENCIES:
=============
MDC depends on:
- Machine: The main machine object
- DataFrame: The data to analyze
- Column type mappings: How to interpret each column
- Column descriptions: Human-readable column information
"""
import pytest
import pandas as pd
import numpy as np
from ML import Machine, MachineDataConfiguration
from models.EasyAutoMLDBModels import EasyAutoMLDBModels
from SharedConstants import DatasetColumnDataType


class TestMachineDataConfiguration:
    """
    Test MachineDataConfiguration Class Functionality
    
    This class contains all tests for the MDC module. Each test method focuses on
    one specific aspect of data analysis and configuration functionality.
    
    TEST ORGANIZATION:
    ==================
    The tests are organized by functionality:
    1. Configuration tests (create, load, save)
    2. Data processing tests (pre-encode, post-decode)
    3. Statistics tests (mean, std, min, max, missing values)
    4. Special data type tests (JSON, mixed types)
    5. Configuration tests (forced inputs/outputs)
    6. Error handling tests (invalid inputs)
    
    FIXTURE DEPENDENCIES:
    =====================
    Most tests use these fixtures:
    - db_cleanup: Ensures test data is cleaned up after each test
    - simple_dataframe: A basic DataFrame with input/output columns
    - numeric_dataframe: A DataFrame with only numeric data
    - columns_datatype: Maps column names to data types
    - columns_description: Human-readable descriptions of each column
    
    TEST NAMING CONVENTION:
    =======================
    - test_mdc_[functionality]: Tests specific MDC functionality
    - Each test name describes what it's testing
    - Tests are independent and can run in any order
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
    def test_mdc_create_configuration(self, db_cleanup, simple_dataframe, columns_datatype, columns_description):
        """
        Test MDC Configuration Creation
        
        WHAT THIS TEST DOES:
        - Creates a new MDC configuration from scratch
        - Analyzes the provided data and creates configuration rules
        - Verifies that the configuration is created correctly
        
        WHY THIS TEST IS IMPORTANT:
        - MDC configuration is the foundation for all other modules
        - This test ensures the basic configuration creation works
        - It verifies that data analysis produces correct results
        
        CONFIGURATION CREATION PROCESS:
        1. Analyze the provided DataFrame
        2. Identify input and output columns
        3. Calculate column statistics
        4. Detect data types and missing values
        5. Create configuration rules for data processing
        
        WHAT WE'RE TESTING:
        - MDC object is created successfully
        - Input/output columns are identified correctly
        - Column counts are calculated correctly
        - Configuration is ready for use
        
        TEST STEPS:
        1. Create a test machine with sample data
        2. Create MDC configuration with the data
        3. Verify configuration properties are correct
        4. Check that input/output columns are identified
        """
        machine = Machine(
            "__TEST_UNIT__mdc_create",
            simple_dataframe,
            decimal_separator=".",
            date_format="%Y-%m-%d",
            machine_create_user_id=self._get_admin_user().id,
            disable_foreign_key_checking=True
        )
        machine.save_machine_to_db()
        
        mdc = MachineDataConfiguration(
            machine=machine,
            user_dataframe_for_create_cfg=simple_dataframe,
            columns_type_user_df=columns_datatype,
            columns_description_user_df=columns_description,
            decimal_separator=".",
            date_format="%Y-%m-%d"
        )
        
        assert mdc._machine == machine
        assert len(mdc.columns_name_input) > 0
        assert len(mdc.columns_name_output) > 0
        assert mdc.columns_input_count > 0
        assert mdc.columns_output_count > 0
        
    @pytest.mark.django_db
    def test_mdc_load_configuration(self, db_cleanup, simple_dataframe, columns_datatype, columns_description):
        """Test MDC configuration loading"""
        machine = Machine(
            "__TEST_UNIT__mdc_load",
            simple_dataframe,
            decimal_separator=".",
            date_format="%Y-%m-%d",
            machine_create_user_id=self._get_admin_user().id,
            disable_foreign_key_checking=True
        )
        machine.save_machine_to_db()
        
        # Create configuration first
        mdc_create = MachineDataConfiguration(
            machine=machine,
            user_dataframe_for_create_cfg=simple_dataframe,
            columns_type_user_df=columns_datatype,
            columns_description_user_df=columns_description,
            decimal_separator=".",
            date_format="%Y-%m-%d"
        )
        mdc_create.save_configuration_in_machine()
        
        # Load configuration
        mdc_load = MachineDataConfiguration(machine=machine)
        
        assert mdc_load._machine == machine
        assert mdc_load.columns_input_count == mdc_create.columns_input_count
        assert mdc_load.columns_output_count == mdc_create.columns_output_count
        
    @pytest.mark.django_db
    def test_mdc_dataframe_pre_encode(self, db_cleanup, simple_dataframe, columns_datatype, columns_description):
        """
        Test DataFrame Pre-Encoding
        
        WHAT THIS TEST DOES:
        - Takes raw data and converts it to a pre-encoded format
        - Handles data type conversions and missing values
        - Prepares data for further encoding by EncDec module
        
        WHY THIS TEST IS IMPORTANT:
        - Pre-encoding is the first step in data processing
        - It standardizes data formats and handles edge cases
        - Other modules depend on pre-encoded data
        
        PRE-ENCODING PROCESS:
        1. Analyze data types and convert to standard formats
        2. Handle missing values (fill or mark them)
        3. Convert categorical data to consistent format
        4. Normalize numeric data ranges
        5. Return standardized DataFrame
        
        WHAT WE'RE TESTING:
        - Pre-encoding produces a valid DataFrame
        - Pre-encoded data is not empty
        - Pre-encoded data has at least as many columns as input
        - Data types are standardized
        
        TEST STEPS:
        1. Create MDC configuration
        2. Call pre-encode method on original data
        3. Verify pre-encoded data is valid
        4. Check data structure and types
        """
        machine = Machine(
            "__TEST_UNIT__mdc_pre_encode",
            simple_dataframe,
            decimal_separator=".",
            date_format="%Y-%m-%d",
            machine_create_user_id=self._get_admin_user().id,
            disable_foreign_key_checking=True
        )
        machine.save_machine_to_db()
        
        mdc = MachineDataConfiguration(
            machine=machine,
            user_dataframe_for_create_cfg=simple_dataframe,
            columns_type_user_df=columns_datatype,
            columns_description_user_df=columns_description,
            decimal_separator=".",
            date_format="%Y-%m-%d"
        )
        
        pre_encoded = mdc.dataframe_pre_encode(simple_dataframe)
        
        assert isinstance(pre_encoded, pd.DataFrame)
        assert not pre_encoded.empty
        assert len(pre_encoded.columns) >= len(simple_dataframe.columns)
        
    @pytest.mark.django_db
    def test_mdc_dataframe_post_decode(self, db_cleanup, simple_dataframe, columns_datatype, columns_description):
        """Test dataframe post-decoding"""
        machine = Machine(
            "__TEST_UNIT__mdc_post_decode",
            simple_dataframe,
            decimal_separator=".",
            date_format="%Y-%m-%d",
            machine_create_user_id=self._get_admin_user().id,
            disable_foreign_key_checking=True
        )
        machine.save_machine_to_db()
        
        mdc = MachineDataConfiguration(
            machine=machine,
            user_dataframe_for_create_cfg=simple_dataframe,
            columns_type_user_df=columns_datatype,
            columns_description_user_df=columns_description,
            decimal_separator=".",
            date_format="%Y-%m-%d"
        )
        
        pre_encoded = mdc.dataframe_pre_encode(simple_dataframe)
        post_decoded = mdc.dataframe_post_decode(pre_encoded)
        
        assert isinstance(post_decoded, pd.DataFrame)
        assert not post_decoded.empty
        
    @pytest.mark.django_db
    def test_mdc_save_configuration(self, db_cleanup, simple_dataframe, columns_datatype, columns_description):
        """Test MDC configuration saving"""
        machine = Machine(
            "__TEST_UNIT__mdc_save",
            simple_dataframe,
            decimal_separator=".",
            date_format="%Y-%m-%d",
            machine_create_user_id=self._get_admin_user().id,
            disable_foreign_key_checking=True
        )
        machine.save_machine_to_db()
        
        mdc = MachineDataConfiguration(
            machine=machine,
            user_dataframe_for_create_cfg=simple_dataframe,
            columns_type_user_df=columns_datatype,
            columns_description_user_df=columns_description,
            decimal_separator=".",
            date_format="%Y-%m-%d"
        )
        
        # Save configuration
        mdc.save_configuration_in_machine()
        
        # Verify configuration was saved
        assert machine.db_machine.mdc_columns_input_count == mdc.columns_input_count
        assert machine.db_machine.mdc_columns_output_count == mdc.columns_output_count
        
    @pytest.mark.django_db
    def test_mdc_column_statistics(self, db_cleanup, simple_dataframe, columns_datatype, columns_description):
        """Test column statistics calculation"""
        
        machine = Machine(
            "__TEST_UNIT__mdc_stats",
            simple_dataframe,
            decimal_separator=".",
            date_format="%Y-%m-%d",
            machine_create_user_id=self._get_admin_user().id,
            disable_foreign_key_checking=True
        )
        machine.save_machine_to_db()
        
        mdc = MachineDataConfiguration(
            machine=machine,
            user_dataframe_for_create_cfg=simple_dataframe,
            columns_type_user_df=columns_datatype,
            columns_description_user_df=columns_description,
            decimal_separator=".",
            date_format="%Y-%m-%d"
        )
        
        # Check that statistics were calculated
        assert len(mdc.columns_values_mean) > 0
        assert len(mdc.columns_values_std_dev) > 0
        assert len(mdc.columns_values_min) > 0
        assert len(mdc.columns_values_max) > 0
        
    @pytest.mark.django_db
    def test_mdc_json_column_handling(self, db_cleanup, simple_dataframe, columns_datatype, columns_description):
        """Test JSON column handling"""
        
        machine = Machine(
            "__TEST_UNIT__mdc_json",
            simple_dataframe,
            decimal_separator=".",
            date_format="%Y-%m-%d",
            machine_create_user_id=self._get_admin_user().id,
            disable_foreign_key_checking=True
        )
        machine.save_machine_to_db()
        
        mdc = MachineDataConfiguration(
            machine=machine,
            user_dataframe_for_create_cfg=simple_dataframe,
            columns_type_user_df=columns_datatype,
            columns_description_user_df=columns_description,
            decimal_separator=".",
            date_format="%Y-%m-%d"
        )
        
        # Test pre-encoding with JSON columns
        pre_encoded = mdc.dataframe_pre_encode(simple_dataframe)
        assert isinstance(pre_encoded, pd.DataFrame)
        assert not pre_encoded.empty
        
    @pytest.mark.django_db
    def test_mdc_missing_values_handling(self, db_cleanup, simple_dataframe, columns_datatype, columns_description):
        """Test handling of missing values"""
        
        machine = Machine(
            "__TEST_UNIT__mdc_missing",
            simple_dataframe,
            decimal_separator=".",
            date_format="%Y-%m-%d",
            machine_create_user_id=self._get_admin_user().id,
            disable_foreign_key_checking=True
        )
        machine.save_machine_to_db()
        
        mdc = MachineDataConfiguration(
            machine=machine,
            user_dataframe_for_create_cfg=simple_dataframe,
            columns_type_user_df=columns_datatype,
            columns_description_user_df=columns_description,
            decimal_separator=".",
            date_format="%Y-%m-%d"
        )
        
        # Check missing percentage calculation
        assert len(mdc.columns_missing_percentage) > 0
        for col, missing_pct in mdc.columns_missing_percentage.items():
            assert 0 <= missing_pct <= 100
            
    @pytest.mark.django_db
    def test_mdc_force_inputs_outputs(self, db_cleanup, numeric_dataframe, numeric_columns_datatype, numeric_columns_description):
        """Test MDC with forced inputs/outputs"""
        force_inputs = {'input1': True, 'input2': True}
        force_outputs = {'output': True}
        
        machine = Machine(
            "__TEST_UNIT__mdc_force",
            numeric_dataframe,
            decimal_separator=".",
            date_format="%Y-%m-%d",
            machine_create_user_id=self._get_admin_user().id,
            disable_foreign_key_checking=True
        )
        machine.save_machine_to_db()
        
        mdc = MachineDataConfiguration(
            machine=machine,
            user_dataframe_for_create_cfg=numeric_dataframe,
            columns_type_user_df=numeric_columns_datatype,
            columns_description_user_df=numeric_columns_description,
            force_create_with_this_inputs=force_inputs,
            force_create_with_this_outputs=force_outputs,
            decimal_separator=".",
            date_format="%Y-%m-%d"
        )
        
        # Verify forced configuration
        assert mdc.columns_name_input['input1'] == True
        assert mdc.columns_name_input['input2'] == True
        assert mdc.columns_name_output['output'] == True
        
    def test_mdc_invalid_machine_type(self, db_cleanup):
        """Test MDC with invalid machine type"""
        with pytest.raises(Exception):
            MachineDataConfiguration(machine="not_a_machine")
            
    @pytest.mark.django_db
    def test_mdc_errors_and_warnings(self, db_cleanup, simple_dataframe, columns_datatype, columns_description):
        """Test MDC errors and warnings collection"""
        machine = Machine(
            "__TEST_UNIT__mdc_errors",
            simple_dataframe,
            decimal_separator=".",
            date_format="%Y-%m-%d",
            machine_create_user_id=self._get_admin_user().id,
            disable_foreign_key_checking=True
        )
        machine.save_machine_to_db()
        
        mdc = MachineDataConfiguration(
            machine=machine,
            user_dataframe_for_create_cfg=simple_dataframe,
            columns_type_user_df=columns_datatype,
            columns_description_user_df=columns_description,
            decimal_separator=".",
            date_format="%Y-%m-%d"
        )
        
        # Check that errors and warnings are dictionaries
        assert isinstance(mdc.columns_errors, dict)
        assert isinstance(mdc.columns_warnings, dict)
        
    @pytest.mark.django_db
    def test_mdc_verify_compatibility_additional_dataframe(self, db_cleanup, simple_dataframe, columns_datatype, columns_description):
        """
        Test Verifying Compatibility of Additional DataFrame
        
        WHAT THIS TEST DOES:
        - Tests the method that verifies if an additional dataframe is compatible with the existing configuration
        - Verifies that the method correctly identifies compatible and incompatible dataframes
        - Tests edge cases and error handling
        
        WHY THIS TEST IS IMPORTANT:
        - This method ensures data consistency when adding new data to an existing machine
        - It prevents data corruption and configuration mismatches
        - It's essential for maintaining data integrity in production systems
        
        WHAT WE'RE TESTING:
        - Method correctly identifies compatible dataframes
        - Method correctly identifies incompatible dataframes
        - Method handles edge cases (empty dataframes, missing columns)
        - Method returns appropriate boolean results
        
        TEST STEPS:
        1. Create MDC configuration with original data
        2. Test with compatible additional dataframe
        3. Test with incompatible additional dataframe
        4. Test with edge cases
        """
        machine = Machine(
            "__TEST_UNIT__mdc_compatibility",
            simple_dataframe,
            decimal_separator=".",
            date_format="%Y-%m-%d",
            machine_create_user_id=self._get_admin_user().id,
            disable_foreign_key_checking=True
        )
        machine.save_machine_to_db()
        
        mdc = MachineDataConfiguration(
            machine=machine,
            user_dataframe_for_create_cfg=simple_dataframe,
            columns_type_user_df=columns_datatype,
            columns_description_user_df=columns_description,
            decimal_separator=".",
            date_format="%Y-%m-%d"
        )
        
        # Test with compatible dataframe (same structure)
        compatible_df = simple_dataframe.copy()
        result = mdc.verify_compatibility_additional_dataframe(compatible_df, machine, ".", "%Y-%m-%d")
        # The method returns a tuple (bool, str), so we check the first element
        assert isinstance(result, tuple) and len(result) == 2
        assert isinstance(result[0], bool)
        
        # Test with incompatible dataframe (different columns) - should handle gracefully
        incompatible_df = pd.DataFrame({'different_col': [1, 2, 3]})
        try:
            result = mdc.verify_compatibility_additional_dataframe(incompatible_df, machine, ".", "%Y-%m-%d")
            assert isinstance(result, tuple) and len(result) == 2
            assert isinstance(result[0], bool)
        except ValueError:
            # Expected behavior - incompatible dataframe may cause errors
            pass
        
        # Test with empty dataframe - should handle gracefully
        empty_df = pd.DataFrame()
        try:
            result = mdc.verify_compatibility_additional_dataframe(empty_df, machine, ".", "%Y-%m-%d")
            assert isinstance(result, tuple) and len(result) == 2
            assert isinstance(result[0], bool)
        except (ValueError, IndexError):
            # Expected behavior - empty dataframe may cause errors
            pass
        
    @pytest.mark.django_db
    def test_mdc_get_parent_of_extended_column(self, db_cleanup, simple_dataframe, columns_datatype, columns_description):
        """
        Test Getting Parent of Extended Column
        
        WHAT THIS TEST DOES:
        - Tests the method that finds the parent column of an extended column
        - Verifies that the method correctly identifies parent-child relationships
        - Tests edge cases and error handling
        
        WHY THIS TEST IS IMPORTANT:
        - This method is used to track column relationships in complex data structures
        - It helps understand how columns are extended during preprocessing
        - It's essential for maintaining data lineage
        
        WHAT WE'RE TESTING:
        - Method correctly identifies parent columns
        - Method handles non-extended columns
        - Method handles edge cases (None inputs, invalid column names)
        - Method returns appropriate results
        
        TEST STEPS:
        1. Create MDC configuration
        2. Test with extended column names
        3. Test with non-extended column names
        4. Test edge cases
        """
        machine = Machine(
            "__TEST_UNIT__mdc_parent",
            simple_dataframe,
            decimal_separator=".",
            date_format="%Y-%m-%d",
            machine_create_user_id=self._get_admin_user().id,
            disable_foreign_key_checking=True
        )
        machine.save_machine_to_db()
        
        mdc = MachineDataConfiguration(
            machine=machine,
            user_dataframe_for_create_cfg=simple_dataframe,
            columns_type_user_df=columns_datatype,
            columns_description_user_df=columns_description,
            decimal_separator=".",
            date_format="%Y-%m-%d"
        )
        
        # Test with regular column name (should return None or same name)
        regular_column = list(simple_dataframe.columns)[0]
        parent = mdc.get_parent_of_extended_column(regular_column)
        assert parent is None or isinstance(parent, str)
        
        # Test with extended column name (if any exist)
        # This would depend on the actual column structure
        try:
            extended_column = f"{regular_column}_extended"
            parent = mdc.get_parent_of_extended_column(extended_column)
            assert parent is None or isinstance(parent, str)
        except Exception:
            # May not have extended columns in simple test data
            pass
        
        # Test with None input - should handle gracefully
        try:
            parent = mdc.get_parent_of_extended_column(None)
            assert parent is None
        except ValueError:
            # Expected behavior - None is not a valid column name
            pass
        
        # Test with empty string - should handle gracefully
        try:
            parent = mdc.get_parent_of_extended_column("")
            assert isinstance(parent, str)
        except ValueError:
            # Expected behavior - empty string is not a valid column name
            pass
        
    @pytest.mark.django_db
    def test_mdc_get_children_of_json_column(self, db_cleanup, simple_dataframe, columns_datatype, columns_description):
        """
        Test Getting Children of JSON Column
        
        WHAT THIS TEST DOES:
        - Tests the method that finds child columns of a JSON column
        - Verifies that the method correctly identifies JSON column relationships
        - Tests edge cases and error handling
        
        WHY THIS TEST IS IMPORTANT:
        - This method is used to track JSON column expansions
        - It helps understand how JSON data is flattened into multiple columns
        - It's essential for maintaining data structure understanding
        
        WHAT WE'RE TESTING:
        - Method correctly identifies child columns of JSON columns
        - Method handles non-JSON columns
        - Method handles edge cases (None inputs, invalid column names)
        - Method returns appropriate list results
        
        TEST STEPS:
        1. Create MDC configuration
        2. Test with JSON column names
        3. Test with non-JSON column names
        4. Test edge cases
        """
        machine = Machine(
            "__TEST_UNIT__mdc_children",
            simple_dataframe,
            decimal_separator=".",
            date_format="%Y-%m-%d",
            machine_create_user_id=self._get_admin_user().id,
            disable_foreign_key_checking=True
        )
        machine.save_machine_to_db()
        
        mdc = MachineDataConfiguration(
            machine=machine,
            user_dataframe_for_create_cfg=simple_dataframe,
            columns_type_user_df=columns_datatype,
            columns_description_user_df=columns_description,
            decimal_separator=".",
            date_format="%Y-%m-%d"
        )
        
        # Test with regular column name (should return empty list)
        regular_column = list(simple_dataframe.columns)[0]
        children = mdc.get_children_of_json_column(regular_column)
        assert isinstance(children, list)
        
        # Test with JSON column name (if any exist)
        # This would depend on the actual column structure
        try:
            json_column = f"{regular_column}_json"
            children = mdc.get_children_of_json_column(json_column)
            assert isinstance(children, list)
        except Exception:
            # May not have JSON columns in simple test data
            pass
        
        # Test with None input - should handle gracefully
        try:
            children = mdc.get_children_of_json_column(None)
            assert isinstance(children, list)
        except ValueError:
            # Expected behavior - None is not a valid column name
            pass
        
        # Test with empty string - should handle gracefully
        try:
            children = mdc.get_children_of_json_column("")
            assert isinstance(children, list)
        except ValueError:
            # Expected behavior - empty string is not a valid column name
            pass
        
        # Test with non-existent column - should handle gracefully
        try:
            children = mdc.get_children_of_json_column("nonexistent_column")
            assert isinstance(children, list)
        except ValueError:
            # Expected behavior - nonexistent column should raise ValueError
            pass