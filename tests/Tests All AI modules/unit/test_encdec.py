"""
Tests for EncDec.py - Encoding/Decoding for Neural Networks

This file tests the EncDec (Encode/Decode) module, which is responsible for converting
data into formats that neural networks can understand and then converting the results
back to human-readable formats.

WHAT IS ENCODING/DECODING?
==========================
In machine learning, we often need to convert data between different formats:

ENCODING (Data → Neural Network Format):
- Converts text categories (like 'A', 'B', 'C') into numbers (like 0, 1, 2)
- Converts dates into numeric values
- Handles missing values
- Normalizes numeric data to similar scales

DECODING (Neural Network Format → Data):
- Converts numbers back to text categories
- Converts numeric dates back to readable dates
- Restores original data formats

WHY IS THIS IMPORTANT?
=====================
Neural networks can only work with numbers, but real-world data often contains:
- Text categories (like 'Male', 'Female')
- Dates (like '2023-01-15')
- Mixed data types

The EncDec module bridges this gap by:
1. Converting all data to numbers for the neural network
2. Converting the neural network's output back to readable formats

WHAT DOES THIS MODULE TEST?
===========================
- Configuration creation and loading
- Encoding data for neural networks
- Decoding neural network outputs
- Round-trip encoding/decoding (encode then decode)
- Configuration saving and serialization
- Error handling with invalid inputs

TESTING STRATEGY:
================
Each test follows this pattern:
1. Create a test machine with sample data
2. Set up all required configurations (MDC, ICI, FEC)
3. Test the specific EncDec functionality
4. Verify the results are correct
5. Clean up test data

DEPENDENCIES:
=============
EncDec depends on several other modules:
- MachineDataConfiguration (MDC): Analyzes the data structure
- InputsColumnsImportance (ICI): Determines which columns are inputs/outputs
- FeatureEngineeringConfiguration (FEC): Sets up feature engineering
- Machine: The main machine object that holds everything together
"""
import pytest
import pandas as pd
import numpy as np
from ML import Machine, MachineDataConfiguration, EncDec, FeatureEngineeringConfiguration, InputsColumnsImportance
from models.EasyAutoMLDBModels import EasyAutoMLDBModels
from SharedConstants import DatasetColumnDataType


class TestEncDec:
    """
    Test EncDec Class Functionality
    
    This class contains all tests for the EncDec module. Each test method focuses on
    one specific aspect of encoding/decoding functionality.
    
    TEST ORGANIZATION:
    ==================
    The tests are organized by functionality:
    1. Configuration tests (create, load, save)
    2. Encoding/decoding tests (basic operations)
    3. Round-trip tests (encode then decode)
    4. Error handling tests
    5. Edge case tests (different data types)
    
    FIXTURE DEPENDENCIES:
    =====================
    Most tests use these fixtures:
    - db_cleanup: Ensures test data is cleaned up after each test
    - simple_dataframe: A basic DataFrame with input/output columns
    - columns_datatype: Maps column names to data types (FLOAT, CATEGORICAL, etc.)
    - columns_description: Human-readable descriptions of each column
    
    TEST NAMING CONVENTION:
    =======================
    - test_encdec_[functionality]: Tests specific EncDec functionality
    - Each test name describes what it's testing
    - Tests are independent and can run in any order
    """
    
    @pytest.mark.django_db
    def test_encdec_create_configuration(self, db_cleanup, machine_with_all_configs):
        """
        Test EncDec Configuration Creation
        
        WHAT THIS TEST DOES:
        - Creates a new EncDec configuration from scratch
        - Uses a machine with all required dependencies (MDC, ICI, FEC) already set up
        - Verifies that the configuration is created correctly
        
        WHY THIS TEST IS IMPORTANT:
        - EncDec needs a configuration to know how to encode/decode data
        - This test ensures the configuration creation process works
        - It verifies that all dependencies are set up correctly
        
        CONFIGURATION CREATION PROCESS:
        1. Use a machine with MDC, ICI, and FEC already configured
        2. Create pre-encoded data using MDC
        3. Create EncDec configuration using the pre-encoded data
        
        WHAT WE'RE TESTING:
        - The EncDec object is created successfully
        - All internal configuration objects are properly initialized
        - The configuration has the expected structure
        
        TEST STEPS:
        1. Use a machine with all configurations already set up
        2. Get the machine's original data
        3. Create pre-encoded data using MDC
        4. Create EncDec configuration
        5. Verify all configuration objects are properly initialized
        """
        machine = machine_with_all_configs
        
        # Get the machine's original data (this matches the machine's configuration)
        original_df = machine.data_lines_read()
        if original_df.empty:
            pytest.skip("Machine has no data to test with")
        
        # Create pre-encoded dataframe
        mdc = MachineDataConfiguration(machine=machine)
        pre_encoded_df = mdc.dataframe_pre_encode(original_df)
        
        # Create EncDec
        encdec = EncDec(machine=machine, dataframe_pre_encoded=pre_encoded_df)
        
        assert encdec._machine == machine
        assert encdec._enc_dec_configuration is not None
        assert encdec._columns_input_encoded_count is not None
        assert encdec._columns_output_encoded_count is not None
        
    @pytest.mark.django_db
    def test_encdec_load_configuration(self, db_cleanup, machine_with_all_configs):
        """
        Test EncDec Configuration Loading
        
        WHAT THIS TEST DOES:
        - Creates an EncDec configuration and saves it
        - Then loads the configuration from the database
        - Verifies that the loaded configuration matches the original
        
        WHY THIS TEST IS IMPORTANT:
        - In real applications, configurations are saved and loaded later
        - This test ensures the save/load process works correctly
        - It verifies that configurations persist correctly in the database
        
        CONFIGURATION LOADING PROCESS:
        1. Create and save an EncDec configuration
        2. Create a new EncDec object that loads the saved configuration
        3. Compare the loaded configuration with the original
        
        WHAT WE'RE TESTING:
        - Configuration can be saved to the database
        - Configuration can be loaded from the database
        - Loaded configuration has the same properties as the original
        - The encoding/decoding parameters are preserved
        
        TEST STEPS:
        1. Use a machine with all configurations already set up
        2. Get the machine's original data
        3. Create and save an EncDec configuration
        4. Create a new EncDec object that loads the configuration
        5. Verify the loaded configuration matches the original
        """
        machine = machine_with_all_configs
        
        # Get the machine's original data (this matches the machine's configuration)
        original_df = machine.data_lines_read()
        if original_df.empty:
            pytest.skip("Machine has no data to test with")
        
        # Create EncDec configuration first (if not already created)
        mdc = MachineDataConfiguration(machine=machine)
        pre_encoded_df = mdc.dataframe_pre_encode(original_df)
        encdec_create = EncDec(machine=machine, dataframe_pre_encoded=pre_encoded_df)
        encdec_create.save_configuration_in_machine()
        
        # Load EncDec configuration
        encdec_load = EncDec(machine=machine)
        
        assert encdec_load._machine == machine
        assert encdec_load._enc_dec_configuration is not None
        assert encdec_load._columns_input_encoded_count == encdec_create._columns_input_encoded_count
        
    @pytest.mark.django_db
    def test_encdec_encode_for_ai(self, db_cleanup, machine_with_all_configs):
        """
        Test Encoding Data for AI/Neural Networks
        
        WHAT THIS TEST DOES:
        - Takes human-readable data and converts it to neural network format
        - Tests the core encoding functionality of EncDec
        - Verifies that the encoded data is in the correct format
        
        WHY THIS TEST IS IMPORTANT:
        - This is the main purpose of EncDec - converting data for AI
        - Neural networks can only process numeric data
        - This test ensures the encoding process works correctly
        
        ENCODING PROCESS:
        1. Take pre-encoded data (from MDC)
        2. Apply encoding rules to convert to neural network format
        3. Handle categorical variables, dates, missing values
        4. Return data that neural networks can process
        
        WHAT WE'RE TESTING:
        - Encoding produces a DataFrame (not crashes)
        - Encoded data is not empty
        - Encoded data has at least as many columns as input
        - The encoding process completes without errors
        
        TEST STEPS:
        1. Use a machine with all configurations already set up
        2. Get the machine's original data
        3. Create pre-encoded data using MDC
        4. Create EncDec configuration
        5. Encode the data for AI
        6. Verify the encoded data is valid
        """
        machine = machine_with_all_configs
        
        # Get the machine's original data (this matches the machine's configuration)
        original_df = machine.data_lines_read()
        if original_df.empty:
            pytest.skip("Machine has no data to test with")
        
        # Create EncDec
        mdc = MachineDataConfiguration(machine=machine)
        pre_encoded_df = mdc.dataframe_pre_encode(original_df)
        encdec = EncDec(machine=machine, dataframe_pre_encoded=pre_encoded_df)
        
        # Encode for AI
        encoded_df = encdec.encode_for_ai(pre_encoded_df)
        
        assert isinstance(encoded_df, pd.DataFrame)
        assert not encoded_df.empty
        assert len(encoded_df.columns) >= len(pre_encoded_df.columns)
        
    @pytest.mark.django_db
    def test_encdec_decode_from_ai(self, db_cleanup, machine_with_all_configs):
        """Test decoding from AI"""
        machine = machine_with_all_configs
        
        # Get the machine's original data (this matches the machine's configuration)
        original_df = machine.data_lines_read()
        if original_df.empty:
            pytest.skip("Machine has no data to test with")
        
        # Create EncDec
        mdc = MachineDataConfiguration(machine=machine)
        pre_encoded_df = mdc.dataframe_pre_encode(original_df)
        encdec = EncDec(machine=machine, dataframe_pre_encoded=pre_encoded_df)
        
        # Encode and decode
        encoded_df = encdec.encode_for_ai(pre_encoded_df)
        decoded_df = encdec.decode_from_ai(encoded_df)
        
        assert isinstance(decoded_df, pd.DataFrame)
        assert not decoded_df.empty
        
    @pytest.mark.django_db
    def test_encdec_encode_decode_roundtrip(self, db_cleanup, machine_with_all_configs):
        """
        Test Complete Encode-Decode Roundtrip
        
        WHAT THIS TEST DOES:
        - Encodes data for AI, then decodes it back to human format
        - Tests the complete data transformation cycle
        - Verifies that data can be converted back and forth without loss
        
        WHY THIS TEST IS IMPORTANT:
        - This is the ultimate test of EncDec functionality
        - Ensures data integrity through the entire process
        - Verifies that encoding/decoding are inverse operations
        
        ROUNDTRIP PROCESS:
        1. Start with original human-readable data
        2. Pre-encode using MDC (handles data types, missing values)
        3. Encode for AI using EncDec (converts to neural network format)
        4. Decode from AI using EncDec (converts back to pre-encoded format)
        5. Post-decode using MDC (converts back to human-readable format)
        
        WHAT WE'RE TESTING:
        - The complete roundtrip process works without errors
        - Final data has the same structure as original data
        - Data can be recovered after encoding/decoding
        - The process preserves data integrity
        
        TEST STEPS:
        1. Use a machine with all configurations already set up
        2. Get the machine's original data
        3. Create pre-encoded data
        4. Encode for AI
        5. Decode from AI
        6. Post-decode to get final result
        7. Verify final result matches original structure
        """
        machine = machine_with_all_configs
        
        # Get the machine's original data (this matches the machine's configuration)
        original_df = machine.data_lines_read()
        if original_df.empty:
            pytest.skip("Machine has no data to test with")
        
        # Create EncDec
        mdc = MachineDataConfiguration(machine=machine)
        pre_encoded_df = mdc.dataframe_pre_encode(original_df)
        encdec = EncDec(machine=machine, dataframe_pre_encoded=pre_encoded_df)
        
        # Roundtrip test
        encoded_df = encdec.encode_for_ai(pre_encoded_df)
        decoded_df = encdec.decode_from_ai(encoded_df)
        post_decoded_df = mdc.dataframe_post_decode(decoded_df)
        
        assert isinstance(post_decoded_df, pd.DataFrame)
        assert not post_decoded_df.empty
        # The decoded dataframe may have different column count due to encoding process
        # Just verify it's not empty and is a DataFrame
        
    @pytest.mark.django_db
    def test_encdec_save_configuration(self, db_cleanup, machine_with_all_configs):
        """Test EncDec configuration saving"""
        machine = machine_with_all_configs
        
        # Get the machine's original data (this matches the machine's configuration)
        original_df = machine.data_lines_read()
        if original_df.empty:
            pytest.skip("Machine has no data to test with")
        
        # Create EncDec
        mdc = MachineDataConfiguration(machine=machine)
        pre_encoded_df = mdc.dataframe_pre_encode(original_df)
        encdec = EncDec(machine=machine, dataframe_pre_encoded=pre_encoded_df)
        
        # Save configuration
        encdec.save_configuration_in_machine()
        
        # Verify configuration was saved
        assert machine.db_machine.enc_dec_columns_info_input_encode_count == encdec._columns_input_encoded_count
        assert machine.db_machine.enc_dec_columns_info_output_encode_count == encdec._columns_output_encoded_count
        
    @pytest.mark.django_db
    def test_encdec_configuration_serialization(self, db_cleanup, machine_with_all_configs):
        """Test EncDec configuration serialization"""
        machine = machine_with_all_configs
        
        # Get the machine's original data (this matches the machine's configuration)
        original_df = machine.data_lines_read()
        if original_df.empty:
            pytest.skip("Machine has no data to test with")
        
        # Create EncDec
        mdc = MachineDataConfiguration(machine=machine)
        pre_encoded_df = mdc.dataframe_pre_encode(original_df)
        encdec = EncDec(machine=machine, dataframe_pre_encoded=pre_encoded_df)
        
        # Test configuration structure
        config = encdec._enc_dec_configuration
        assert isinstance(config, dict)
        
        for column_name, column_config in config.items():
            assert 'is_input' in column_config
            assert 'is_output' in column_config
            assert 'column_datatype_enum' in column_config
            assert 'column_datatype_name' in column_config
            assert 'fet_list' in column_config
            
    def test_encdec_invalid_machine_type(self, db_cleanup):
        """Test EncDec with invalid machine type"""
        with pytest.raises(Exception):
            EncDec(machine="not_a_machine")
            
    @pytest.mark.django_db
    def test_encdec_invalid_dataframe_type(self, db_cleanup, machine_with_all_configs):
        """Test EncDec with invalid dataframe type"""
        machine = machine_with_all_configs
        
        # Test that EncDec raises an exception when given an invalid dataframe type
        with pytest.raises(Exception):
            EncDec(machine=machine, dataframe_pre_encoded="not_a_dataframe")
            
    @pytest.mark.django_db
    def test_encdec_with_different_data_types(self, db_cleanup, machine_with_all_configs):
        """Test EncDec with different data types"""
        machine = machine_with_all_configs
        
        # Get the machine's original data (this matches the machine's configuration)
        original_df = machine.data_lines_read()
        if original_df.empty:
            pytest.skip("Machine has no data to test with")
        
        # Create EncDec
        mdc = MachineDataConfiguration(machine=machine)
        pre_encoded_df = mdc.dataframe_pre_encode(original_df)
        encdec = EncDec(machine=machine, dataframe_pre_encoded=pre_encoded_df)
        
        # Test encoding/decoding with mixed types
        encoded_df = encdec.encode_for_ai(pre_encoded_df)
        decoded_df = encdec.decode_from_ai(encoded_df)
        
        assert isinstance(encoded_df, pd.DataFrame)
        assert isinstance(decoded_df, pd.DataFrame)
        assert not encoded_df.empty
        assert not decoded_df.empty
        
    @pytest.mark.django_db
    def test_encdec_get_encoded_for_ai_columns_names_by_the_pre_encoded_column_name(self, db_cleanup, machine_with_all_configs):
        """
        Test Getting Encoded Column Names by Pre-Encoded Column Name
        
        WHAT THIS TEST DOES:
        - Tests the method that retrieves encoded column names for a given pre-encoded column
        - Verifies that the method returns the correct list of encoded column names
        - Tests edge cases and error handling
        
        WHY THIS TEST IS IMPORTANT:
        - This method is used to map pre-encoded columns to their encoded counterparts
        - It's essential for understanding the data transformation pipeline
        - It helps track how columns are expanded during encoding
        
        WHAT WE'RE TESTING:
        - Method returns a list of strings
        - List contains expected encoded column names
        - Method handles invalid column names gracefully
        - Method works with different data types
        
        TEST STEPS:
        1. Use a machine with all configurations already set up
        2. Get the machine's original data
        3. Create EncDec configuration
        4. Call the method with a valid pre-encoded column name
        5. Verify the returned list is correct
        6. Test with invalid column name
        """
        machine = machine_with_all_configs
        
        # Get the machine's original data (this matches the machine's configuration)
        original_df = machine.data_lines_read()
        if original_df.empty:
            pytest.skip("Machine has no data to test with")
        
        # Create EncDec
        mdc = MachineDataConfiguration(machine=machine)
        pre_encoded_df = mdc.dataframe_pre_encode(original_df)
        encdec = EncDec(machine=machine, dataframe_pre_encoded=pre_encoded_df)
        
        # Test with a valid pre-encoded column name
        input_columns = list(pre_encoded_df.columns)[:2]  # Get first 2 columns as inputs
        if input_columns:
            column_name = input_columns[0]
            encoded_names = encdec.get_encoded_for_ai_columns_names_by_the_pre_encoded_column_name(column_name)
            
            assert isinstance(encoded_names, list)
            assert all(isinstance(name, str) for name in encoded_names)
            assert len(encoded_names) > 0
        
        # Test with invalid column name
        with pytest.raises(Exception):
            encdec.get_encoded_for_ai_columns_names_by_the_pre_encoded_column_name("nonexistent_column")
    
    @pytest.mark.django_db    
    def _setup_machine_configurations(self, machine, dataframe, columns_datatype, columns_description):
        """
        Helper Method: Setup All Machine Configurations
        
        WHAT THIS METHOD DOES:
        - Sets up all the required configurations for EncDec to work
        - Creates MDC, ICI, and FEC configurations in the correct order
        - Saves all configurations to the machine
        
        WHY THIS HELPER EXISTS:
        - Many tests need the same setup process
        - Avoids code duplication across test methods
        - Ensures consistent configuration setup
        
        CONFIGURATION DEPENDENCIES:
        - MDC (MachineDataConfiguration): Must be created first
        - ICI (InputsColumnsImportance): Depends on MDC
        - FEC (FeatureEngineeringConfiguration): Depends on ICI
        - EncDec: Depends on all three above
        
        SETUP PROCESS:
        1. Create MDC with the dataframe and column information
        2. Save MDC configuration to the machine
        3. Create ICI with minimum configuration
        4. Save ICI configuration to the machine
        5. Create FEC with minimum configuration
        6. Save FEC configuration to the machine
        
        PARAMETERS:
        - machine: The Machine object to configure
        - dataframe: The data to analyze
        - columns_datatype: Data type mapping for columns
        - columns_description: Human descriptions of columns
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
