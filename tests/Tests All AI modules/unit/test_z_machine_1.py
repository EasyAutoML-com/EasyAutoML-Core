"""
Test Machine 1 - Complete Machine Workflow Test

This test performs a complete machine workflow:
1. Create machine from CSV file
2. Test machine properties
3. Create all configurations
4. Run training
5. Test solving

This test is based on the Iris flowers dataset and follows the same
pattern as the Django management command test-machine.py.
"""
import pytest
import pandas as pd
import os
import sys
from ML import Machine, NNEngine, DataFileReader

# Add parent directory to path to import from conftest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from conftest import get_admin_user


class TestMachine1:
    """
    Test complete machine workflow:
    - Create machine from CSV
    - Test machine properties
    - Create all configurations
    - Run training
    - Test solving
    """
    
    @pytest.mark.django_db
    def test_machine_1(self, test_database_with_verification):
        """
        Complete machine workflow test:
        1. Create machine from CSV file
        2. Test machine properties
        3. Create all configurations
        4. Run training
        5. Test solving
        """
        # Step 1: Create machine from CSV
        print('\n1. Creating machine from CSV file...')
        machine = self.create_machine_from_csv()
        
        # Step 2: Test machine properties
        print('\n2. Testing machine properties...')
        self._verify_machine_properties(machine)
        
        # Step 3: Create all configurations
        print('\n3. Creating all configurations...')
        nn_engine = self.create_configurations(machine)
        
        # Step 4: Run training
        print('\n4. Running training...')
        self.run_training(nn_engine, machine)
        
        # Step 5: Test solving
        print('\n5. Testing solving...')
        self._verify_solving(nn_engine, machine)
    
    def create_machine_from_csv(self):
        """Create machine from Iris flowers CSV"""
        User = get_admin_user()
        if User is None:
            from django.contrib.auth import get_user_model
            User = get_user_model()
            User = User.objects.get(email='SuperSuperAdmin@easyautoml.com')
        
        # Get the CSV file path
        # __file__ is in tests/Tests All AI modules/unit/test_z_machine_1.py
        # CSV is in tests/Tests All AI modules/Test Data - Iris flowers.csv
        test_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_file_path = os.path.join(
            test_dir,
            "Test Data - Iris flowers.csv"
        )
        
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
        
        print(f'   Reading CSV file: {csv_file_path}')
        
        # Read CSV file using DataFileReader
        dfr = DataFileReader(
            csv_file_path,
            date_format="MDY",
            decimal_separator=".",
        )
        
        # Ensure all columns have descriptions
        formatted_df = dfr.get_formatted_user_dataframe
        columns_description = dfr.get_user_columns_description or {}
        
        # Add missing column descriptions
        for col in formatted_df.columns:
            if col not in columns_description:
                columns_description[col] = f"Column {col}"
        
        print(f'   Dataframe shape: {formatted_df.shape}')
        print(f'   Columns: {list(formatted_df.columns)}')
        
        # Create machine
        machine_name = "__TEST_CMD__iris_flowers"
        machine = Machine(
            machine_name,
            dfr=dfr,
            decimal_separator=".",
            date_format="MDY",
            machine_create_user_id=User.id,
            machine_create_team_id=1,
            machine_description="Test machine for Iris flowers dataset",
            force_create_with_this_descriptions=columns_description,
            machine_level=1,
            disable_foreign_key_checking=True
        )
        
        machine.save_machine_to_db()
        
        print(f'   ✓ Machine created with ID: {machine.id}')
        print(f'   Machine name: {machine.db_machine.machine_name}')
        
        return machine
    
    def _verify_machine_properties(self, machine):
        """Verify various machine properties (helper method, not a test)"""
        print(f'   Machine ID: {machine.id}')
        print(f'   Machine name: {machine.db_machine.machine_name}')
        print(f'   Machine level: {machine.db_machine.machine_level}')
        
        # Test data counts
        input_count = machine.data_input_lines_count()
        output_count = machine.data_output_lines_count()
        print(f'   Input lines: {input_count}')
        print(f'   Output lines: {output_count}')
        
        # Test configuration status
        print(f'   MDC ready: {machine.is_config_ready_mdc}')
        print(f'   ICI ready: {machine.is_config_ready_ici}')
        print(f'   FE ready: {machine.is_config_ready_fe}')
        print(f'   EncDec ready: {machine.is_config_ready_enc_dec}')
        print(f'   NN Config ready: {machine.is_config_ready_nn_configuration}')
        print(f'   NN Model ready: {machine.is_config_ready_nn_model}')
        
        print('   ✓ All machine properties accessible')
    
    def create_configurations(self, machine):
        """Create all machine configurations via NNEngine"""
        print('   Creating NNEngine (this will create all configurations)...')
        
        try:
            nn_engine = NNEngine(machine, allow_re_run_configuration=True)
            print('   ✓ NNEngine created successfully')
            return nn_engine
        except Exception as e:
            print(f'   ✗ Failed to create NNEngine: {e}')
            raise
    
    def run_training(self, nn_engine, machine):
        """Run machine training"""
        print('   Starting training (this may take a while)...')
        
        try:
            nn_engine.do_training_and_save()
            print('   ✓ Training completed successfully')
            
            # Check if machine is ready for solving
            if machine.is_nn_solving_ready():
                print('   ✓ Machine is ready for solving')
            else:
                print('   ⚠ Machine may not be fully ready for solving')
                
        except Exception as e:
            print(f'   ✗ Training failed: {e}')
            raise
    
    def _verify_solving(self, nn_engine, machine):
        """Verify solving on rows 148-149 (last two rows of Iris dataset) - helper method, not a test"""
        # Get the CSV file path
        test_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_file_path = os.path.join(
            test_dir,
            "Test Data - Iris flowers.csv"
        )
        
        # DIAGNOSTIC TEST 0: Get expected columns first
        from SharedConstants import ColumnDirectionType, DataframeEncodingType
        expected_input_columns_user = machine.get_list_of_columns_name(
            ColumnDirectionType.INPUT, 
            DataframeEncodingType.USER
        )
        
        # Read the full CSV to get specific rows - but we need to normalize column names
        # because DataFileReader normalizes them (e.g., ':' becomes '-')
        # Skip the first row (type row: input,input,input,input,output) since pandas treats it as data
        df_full = pd.read_csv(csv_file_path, skiprows=[1])
        
        # Normalize column names to match what the machine expects
        # This matches the normalization in DataFileReader.correct_dataset_columns_names
        import re
        def normalize_column_name(s):
            s = re.sub(r' +', ' ', s).strip()
            s = re.sub(r'_+', '_', s).strip()
            s = re.sub(r'[&"#%$*!:;,?]', '-', s).strip()
            if len(s) <= 64:
                return s
            else:
                return s[:62] + ".."
        
        df_full.columns = df_full.columns.map(normalize_column_name)
        
        # Get rows 148-149 (indices 148-149) - last two rows of Iris dataset (150 rows total)
        solving_rows = df_full.iloc[148:150]
        
        # Extract only input columns - use machine's expected columns to ensure we have the right ones
        # This ensures we're using the same columns that were used during training
        solving_dataframe = solving_rows[expected_input_columns_user].copy()
        
        print(f'   Solving {len(solving_dataframe)} rows...')
        print(f'   Input columns: {list(solving_dataframe.columns)}')
        
        # DIAGNOSTIC TEST 1: Check expected input columns from machine (already retrieved above)
        print(f'\n   DIAGNOSTIC: Expected input columns (USER): {expected_input_columns_user}')
        print(f'   DIAGNOSTIC: Expected input columns count (USER): {len(expected_input_columns_user)}')
        
        expected_input_columns_encoded = machine.get_list_of_columns_name(
            ColumnDirectionType.INPUT, 
            DataframeEncodingType.ENCODED_FOR_AI
        )
        print(f'   DIAGNOSTIC: Expected input columns (ENCODED_FOR_AI): {expected_input_columns_encoded}')
        print(f'   DIAGNOSTIC: Expected input columns count (ENCODED_FOR_AI): {len(expected_input_columns_encoded)}')
        print(f'   DIAGNOSTIC: Machine enc_dec_columns_info_input_encode_count: {machine.db_machine.enc_dec_columns_info_input_encode_count}')
        
        # DIAGNOSTIC TEST 2: Check what columns are in solving dataframe
        solving_columns = list(solving_dataframe.columns)
        print(f'   DIAGNOSTIC: Solving dataframe columns: {solving_columns}')
        print(f'   DIAGNOSTIC: Solving dataframe columns count: {len(solving_columns)}')
        
        # DIAGNOSTIC TEST 3: Check for missing columns
        missing_columns = set(expected_input_columns_user) - set(solving_columns)
        extra_columns = set(solving_columns) - set(expected_input_columns_user)
        print(f'   DIAGNOSTIC: Missing columns in solving dataframe: {missing_columns}')
        print(f'   DIAGNOSTIC: Extra columns in solving dataframe: {extra_columns}')
        
        # DIAGNOSTIC TEST 4: Try encoding the solving dataframe and check column count
        try:
            pre_encoded_df = nn_engine._mdc.dataframe_pre_encode(solving_dataframe)
            print(f'   DIAGNOSTIC: Pre-encoded dataframe columns: {list(pre_encoded_df.columns)}')
            print(f'   DIAGNOSTIC: Pre-encoded dataframe columns count: {len(pre_encoded_df.columns)}')
            
            encoded_df = nn_engine._enc_dec.encode_for_ai(pre_encoded_df)
            print(f'   DIAGNOSTIC: Encoded dataframe columns: {list(encoded_df.columns)}')
            print(f'   DIAGNOSTIC: Encoded dataframe columns count: {len(encoded_df.columns)}')
            print(f'   DIAGNOSTIC: Expected encoded columns count: {machine.db_machine.enc_dec_columns_info_input_encode_count}')
            
            if len(encoded_df.columns) != machine.db_machine.enc_dec_columns_info_input_encode_count:
                print(f'   ⚠ BUG DETECTED: Encoded column count mismatch!')
                print(f'      Expected: {machine.db_machine.enc_dec_columns_info_input_encode_count}')
                print(f'      Got: {len(encoded_df.columns)}')
                
                # Check which columns are missing
                expected_encoded_cols = set(expected_input_columns_encoded)
                actual_encoded_cols = set(encoded_df.columns)
                missing_encoded = expected_encoded_cols - actual_encoded_cols
                extra_encoded = actual_encoded_cols - expected_encoded_cols
                print(f'      Missing encoded columns: {missing_encoded}')
                print(f'      Extra encoded columns: {extra_encoded}')
        except Exception as e:
            print(f'   DIAGNOSTIC: Error during encoding test: {e}')
            import traceback
            traceback.print_exc()
        
        # DIAGNOSTIC TEST 5: Compare with training data columns
        try:
            training_df = machine.get_random_user_dataframe_for_training_trial(is_for_learning=True, force_row_count_same_as_for_evaluation=True)
            training_input_cols = [col for col in training_df.columns if col in expected_input_columns_user]
            print(f'   DIAGNOSTIC: Training dataframe input columns: {training_input_cols}')
            print(f'   DIAGNOSTIC: Training dataframe input columns count: {len(training_input_cols)}')
            
            # Encode training data to see what it produces
            training_pre_encoded = nn_engine._mdc.dataframe_pre_encode(training_df)
            training_encoded = nn_engine._enc_dec.encode_for_ai(training_pre_encoded)
            training_input_encoded_cols = [col for col in training_encoded.columns if col in expected_input_columns_encoded]
            print(f'   DIAGNOSTIC: Training encoded input columns count: {len(training_input_encoded_cols)}')
            print(f'   DIAGNOSTIC: Training encoded ALL columns: {list(training_encoded.columns)}')
        except Exception as e:
            print(f'   DIAGNOSTIC: Error checking training data: {e}')
            import traceback
            traceback.print_exc()
        
        # DIAGNOSTIC TEST 6: Check the actual model's input shape
        try:
            nn_model = nn_engine._get_nn_model_from_db()
            if nn_model is not None:
                model_input_shape = nn_model.input_shape
                print(f'   DIAGNOSTIC: Model input shape: {model_input_shape}')
                if isinstance(model_input_shape, tuple) and len(model_input_shape) > 1:
                    print(f'   DIAGNOSTIC: Model expects {model_input_shape[1]} input features')
        except Exception as e:
            print(f'   DIAGNOSTIC: Error checking model input shape: {e}')
            import traceback
            traceback.print_exc()
        
        # DIAGNOSTIC TEST 7: Check if dataframe_full_encode produces correct columns
        try:
            full_encoded = nn_engine.dataframe_full_encode(solving_dataframe)
            print(f'   DIAGNOSTIC: Full encoded dataframe columns count: {len(full_encoded.columns)}')
            print(f'   DIAGNOSTIC: Full encoded dataframe ALL columns: {list(full_encoded.columns)}')
            
            # Split into input and output to see what we get
            input_encoded_cols = [col for col in full_encoded.columns if col in expected_input_columns_encoded]
            print(f'   DIAGNOSTIC: Full encoded INPUT columns: {input_encoded_cols}')
            print(f'   DIAGNOSTIC: Full encoded INPUT columns count: {len(input_encoded_cols)}')
            
            # Try to split the encoded dataframe like the solving method does
            try:
                input_df, output_df = nn_engine._split_dataframe_into_input_and_output(
                    full_encoded, 
                    DataframeEncodingType.ENCODED_FOR_AI
                )
                print(f'   DIAGNOSTIC: After split - Input columns count: {len(input_df.columns)}')
                print(f'   DIAGNOSTIC: After split - Input columns: {list(input_df.columns)}')
                print(f'   DIAGNOSTIC: After split - Output columns count: {len(output_df.columns)}')
            except Exception as split_e:
                print(f'   DIAGNOSTIC: Error splitting encoded dataframe: {split_e}')
        except Exception as e:
            print(f'   DIAGNOSTIC: Error in dataframe_full_encode: {e}')
            import traceback
            traceback.print_exc()
        
        try:
            # Solve the rows
            solved_dataframe = nn_engine.do_solving_direct_dataframe_user(solving_dataframe)
            
            print(f'   ✓ Solved {len(solved_dataframe)} rows')
            print(f'   Output columns: {list(solved_dataframe.columns)}')
            
            # Display results
            print('\n   Results:')
            print('   ' + '-'*70)
            for idx, row in solved_dataframe.iterrows():
                print(f'   Row {idx}: {dict(row)}')
            print('   ' + '-'*70)
            
            # Verify results
            assert solved_dataframe is not None, "Solving should return a result"
            assert len(solved_dataframe) == 2, f"Solved dataframe should have 2 rows, got {len(solved_dataframe)}"
            
        except Exception as e:
            print(f'   ✗ Solving failed: {e}')
            import traceback
            traceback.print_exc()
            raise


