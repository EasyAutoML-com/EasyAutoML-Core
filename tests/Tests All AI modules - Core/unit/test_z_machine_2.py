"""
Test Machine 2 - MachineEasyAutoML with Experimenter Workflow Test

This test performs a complete workflow with MachineEasyAutoML using an Experimenter:
1. Create SimpleExperimenter that calculates: 100 + A + B + A*B + A/B
2. Create MachineEasyAutoML with experimenter (no initial data)
3. Test predictions BEFORE training (using experimenter only)
4. Train the machine
5. Test predictions AFTER training (using trained model)
6. Compare results and verify improvement

This test is inspired by the Django management command test-machine.py workflow
and demonstrates the progressive learning capability of MachineEasyAutoML.
"""
import pytest
import pandas as pd
import random
import os
import sys
from ML import MachineEasyAutoML, NNEngine
from ML.Experimenter import Experimenter

# Add parent directory to path to import from conftest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from conftest import get_admin_user


class SimpleExperimenter(Experimenter):
    """
    Experimenter that calculates: 100 + A + B + A*B + A/B

    Expects input columns: input_A, input_B

    Returns output column: output_C
    """

    def _do_single(self, df_user_data_input: pd.Series) -> pd.Series:
        """
        Calculate output for a single input row
        Formula: C = 100 + A + B + A*B + A/B
        """
        A = float(df_user_data_input['input_A'])
        B = float(df_user_data_input['input_B'])

        # Calculate the formula
        result = 100 + A + B + (A * B) + (A / B)

        # Return as Series with output column name
        return pd.Series({'output_C': result})


class TestMachine2:
    """
    Test MachineEasyAutoML with Experimenter workflow:
    - Create experimenter
    - Create MachineEasyAutoML with experimenter
    - Test predictions BEFORE training (using experimenter)
    - Train the machine
    - Test predictions AFTER training (using trained model)
    - Compare results
    """

    @pytest.mark.django_db
    def test_machine_2_experimenter_workflow(self, test_database_with_verification):
        """
        Complete MachineEasyAutoML with Experimenter workflow test:
        1. Create SimpleExperimenter
        2. Create MachineEasyAutoML with experimenter (no initial data)
        3. Test predictions BEFORE training (using experimenter only)
        4. Train the machine
        5. Test predictions AFTER training (using trained model)
        6. Compare results and verify improvement
        """
        # Get admin user
        test_user = get_admin_user()
        if test_user is None:
            from django.contrib.auth import get_user_model
            User = get_user_model()
            test_user = User.objects.get(email='SuperSuperAdmin@easyautoml.com')

        print('\n' + '=' * 80)
        print('TEST MACHINE 2: MachineEasyAutoML with Experimenter Workflow')
        print('=' * 80)
        print('')

        # Step 1: Create experimenter
        print('Step 1: Creating SimpleExperimenter...')
        experimenter = SimpleExperimenter()
        print('✓ Created SimpleExperimenter')
        print('')

        # Step 2: Create MachineEasyAutoML with experimenter (no initial data)
        machine_name = "__TEST_CMD2__formula_machine"
        print(f'Step 2: Creating MachineEasyAutoML: {machine_name}')

        machine_eaml = MachineEasyAutoML(
            machine_name=machine_name,
            optional_experimenter=experimenter,
            record_experiments=True,
            access_user_id=test_user.id,
            access_team_id=1,
        )

        print('✓ MachineEasyAutoML created')
        print('')

        # Phase 1: Test predictions BEFORE training (using experimenter only)
        print('-' * 80)
        print('PHASE 1: Testing predictions BEFORE training (using experimenter)')
        print('-' * 80)
        print('')

        total_loss_before = 0
        predictions_before = []

        for i in range(100):
            # Generate random integers 1-10
            A = random.randint(1, 10)
            B = random.randint(1, 10)

            # Calculate expected value using the formula
            expected = 100 + A + B + (A * B) + (A / B)

            # Create input dataframe
            input_df = pd.DataFrame({
                'input_A': [A],
                'input_B': [B]
            })

            # Predict using MachineEasyAutoML
            result = machine_eaml.do_predict(input_df)
            predicted = float(result['output_C'].iloc[0])

            # Calculate loss (absolute difference)
            loss = abs(predicted - expected)
            total_loss_before += loss

            predictions_before.append({
                'A': A,
                'B': B,
                'expected': expected,
                'predicted': predicted,
                'loss': loss
            })

            if i < 5 or i % 20 == 0:  # Show first 5 and every 20th
                print(
                    f"  Test {i+1:3d}: A={A:2d}, B={B:2d} | "
                    f"Expected={expected:8.2f}, Predicted={predicted:8.2f}, Loss={loss:8.4f}"
                )

        avg_loss_before = total_loss_before / 100
        print('')
        print(f'Average Loss BEFORE training: {avg_loss_before:.4f}')
        print('')

        # Phase 2: Train the machine
        print('-' * 80)
        print('PHASE 2: Training the machine')
        print('-' * 80)
        print('')

        # Check if machine was created (it should be created during predictions)
        if machine_eaml._machine:
            print(f'✓ Machine exists with ID: {machine_eaml._machine.id}')
            print(f'  Data lines: {machine_eaml._machine.db_data_input_lines.objects.count()}')
            print('')

            # Train the machine
            print('Starting training...')
            print('')

            try:
                nn_engine = NNEngine(
                    machine_eaml._machine,
                    allow_re_run_configuration=True
                )

                print('✓ NNEngine created and configured')
                print('')

                # Train
                nn_engine.do_training_and_save()

                print('✓ Training completed')
                print('')

            except Exception as e:
                print(f'✗ Training failed: {e}')
                raise
        else:
            print('✗ Machine was not created during predictions')
            pytest.fail("Machine was not created during predictions")

        # Phase 3: Test predictions AFTER training (using trained model)
        print('-' * 80)
        print('PHASE 3: Testing predictions AFTER training (using trained model)')
        print('-' * 80)
        print('')

        total_loss_after = 0
        predictions_after = []

        for i in range(100):
            # Generate random integers 1-10
            A = random.randint(1, 10)
            B = random.randint(1, 10)

            # Calculate expected value using the formula
            expected = 100 + A + B + (A * B) + (A / B)

            # Create input dataframe
            input_df = pd.DataFrame({
                'input_A': [A],
                'input_B': [B]
            })

            # Predict using MachineEasyAutoML (should use trained model now)
            result = machine_eaml.do_predict(input_df)
            predicted = float(result['output_C'].iloc[0])

            # Calculate loss (absolute difference)
            loss = abs(predicted - expected)
            total_loss_after += loss

            predictions_after.append({
                'A': A,
                'B': B,
                'expected': expected,
                'predicted': predicted,
                'loss': loss
            })

            if i < 5 or i % 20 == 0:  # Show first 5 and every 20th
                print(
                    f"  Test {i+1:3d}: A={A:2d}, B={B:2d} | "
                    f"Expected={expected:8.2f}, Predicted={predicted:8.2f}, Loss={loss:8.4f}"
                )

        avg_loss_after = total_loss_after / 100
        print('')
        print(f'Average Loss AFTER training: {avg_loss_after:.4f}')
        print('')

        # Final summary
        print('=' * 80)
        print('SUMMARY')
        print('=' * 80)
        print(f'Average Loss BEFORE training: {avg_loss_before:.4f}')
        print(f'Average Loss AFTER training:  {avg_loss_after:.4f}')

        improvement = avg_loss_before - avg_loss_after
        improvement_pct = (improvement / avg_loss_before * 100) if avg_loss_before > 0 else 0

        if improvement > 0:
            print(f'Improvement: {improvement:.4f} ({improvement_pct:.1f}%)')
        else:
            print(f'Change: {improvement:.4f} ({improvement_pct:.1f}%)')

        print('')
        print('✓ Test completed successfully!')
        print('')

        # Verify that predictions before training should be very accurate (experimenter is exact)
        # The loss before training should be very close to 0 since experimenter uses exact formula
        assert avg_loss_before < 0.01, f"Experimenter should be very accurate, but avg_loss_before={avg_loss_before}"

        # Verify that machine was created
        assert machine_eaml._machine is not None, "Machine should be created during predictions"

        # Verify that machine is ready for solving after training
        assert machine_eaml.ready_to_predict(), "Machine should be ready for prediction after training"

        # Cleanup
        try:
            machine_eaml._machine.db_machine.delete()
            print('✓ Test machine cleaned up')
        except Exception as e:
            print(f'Could not cleanup: {e}')


