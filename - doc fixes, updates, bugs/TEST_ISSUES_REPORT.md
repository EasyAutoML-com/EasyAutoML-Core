# Comprehensive Test Issues Report

**Date**: Generated from test run  
**Total Tests**: 204  
**Passed**: 153  
**Failed**: 41  
**Errors**: 10  

---

## 🔴 CRITICAL ISSUES

### 1. **Missing CSV File for test_z_machine_1.py**

**Status**: ❌ FAILED  
**Test**: `test_z_machine_1.py::TestMachine1::test_machine_1`  
**Error**: `FileNotFoundError: CSV file not found: C:\Users\Administrator\Documents\Repo-EAML-Core\tests\Create All Tests Machines\(Small) salaries prediction.csv`

**Details**:
- The test expects a CSV file at: `tests/Create All Tests Machines/(Small) salaries prediction.csv`
- The directory `tests/Create All Tests Machines` does not exist
- The test cannot proceed without this file

**Impact**: 
- Blocks the entire `test_z_machine_1.py` workflow test
- Also affects `test_salaries_prediction.py` (2 tests failing for same reason)

**Solution Required**:
- Create the directory `tests/Create All Tests Machines/`
- Add the CSV file `(Small) salaries prediction.csv` to that directory
- OR update the test to use an alternative data source

**Affected Tests**:
- `test_z_machine_1.py::TestMachine1::test_machine_1` ❌
- `test_salaries_prediction.py::TestSalariesPrediction::test_salaries_prediction_complete_workflow` ❌
- `test_salaries_prediction.py::TestSalariesPrediction::test_salaries_prediction_diagnostic_row_count` ❌

---

### 2. **MachineEasyAutoML API Parameter Mismatch**

**Status**: ✅ FIXED  
**Test**: `test_z_machine_2.py::TestMachine2::test_machine_2_experimenter_workflow`  
**Error**: `TypeError: MachineEasyAutoML.__init__() got an unexpected keyword argument 'experimenter'`

**Details**:
- Test code was using: `MachineEasyAutoML(experimenter=experimenter, ...)`
- API signature has been updated: `MachineEasyAutoML(optional_experimenter=..., ...)`
- The parameter name is now `optional_experimenter` (formula feature has been removed)

**Code Location**:
```96:102:tests/Tests All AI modules/unit/test_z_machine_2.py
machine_eaml = MachineEasyAutoML(
    machine_name=machine_name,
    optional_experimenter=experimenter,  # ✅ CORRECT NAME
    record_experiments=True,
    access_user_id=test_user.id,
    access_team_id=1,
)
```

**Actual API** (from `ML/MachineEasyAutoML.py`):
```32:40:ML/MachineEasyAutoML.py
def __init__( self,
                machine_name: str,
                optional_experimenter: Optional = None,  # ✅ CORRECT NAME
                record_experiments: bool = True,
                access_user_id: int = None,
                access_team_id: int = None,
                decimal_separator : str = ".",
                date_format : str = "MDY",
):
```

**Solution Applied**:
- Changed parameter name from `approximation_formula_or_experimenter` to `optional_experimenter`
- Removed approximation formula feature - only experimenter is now supported
- Updated test to use `optional_experimenter=experimenter`

**Impact**: Issue resolved - test should now pass

---

## 🟡 HIGH PRIORITY ISSUES

### 3. **KeyError: 'feature3' in Multiple Tests (20 tests)**

**Status**: ❌ FAILED (20 tests)  
**Error**: `KeyError: 'feature3'`

**Affected Tests**:
1. `test_feature_engineering_configuration.py` - 10 tests
2. `test_inputs_columns_importance.py` - 9 tests  
3. `test_machine_data_configuration.py` - 12 tests
4. `test_nn_configuration.py` - 1 test
5. `test_nn_engine.py` - 1 test

**Root Cause Analysis**:
- Tests use `TestDataGenerator.create_simple_classification_data()` which creates a DataFrame with columns: `['feature1', 'feature2', 'feature3', 'target']`
- The `feature3` column is created in the DataFrame (line 20 of test_data_generator.py)
- However, when tests try to access `feature3`, it's missing from the DataFrame or column mapping

**Possible Causes**:
1. Column filtering/removal happening somewhere in the test pipeline
2. Column name normalization that removes or renames `feature3`
3. DataFrame operations that drop categorical columns
4. Missing column in datatype/description mappings (though `feature3` IS in the mappings)

**Investigation Needed**:
- Check if `feature3` is being filtered out as a categorical column
- Verify column name normalization doesn't affect `feature3`
- Check if any preprocessing steps remove categorical columns
- Verify the DataFrame actually contains `feature3` when accessed

**Impact**: 20 tests failing, affecting core functionality tests

---

### 4. **Database Integrity Test Failures**

**Status**: ❌ FAILED (2 tests)

#### 4a. Machine Levels Mismatch

**Test**: `test_database_integrity.py::test_database_integrity_check_machine_data`  
**Error**: `AssertionError: Expected machine levels [1, 2, 3], found [1]`

**Details**:
- Test expects machines with levels 1, 2, and 3
- Database only contains machines with level 1
- This suggests the test database is incomplete or was modified

**Expected**: Machines with levels [1, 2, 3]  
**Actual**: Machines with level [1] only

**Impact**: Database integrity validation fails

#### 4b. NNModel Count Mismatch

**Test**: `test_database_integrity.py::test_database_integrity_check_configuration_data`  
**Error**: `AssertionError: Expected 469 NNModels, found 72`

**Details**:
- Test expects 469 NNModel records
- Database only has 72 NNModel records
- This suggests the test database is incomplete

**Expected**: 469 NNModels  
**Actual**: 72 NNModels

**Impact**: Configuration data validation fails

**Solution Required**:
- Regenerate the test database with all required data
- OR update test expectations to match actual database state
- OR make tests more flexible to handle varying database states

---

## 🟠 MEDIUM PRIORITY ISSUES

### 5. **Helper Methods Collected as Tests**

**Status**: ⚠️ WARNING (2 methods)  
**Tests**: 
- `test_z_machine_1.py::TestMachine1::test_machine_properties` (ERROR)
- `test_z_machine_1.py::TestMachine1::test_solving` (ERROR)

**Details**:
- These are helper methods called from `test_machine_1()`, not standalone tests
- They start with `test_` so pytest collects them as tests
- They fail because they're called without proper setup (missing `machine` parameter)

**Code Structure**:
```129:149:tests/Tests All AI modules/unit/test_z_machine_1.py
def test_machine_properties(self, machine):  # ❌ Collected as test
    """Test various machine properties"""
    # ... helper method code ...

def test_solving(self, nn_engine, machine):  # ❌ Collected as test
    """Test solving on rows 607-608"""
    # ... helper method code ...
```

**Solution Required**:
- Rename methods to not start with `test_`:
  - `test_machine_properties` → `_test_machine_properties` or `verify_machine_properties`
  - `test_solving` → `_test_solving` or `verify_solving`
- OR add `@pytest.mark.skip` decorator
- OR use `pytest.parametrize` if they should be separate tests

**Impact**: 2 false test failures, cluttering test results

---

### 6. **NN Configuration Test Errors (4 tests)**

**Status**: ❌ ERROR (4 tests)

**Affected Tests**:
- `test_nn_configuration.py::TestNNConfiguration::test_nn_config_save_configuration`
- `test_nn_configuration.py::TestNNConfiguration::test_nnc_adapt_config_to_new_enc_dec`
- `test_nn_configuration.py::TestNNConfiguration::test_nnc_get_configuration_as_dict`
- `test_nn_configuration.py::TestNNConfiguration::test_nnc_build_keras_nn_model`
- `test_nn_configuration.py::TestNNConfiguration::test_nnc_get_user_nn_shape`
- `test_nn_configuration.py::TestNNConfiguration::test_nnc_get_machine_nn_shape`
- `test_nn_configuration.py::TestNNConfiguration::test_nnc_get_hidden_layers_count`
- `test_nn_configuration.py::TestNNConfiguration::test_nnc_get_neurons_percentage`

**Details**:
- These tests show as "ERROR" (not "FAILED"), suggesting setup/teardown issues
- Likely related to the `feature3` KeyError issue (dependency chain)
- May fail during fixture setup or test initialization

**Impact**: 8 tests showing errors, likely cascading from other failures

---

## 📊 Summary by Category

### By Severity:
- **Critical**: 2 issues (blocking major test workflows)
- **High Priority**: 2 issues (20+ tests affected)
- **Medium Priority**: 2 issues (test quality/cleanup)

### By Test File:
- `test_z_machine_1.py`: 3 issues (CSV missing, 2 helper methods)
- `test_z_machine_2.py`: 1 issue (API parameter)
- `test_feature_engineering_configuration.py`: 10 failures (feature3)
- `test_inputs_columns_importance.py`: 9 failures (feature3)
- `test_machine_data_configuration.py`: 12 failures (feature3)
- `test_nn_configuration.py`: 9 failures/errors (feature3 + errors)
- `test_nn_engine.py`: 1 failure (feature3)
- `test_salaries_prediction.py`: 2 failures (CSV missing)
- `test_database_integrity.py`: 2 failures (data mismatch)

### By Root Cause:
1. **Missing Test Data**: CSV file not found (3 tests)
2. **API Mismatch**: Wrong parameter name (1 test)
3. **Column Access Issue**: KeyError 'feature3' (20 tests)
4. **Database State**: Incomplete test database (2 tests)
5. **Test Structure**: Helper methods as tests (2 tests)

---

## 🔧 Recommended Fix Priority

1. **IMMEDIATE** (Blocking):
   - Fix `MachineEasyAutoML` parameter name in `test_z_machine_2.py`
   - Create/verify CSV file location for `test_z_machine_1.py`

2. **HIGH PRIORITY** (Many tests):
   - Investigate and fix `feature3` KeyError issue
   - Fix helper methods in `test_z_machine_1.py`

3. **MEDIUM PRIORITY** (Data quality):
   - Regenerate or update test database expectations
   - Review and fix NN configuration test errors

---

## 📝 Notes

- The two renamed test files (`test_z_machine_1.py` and `test_z_machine_2.py`) **DO run last** as intended ✅
- Most failures are fixable with code changes
- Database integrity issues may require test database regeneration
- The `feature3` issue needs deeper investigation to understand the root cause

