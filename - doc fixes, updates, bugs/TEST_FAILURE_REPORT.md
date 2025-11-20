# Test Failure Report - All AI Modules

**Generated:** 2025-11-20 07:46:36  
**Test Run Duration:** 322.38 seconds (5 minutes 22 seconds)  
**Total Tests:** 204  
**Passed:** 187 ✅  
**Failed:** 7 ❌  
**Errors:** 10 ⚠️  
**Success Rate:** 91.67%

---

## Executive Summary

The test suite ran successfully with **187 tests passing** out of 204 total tests. However, there are **7 failures** and **10 errors** that need attention. The failures are primarily related to:

1. **Database column mismatches** - Tests expecting specific column names that don't exist in the database tables
2. **User management issues** - Multiple user objects or missing admin users
3. **Configuration dependencies** - Tests requiring existing machine configurations that may not be properly set up
4. **Data type handling** - Issues with column data type detection and handling

---

## Failed Tests (7)

### 1. `test_fec_get_all_column_datas_infos` ❌
**File:** `test_feature_engineering_configuration.py`  
**Test Class:** `TestFeatureEngineeringConfiguration`  
**Error Type:** `KeyError: 'input1'`

**Description:**
The test is trying to access a column named `'input1'` that doesn't exist in the test data. The test uses `simple_dataframe` fixture which has columns: `feature1`, `feature2`, `feature3`, `target`.

**Root Cause:**
The test is using a fixture (`numeric_dataframe` or similar) that has columns named `input1`, `input2`, `output`, but the actual fixture being used (`simple_dataframe`) has different column names.

**Recommendation:**
- Update the test to use the correct fixture (`numeric_dataframe`) instead of `simple_dataframe`
- Or update the test to use the actual column names from `simple_dataframe` (`feature1`, `feature2`, `feature3`, `target`)

---

### 2. `test_mdc_force_inputs_outputs` ❌
**File:** `test_machine_data_configuration.py`  
**Test Class:** `TestMachineDataConfiguration`  
**Error Type:** `KeyError: 'input1'`

**Description:**
Similar to the previous failure, this test is trying to access a column named `'input1'` that doesn't exist in the test data.

**Root Cause:**
Same as above - column name mismatch between expected and actual test data.

**Recommendation:**
- Use the correct fixture with matching column names
- Or update the test to reference the correct column names

---

### 3. `test_mdc_get_parent_of_extended_column` ❌
**File:** `test_machine_data_configuration.py`  
**Test Class:** `TestMachineDataConfiguration`  
**Error Type:** `assert False` - `isinstance(None, str)`

**Description:**
The test expects `get_parent_of_extended_column()` to return a string, but it's returning `None`.

**Root Cause:**
The method is not finding a parent column for the given extended column name, possibly because:
- The extended column doesn't have a parent
- The parent lookup logic is incorrect
- The column name format doesn't match expected patterns

**Recommendation:**
- Verify the test data includes extended columns with proper parent relationships
- Check the `get_parent_of_extended_column()` implementation
- Ensure the column naming convention matches what the method expects

---

### 4. `test_salaries_prediction_complete_workflow` ❌
**File:** `test_salaries_prediction.py`  
**Test Class:** `TestSalariesPrediction`  
**Error Type:** `models.user.User.MultipleObjectsReturned: get() returned more than one User -- it returned 2!`

**Description:**
The test is trying to get a single user with `User.objects.get(email='SuperSuperAdmin@easyautoml.com')`, but there are multiple users with that email in the database.

**Root Cause:**
Duplicate user records exist in the test database with the same email address. This violates the unique constraint that should exist on the email field.

**Recommendation:**
- Clean up duplicate users in the test database
- Update `get_admin_user()` function in `conftest.py` to handle duplicates (it already has some logic for this, but may need improvement)
- Use `filter().first()` instead of `get()` when duplicates might exist
- Ensure database constraints prevent duplicate emails

---

### 5. `test_salaries_prediction_diagnostic_row_count` ❌
**File:** `test_salaries_prediction.py`  
**Test Class:** `TestSalariesPrediction`  
**Error Type:** `pandas.errors.DatabaseError: no such column: SepalLengthCm`

**Description:**
The test is trying to query columns from the Iris flowers dataset (`SepalLengthCm`, `SepalWidthCm`, `PetalLengthCm`, `PetalWidthCm`, `Species`), but these columns don't exist in the database table `Machine_957_DataInputLines`.

**Root Cause:**
The machine was created with different column names than expected, or the data wasn't properly loaded into the database tables. The test is using the Iris flowers CSV file, but the actual column names in the database might be different.

**SQL Query That Failed:**
```sql
SELECT `SepalLengthCm`, `SepalWidthCm`, `PetalLengthCm`, `PetalWidthCm`, `Species` 
FROM Machine_957_DataInputLines 
LEFT JOIN Machine_957_DataOutputLines 
ON Machine_957_DataInputLines.Line_ID = Machine_957_DataOutputLines.Line_ID
```

**Recommendation:**
- Verify the actual column names in the database after machine creation
- Check if the CSV file column names match what's expected
- Ensure the machine's `get_random_dataframe()` method returns columns with the correct names
- Add logging to see what columns are actually created in the database

---

### 6. `test_machine_1` ❌
**File:** `test_z_machine_1.py`  
**Test Class:** `TestMachine1`  
**Error Type:** `pandas.errors.DatabaseError: no such column: SepalLengthCm`

**Description:**
Same issue as test #5 - the test expects Iris flower column names that don't exist in the database.

**Root Cause:**
Same as test #5 - column name mismatch between expected and actual database columns.

**Recommendation:**
- Same as test #5 recommendations
- This is a workflow test that creates a machine from CSV, so verify the CSV loading process creates columns with the expected names

---

### 7. `test_machine_2_experimenter_workflow` ❌
**File:** `test_z_machine_2.py`  
**Test Class:** `TestMachine2`  
**Error Type:** `ValueError: Unable to find super admin 'SuperSuperAdmin@easyautoml.com' in the user database`

**Description:**
The test is trying to create a `MachineEasyAutoML` instance, which requires finding a super admin user, but the user lookup is failing.

**Root Cause:**
The `MachineEasyAutoML.__init__()` method tries to find the super admin user, but:
- The user doesn't exist in the database
- The user lookup logic is failing
- There's a database connection issue

**Code Location:**
`models/user.py:161` - The code raises a ValueError when the super admin user cannot be found.

**Recommendation:**
- Ensure `get_admin_user()` from `conftest.py` is called before creating `MachineEasyAutoML`
- The test does call `get_admin_user()`, but it might be returning `None` or the user might not be properly saved
- Verify the user is created and saved before the `MachineEasyAutoML` initialization
- Check if there's a timing issue where the user needs to be committed to the database first

---

## Error Tests (10)

All errors are in `test_nn_configuration.py` and `test_z_machine_1.py`. These are likely setup/teardown errors or missing fixture dependencies.

### NN Configuration Tests (8 errors)

1. `test_nn_config_save_configuration` ⚠️
2. `test_nnc_adapt_config_to_new_enc_dec` ⚠️
3. `test_nnc_get_configuration_as_dict` ⚠️
4. `test_nnc_build_keras_nn_model` ⚠️
5. `test_nnc_get_user_nn_shape` ⚠️
6. `test_nnc_get_machine_nn_shape` ⚠️
7. `test_nnc_get_hidden_layers_count` ⚠️
8. `test_nnc_get_neurons_percentage` ⚠️

**Common Issue:**
All these tests use the `existing_machine_with_nn_configuration` fixture, which likely:
- Doesn't exist or isn't properly defined
- Fails to find/create a machine with NN configuration
- Has dependency issues with other fixtures

**Recommendation:**
- Verify the `existing_machine_with_nn_configuration` fixture exists in `conftest.py`
- Check if the fixture can properly find/create a machine with NN configuration
- Ensure the test database has machines with NN configurations ready
- Review fixture dependencies and execution order

---

### Machine Workflow Tests (2 errors)

9. `test_machine_properties` ⚠️ (from `test_z_machine_1.py`)
10. `test_solving` ⚠️ (from `test_z_machine_1.py`)

**Common Issue:**
These are likely dependent on `test_machine_1` which failed, or they have similar database column issues.

**Recommendation:**
- Fix the underlying `test_machine_1` failure first
- These tests may pass once the main workflow test is fixed

---

## Common Error Patterns

### 1. Column Name Mismatches
**Affected Tests:** 3 tests  
**Pattern:** Tests expect specific column names (`input1`, `SepalLengthCm`, etc.) that don't match the actual database columns.

**Solution:**
- Standardize test data fixtures
- Use dynamic column name detection instead of hardcoded names
- Verify column names after machine creation

---

### 2. User Management Issues
**Affected Tests:** 2 tests  
**Pattern:** Problems with finding/creating admin users, duplicate users.

**Solution:**
- Improve `get_admin_user()` function to handle edge cases
- Add database cleanup to remove duplicates
- Use transactions to ensure user creation is atomic

---

### 3. Missing Fixtures/Dependencies
**Affected Tests:** 8 tests  
**Pattern:** Tests fail because required fixtures don't exist or fail to set up.

**Solution:**
- Verify all fixtures are properly defined
- Check fixture dependencies and execution order
- Add better error messages when fixtures fail

---

## Test Statistics by Module

| Module | Passed | Failed | Errors | Total |
|--------|--------|--------|--------|-------|
| Database Integrity | 8 | 0 | 0 | 8 |
| EAML DB Models | 20 | 0 | 0 | 20 |
| EncDec | 10 | 0 | 0 | 10 |
| Feature Engineering | 11 | 1 | 0 | 12 |
| Inputs Columns Importance | 10 | 0 | 0 | 10 |
| Machine | 40 | 0 | 0 | 40 |
| Machine Data Configuration | 12 | 2 | 0 | 14 |
| Machine EasyAutoML | 18 | 0 | 0 | 18 |
| NN Configuration | 2 | 0 | 8 | 10 |
| NN Engine | 4 | 0 | 0 | 4 |
| Solution Finder | 8 | 0 | 0 | 8 |
| Solution Score | 24 | 0 | 0 | 24 |
| Salaries Prediction | 0 | 2 | 0 | 2 |
| Machine Workflow 1 | 0 | 1 | 2 | 3 |
| Machine Workflow 2 | 0 | 1 | 0 | 1 |
| **TOTAL** | **187** | **7** | **10** | **204** |

---

## Recommendations

### High Priority (Fix Immediately)

1. **Fix User Management**
   - Resolve duplicate user issue in test database
   - Improve `get_admin_user()` function
   - Ensure users are properly created before tests run

2. **Fix Column Name Issues**
   - Standardize test data fixtures
   - Fix Iris flowers dataset column name handling
   - Update tests to use correct column names

3. **Fix NN Configuration Fixtures**
   - Create/verify `existing_machine_with_nn_configuration` fixture
   - Ensure test database has machines with NN configurations

### Medium Priority

4. **Improve Error Messages**
   - Add more descriptive error messages in tests
   - Log actual vs. expected values when assertions fail
   - Add fixture validation messages

5. **Database Cleanup**
   - Ensure proper cleanup between tests
   - Remove duplicate records
   - Verify database state before tests

### Low Priority

6. **Test Documentation**
   - Add more comments explaining test expectations
   - Document fixture requirements
   - Add troubleshooting guides for common failures

---

## Next Steps

1. **Immediate Actions:**
   - Fix the `get_admin_user()` function to handle duplicates
   - Fix column name mismatches in Iris flowers tests
   - Create/verify NN configuration fixtures

2. **Short-term:**
   - Run tests again after fixes
   - Verify all 204 tests pass
   - Update test documentation

3. **Long-term:**
   - Add test coverage reporting
   - Implement continuous integration
   - Add automated test result reporting

---

## Additional Notes

- All database integrity tests passed ✅
- Most core functionality tests passed ✅
- Errors are primarily in integration/workflow tests
- The test suite is generally healthy with 91.67% success rate

---

## Error Details

### Detailed Error Messages

#### Test: `test_fec_get_all_column_datas_infos`
```
KeyError: 'input1'
```
**Location:** `test_feature_engineering_configuration.py:271`

#### Test: `test_mdc_force_inputs_outputs`
```
KeyError: 'input1'
```
**Location:** `test_machine_data_configuration.py:419`

#### Test: `test_mdc_get_parent_of_extended_column`
```
assert False
 +  where False = isinstance(None, str)
```
**Location:** `test_machine_data_configuration.py:555`

#### Test: `test_salaries_prediction_complete_workflow`
```
models.user.User.MultipleObjectsReturned: get() returned more than one User -- it returned 2!
```
**Location:** `test_salaries_prediction.py:31`

#### Test: `test_salaries_prediction_diagnostic_row_count`
```
pandas.errors.DatabaseError: Execution failed on sql: 
SELECT `SepalLengthCm`, `SepalWidthCm`, `PetalLengthCm`, `PetalWidthCm`, `Species` 
FROM Machine_957_DataInputLines 
LEFT JOIN Machine_957_DataOutputLines 
ON Machine_957_DataInputLines.Line_ID = Machine_957_DataOutputLines.Line_ID;
no such column: SepalLengthCm
unable to rollback
```
**Location:** `test_salaries_prediction.py`

#### Test: `test_machine_1`
```
pandas.errors.DatabaseError: Execution failed on sql: 
SELECT `SepalLengthCm`, `SepalWidthCm`, `PetalLengthCm`, `PetalWidthCm`, `Species` 
FROM Machine_957_DataInputLines 
LEFT JOIN Machine_957_DataOutputLines 
ON Machine_957_DataInputLines.Line_ID = Machine_957_DataOutputLines.Line_ID;
no such column: SepalLengthCm
unable to rollback
```
**Location:** `test_z_machine_1.py:55`

#### Test: `test_machine_2_experimenter_workflow`
```
ValueError: Unable to find super admin 'SuperSuperAdmin@easyautoml.com' in the user database
```
**Location:** `models/user.py:161` (called from `test_z_machine_2.py:96`)

---

## Conclusion

The test suite shows a **91.67% success rate**, which is good, but the 7 failures and 10 errors need to be addressed. The main issues are:

1. **Column name mismatches** - Need to standardize test data
2. **User management** - Need to fix duplicate user handling
3. **Missing fixtures** - Need to create/verify NN configuration fixtures

Once these issues are resolved, the test suite should achieve close to 100% pass rate.

---

*Report generated automatically from pytest test run results*

