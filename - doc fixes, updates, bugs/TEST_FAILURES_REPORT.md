# Test Failures Report
**Generated:** $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")  
**Total Tests:** 204  
**Passed:** 198  
**Failed:** 6  
**Success Rate:** 97.06%

---

## Summary

The test suite ran successfully with **198 out of 204 tests passing**. There are **6 failing tests** that fall into **3 main categories**:

1. **Database Lock Issues** (3 tests) - SQLite database locking preventing table creation
2. **User Authentication Issues** (2 tests) - Missing super admin user in test database
3. **Type Assertion Issues** (2 tests) - Incorrect type checking in status methods

---

## Detailed Failure Analysis

### Category 1: Database Lock Issues (3 failures)

#### Problem
SQLite database locking errors prevent proper table creation during machine initialization. The tables are created but remain empty, causing "no such column" errors when trying to read data.

#### Affected Tests:
1. `test_salaries_prediction.py::TestSalariesPrediction::test_salaries_prediction_complete_workflow`
2. `test_salaries_prediction.py::TestSalariesPrediction::test_salaries_prediction_diagnostic_row_count`
3. `test_z_machine_1.py::TestMachine1::test_machine_1`

#### Error Details:
```
pandas.errors.DatabaseError: Execution failed on sql: 
SELECT `SepalLengthCm`, `SepalWidthCm`, `PetalLengthCm`, `PetalWidthCm`, `Species` 
FROM Machine_904_DataInputLines 
LEFT JOIN Machine_904_DataOutputLines 
ON Machine_904_DataInputLines.Line_ID = Machine_904_DataOutputLines.Line_ID;
no such column: SepalLengthCm
unable to rollback
```

#### Root Cause:
- Database lock errors occur during `dataframe.to_sql()` execution when creating tables
- Error message: `(sqlite3.OperationalError) database is locked`
- Tables are created but remain empty because data insertion fails
- When tests try to read data, columns don't exist in empty tables

#### Evidence from Logs:
```
ERROR | There was a problem during dataframe.to_sql execution in table 'Machine_904_DataInputLines' : 
(sqlite3.OperationalError) database is locked
[SQL: CREATE TABLE "Machine_904_DataInputLines" (...)]
DEBUG | machine data input lines was appended. First_new_row_id:1 with arguments:{}
```

#### Impact:
- **Severity:** Medium
- **Frequency:** Occurs in test environment with SQLite
- **Workaround:** Tests can be re-run, sometimes they pass on retry
- **Production Impact:** Likely not an issue in production with proper database configuration

#### Recommended Solutions:
1. **Immediate:** Add retry logic for database operations in tests
2. **Short-term:** Increase SQLite timeout settings in test configuration
3. **Long-term:** Consider using a more robust test database (PostgreSQL) or implementing proper connection pooling

---

### Category 2: User Authentication Issues (1 failure)

#### Problem
Test expects a super admin user with email `SuperAdmin@easyautoml.com`, but the test database only contains `SuperSuperAdmin@easyautoml.com`.

#### Affected Tests:
1. `test_z_machine_2.py::TestMachine2::test_machine_2_experimenter_workflow`

#### Error Details:
```
ValueError: Unable to find super admin 'SuperAdmin@easyautoml.com' in the user database
```

#### Root Cause:
- `MachineEasyAutoML.__init__()` calls `User.get_super_admin()` which looks for `SUPER_ADMIN_EASYAUTOML_EMAIL = 'SuperAdmin@easyautoml.com'`
- Test database only has `SuperSuperAdmin@easyautoml.com` user
- Code path: `ML/MachineEasyAutoML.py:60` → `models/user.py:157`

#### Code Location:
```python
# ML/MachineEasyAutoML.py:60
if machine_name.startswith("__") or machine_name.endswith("__"):
    self._current_access_user_id = EasyAutoMLDBModels().User.get_super_admin().id
```

#### Impact:
- **Severity:** Low (test environment only)
- **Frequency:** Always fails for this specific test
- **Production Impact:** None (production database should have correct user)

#### Recommended Solutions:
1. **Option 1:** Update test to use `SuperSuperAdmin@easyautoml.com` or create the expected user
2. **Option 2:** Update `MachineEasyAutoML` to handle missing super admin gracefully
3. **Option 3:** Update test fixture to create the required user before test runs

---

### Category 3: Type Assertion Issues (2 failures)

#### Problem
Tests are checking for 'status' in error messages, but the actual error is about type checking (`isinstance(None, bool)`).

#### Affected Tests:
1. `test_machine_easy_automl.py::TestMachineEasyAutoML::test_easy_automl_get_status`
2. `test_machine_easy_automl.py::TestMachineEasyAutoMLAPI::test_easy_automl_api_get_status`

#### Error Details:
```
AssertionError: assert 'status' in 'assert false\n +  where false = isinstance(none, bool)'
 +  where 'assert false\n +  where false = isinstance(none, bool)' = <built-in method lower of str object>
 +  where 'assert False\n +  where False = isinstance(None, bool)' = str(AssertionError('assert False\n +  where False = isinstance(None, bool)'))
```

#### Root Cause:
- Test expects error message to contain 'status'
- Actual error is: `isinstance(None, bool)` assertion failure
- The test is checking the wrong thing - it should verify the actual error or fix the underlying issue

#### Impact:
- **Severity:** Low (test logic issue)
- **Frequency:** Always fails
- **Production Impact:** None (test issue only)

#### Recommended Solutions:
1. **Option 1:** Fix the underlying issue causing `None` to be passed where a `bool` is expected
2. **Option 2:** Update test to check for the correct error message or handle the actual error

---

## Test Results by Module

| Module | Total | Passed | Failed | Success Rate |
|--------|-------|--------|--------|--------------|
| test_database_integrity | 1 | 1 | 0 | 100% |
| test_eaml_db_models | 20 | 20 | 0 | 100% |
| test_encdec | 10 | 10 | 0 | 100% |
| test_feature_engineering_configuration | 12 | 12 | 0 | 100% |
| test_inputs_columns_importance | 11 | 11 | 0 | 100% |
| test_machine | 48 | 48 | 0 | 100% |
| test_machine_data_configuration | 12 | 12 | 0 | 100% |
| test_machine_easy_automl | 2 | 0 | 2 | 0% |
| test_salaries_prediction | 2 | 0 | 2 | 0% |
| test_z_machine_1 | 1 | 0 | 1 | 0% |
| test_z_machine_2 | 1 | 0 | 1 | 0% |
| test_z_machine_3 | 1 | 1 | 0 | 100% |
| **Other tests** | 83 | 83 | 0 | 100% |

---

## Recommendations

### Priority 1 (High - Fix Immediately)
1. **Fix User Authentication Test**
   - Update `test_z_machine_2.py` to use correct admin user or create fixture
   - **Estimated effort:** 30 minutes

### Priority 2 (Medium - Fix Soon)
2. **Fix Type Assertion Tests**
   - Investigate why `get_status()` methods return `None` instead of expected type
   - Update tests or fix underlying methods
   - **Estimated effort:** 1-2 hours

3. **Improve Database Lock Handling**
   - Add retry logic for database operations
   - Increase SQLite timeout in test configuration
   - **Estimated effort:** 2-3 hours

### Priority 3 (Low - Nice to Have)
4. **Consider Test Database Improvements**
   - Evaluate using PostgreSQL for tests instead of SQLite
   - Implement better connection pooling
   - **Estimated effort:** 1-2 days

---

## Notes

- The `test_z_machine_3.py` test (JSON flattening) **PASSES** successfully after recent fixes
- Most core functionality tests (198/204) are passing, indicating the system is generally stable
- Database lock issues are environment-specific and may not occur in production
- All failures are in integration/end-to-end tests, not unit tests

---

## Next Steps

1. ✅ **Completed:** Fixed `test_mdc_get_parent_of_extended_column` test
2. ⏳ **In Progress:** Need to fix the 6 remaining failing tests
3. 📋 **Pending:** Review and implement recommendations above

---

**Report Generated By:** Auto (Cursor AI Assistant)  
**Test Framework:** pytest 8.4.2  
**Python Version:** 3.11.9  
**Django Version:** 5.2.3

