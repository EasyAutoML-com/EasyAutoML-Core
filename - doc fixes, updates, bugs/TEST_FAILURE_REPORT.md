# Test Failure Report

**Date:** November 20, 2025  
**Test Suite:** Tests All AI modules  
**Total Tests Run:** 80  
**Passed:** 30  
**Failed:** 50  
**Execution Time:** 306.16s (5 minutes 6 seconds)

---

## Executive Summary

The test suite has **50 failures** out of 80 tests, with a **62.5% failure rate**. The failures fall into two distinct categories:

1. **Database User Duplication Issue** (49 failures) - Multiple User objects with the same email
2. **Missing Database Models** (1 failure) - EasyAutoMLDBModels cannot access models

---

## Critical Issues

### Issue #1: Multiple User Objects with Same Email (49 failures)

**Error Type:** `models.user.User.MultipleObjectsReturned`

**Error Message:**
```
get() returned more than one User -- it returned 2!
```

**Root Cause:**
The test database contains duplicate User records with the email `SuperSuperAdmin@easyautoml.com`. When tests try to use `get_or_create()`, it fails because `get()` expects a unique result but finds 2 users.

**Affected Test Files:**
- `test_feature_engineering_configuration.py` (13 failures)
- `test_inputs_columns_importance.py` (11 failures)
- `test_machine.py` (25 failures)

**Affected Tests:**
1. `test_fec_create_minimum_configuration`
2. `test_fec_load_configuration`
3. `test_fec_save_configuration`
4. `test_fec_get_all_column_datas_infos`
5. `test_fec_fet_activation_logic`
6. `test_fec_cost_per_columns`
7. `test_fec_invalid_parameters`
8. `test_fec_with_different_data_types`
9. `test_fec_find_delay_tracking`
10. `test_fec_set_this_fec_in_columns_configuration`
11. `test_fec_store_this_fec_to_fet_list_configuration`
12. `test_fec_get_column_data_overview_information`
13. `test_ici_create_minimum_configuration`
14. `test_ici_load_configuration`
15. `test_ici_save_configuration`
16. `test_ici_importance_evaluation_structure`
17. `test_ici_input_output_columns_separation`
18. `test_ici_find_delay_tracking`
19. `test_ici_invalid_parameters`
20. `test_ici_with_different_data_types`
21. `test_ici_minimum_configuration_equal_importance`
22. `test_ici_with_numeric_data`
23. `test_machine_create_with_dataframe`
24. `test_machine_save_and_load_by_id`
25. `test_machine_load_by_name`
26. `test_machine_get_random_dataframe`
27. `test_machine_config_ready_flags`
28. `test_machine_clear_config_methods`
29. `test_machine_repr`
30. `test_machine_with_machine_level`
31. `test_machine_access_check`
32. `test_machine_data_lines_get_last_id`
33. `test_machine_data_lines_create_both_tables`
34. `test_machine_data_lines_read`
35. `test_machine_data_lines_update`
36. `test_machine_data_lines_delete_all`
37. `test_machine_data_lines_append`
38. `test_machine_data_lines_mark`
39. `test_machine_data_input_lines_read`
40. `test_machine_data_input_lines_append`
41. `test_machine_data_input_lines_mark`
42. `test_machine_data_input_lines_mark_all_IsForLearning_as_IsLearned`
43. `test_machine_data_input_lines_update`
44. `test_machine_data_input_lines_count`
45. `test_machine_data_output_lines_read`
46. `test_machine_data_output_lines_append`
47. `test_machine_data_output_lines_mark`
48. `test_machine_data_output_lines_update`
49. `test_machine_data_output_lines_count`

**Problem Location:**
```python
# File: tests/Tests All AI modules/unit/test_machine.py (line 101)
def _get_admin_user(self):
    """Helper method to get or create admin user for machine ownership"""
    from django.contrib.auth import get_user_model
    User = get_user_model()
    admin_user, created = User.objects.get_or_create(
        email='SuperSuperAdmin@easyautoml.com',  # ← This email exists twice!
        defaults={
            'first_name': 'Test',
            'last_name': 'EasyAutoML',
            'is_staff': True,
            'is_superuser': True,
            'is_active': True,
        }
    )
    return admin_user
```

**Database State:**
The test database (`start_set_databases.sqlite3`) contains 2 User records:
1. User: Admin User SuperSuperAdmin@easyautoml.com
2. User: Test EasyAutoML SuperSuperAdmin@easyautoml.com

**Solution Options:**

**Option A: Fix the Test Database (RECOMMENDED)**
```python
# Clean up duplicate users in start_set_databases.sqlite3
# Keep only one user with email 'SuperSuperAdmin@easyautoml.com'
```

**Option B: Fix the Helper Method**
```python
def _get_admin_user(self):
    """Helper method to get or create admin user for machine ownership"""
    from django.contrib.auth import get_user_model
    User = get_user_model()
    # Use filter().first() instead of get_or_create()
    admin_user = User.objects.filter(
        email='SuperSuperAdmin@easyautoml.com'
    ).first()
    
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
```

**Option C: Use a Unique Test Email**
```python
def _get_admin_user(self):
    """Helper method to get or create admin user for machine ownership"""
    from django.contrib.auth import get_user_model
    User = get_user_model()
    # Use a unique email for tests
    admin_user, created = User.objects.get_or_create(
        email='test_admin_unique@easyautoml.com',  # ← Different email
        defaults={
            'first_name': 'Test',
            'last_name': 'EasyAutoML',
            'is_staff': True,
            'is_superuser': True,
            'is_active': True,
        }
    )
    return admin_user
```

---

### Issue #2: EasyAutoMLDBModels Cannot Access Models (1 failure)

**Error Type:** `AssertionError`

**Error Message:**
```
assert None is not None
```

**Test:** `test_eaml_db_models.py::TestEasyAutoMLDBModels::test_eaml_db_models_all_models_access`

**Root Cause:**
The `EasyAutoMLDBModels` class is returning `None` when trying to access certain models. This suggests that the models are not properly registered or the accessor methods are not working correctly.

**Problem Location:**
```python
# File: tests/Tests All AI modules/unit/test_eaml_db_models.py
def test_eaml_db_models_all_models_access(self):
    """Test that all models can be accessed through EasyAutoMLDBModels"""
    # Some model accessor is returning None
    assert some_model is not None  # ← This fails
```

**Possible Causes:**
1. Model not imported in `EasyAutoMLDBModels.py`
2. Model accessor method returns `None` instead of the model class
3. Django models not properly initialized in test environment

**Solution:**
Need to investigate `ML/EasyAutoMLDBModels.py` to identify which model accessor is failing and fix the implementation.

---

## Passed Tests (30 tests)

The following test categories are working correctly:

### Database Integrity Tests (5 passed)
- ✅ `test_database_integrity_check_tables`
- ✅ `test_database_integrity_check_machine_data`
- ✅ `test_database_integrity_check_configuration_data`
- ✅ `test_database_integrity_check_data_lines_tables`
- ✅ `test_database_integrity_check_constraints`

### Dependency Tests (1 passed)
- ✅ `test_dependencies_all_imports`

### EncDec Tests (7 passed)
- ✅ `test_encdec_initialization`
- ✅ `test_encdec_encode_decode_numeric`
- ✅ `test_encdec_encode_decode_categorical`
- ✅ `test_encdec_encode_decode_mixed`
- ✅ `test_encdec_save_load_configuration`
- ✅ `test_encdec_handle_missing_values`
- ✅ `test_encdec_invalid_data_types`

### Machine Data Configuration Tests (6 passed)
- ✅ `test_mdc_create_minimum_configuration`
- ✅ `test_mdc_load_configuration`
- ✅ `test_mdc_save_configuration`
- ✅ `test_mdc_column_configuration`
- ✅ `test_mdc_invalid_parameters`
- ✅ `test_mdc_with_different_data_types`

### Neural Network Configuration Tests (5 passed)
- ✅ `test_nn_config_create_minimum`
- ✅ `test_nn_config_load_save`
- ✅ `test_nn_config_layer_configuration`
- ✅ `test_nn_config_activation_functions`
- ✅ `test_nn_config_optimizer_settings`

### Neural Network Engine Tests (3 passed)
- ✅ `test_nn_engine_initialization`
- ✅ `test_nn_engine_build_model`
- ✅ `test_nn_engine_compile_model`

### Solution Finder Tests (2 passed)
- ✅ `test_solution_finder_initialization`
- ✅ `test_solution_finder_find_solutions`

### Solution Score Tests (1 passed)
- ✅ `test_solution_score_calculation`

---

## Impact Analysis

### High Priority (Blocking)
- **User Duplication Issue**: Affects 49 tests across 3 test files. This is the primary blocker preventing most tests from running.

### Medium Priority
- **EasyAutoMLDBModels Issue**: Affects 1 test but may indicate a deeper problem with model registration.

### Low Priority
- None identified

---

## Recommendations

### Immediate Actions (Priority 1)

1. **Fix User Duplication in Test Database**
   - Open `start_set_databases.sqlite3`
   - Query: `SELECT * FROM user WHERE email='SuperSuperAdmin@easyautoml.com'`
   - Delete duplicate user records, keep only one
   - Or update the `_get_admin_user()` helper method to handle duplicates

2. **Update Test Helper Method**
   - Modify `_get_admin_user()` in test files to use `.filter().first()` instead of `.get_or_create()`
   - This will prevent the MultipleObjectsReturned error

### Short-term Actions (Priority 2)

3. **Fix EasyAutoMLDBModels**
   - Investigate which model accessor is returning `None`
   - Ensure all models are properly imported and registered
   - Add proper error handling for missing models

4. **Add Database Validation**
   - Add a pre-test check to verify no duplicate users exist
   - Add fixture to clean up duplicate users before tests run

### Long-term Actions (Priority 3)

5. **Improve Test Isolation**
   - Ensure each test creates its own unique users
   - Use unique identifiers (UUIDs) for test data
   - Improve cleanup fixtures to remove all test data

6. **Add Test Documentation**
   - Document the expected state of `start_set_databases.sqlite3`
   - Add validation scripts to verify database integrity
   - Create setup scripts to regenerate clean test databases

---

## Test Execution Details

### Environment
- **OS:** Windows 10 (Build 26100)
- **Python:** 3.11+
- **Test Framework:** pytest
- **Database:** SQLite (start_set_databases.sqlite3)
- **Django:** Configured with test database

### Command Used
```bash
python -m pytest "tests/Tests All AI modules/unit" -v --tb=short --maxfail=50
```

### Execution Time Breakdown
- Total Time: 306.16 seconds (5 minutes 6 seconds)
- Average per test: ~3.8 seconds
- Database setup overhead: Significant (each test copies database)

---

## Next Steps

1. ✅ **Identify root cause** - COMPLETED (User duplication in database)
2. ⏳ **Fix test database** - PENDING (Remove duplicate users)
3. ⏳ **Update helper methods** - PENDING (Handle duplicates gracefully)
4. ⏳ **Fix EasyAutoMLDBModels** - PENDING (Investigate None returns)
5. ⏳ **Re-run tests** - PENDING (Verify fixes)
6. ⏳ **Document fixes** - PENDING (Update test documentation)

---

## Appendix

### Failed Test Summary by Category

| Category | Failed | Total | Pass Rate |
|----------|--------|-------|-----------|
| Feature Engineering Configuration | 13 | 13 | 0% |
| Inputs Columns Importance | 11 | 11 | 0% |
| Machine | 25 | 25 | 0% |
| EAML DB Models | 1 | 1 | 0% |
| **Total** | **50** | **50** | **0%** |

### Passed Test Summary by Category

| Category | Passed | Total | Pass Rate |
|----------|--------|-------|-----------|
| Database Integrity | 5 | 5 | 100% |
| Dependencies | 1 | 1 | 100% |
| EncDec | 7 | 7 | 100% |
| Machine Data Configuration | 6 | 6 | 100% |
| Neural Network Configuration | 5 | 5 | 100% |
| Neural Network Engine | 3 | 3 | 100% |
| Solution Finder | 2 | 2 | 100% |
| Solution Score | 1 | 1 | 100% |
| **Total** | **30** | **30** | **100%** |

### Overall Statistics

- **Total Tests:** 80
- **Passed:** 30 (37.5%)
- **Failed:** 50 (62.5%)
- **Success Rate:** 37.5%

---

**Report Generated:** November 20, 2025  
**Report Version:** 1.0
