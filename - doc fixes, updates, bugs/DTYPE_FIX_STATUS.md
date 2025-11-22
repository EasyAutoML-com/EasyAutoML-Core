# Database Dtype Fix Status

## Problem Summary

The tests are failing because of a datatype mismatch between:
1. How data is written to the database (as strings)
2. How Django expects to read it back (as numeric types for DecimalField)

## Root Cause

When `pandas.DataFrame.to_sql()` writes data to SQLite using a raw DB-API connection (not SQLAlchemy), it converts all values to strings unless the DataFrame columns have the correct pandas dtype.

The issue chain:
1. Data is read from CSV → DataFrame columns are float64
2. Machine creates dynamic tables with DecimalField for FLOAT columns
3. Data is appended using `to_sql()` with raw SQLite connection
4. **BUG**: `to_sql()` writes float values as TEXT strings ('5.9') instead of REAL numbers (5.9)
5. When Django queries the data back, it tries to convert TEXT '5.9' to Decimal
6. SQLite operations.py line 346 fails: `TypeError: argument must be int or float`

## What We Fixed

### 1. Removed dtype parameter from to_sql() ✅
- **File**: `models/dynamic_model.py`
- **Change**: Removed `dtype=dtype_dict` parameter from `to_sql()` calls
- **Reason**: When appending to existing tables, dtype parameter causes conflicts

### 2. Added DataFrame type conversion ✅
- **File**: `models/dynamic_model.py` lines 68-115
- **Change**: Convert DataFrame columns to match Django model field types before writing
- **Types handled**:
  - DecimalField → float64
  - FloatField → float64
  - BooleanField → int (0/1 for SQLite)
  - IntegerField → int
  - BigIntegerField → int

## Current Status

**Status**: ⚠️ PARTIAL FIX - Data writes successfully but query fails

### What Works:
- ✅ No more "not a string" errors during `to_sql()`
- ✅ Data is written to database successfully
- ✅ `machine data input lines was appended` message appears

### What Doesn't Work:
- ❌ Querying data back fails with `TypeError: argument must be int or float`
- ❌ Django tries to convert TEXT strings to Decimal and fails
- ❌ The data is still being written as TEXT instead of REAL/INTEGER

## The Real Problem

The issue is that pandas `to_sql()` with a raw SQLite connection **ignores DataFrame dtypes** and writes everything as TEXT.

From the error traceback:
```python
results = [[(150, 0, 0, 1, 0, 0, '5.9', '3.0', '5.1', '1.8')]]
                                  ^^^^^  ^^^^^  ^^^^^  ^^^^^ 
                                  These should be floats, not strings!
```

## Potential Solutions

### Option 1: Use SQLAlchemy Engine (RECOMMENDED)
Force SQLite to also use SQLAlchemy engine instead of raw connection:

```python
# In models/dynamic_model.py
if 'sqlite' in engine_name:
    # Use SQLAlchemy engine for SQLite too (like MySQL/PostgreSQL)
    dataframe.to_sql(
        name=name_of_table,
        con=_get_alchemy_engine(),  # Use SQLAlchemy, not raw connection
        if_exists="append",
        method="multi"
    )
```

**Pros**:
- SQLAlchemy properly handles type conversions
- Consistent behavior across all databases
- Respects DataFrame dtypes

**Cons**:
- Might have different locking behavior
- Need to test for database locking issues

### Option 2: Manual SQL INSERT
Write data using parameterized SQL INSERT statements:

```python
# Build INSERT statement with proper parameter binding
# This ensures types are preserved
```

**Pros**:
- Full control over types
- No ambiguity

**Cons**:
- More complex code
- Slower for bulk inserts

### Option 3: Fix Table Schema
Change DecimalField to FloatField in TYPE_MAPPING:

```python
# In models/machine.py
TYPE_MAPPING = {
    "FLOAT": (models.FloatField, {"null": True}),  # Changed from DecimalField
}
```

**Pros**:
- Simpler schema
- Float is more natural for ML data

**Cons**:
- Requires database migration
- Might affect existing data

## Recommendation

**Use Option 1**: Switch SQLite to use SQLAlchemy engine like MySQL/PostgreSQL.

This is the cleanest solution because:
1. It's a minimal code change
2. It makes all databases behave consistently
3. SQLAlchemy is designed to handle type conversions properly
4. The locking issue mentioned in comments might be outdated or fixable

## Files Modified

1. `models/dynamic_model.py`:
   - Lines 68-115: Added DataFrame type conversion
   - Lines 117-184: Removed dtype_dict building (no longer needed)
   - Lines 185-198: Removed dtype parameter from to_sql()
   - Lines 206-215: Removed dtype parameter from MySQL/PostgreSQL path

## Next Steps

1. Try Option 1 (use SQLAlchemy for SQLite)
2. If locking issues occur, investigate connection pooling settings
3. Test all 4 failing tests to ensure they pass
4. Document the fix in the codebase

## Test Status

- `test_salaries_prediction_complete_workflow`: ❌ FAILED (TypeError: argument must be int or float)
- `test_salaries_prediction_diagnostic_row_count`: ❌ NOT TESTED YET
- `test_machine_1`: ❌ NOT TESTED YET
- `test_machine_2_experimenter_workflow`: ❌ FAILED (Different issue: User lookup)

