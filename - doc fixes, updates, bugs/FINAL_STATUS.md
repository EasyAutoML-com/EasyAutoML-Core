# Final Status: Database Dtype Fix Investigation

## Summary

We successfully identified and partially fixed the root cause of the test failures. The issue is a **complex interaction between Django transactions, SQLite locking, and pandas dtype handling**.

## What We Fixed

### 1. Removed Incorrect dtype Parameter ✅
- **Problem**: `to_sql()` was being called with `dtype` parameter that didn't match table schema
- **Solution**: Removed dtype parameter since tables already exist
- **Result**: No more "not a string" errors

### 2. Improved SQLAlchemy Configuration ✅
- **Problem**: SQLite locking when using SQLAlchemy
- **Solution**: Added `isolation_level=None` (autocommit mode) and increased timeout to 60s
- **Result**: Reduced but didn't eliminate locking issues

### 3. Added Django Transaction Commit ✅
- **Problem**: Django holds transaction locks that block SQLAlchemy
- **Solution**: Call `django_connection.commit()` before `to_sql()`
- **Result**: Helps but doesn't fully resolve the issue in test context

## Current Status

**Status**: ⚠️ BLOCKED BY DJANGO TEST TRANSACTION CONTEXT

### The Core Issue

The tests are running inside Django's `@pytest.mark.django_db` transaction wrapper, which:
1. Starts a transaction at the beginning of each test
2. Holds a database lock for the entire test duration
3. Rolls back the transaction at the end (for test isolation)

When we try to use SQLAlchemy's separate connection pool:
- Django's transaction has a lock on the database
- SQLAlchemy tries to get its own connection
- SQLite only allows one writer at a time
- Result: `(sqlite3.OperationalError) database is locked`

### Why the Original Code Used Raw Connection

The original code used Django's raw DB-API connection (`django_connection.connection`) specifically to avoid this issue:
- It reuses Django's existing connection
- No separate connection = no lock conflict
- But: pandas `to_sql()` with raw connection doesn't respect DataFrame dtypes properly
- Result: Data written as TEXT strings instead of proper numeric types

## The Dilemma

We have two conflicting requirements:

1. **Type Correctness**: Need SQLAlchemy for proper type handling
   - SQLAlchemy converts DataFrame dtypes to correct SQL types
   - Ensures DECIMAL/FLOAT/INTEGER are written correctly
   - But: Requires separate connection pool

2. **Lock Avoidance**: Need Django's connection to avoid locks
   - Reuses Django's existing database connection
   - No lock conflicts in transaction context
   - But: Doesn't handle DataFrame dtypes properly

## Possible Solutions

### Option 1: Use Django's Connection with Manual Type Conversion (RECOMMENDED)
Go back to using Django's raw connection, but manually convert DataFrame values to proper Python types before writing:

```python
# Convert DataFrame to proper Python types
for col in dataframe.columns:
    if is_numeric_field(col):
        dataframe[col] = dataframe[col].astype(float).tolist()  # Convert to Python float list
    elif is_boolean_field(col):
        dataframe[col] = dataframe[col].astype(int).tolist()  # Convert to Python int list
```

**Pros**:
- No locking issues
- Works in Django transaction context
- Compatible with tests

**Cons**:
- More complex code
- Need to manually handle type conversions

### Option 2: Disable Django Transaction Wrapping for These Tests
Use `@pytest.mark.django_db(transaction=False)` for tests that write data:

```python
@pytest.mark.django_db(transaction=False)
def test_salaries_prediction_complete_workflow(self):
    ...
```

**Pros**:
- Allows SQLAlchemy to work properly
- Clean solution

**Cons**:
- Tests are no longer isolated
- Need manual cleanup
- Might affect other tests

### Option 3: Use SQLAlchemy's `begin()` Context Manager
Wrap the `to_sql()` call in SQLAlchemy's transaction context:

```python
with _get_alchemy_engine().begin() as connection:
    dataframe.to_sql(name=name_of_table, con=connection, if_exists="append")
```

**Pros**:
- Proper transaction handling
- Might reduce lock contention

**Cons**:
- Still might conflict with Django's transaction
- Unclear if it solves the core issue

### Option 4: Write Data Using Raw SQL INSERT
Manually construct INSERT statements with proper type binding:

```python
# Build parameterized INSERT statement
sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
for row in dataframe.itertuples():
    cursor.execute(sql, row)
```

**Pros**:
- Full control over types
- Uses Django's connection
- No locking issues

**Cons**:
- Much more complex code
- Slower for bulk inserts
- Need to handle all edge cases

## Recommendation

**Use Option 1**: Django's connection with manual type conversion.

This is the most pragmatic solution because:
1. It works within Django's transaction context (test-friendly)
2. It avoids all locking issues
3. We have full control over type conversions
4. It's compatible with all databases (SQLite, MySQL, PostgreSQL)

## Implementation Plan for Option 1

1. Revert to using `django_connection.connection`
2. Before calling `to_sql()`, convert DataFrame columns to proper Python types:
   - Float columns: Convert to Python `float` (not numpy.float64)
   - Integer columns: Convert to Python `int` (not numpy.int64)
   - Boolean columns: Convert to Python `int` (0 or 1)
   - String columns: Convert to Python `str`
3. Let pandas `to_sql()` handle the SQL generation with proper type binding

## Files Modified

- `models/dynamic_model.py`:
  - Lines 10-38: Updated `_get_alchemy_engine()` with better SQLite settings
  - Lines 60-75: Added Django transaction commit before SQLAlchemy
  - Lines 77-84: Switched to using SQLAlchemy for all databases

## Next Steps

1. Implement Option 1 (manual type conversion with Django's connection)
2. Test with all 4 failing tests
3. Verify data is written and read correctly
4. Document the solution in code comments

## Test Results

- `test_salaries_prediction_complete_workflow`: ❌ FAILED (database locked)
- `test_salaries_prediction_diagnostic_row_count`: ❌ NOT TESTED
- `test_machine_1`: ❌ NOT TESTED
- `test_machine_2_experimenter_workflow`: ❌ FAILED (different issue: user lookup)

---

**Conclusion**: The dtype issue is solvable, but requires careful handling of Django's transaction context and SQLite's single-writer limitation. Option 1 (manual type conversion) is the most reliable path forward.

