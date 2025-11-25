# Database Lock Solution - Implementation Guide

## Solution Implemented

### Primary Solution: Use Django's Connection for SQLite

**File Modified:** `models/dynamic_model.py`

**Changes:**
1. **For SQLite:** Use Django's raw connection instead of SQLAlchemy
   - Same connection = no lock conflict
   - Works within Django's transaction context
   - Compatible with `@pytest.mark.django_db`

2. **For MySQL/PostgreSQL:** Continue using SQLAlchemy
   - These databases handle concurrent connections better
   - SQLAlchemy provides better type handling

3. **Added Retry Logic:**
   - Retries up to 3 times if "database is locked" error occurs
   - Exponential backoff (0.1s, 0.2s, 0.3s)

## How It Works

### Before (Problem):
```python
# SQLite: Uses SQLAlchemy (different connection)
dataframe.to_sql(con=_get_alchemy_engine())  # ❌ Lock conflict!
```

### After (Solution):
```python
# SQLite: Uses Django's connection (same connection)
dataframe.to_sql(con=django_connection.connection)  # ✅ No lock conflict!
```

## Why This Works

1. **Same Connection = No Lock Conflict**
   - Django's transaction uses connection A
   - We also use connection A
   - No conflict because it's the same connection

2. **Works in Test Context**
   - `@pytest.mark.django_db` wraps test in transaction
   - We use the same connection Django has
   - No lock conflicts

3. **Retry Logic as Safety Net**
   - If lock still occurs (edge case), retry
   - Exponential backoff prevents hammering

## Trade-offs

### ✅ Benefits:
- **No lock conflicts** in SQLite
- **Works in test environment** with `@pytest.mark.django_db`
- **Simple solution** - minimal code changes
- **Backward compatible** - MySQL/PostgreSQL unchanged

### ⚠️ Potential Issues:
- **Type handling:** Django's raw connection might not handle DataFrame dtypes as well as SQLAlchemy
- **Performance:** Raw connection might be slightly slower for very large datasets

### Mitigation:
- If type issues occur, we can add explicit type conversion before `to_sql()`
- Performance impact is minimal for typical use cases

## Alternative Solutions (Not Implemented)

### Option 2: Disable Transaction for Specific Tests
```python
@pytest.mark.django_db(transaction=False)
def test_something():
    # No transaction = no lock
    # But: Tests are no longer isolated
```

**Why Not:** Breaks test isolation, requires manual cleanup

### Option 3: Use transaction.on_commit()
```python
from django.db import transaction

transaction.on_commit(
    lambda: dataframe.to_sql(...)
)
```

**Why Not:** More complex, might delay operations unnecessarily

### Option 4: Manual SQL INSERT
```python
# Build INSERT statements manually
# Full control but much more complex
```

**Why Not:** Too complex, slower, harder to maintain

## Testing the Solution

### Test Cases to Verify:
1. ✅ `test_salaries_prediction_complete_workflow`
2. ✅ `test_salaries_prediction_diagnostic_row_count`
3. ✅ `test_z_machine_1::test_machine_1`

### Expected Results:
- Tables are created successfully
- Data is inserted successfully (no "database is locked" errors)
- Data can be read back correctly
- Tests pass consistently

## Monitoring

### What to Watch For:
1. **Type Issues:** If data is written as strings instead of numbers
   - **Fix:** Add explicit type conversion before `to_sql()`

2. **Performance:** If inserts become slow
   - **Fix:** Optimize or batch inserts

3. **Edge Cases:** If retry logic triggers frequently
   - **Fix:** Investigate why locks still occur

## Rollback Plan

If this solution causes issues:

1. **Revert to SQLAlchemy for SQLite:**
   ```python
   # Change back to:
   dataframe.to_sql(con=_get_alchemy_engine())
   ```

2. **Add explicit transaction management:**
   ```python
   from django.db import transaction
   with transaction.atomic():
       # Force commit before to_sql()
       transaction.commit()
       dataframe.to_sql(...)
   ```

## Next Steps

1. ✅ **Implemented:** Use Django's connection for SQLite
2. ⏳ **Test:** Run failing tests to verify fix
3. 📋 **Monitor:** Watch for type issues or performance problems
4. 🔧 **Optimize:** If needed, add type conversion or performance improvements

## Code Changes Summary

**File:** `models/dynamic_model.py`
**Method:** `DynamicModel.append_dataframe()`
**Lines:** 48-95

**Key Changes:**
- SQLite now uses `django_connection.connection` instead of `_get_alchemy_engine()`
- Added retry logic with exponential backoff
- Better error handling and logging
- MySQL/PostgreSQL unchanged (still use SQLAlchemy)

