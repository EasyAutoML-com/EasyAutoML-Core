# Database Lock Issues - Detailed Explanation

## What is SQLite Database Locking?

SQLite uses a **file-based locking mechanism** to ensure data integrity. When a write operation (INSERT, UPDATE, DELETE, CREATE TABLE) is in progress, SQLite locks the entire database file. This means:

- **Only ONE write operation can happen at a time**
- Other write operations must **wait** until the lock is released
- If a lock timeout is exceeded, you get: `sqlite3.OperationalError: database is locked`

## The Problem in Our Code

### Step-by-Step Sequence of Events

#### Step 1: Table Creation (Succeeds)
```python
# ML/Machine.py:374 - data_lines_create_both_tables()
# Creates table structure using Django's schema editor
schema_editor.create_model(input_model)  # ✅ SUCCESS - Table structure created
schema_editor.create_model(output_model)  # ✅ SUCCESS - Table structure created
```

**Result:** Tables `Machine_904_DataInputLines` and `Machine_904_DataOutputLines` are created with the correct schema (columns defined).

#### Step 2: Data Insertion Attempt (Fails Silently)
```python
# ML/Machine.py:247 - data_lines_append()
# Tries to insert data using pandas to_sql()
dataframe.to_sql(
    name="Machine_904_DataInputLines",
    con=_get_alchemy_engine(),  # Uses SQLAlchemy engine (different connection!)
    if_exists="append"
)
```

**What Happens:**
1. Django's schema editor **still holds a lock** on the database (transaction not committed)
2. SQLAlchemy engine tries to **acquire a write lock** for `to_sql()`
3. SQLite says: **"Database is locked!"** ❌
4. The error is **caught and logged**, but **NOT raised** (see line 82-83 in dynamic_model.py)

**Code Evidence:**
```python
# models/dynamic_model.py:82-83
except Exception as error:
    logger.error(f"There was a problem during dataframe.to_sql execution...")  
    # ⚠️ ERROR IS LOGGED BUT NOT RAISED - execution continues!
```

#### Step 3: Test Tries to Read Data (Fails)
```python
# Test tries to read data
exported_df = machine.data_lines_read()
# SQL Query: SELECT `SepalLengthCm`, ... FROM Machine_904_DataInputLines ...
```

**What Happens:**
1. Table structure exists (from Step 1) ✅
2. But table is **EMPTY** (data insertion failed in Step 2) ❌
3. SQLite returns an empty result set
4. OR if the query tries to access columns that don't exist in an empty table context, you get: `no such column: SepalLengthCm`

## Why "No Such Column" Error?

This is the confusing part! The error says "no such column" but the table structure was created correctly. Here's why:

### Scenario A: Table is Empty
- Table structure exists with columns: `Line_ID`, `SepalLengthCm`, `SepalWidthCm`, etc.
- But table has **0 rows**
- When pandas tries to read, it might not find the expected structure if the table is truly empty
- SQLite might return an error if the table metadata is inconsistent

### Scenario B: Transaction Rollback
- Django's transaction might have been rolled back
- Table structure might have been reverted
- But the code thinks the table exists

### Scenario C: Connection Mismatch
- Django creates table using one connection
- SQLAlchemy tries to read using a different connection
- Connection isolation might cause metadata inconsistencies

## Visual Timeline

```
Time →
│
├─ [0ms] Django Schema Editor: CREATE TABLE Machine_904_DataInputLines (...)
│         ✅ Table structure created
│         🔒 Database LOCKED by Django connection
│
├─ [10ms] Code tries: dataframe.to_sql(..., if_exists="append")
│         🔒 Tries to acquire lock (but Django still has it!)
│         ❌ ERROR: "database is locked"
│         ⚠️  Error logged but NOT raised - execution continues
│
├─ [20ms] Django transaction might commit/rollback
│         🔓 Lock released (but too late - data wasn't inserted)
│
├─ [30ms] Test tries: machine.data_lines_read()
│         SQL: SELECT SepalLengthCm FROM Machine_904_DataInputLines
│         ❌ ERROR: "no such column: SepalLengthCm"
│         OR: Returns empty DataFrame (0 rows)
│
└─ [40ms] Test fails: AssertionError: Should have 150 rows, got 0
```

## Why This Happens in Tests

### 1. **Multiple Database Connections**
- Django uses its own connection pool
- SQLAlchemy (used by pandas `to_sql()`) uses a separate connection
- Both try to access the same SQLite file simultaneously

### 2. **Transaction Management**
- Django wraps operations in transactions
- SQLAlchemy also uses transactions
- SQLite can't handle concurrent transactions well

### 3. **Test Environment**
- Tests run quickly, one after another
- Database connections might not be properly closed
- Locks accumulate and cause conflicts

### 4. **SQLite Limitations**
- SQLite is designed for single-user/single-connection scenarios
- Multiple concurrent connections can cause locking issues
- Production databases (PostgreSQL, MySQL) handle this better

## Code Locations

### Where Tables Are Created:
```python
# ML/Machine.py:374-419
def data_lines_create_both_tables(self):
    # Uses Django's schema editor
    with connections["default"].schema_editor() as schema_editor:
        schema_editor.create_model(input_model)  # Creates table structure
```

### Where Data Is Inserted:
```python
# models/dynamic_model.py:49-83
@classmethod
def append_dataframe(cls, dataframe):
    # Uses SQLAlchemy engine (different connection!)
    dataframe.to_sql(
        name=name_of_table,
        con=_get_alchemy_engine(),  # ← Different connection!
        if_exists="append"
    )
```

### Where Error Is Silently Swallowed:
```python
# models/dynamic_model.py:82-83
except Exception as error:
    logger.error(f"There was a problem...")  
    # ⚠️ Error logged but NOT raised - code continues!
```

## Why Tables Are Created But Empty

1. **Table creation succeeds** because it happens first and gets the lock
2. **Data insertion fails** because it can't get the lock (table creation still has it)
3. **Error is caught** but not raised, so code thinks everything is OK
4. **Test runs** and finds an empty table
5. **Test fails** when trying to read data that doesn't exist

## Solutions

### Solution 1: Commit Django Transaction Before to_sql()
```python
# models/dynamic_model.py (already attempted, but might need improvement)
if 'sqlite' in engine_name:
    django_connection.commit()  # Release Django's lock
    # Then do to_sql()
```

### Solution 2: Use Same Connection for Everything
```python
# Use Django's connection for to_sql() instead of SQLAlchemy
from django.db import connection
dataframe.to_sql(name=name_of_table, con=connection.connection)
```

### Solution 3: Add Retry Logic
```python
import time
max_retries = 5
for attempt in range(max_retries):
    try:
        dataframe.to_sql(...)
        break
    except sqlite3.OperationalError as e:
        if "database is locked" in str(e) and attempt < max_retries - 1:
            time.sleep(0.1)  # Wait and retry
            continue
        raise
```

### Solution 4: Increase SQLite Timeout
```python
# In _get_alchemy_engine() in dynamic_model.py
connect_args = {
    "timeout": 60,  # Wait up to 60 seconds for lock (already set)
    "check_same_thread": False,
}
```

### Solution 5: Use PostgreSQL for Tests
- PostgreSQL handles concurrent connections much better
- No file-based locking issues
- More production-like environment

## Current Status

The code **already attempts** to handle this (see `dynamic_model.py:67-72`):
```python
if 'sqlite' in engine_name:
    try:
        django_connection.commit()  # Try to release lock
    except Exception:
        pass  # Ignore if no transaction is active
```

But this might not be sufficient because:
- The commit might happen too early
- The lock might be held by a different part of Django's transaction system
- SQLAlchemy might still conflict

## Summary

**The Issue:**
- Tables are created successfully ✅
- Data insertion fails due to database lock ❌
- Error is logged but not raised ⚠️
- Tests find empty tables and fail ❌

**Root Cause:**
- SQLite file-based locking
- Multiple database connections (Django + SQLAlchemy)
- Transaction conflicts
- Silent error handling

**Impact:**
- Tests fail intermittently
- Production might be OK (if using better database or proper connection management)
- Test reliability is reduced

