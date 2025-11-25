# Why Django Locks the Database 🔒

## Django's Transaction Management

Django uses **database transactions** to ensure data consistency. Here's how it works:

### 1. Automatic Transaction Wrapping

When you use `@pytest.mark.django_db`, Django automatically:

```python
@pytest.mark.django_db
def test_something():
    # Django automatically starts a transaction here
    machine = Machine(...)  # All DB operations are in a transaction
    machine.save_machine_to_db()
    # Transaction is still active!
    # Django will rollback at the end of the test
```

**What happens:**
- Django **starts a transaction** when the test begins
- **All database operations** (CREATE TABLE, INSERT, UPDATE) are part of this transaction
- The transaction **stays open** for the entire test
- At the end, Django **rolls back** the transaction (to keep tests isolated)

### 2. SQLite Transaction Locking

SQLite uses **file-based locking** with these rules:

```
┌─────────────────────────────────────────┐
│  SQLite Lock Levels (from least to most)│
├─────────────────────────────────────────┤
│ 1. UNLOCKED    - No lock                │
│ 2. SHARED      - Multiple readers       │
│ 3. RESERVED    - One writer preparing   │
│ 4. PENDING     - Writer waiting         │
│ 5. EXCLUSIVE   - Writer writing         │
└─────────────────────────────────────────┘
```

**When Django creates a table:**
1. Django starts a transaction → SQLite acquires **RESERVED lock**
2. `CREATE TABLE` executes → SQLite upgrades to **EXCLUSIVE lock**
3. Table is created → Lock is **still held** (transaction not committed)
4. Django keeps the transaction open → **Lock remains active**

### 3. The Lock Timeline

Here's exactly what happens in your code:

```python
# ML/Machine.py:240-250 - save_machine_to_db()

# Step 1: Save machine model
self.db_machine.save()  # ← Django starts/continues transaction
                        # 🔒 SQLite: RESERVED lock acquired

# Step 2: Create tables
self.data_lines_create_both_tables()
    # Inside: schema_editor.create_model()
    # 🔒 SQLite: EXCLUSIVE lock (for CREATE TABLE)
    # ✅ Table created successfully
    # ⚠️  But transaction NOT committed - lock still held!

# Step 3: Try to insert data
self.data_lines_append(...)
    # Inside: dataframe.to_sql() using SQLAlchemy
    # 🔒 SQLAlchemy tries to get EXCLUSIVE lock
    # ❌ FAILS! Django still has the lock!
    # Error: "database is locked"
```

## Why Django Keeps the Lock

### Reason 1: Test Isolation

Django keeps transactions open during tests to enable **automatic rollback**:

```python
@pytest.mark.django_db
def test_machine_creation():
    # Transaction starts here
    machine = Machine(...)
    machine.save_machine_to_db()  # Creates tables, inserts data
    
    # All changes are in a transaction
    # If test fails, Django can rollback everything
    # If test passes, Django still rolls back (for isolation)
    
    # Transaction ends here (rollback happens)
```

**Benefits:**
- Each test starts with a clean database
- Tests don't interfere with each other
- No need to manually clean up

**Problem:**
- Transaction stays open for the **entire test duration**
- Lock is held the **whole time**

### Reason 2: Django's Schema Editor

When you use Django's schema editor:

```python
# ML/Machine.py:408-416
with connections["default"].schema_editor() as schema_editor:
    schema_editor.create_model(input_model)  # Creates table
    schema_editor.create_model(output_model)  # Creates table
    # Schema editor context ends, but...
    # Transaction is still active!
```

**What happens:**
1. Schema editor starts (part of Django transaction)
2. Creates table (SQLite gets EXCLUSIVE lock)
3. Schema editor context ends
4. **But Django's test transaction is still active!**
5. Lock remains until test ends

### Reason 3: SQLite's Single-Writer Rule

SQLite has a fundamental limitation:

> **"SQLite only allows ONE writer at a time"**

This means:
- If Django has a write lock → SQLAlchemy **cannot** write
- If SQLAlchemy has a write lock → Django **cannot** write
- They must **take turns** or use the **same connection**

## Visual Explanation

```
Test Execution Timeline:
═══════════════════════════════════════════════════════════════

[Test Starts]
    │
    ├─ @pytest.mark.django_db activates
    │  └─ Django starts transaction
    │     └─ 🔒 SQLite: RESERVED lock acquired
    │
    ├─ machine.save_machine_to_db()
    │  │
    │  ├─ self.db_machine.save()
    │  │  └─ 🔒 Lock still held (transaction active)
    │  │
    │  ├─ self.data_lines_create_both_tables()
    │  │  │
    │  │  ├─ schema_editor.create_model()
    │  │  │  └─ 🔒 SQLite: EXCLUSIVE lock (CREATE TABLE)
    │  │  │  └─ ✅ Table created
    │  │  │  └─ ⚠️  Transaction NOT committed
    │  │  │
    │  │  └─ schema_editor context ends
    │  │     └─ 🔒 But Django transaction still active!
    │  │
    │  └─ self.data_lines_append()
    │     │
    │     └─ dataframe.to_sql() [SQLAlchemy connection]
    │        └─ 🔒 Tries to get EXCLUSIVE lock
    │        └─ ❌ FAILS! Django still has lock
    │        └─ Error: "database is locked"
    │        └─ ⚠️  Error caught but not raised
    │
    └─ [Test continues...]
       └─ 🔒 Lock still held by Django
       └─ [Test ends]
          └─ Django rolls back transaction
          └─ 🔓 Lock finally released
```

## Why This Happens in Tests But Maybe Not Production

### In Tests (with @pytest.mark.django_db):

```python
@pytest.mark.django_db  # ← This is the key!
def test_something():
    # Django wraps entire test in ONE transaction
    # Transaction stays open for entire test
    # Lock is held the whole time
```

### In Production (without test wrapper):

```python
def create_machine():
    # Each operation can commit immediately
    machine.save()  # Commits transaction
    # Lock released
    
    create_tables()  # New transaction
    # Lock acquired and released
    
    insert_data()  # New transaction
    # Lock acquired and released
```

**Key Difference:**
- **Tests:** One long transaction = lock held for entire test
- **Production:** Multiple short transactions = locks released quickly

## The Specific Code Path

Let's trace exactly where the lock is held:

```python
# 1. Test starts
@pytest.mark.django_db  # ← Transaction starts here
def test_machine_1():
    
    # 2. Machine creation
    machine = Machine(...)
    machine.save_machine_to_db()
        # ML/Machine.py:241
        self.db_machine.save()  # ← Part of Django transaction
                                # 🔒 Lock acquired
        
        # ML/Machine.py:245
        self.data_lines_create_both_tables()
            # ML/Machine.py:408
            with connections["default"].schema_editor() as schema_editor:
                schema_editor.create_model(input_model)
                    # Django executes: CREATE TABLE ...
                    # 🔒 SQLite: EXCLUSIVE lock
                    # ✅ Table created
                    # ⚠️  Transaction NOT committed
            
            # Schema editor context ends
            # But Django's test transaction is STILL ACTIVE
            # 🔒 Lock still held!
        
        # ML/Machine.py:247
        self.data_lines_append(...)
            # models/dynamic_model.py:76
            dataframe.to_sql(con=_get_alchemy_engine())
                # SQLAlchemy tries to get connection
                # Tries to acquire EXCLUSIVE lock
                # ❌ FAILS - Django still has it!
                # Error: "database is locked"
    
    # Test continues...
    # 🔒 Lock still held by Django
    
# Test ends
# Django rolls back transaction
# 🔓 Lock finally released
```

## Why Django Doesn't Commit Immediately

Django uses **transaction management** for these reasons:

### 1. Atomicity
```python
# All operations succeed or all fail
with transaction.atomic():
    create_table()
    insert_data()
    update_config()
    # If any fails, all are rolled back
```

### 2. Test Isolation
```python
# Each test gets a clean database
@pytest.mark.django_db
def test_1():
    create_data()  # Changes are in transaction
    
@pytest.mark.django_db  
def test_2():
    # test_1's changes were rolled back
    # Database is clean!
```

### 3. Performance
```python
# Multiple operations in one transaction = faster
# One commit instead of many
```

## Solutions to Release the Lock

### Option 1: Explicit Commit (Current Attempt)

```python
# models/dynamic_model.py:67-72
if 'sqlite' in engine_name:
    try:
        django_connection.commit()  # Try to release lock
    except Exception:
        pass
```

**Problem:** This might not work because:
- Django's test transaction manager might prevent commits
- The transaction might be managed at a higher level
- Committing might break test isolation

### Option 2: Use Django's Connection for to_sql()

```python
# Instead of SQLAlchemy engine
from django.db import connection
dataframe.to_sql(
    name=name_of_table,
    con=connection.connection,  # Use Django's connection
    if_exists="append"
)
```

**Benefit:** Same connection = no lock conflict

### Option 3: Disable Transaction for Specific Operations

```python
from django.db import transaction

@transaction.non_atomic_requests
def data_lines_append(self, ...):
    # This operation commits immediately
    # Lock is released
```

**Problem:** Might break test isolation

### Option 4: Use Transaction.on_commit()

```python
from django.db import transaction

def data_lines_append(self, ...):
    # Schedule to run after transaction commits
    transaction.on_commit(
        lambda: dataframe.to_sql(...)
    )
```

**Benefit:** Runs after Django releases the lock

## Summary

**Why Django Locks the Database:**

1. **Test Transaction Wrapper** (`@pytest.mark.django_db`)
   - Wraps entire test in one transaction
   - Transaction stays open for entire test
   - Lock is held the whole time

2. **SQLite's Single-Writer Rule**
   - Only one write operation at a time
   - Django has the lock → SQLAlchemy can't get it

3. **Schema Editor Behavior**
   - Creates tables within Django transaction
   - Transaction not committed immediately
   - Lock remains active

4. **Test Isolation Requirements**
   - Django needs to rollback at test end
   - Keeps transaction open for this purpose
   - But this conflicts with concurrent connections

**The Core Issue:**
- Django's transaction management (for test isolation)
- Conflicts with SQLAlchemy's separate connection
- SQLite can't handle both simultaneously

**The Solution:**
- Use Django's connection for `to_sql()` instead of SQLAlchemy
- OR: Commit Django transaction before using SQLAlchemy
- OR: Use a database that handles concurrent connections better (PostgreSQL)

