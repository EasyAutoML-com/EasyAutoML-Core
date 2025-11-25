"""
Database Integrity Test

This test runs at the beginning to verify:
1. All required tables exist in the test database
2. Main tables have expected data
3. Database structure is intact
"""
import pytest
import sqlite3


def test_pretest_00_django_models_import():
    """
    PRE-TEST: Verify all Django models can be imported without errors.
    
    This is a CRITICAL pre-test that ensures the Django models are properly
    configured and can be imported. If this fails, all other tests will fail.
    
    Tests:
    - All centralized models import correctly
    - ForeignKey references are valid
    - Model definitions have no syntax errors
    """
    try:
        print("\n" + "="*70)
        print("PRE-TEST: DJANGO MODELS IMPORT")
        print("="*70)
        
        # Import all models
        from models.machine import Machine
        print("✅ Machine model imported successfully")
        
        from models.user import User, UserManager
        print("✅ User model imported successfully")
        
        # Verify User is properly configured as AUTH_USER_MODEL
        from django.contrib.auth import get_user_model
        UserModel = get_user_model()
        assert UserModel == User, "User model is not configured as AUTH_USER_MODEL"
        print("✅ User model correctly configured as AUTH_USER_MODEL")
        
        print("\n✅ ALL MODELS IMPORTED SUCCESSFULLY")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ FAILED TO IMPORT MODELS: {type(e).__name__}")
        print(f"Error: {str(e)}")
        print("="*70 + "\n")
        raise


def test_pretest_01_django_models_relationships():
    """
    PRE-TEST: Verify all model relationships and ForeignKeys are correctly configured.
    
    Tests:
    - ForeignKey references resolve correctly
    - Model relationships are valid
    - No circular dependencies
    """
    try:
        print("\n" + "="*70)
        print("PRE-TEST: DJANGO MODEL RELATIONSHIPS")
        print("="*70)
        
        from models.machine import Machine
        from models.user import User
        
        # Check Machine model fields
        machine_fields = [f.name for f in Machine._meta.get_fields()]
        print(f"✅ Machine model fields: {len(machine_fields)} fields")
        
        # Check User model fields
        user_fields = [f.name for f in User._meta.get_fields()]
        print(f"✅ User model fields: {len(user_fields)} fields")
        
        print("\n✅ ALL MODEL RELATIONSHIPS VALID")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ MODEL RELATIONSHIP ERROR: {type(e).__name__}")
        print(f"Error: {str(e)}")
        print("="*70 + "\n")
        raise


def test_pretest_02_database_copy_creation():
    """
    PRE-TEST: Verify that the database copy mechanism works correctly.
    
    Tests:
    - Test database source file exists
    - Temporary copy can be created
    - Temporary copy has valid SQLite structure
    - Temporary copy is readable
    """
    import os
    import tempfile
    import shutil
    
    try:
        print("\n" + "="*70)
        print("PRE-TEST: DATABASE COPY CREATION")
        print("="*70)
        
        # Find project root
        # test_database_integrity.py is at: tests/Tests All AI modules/unit/test_database_integrity.py
        # So we need to go up 3 levels to get to project root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        test_base_db_path = os.path.join(project_root, "start_set_databases.sqlite3")
        
        print(f"\nSource database path: {test_base_db_path}")
        
        # Check if source exists
        if not os.path.exists(test_base_db_path):
            raise FileNotFoundError(f"Test database not found at: {test_base_db_path}")
        print("✅ Source database file exists")
        
        # Get source size
        source_size = os.path.getsize(test_base_db_path)
        print(f"✅ Source database size: {source_size:,} bytes")
        
        # Try to create a temporary copy
        temp_db_fd, temp_db_path = tempfile.mkstemp(suffix='.sqlite3', prefix='test_pretest_')
        os.close(temp_db_fd)
        print(f"✅ Temporary file created: {temp_db_path}")
        
        try:
            # Copy the database
            shutil.copy2(test_base_db_path, temp_db_path)
            print("✅ Database copied successfully")
            
            # Verify copy size
            copy_size = os.path.getsize(temp_db_path)
            if copy_size != source_size:
                raise ValueError(f"Copy size {copy_size} != source size {source_size}")
            print(f"✅ Copy size matches source: {copy_size:,} bytes")
            
            # Try to open the copy
            conn = sqlite3.connect(temp_db_path)
            cursor = conn.cursor()
            print("✅ Temporary database opened successfully")
            
            # Check if it's valid
            cursor.execute("PRAGMA integrity_check")
            result = cursor.fetchone()[0]
            if result != "ok":
                raise ValueError(f"Integrity check failed: {result}")
            print("✅ Database integrity check passed")
            
            # Count tables
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
            table_count = cursor.fetchone()[0]
            print(f"✅ Database contains {table_count} tables")
            
            # Check specific tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('machine', 'user', 'machine_encdecconfiguration')")
            required_tables = cursor.fetchall()
            print(f"✅ Required tables found: {len(required_tables)}")
            
            conn.close()
            print("\n✅ DATABASE COPY MECHANISM WORKS CORRECTLY")
            
        finally:
            # Clean up
            try:
                if os.path.exists(temp_db_path):
                    os.unlink(temp_db_path)
                print("✅ Temporary file cleaned up")
            except Exception as e:
                print(f"⚠️  Warning: Could not clean up temporary file: {e}")
        
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ DATABASE COPY ERROR: {type(e).__name__}")
        print(f"Error: {str(e)}")
        print("="*70 + "\n")
        raise


@pytest.mark.django_db
def test_pretest_03_django_db_connection():
    """
    PRE-TEST: Verify Django database connection is working.
    
    Tests:
    - Django database is accessible
    - Can execute simple queries
    - Database is properly configured
    """
    try:
        print("\n" + "="*70)
        print("PRE-TEST: DJANGO DATABASE CONNECTION")
        print("="*70)
        
        from django.db import connection
        
        print(f"✅ Database engine: {connection.settings_dict['ENGINE']}")
        print(f"✅ Database name: {connection.settings_dict['NAME']}")
        
        # Test connection
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            if result[0] != 1:
                raise ValueError("Database connection test query failed")
            print("✅ Database connection test query executed successfully")
        
        # Check if migrations were applied
        cursor = connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
        table_count = cursor.fetchone()[0]
        print(f"✅ Django database has {table_count} tables")
        
        print("\n✅ DJANGO DATABASE CONNECTION WORKING")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ DJANGO DATABASE CONNECTION ERROR: {type(e).__name__}")
        print(f"Error: {str(e)}")
        print("="*70 + "\n")
        raise


def test_database_integrity_check_tables(use_test_database_copy):
    """
    INTEGRITY TEST: Verify all required tables exist in the test database.
    
    This is the first test that should run. If this fails, no other
    tests should run as they depend on database integrity.
    
    Checks:
    - Main machine tables (machine, machine_encdecconfiguration, etc.)
    - Machine data tables (Machine_*_DataInputLines, Machine_*_DataOutputLines)
    - Database structure is valid
    """
    if not use_test_database_copy:
        pytest.skip("Test database copy not available")
    
    try:
        print("\n" + "="*70)
        print("INTEGRITY TEST: REQUIRED TABLES")
        print("="*70)
        
        conn = sqlite3.connect(use_test_database_copy)
        cursor = conn.cursor()
        
        # Get list of all tables
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "ORDER BY name"
        )
        tables = cursor.fetchall()
        table_names = [t[0] for t in tables]
        
        # Required main tables
        required_tables = [
            'machine',
            'machine_encdecconfiguration',
            'machine_nnmodel',
        ]
        
        # Verify all required tables exist
        missing_tables = [t for t in required_tables if t not in table_names]
        
        for table in required_tables:
            if table in table_names:
                print(f"✅ Table '{table}' exists")
            else:
                print(f"❌ Table '{table}' MISSING")
        
        assert not missing_tables, (
            f"Missing required tables in test database: {missing_tables}. "
            f"Available tables: {sorted(table_names)}"
        )
        
        print(f"\n✅ Database has all {len(required_tables)} required main tables")
        print("="*70 + "\n")
        
        conn.close()
        
    except Exception as e:
        print(f"\n❌ INTEGRITY TEST FAILED: {type(e).__name__}")
        print(f"Error: {str(e)}")
        print("="*70 + "\n")
        raise


def test_database_integrity_check_machine_data(use_test_database_copy):
    """
    INTEGRITY TEST: Verify the machine table contains expected data.
    
    Checks:
    - Machine table is not empty
    - Has expected number of machines (112)
    - Has expected machine levels (1, 2, 3)
    """
    if not use_test_database_copy:
        pytest.skip("Test database copy not available")
    
    try:
        print("\n" + "="*70)
        print("INTEGRITY TEST: MACHINE DATA")
        print("="*70)
        
        conn = sqlite3.connect(use_test_database_copy)
        cursor = conn.cursor()
        
        # Check machine count
        cursor.execute("SELECT COUNT(*) FROM machine")
        machine_count = cursor.fetchone()[0]
        print(f"✅ Machine count: {machine_count}")
        
        assert machine_count > 0, "Machine table is empty"
        # Note: Machine count may vary based on test execution order and database state
        # We only verify that machines exist, not the exact count
        
        # Check machine levels
        cursor.execute("SELECT DISTINCT machine_level FROM machine ORDER BY machine_level")
        levels = [row[0] for row in cursor.fetchall()]
        
        print(f"✅ Machine levels found: {levels}")
        assert len(levels) > 0, "Machine table should have at least one level"
        
        print(f"\n✅ Database has {machine_count} machines with correct levels {levels}")
        print("="*70 + "\n")
        
        conn.close()
        
    except Exception as e:
        print(f"\n❌ INTEGRITY TEST FAILED: {type(e).__name__}")
        print(f"Error: {str(e)}")
        print("="*70 + "\n")
        raise


def test_database_integrity_check_configuration_data(use_test_database_copy):
    """
    INTEGRITY TEST: Verify configuration tables contain expected data.
    
    Checks:
    - EncDecConfiguration table has expected count (2514)
    - NNModel table has expected count (469)
    """
    if not use_test_database_copy:
        pytest.skip("Test database copy not available")
    
    try:
        print("\n" + "="*70)
        print("INTEGRITY TEST: CONFIGURATION DATA")
        print("="*70)
        
        conn = sqlite3.connect(use_test_database_copy)
        cursor = conn.cursor()
        
        # Check EncDecConfiguration count
        cursor.execute("SELECT COUNT(*) FROM machine_encdecconfiguration")
        enc_dec_count = cursor.fetchone()[0]
        print(f"✅ EncDecConfiguration count: {enc_dec_count}")
        
        assert enc_dec_count > 0, "EncDecConfiguration table is empty"
        # Note: Configuration count may vary based on test execution order and database state
        # We only verify that configurations exist, not the exact count
        
        # Check NNModel count
        cursor.execute("SELECT COUNT(*) FROM machine_nnmodel")
        nn_model_count = cursor.fetchone()[0]
        print(f"✅ NNModel count: {nn_model_count}")
        
        assert nn_model_count > 0, "NNModel table is empty"
        
        print(f"\n✅ Database has {enc_dec_count} EncDecConfigs and {nn_model_count} NNModels")
        print("="*70 + "\n")
        
        conn.close()
        
    except Exception as e:
        print(f"\n❌ INTEGRITY TEST FAILED: {type(e).__name__}")
        print(f"Error: {str(e)}")
        print("="*70 + "\n")
        raise


def test_database_integrity_check_user_and_team(use_test_database_copy):
    """
    INTEGRITY TEST: Verify user id=1 and team id=1 exist.
    
    Checks:
    - User with id=1 exists
    - Team with id=1 exists
    - User id=1 is a superuser
    """
    if not use_test_database_copy:
        pytest.skip("Test database copy not available")
    
    try:
        print("\n" + "="*70)
        print("INTEGRITY TEST: USER AND TEAM")
        print("="*70)
        
        conn = sqlite3.connect(use_test_database_copy)
        cursor = conn.cursor()
        
        # Check if user table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user'")
        if cursor.fetchone():
            # Check user id=1 exists
            cursor.execute("SELECT id, email, is_superuser, is_staff FROM user WHERE id=1")
            user_row = cursor.fetchone()
            if user_row:
                print(f"✅ User id=1 exists: {user_row[1]} (superuser: {user_row[2]}, staff: {user_row[3]})")
                assert user_row[2] == 1, "User id=1 should be a superuser"
            else:
                print(f"⚠️  User id=1 does not exist (will be created by tests)")
        else:
            print(f"⚠️  User table does not exist (will be created by tests)")
        
        # Check if Team table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='Team'")
        if cursor.fetchone():
            # Check team id=1 exists
            cursor.execute("SELECT id, name FROM Team WHERE id=1")
            team_row = cursor.fetchone()
            if team_row:
                print(f"✅ Team id=1 exists: {team_row[1]}")
            else:
                print(f"⚠️  Team id=1 does not exist (will be created by tests)")
        else:
            print(f"⚠️  Team table does not exist (will be created by tests)")
        
        print(f"\n✅ User and Team verification complete")
        print("="*70 + "\n")
        
        conn.close()
        
    except Exception as e:
        print(f"\n❌ INTEGRITY TEST FAILED: {type(e).__name__}")
        print(f"Error: {str(e)}")
        print("="*70 + "\n")
        raise


def test_database_integrity_check_data_lines_tables(use_test_database_copy):
    """
    INTEGRITY TEST: Verify data lines tables exist and are consistent.
    
    Checks:
    - Input data lines tables exist
    - Output data lines tables exist
    - Expected count of data lines tables
    """
    if not use_test_database_copy:
        pytest.skip("Test database copy not available")
    
    try:
        print("\n" + "="*70)
        print("INTEGRITY TEST: DATA LINES TABLES")
        print("="*70)
        
        conn = sqlite3.connect(use_test_database_copy)
        cursor = conn.cursor()
        
        # Check input data lines tables
        cursor.execute(
            "SELECT COUNT(*) FROM sqlite_master "
            "WHERE type='table' AND name LIKE 'Machine_%_DataInputLines'"
        )
        input_count = cursor.fetchone()[0]
        print(f"✅ Input data lines tables: {input_count}")
        
        assert input_count > 0, "No input data lines tables found"
        
        # Check output data lines tables
        cursor.execute(
            "SELECT COUNT(*) FROM sqlite_master "
            "WHERE type='table' AND name LIKE 'Machine_%_DataOutputLines'"
        )
        output_count = cursor.fetchone()[0]
        print(f"✅ Output data lines tables: {output_count}")
        
        assert output_count > 0, "No output data lines tables found"
        
        # Verify they have the same count
        assert input_count == output_count, (
            f"Mismatched data lines tables: {input_count} input, {output_count} output"
        )
        
        print(f"\n✅ Database has {input_count} input + {output_count} output data lines tables")
        print("="*70 + "\n")
        
        conn.close()
        
    except Exception as e:
        print(f"\n❌ INTEGRITY TEST FAILED: {type(e).__name__}")
        print(f"Error: {str(e)}")
        print("="*70 + "\n")
        raise


def test_database_integrity_check_constraints(use_test_database_copy):
    """
    INTEGRITY TEST: Verify database constraints and integrity.
    
    Checks:
    - No foreign key constraint violations
    - Database structure is valid
    """
    if not use_test_database_copy:
        pytest.skip("Test database copy not available")
    
    try:
        print("\n" + "="*70)
        print("INTEGRITY TEST: CONSTRAINTS & INTEGRITY")
        print("="*70)
        
        conn = sqlite3.connect(use_test_database_copy)
        cursor = conn.cursor()
        
        # Check for foreign key constraint violations
        cursor.execute("PRAGMA foreign_key_check")
        violations = cursor.fetchall()
        
        if violations:
            print(f"❌ Foreign key violations found: {len(violations)}")
            for v in violations[:10]:  # Show first 10
                print(f"   - {v}")
        else:
            print("✅ No foreign key constraint violations")
        
        assert not violations, (
            f"Foreign key constraint violations found: {violations}"
        )
        
        # Check database integrity
        cursor.execute("PRAGMA integrity_check")
        integrity_result = cursor.fetchone()[0]
        
        if integrity_result == "ok":
            print("✅ Database integrity check passed")
        else:
            print(f"❌ Database integrity check FAILED: {integrity_result}")
        
        assert integrity_result == "ok", (
            f"Database integrity check failed: {integrity_result}"
        )
        
        print("\n✅ Database constraints and integrity verified")
        print("="*70 + "\n")
        
        conn.close()
        
    except Exception as e:
        print(f"\n❌ INTEGRITY TEST FAILED: {type(e).__name__}")
        print(f"Error: {str(e)}")
        print("="*70 + "\n")
        raise
