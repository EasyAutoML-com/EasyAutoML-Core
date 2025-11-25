"""
Pytest configuration and fixtures for AI module tests.

This file provides:
1. Django setup before tests run
2. Database fixtures for test isolation
3. Data fixtures for test data generation
4. Cleanup fixtures for test data management
"""
import os
import sys
import tempfile
import shutil
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Set Django settings module before any Django imports
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "settings")

# Configure Django before any tests run
import django
from django.conf import settings

# Check if eamllogger is available before Django setup
try:
    import eamllogger
    eamllogger_available = True
except ImportError:
    eamllogger_available = False

# Only setup Django if it hasn't been set up yet
if not settings.configured:
    # If eamllogger is not available, we need to modify INSTALLED_APPS before setup
    if not eamllogger_available and hasattr(settings, 'INSTALLED_APPS'):
        # Create a modified copy of INSTALLED_APPS without eamllogger
        original_installed_apps = list(settings.INSTALLED_APPS)
        if 'eamllogger' in original_installed_apps:
            # Temporarily modify settings
            settings.INSTALLED_APPS = [app for app in original_installed_apps if app != 'eamllogger']
    
    try:
        django.setup()
    except Exception as e:
        # If setup fails due to eamllogger, try without it
        if 'eamllogger' in str(e) and hasattr(settings, 'INSTALLED_APPS'):
            if 'eamllogger' in settings.INSTALLED_APPS:
                settings.INSTALLED_APPS = [app for app in settings.INSTALLED_APPS if app != 'eamllogger']
                django.setup()
        else:
            raise

# Import after Django setup
from django.db import connection, connections
from django.test.utils import override_settings
from models.EasyAutoMLDBModels import EasyAutoMLDBModels
from SharedConstants import DatasetColumnDataType
# Import test data generator - handle path with spaces
import importlib.util
fixtures_path = project_root / "tests" / "Tests All AI modules - Core" / "fixtures" / "test_data_generator.py"
spec = importlib.util.spec_from_file_location("test_data_generator", fixtures_path)
test_data_generator_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test_data_generator_module)
TestDataGenerator = test_data_generator_module.TestDataGenerator


# ============================================================================
# DATABASE FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def test_database_source():
    """Get the path to the source test database."""
    db_path = project_root / "start_set_databases.sqlite3"
    if not db_path.exists():
        pytest.skip(f"Test database not found at: {db_path}")
    return str(db_path)


@pytest.fixture(scope="session")
def session_test_database(test_database_source):
    """
    Create a session-level copy of the test database.
    This is used by django_db_setup to configure Django.
    """
    import sqlite3
    
    # Create a session-level copy of the test database
    temp_fd, temp_path = tempfile.mkstemp(suffix='.sqlite3', prefix='session_test_db_')
    os.close(temp_fd)
    
    try:
        # Copy source database to temporary location
        shutil.copy2(test_database_source, temp_path)
        
        # Verify the copy was successful
        with sqlite3.connect(temp_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='machine'")
            if cursor.fetchone()[0] == 0:
                raise ValueError(f"Machine table not found in copied database: {temp_path}")
        
        yield temp_path
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            try:
                connections.close_all()
                os.unlink(temp_path)
            except Exception:
                pass  # Ignore cleanup errors


@pytest.fixture(scope="session")
def django_db_setup(session_test_database):
    """
    Configure pytest-django to use the copied test database instead of creating a new one.
    
    This fixture:
    1. Uses the session-level copy of the test database
    2. Configures Django to use this database
    3. Prevents pytest-django from creating its own in-memory database
    
    This runs before any database operations and ensures all tests use the same
    pre-populated database copy.
    """
    # Configure Django to use the session database
    # Override Django settings to use the copied database
    settings.DATABASES['default'] = {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': session_test_database,
    }
    
    # Close any existing connections
    connections.close_all()
    
    yield
    
    # Cleanup: close connections
    connections.close_all()


@pytest.fixture(scope="session")
def django_db_createdb():
    """
    Tell pytest-django not to create a new database.
    We're using our own pre-populated database copy instead.
    """
    return False


@pytest.fixture(scope="function")
def use_test_database_copy(test_database_source):
    """
    Create a temporary copy of the test database for direct SQLite access.
    
    This fixture creates a temporary copy of the test database that can be
    accessed directly via sqlite3 without Django ORM. Useful for integrity checks.
    
    Returns:
        str: Path to temporary database copy
    """
    # Create temporary database file
    temp_fd, temp_path = tempfile.mkstemp(suffix='.sqlite3', prefix='test_db_')
    os.close(temp_fd)
    
    try:
        # Copy source database to temporary location
        shutil.copy2(test_database_source, temp_path)
        yield temp_path
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass  # Ignore cleanup errors


@pytest.fixture(scope="function")
def test_database(session_test_database):
    """
    Return the session-level test database path.
    
    This fixture now uses the session-level database copy created by django_db_setup
    instead of creating a new copy for each test. This ensures Django uses the
    correct database with all tables.
    
    Returns:
        str: Path to the session-level test database
    """
    # Use the session-level database instead of creating a new copy
    # This ensures Django ORM uses the correct database
    yield session_test_database


@pytest.fixture(scope="function")
def test_database_direct(test_database_source):
    """
    Create a temporary copy of the test database for direct SQLite access.
    
    Similar to use_test_database_copy but with a different name for compatibility.
    """
    temp_fd, temp_path = tempfile.mkstemp(suffix='.sqlite3', prefix='test_db_direct_')
    os.close(temp_fd)
    
    try:
        shutil.copy2(test_database_source, temp_path)
        yield temp_path
    finally:
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass


@pytest.fixture(scope="function")
def test_database_with_verification(test_database):
    """
    Enhanced test database fixture that includes verification.
    
    This fixture:
    1. Uses the session-level test database (already configured by django_db_setup)
    2. Verifies the database has expected content
    3. Ensures Django connections are ready
    
    Use this for tests that need a verified, populated database.
    Note: The database is already configured by django_db_setup, so we don't need
    to override settings here.
    """
    import sqlite3
    
    # Verify database has content using direct SQLite access
    conn = sqlite3.connect(test_database)
    cursor = conn.cursor()
    
    try:
        # Check if machine table exists and has data
        cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='machine'")
        if cursor.fetchone()[0] == 0:
            pytest.skip("Machine table not found in test database")
        
        cursor.execute("SELECT COUNT(*) FROM machine")
        machine_count = cursor.fetchone()[0]
        if machine_count == 0:
            pytest.skip("Test database has no machines")
        
        # Ensure Django connections are ready (database is already configured by django_db_setup)
        # Don't close connections here - they need to be open for the test
        # Just ensure the database path is correct
        if settings.DATABASES['default']['NAME'] != test_database:
            settings.DATABASES['default']['NAME'] = test_database
            connections.close_all()
        
        yield test_database
        
        # Cleanup: close connections after test
        connections.close_all()
    finally:
        conn.close()


# ============================================================================
# DATA FIXTURES
# ============================================================================

@pytest.fixture
def simple_dataframe():
    """
    Create a simple test DataFrame with input and output columns.
    
    Returns:
        pd.DataFrame: DataFrame with columns: feature1, feature2, feature3, target
    """
    np.random.seed(42)
    return pd.DataFrame({
        'feature1': np.random.normal(0, 1, 20),
        'feature2': np.random.normal(0, 1, 20),
        'feature3': np.random.choice(['A', 'B', 'C'], 20),
        'target': np.random.normal(0, 1, 20)
    })


@pytest.fixture
def numeric_dataframe():
    """
    Create a numeric-only test DataFrame.
    
    Returns:
        pd.DataFrame: DataFrame with only numeric columns
    """
    np.random.seed(42)
    return pd.DataFrame({
        'input1': np.random.normal(0, 1, 20),
        'input2': np.random.normal(0, 1, 20),
        'output': np.random.normal(0, 1, 20)
    })


@pytest.fixture
def mixed_dataframe():
    """
    Create a test DataFrame with mixed data types.
    
    Returns:
        pd.DataFrame: DataFrame with numeric, categorical, and text columns
    """
    return TestDataGenerator.create_mixed_types_data(n_samples=15)


@pytest.fixture
def columns_datatype():
    """
    Get standard column datatype mapping for test data (simple_dataframe columns).
    
    Returns:
        dict: Mapping of column names to DatasetColumnDataType values
    """
    # Get full mapping but only return columns that exist in simple_dataframe
    full_mapping = TestDataGenerator.get_standard_datatype_mapping()
    simple_df_columns = ['feature1', 'feature2', 'feature3', 'target']
    return {k: v for k, v in full_mapping.items() if k in simple_df_columns}


@pytest.fixture
def columns_description():
    """
    Get standard column description mapping for test data (simple_dataframe columns).
    
    Returns:
        dict: Mapping of column names to descriptions
    """
    # Get full mapping but only return columns that exist in simple_dataframe
    full_mapping = TestDataGenerator.get_standard_description_mapping()
    simple_df_columns = ['feature1', 'feature2', 'feature3', 'target']
    return {k: v for k, v in full_mapping.items() if k in simple_df_columns}


@pytest.fixture
def numeric_columns_datatype():
    """
    Get column datatype mapping for numeric_dataframe.
    
    Returns:
        dict: Mapping of column names to DatasetColumnDataType values
    """
    full_mapping = TestDataGenerator.get_standard_datatype_mapping()
    numeric_df_columns = ['input1', 'input2', 'output']
    return {k: v for k, v in full_mapping.items() if k in numeric_df_columns}


@pytest.fixture
def numeric_columns_description():
    """
    Get column description mapping for numeric_dataframe.
    
    Returns:
        dict: Mapping of column names to descriptions
    """
    full_mapping = TestDataGenerator.get_standard_description_mapping()
    numeric_df_columns = ['input1', 'input2', 'output']
    return {k: v for k, v in full_mapping.items() if k in numeric_df_columns}


@pytest.fixture
def iris_dataframe():
    """
    Load the Iris flowers dataset from CSV file.
    
    Returns:
        pd.DataFrame: Iris dataset with columns: SepalLengthCm, SepalWidthCm, 
                     PetalLengthCm, PetalWidthCm, Species
    """
    csv_path = project_root / "tests" / "Test Data - Iris flowers.csv"
    if not csv_path.exists():
        pytest.skip(f"Iris dataset not found at: {csv_path}")
    
    df = pd.read_csv(csv_path)
    # Remove the second row which contains "input"/"output" labels
    if len(df) > 0 and df.iloc[0, 0] in ['input', 'output']:
        df = df.iloc[1:].reset_index(drop=True)
    
    return df


# ============================================================================
# CLEANUP FIXTURES
# ============================================================================

def _delete_test_unit_machines():
    """
    Helper function to delete all machines matching '__TEST_UNIT_*' pattern.
    Uses ML.Machine class to properly clean up both Django model and data tables.
    """
    try:
        from models.machine import Machine as MachineModel
        from ML.Machine import Machine
        from django.db import transaction
        
        with transaction.atomic():
            # Find all machines with names starting with '__TEST_UNIT_'
            test_machines = MachineModel.objects.filter(machine_name__startswith='__TEST_UNIT_')
            count = test_machines.count()
            
            if count > 0:
                # Load each machine using ML.Machine class and delete properly
                for db_machine in test_machines:
                    try:
                        # Load machine using ML.Machine class to ensure proper cleanup
                        machine = Machine(db_machine.id)
                        machine.delete()
                    except Exception as e:
                        # If loading fails, try direct Django deletion as fallback
                        try:
                            db_machine.delete()
                        except Exception:
                            pass  # Ignore individual deletion errors
    except Exception:
        # Ignore cleanup errors (database might not be available)
        pass


@pytest.fixture(scope="session", autouse=True)
def session_cleanup_start():
    """
    Session-scoped fixture that runs at the beginning of the test session.
    Deletes all machines matching '__TEST_UNIT_*' pattern before tests start.
    """
    _delete_test_unit_machines()
    yield
    # Cleanup at end of session is handled by session_cleanup_end


@pytest.fixture(scope="session", autouse=True)
def session_cleanup_end():
    """
    Session-scoped fixture that runs at the end of the test session.
    Deletes all machines matching '__TEST_UNIT_*' pattern after all tests complete.
    """
    yield
    _delete_test_unit_machines()


@pytest.fixture(autouse=True)
def db_cleanup():
    """
    Automatic cleanup fixture that runs before and after each test.
    
    This fixture:
    1. Runs before each test (setup) - deletes test machines
    2. Yields control to the test
    3. Runs after each test (cleanup) - deletes test machines
    
    It deletes all machines with names starting with '__TEST_UNIT_'
    """
    # Setup: cleanup before test
    _delete_test_unit_machines()
    
    yield
    
    # Cleanup: remove test machines after test
    _delete_test_unit_machines()


# ============================================================================
# FACTORY FIXTURES
# ============================================================================

@pytest.fixture
def test_machine_name():
    """
    Generate a unique test machine name.
    
    Returns:
        str: Unique machine name prefixed with '__TEST_UNIT__'
    """
    import uuid
    return f'__TEST_UNIT__{uuid.uuid4().hex[:8]}'


@pytest.fixture
def machine_factory(test_database_with_verification):
    """
    Factory fixture for creating test machines.
    
    Usage:
        machine = machine_factory(
            name="test_machine",
            dataframe=my_df,
            columns_datatype=my_datatypes,
            columns_description=my_descriptions
        )
    
    Returns:
        function: Factory function that creates Machine instances
    """
    def _create_machine(name=None, dataframe=None, columns_datatype=None, 
                       columns_description=None, level=1):
        """Create a test machine with the given parameters."""
        from ML import Machine, MachineLevel
        from django.contrib.auth import get_user_model
        
        User = get_user_model()
        
        # Get or create admin user
        admin_user, _ = User.objects.get_or_create(
            email='SuperSuperAdmin@easyautoml.com',
            defaults={
                'first_name': 'Test',
                'last_name': 'EasyAutoML',
                'is_staff': True,
                'is_superuser': True,
                'is_active': True,
            }
        )
        if not admin_user.has_usable_password():
            admin_user.set_password('easyautoml')
            admin_user.save()
        
        # Use default values if not provided
        if dataframe is None:
            dataframe = TestDataGenerator.create_simple_classification_data()
        if columns_datatype is None:
            columns_datatype = TestDataGenerator.get_standard_datatype_mapping()
        if columns_description is None:
            columns_description = TestDataGenerator.get_standard_description_mapping()
        if name is None:
            import uuid
            name = f'__TEST_UNIT__{uuid.uuid4().hex[:8]}'
        
        # Create machine
        machine = Machine(
            name=name,
            user=admin_user,
            level=MachineLevel(level),
            dataframe=dataframe,
            columns_datatype=columns_datatype,
            columns_description=columns_description
        )
        
        return machine
    
    return _create_machine


@pytest.fixture
def logger():
    """
    Get a logger instance for tests.
    
    Returns:
        Logger: Logger instance from EasyAutoMLDBModels
    """
    return EasyAutoMLDBModels().logger


def get_admin_user():
    """
    Get or create the admin user for tests with ID=1.
    
    This function:
    1. Gets or creates a superuser with ID=1 and email 'SuperSuperAdmin@easyautoml.com'
    2. Sets password if the user was just created
    3. Returns the user instance
    
    Returns:
        User: Django User instance with admin privileges (ID=1)
    """
    from django.contrib.auth import get_user_model
    from django.db import connection
    from django.contrib.auth.hashers import make_password
    import json
    
    User = get_user_model()
    
    # Clean up duplicate users first
    try:
        duplicate_users = User.objects.filter(email='SuperSuperAdmin@easyautoml.com')
        if duplicate_users.count() > 1:
            # Keep the first one (preferably ID=1), delete others
            keep_user = duplicate_users.filter(id=1).first() or duplicate_users.first()
            duplicate_users.exclude(id=keep_user.id).delete()
    except Exception:
        # If this fails, continue with normal logic
        pass
    
    # Check if user table exists
    table_exists = False
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user'")
            table_exists = cursor.fetchone() is not None
    except Exception:
        pass
    
    if not table_exists:
        # Table doesn't exist - we can't create user via ORM
        # Try to create using raw SQL (this assumes we know the table structure)
        try:
            with connection.cursor() as cursor:
                # Check what tables exist
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                if 'user' not in tables:
                    # Can't proceed without user table
                    return None
        except Exception:
            return None
    
    # Try to get user with ID=1 first
    try:
        admin_user = User.objects.get(id=1)
        if admin_user.email != 'SuperSuperAdmin@easyautoml.com':
            # Update email if different
            admin_user.email = 'SuperSuperAdmin@easyautoml.com'
            admin_user.is_staff = True
            admin_user.is_superuser = True
            admin_user.is_active = True
            admin_user.save()
        if not admin_user.has_usable_password():
            admin_user.set_password('easyautoml')
            admin_user.save()
        return admin_user
    except User.DoesNotExist:
        pass
    except Exception:
        # If ORM fails, try raw SQL
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT id FROM user WHERE id=1")
                if cursor.fetchone():
                    # User exists, try to load via ORM again
                    pass
        except Exception:
            pass
    
    # Try to get by email - use filter().first() to avoid MultipleObjectsReturned
    try:
        admin_user = User.objects.filter(email='SuperSuperAdmin@easyautoml.com').first()
        if admin_user:
            # If user exists but doesn't have ID=1, we'll use it as-is
            if not admin_user.has_usable_password():
                admin_user.set_password('easyautoml')
                admin_user.save()
            return admin_user
    except Exception:
        # If ORM fails, try raw SQL to check
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT id FROM user WHERE email=?", ('SuperSuperAdmin@easyautoml.com',))
                row = cursor.fetchone()
                if row:
                    user_id = row[0]
                    admin_user = User.objects.get(id=user_id)
                    if not admin_user.has_usable_password():
                        admin_user.set_password('easyautoml')
                        admin_user.save()
                    return admin_user
        except Exception:
            pass
    
    # Create new user - try to set ID=1 if possible
    try:
        # Use raw SQL to check if table is empty
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM user")
            count = cursor.fetchone()[0]
            if count == 0:
                # Table is empty, we can set ID=1
                admin_user = User(
                    id=1,
                    email='SuperSuperAdmin@easyautoml.com',
                    first_name='Test',
                    last_name='EasyAutoML',
                    is_staff=True,
                    is_superuser=True,
                    is_active=True,
                )
                admin_user.set_password('easyautoml')
                admin_user.save()
                return admin_user
    except Exception as e:
        # If ORM fails, try raw SQL insert
        try:
            password_hash = make_password('easyautoml')
            with connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM user")
                count = cursor.fetchone()[0]
                if count == 0:
                    # Insert user with ID=1 using raw SQL
                    cursor.execute("""
                        INSERT INTO user (id, email, password, first_name, last_name, is_staff, is_superuser, is_active, date_joined)
                        VALUES (1, ?, ?, ?, ?, 1, 1, 1, datetime('now'))
                    """, ('SuperSuperAdmin@easyautoml.com', password_hash, 'Test', 'EasyAutoML'))
                    # Now try to load via ORM
                    admin_user = User.objects.get(id=1)
                    return admin_user
        except Exception:
            pass
    
    # Fallback: create normally
    try:
        admin_user = User.objects.create(
            email='SuperSuperAdmin@easyautoml.com',
            first_name='Test',
            last_name='EasyAutoML',
            is_staff=True,
            is_superuser=True,
            is_active=True,
        )
        admin_user.set_password('easyautoml')
        admin_user.save()
        return admin_user
    except Exception:
        return None


def get_admin_team():
    """
    Get or create the admin team for tests with ID=1.
    
    This function:
    1. Gets or creates a team with ID=1
    2. Sets admin_user to the admin user
    3. Returns the team instance
    
    Returns:
        Team: Team instance with admin privileges (ID=1)
    """
    from models.team import Team
    from django.db import connection
    
    # Check if Team table exists
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='Team'")
            if not cursor.fetchone():
                # Table doesn't exist, return None
                return None
    except Exception:
        pass
    
    admin_user = get_admin_user()
    if admin_user is None:
        return None
    
    # Try to get team with ID=1 first
    try:
        admin_team = Team.objects.get(id=1)
        if admin_team.admin_user != admin_user:
            admin_team.admin_user = admin_user
            admin_team.save()
        return admin_team
    except Team.DoesNotExist:
        pass
    
    # Try to get by name - use filter().first() to avoid MultipleObjectsReturned
    try:
        admin_team = Team.objects.filter(name='Admin Team').first()
        if admin_team:
            if admin_team.admin_user != admin_user:
                admin_team.admin_user = admin_user
                admin_team.save()
            return admin_team
    except Exception:
        pass
    
    # Create new team - try to set ID=1 if possible
    try:
        # Use raw SQL to insert with specific ID if table is empty
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM Team")
            count = cursor.fetchone()[0]
            if count == 0:
                # Table is empty, we can set ID=1
                admin_team = Team(
                    id=1,
                    name='Admin Team',
                    admin_user=admin_user,
                )
                admin_team.save()
                admin_team.create_permission()
                return admin_team
    except Exception:
        pass
    
    # Fallback: create normally
    admin_team = Team.objects.create(
        name='Admin Team',
        admin_user=admin_user,
    )
    admin_team.create_permission()
    return admin_team


@pytest.fixture
def machine_with_all_configs(test_database_with_verification):
    """
    Get an existing machine from the test database with all configurations (MDC, ICI, FEC, EncDec) already set up.
    
    This fixture:
    1. Ensures admin user (ID=1) and admin team (ID=1) exist
    2. Loads an existing machine from the test database that has all configs ready
    3. Raises an error if no suitable machine is found
    
    Usage:
        def test_something(machine_with_all_configs):
            machine = machine_with_all_configs
            # Machine already has MDC, ICI, FEC, EncDec configured
            encdec = EncDec(machine=machine)  # Will load existing config
    
    Returns:
        Machine: Machine instance with all configurations ready
    
    Raises:
        ValueError: If no machine with all configurations is found in the test database
    """
    from ML import Machine
    from django.db import connection
    
    # Ensure Django connection is open
    try:
        # Test if connection is open by getting the database name
        _ = connection.settings_dict['NAME']
    except Exception:
        # Connection is closed, reopen it
        connections.close_all()
        # Connection will be opened automatically on next use
    
    # Ensure admin user and team exist
    admin_user = get_admin_user()
    if admin_user is None:
        # Try to create user using raw SQL if user table exists
        try:
            # Ensure connection is open
            connections.close_all()  # Close any stale connections
            with connection.cursor() as cursor:
                # Check if user table exists
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user'")
                table_exists = cursor.fetchone()
                if table_exists:
                    # Table exists, check if user with ID=1 exists
                    cursor.execute("SELECT id FROM user WHERE id=1")
                    user_exists = cursor.fetchone()
                    if not user_exists:
                        # Create user with ID=1 using raw SQL
                        from django.contrib.auth.hashers import make_password
                        password_hash = make_password('easyautoml')
                        try:
                            cursor.execute("""
                                INSERT INTO user (id, email, password, first_name, last_name, is_staff, is_superuser, is_active, date_joined)
                                VALUES (1, ?, ?, ?, ?, 1, 1, 1, datetime('now'))
                            """, ('SuperSuperAdmin@easyautoml.com', password_hash, 'Test', 'EasyAutoML'))
                        except Exception as e:
                            # User might already exist with different ID, try to find it
                            cursor.execute("SELECT id FROM user WHERE email=?", ('SuperSuperAdmin@easyautoml.com',))
                            row = cursor.fetchone()
                            if row:
                                user_id = row[0]
                            else:
                                raise e
                    
                    # Reload user via ORM
                    from django.contrib.auth import get_user_model
                    User = get_user_model()
                    admin_user = User.objects.get(id=1)
                else:
                    # User table doesn't exist - create it using the migration schema
                    try:
                        # Create table
                        cursor.execute("""
                            CREATE TABLE "user" (
                                "id" integer NOT NULL PRIMARY KEY AUTOINCREMENT,
                                "password" varchar(128) NOT NULL,
                                "last_login" datetime NULL,
                                "email" varchar(254) NULL UNIQUE,
                                "first_name" varchar(140) NULL,
                                "last_name" varchar(140) NULL,
                                "user_profile" text NULL,
                                "time_format" varchar(3) NOT NULL DEFAULT '24H',
                                "date_format" varchar(3) NOT NULL DEFAULT 'DMY',
                                "date_separator" varchar(1) NOT NULL DEFAULT '/',
                                "datetime_separator" varchar(1) NOT NULL DEFAULT ' ',
                                "decimal_separator" varchar(1) NOT NULL DEFAULT ',',
                                "coupons_activated_date" text NOT NULL DEFAULT '{}',
                                "is_super_admin" bool NOT NULL DEFAULT 0,
                                "is_superuser" bool NOT NULL DEFAULT 0,
                                "is_staff" bool NOT NULL DEFAULT 0,
                                "is_active" bool NOT NULL DEFAULT 1,
                                "date_joined" datetime NOT NULL,
                                "coupon_balance" decimal(32, 10) NULL DEFAULT 0,
                                "user_balance" decimal(32, 10) NULL DEFAULT 0,
                                "user_ixioo_balance" decimal(32, 10) NULL DEFAULT 0,
                                "last_billing_time" datetime NULL
                            )
                        """)
                        # Create index
                        cursor.execute('CREATE INDEX "user_email_idx" ON "user" ("email")')
                        # Now create user with ID=1
                        from django.contrib.auth.hashers import make_password
                        password_hash = make_password('easyautoml')
                        cursor.execute("""
                            INSERT INTO user (id, email, password, first_name, last_name, is_staff, is_superuser, is_active, is_super_admin, date_joined)
                            VALUES (1, ?, ?, ?, ?, 1, 1, 1, 1, datetime('now'))
                        """, ('SuperSuperAdmin@easyautoml.com', password_hash, 'Test', 'EasyAutoML'))
                        # Reload user via ORM
                        from django.contrib.auth import get_user_model
                        User = get_user_model()
                        admin_user = User.objects.get(id=1)
                    except Exception as e:
                        # If creation fails, try to get existing user
                        try:
                            from django.contrib.auth import get_user_model
                            User = get_user_model()
                            admin_user = User.objects.get(id=1)
                        except Exception:
                            raise ValueError(
                                f"Cannot create user table or user: {str(e)}. "
                                "Please ensure the test database has been properly set up."
                            ) from e
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(
                f"Cannot create admin user: {str(e)}. "
                "Please ensure the test database has been properly set up with migrations."
            ) from e
    
    if admin_user is None:
        raise ValueError(
            "Cannot create admin user. Database tables may not be initialized. "
            "Please ensure the test database has been properly set up with migrations."
        )
    
    get_admin_team()
    
    # Find an existing machine with all configs ready from test database
    # First, verify the database file has the machine table using direct SQLite access
    import sqlite3
    db_path = test_database_with_verification
    try:
        with sqlite3.connect(db_path) as sqlite_conn:
            sqlite_cursor = sqlite_conn.cursor()
            # Check if machine table exists in the actual database file
            sqlite_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='machine'")
            if not sqlite_cursor.fetchone():
                raise ValueError(
                    f"Machine table does not exist in test database file: {db_path}. "
                    "The database file may be empty or corrupted."
                )
            
            # Get machine IDs from the actual database file
            sqlite_cursor.execute("""
                SELECT id FROM machine
                WHERE mdc_columns_name_input IS NOT NULL
                AND mdc_columns_name_input != '{}'
                AND mdc_columns_name_input != ''
                AND enc_dec_columns_info_input_encode_count IS NOT NULL
                LIMIT 20
            """)
            machine_ids = [row[0] for row in sqlite_cursor.fetchall()]
            
            if not machine_ids:
                raise ValueError(
                    f"No machines with required configurations found in test database file: {db_path}. "
                    "Please ensure the test database contains machines with MDC and EncDec configurations."
                )
    except sqlite3.Error as e:
        raise ValueError(
            f"Cannot access test database file {db_path}: {str(e)}. "
            "Please ensure the database file exists and is accessible."
        ) from e
    
    # Now try to load machines using Django ORM or Machine class
    from models.machine import Machine as MachineModel
    
    # Try to find machines using Django ORM
    try:
        existing_machines = MachineModel.objects.filter(id__in=machine_ids)
        
        for machine_db in existing_machines:
            try:
                # Try to load with admin user first
                try:
                    machine = Machine(machine_db.id, machine_access_check_with_user_id=admin_user.id)
                except Exception:
                    # If that fails, try without user check (if machine allows it)
                    try:
                        machine = Machine(machine_db.id)
                    except Exception:
                        continue
                
                # Verify all configs are ready using Machine's methods
                if (machine.is_config_ready_mdc() and 
                    machine.is_config_ready_ici() and 
                    machine.is_config_ready_fe() and 
                    machine.is_config_ready_enc_dec()):
                    return machine
            except Exception:
                # Skip this machine if loading fails
                continue
    except Exception as e:
        # If ORM fails, try loading machines directly by ID
        for machine_id in machine_ids:
            try:
                machine = Machine(machine_id, machine_access_check_with_user_id=admin_user.id)
                if (machine.is_config_ready_mdc() and 
                    machine.is_config_ready_ici() and 
                    machine.is_config_ready_fe() and 
                    machine.is_config_ready_enc_dec()):
                    return machine
            except Exception:
                continue
    
    # No suitable machine found
    raise ValueError(
        "No machine with all configurations (MDC, ICI, FEC, EncDec) found in test database. "
        "Please ensure the test database contains at least one machine with all configurations ready."
    )


@pytest.fixture
def existing_machine_with_nn_configuration(test_database_with_verification):
    """
    Get an existing machine from the test database with NN configuration already set up.
    
    This fixture:
    1. Ensures admin user (ID=1) and admin team (ID=1) exist
    2. Loads an existing machine from the test database that has NN configuration ready
    3. Raises an error if no suitable machine is found
    
    Usage:
        def test_something(existing_machine_with_nn_configuration):
            machine = existing_machine_with_nn_configuration
            # Machine already has NN configuration ready
            nn_config = NNConfiguration(machine=machine)  # Will load existing config
    
    Returns:
        Machine: Machine instance with NN configuration ready
    
    Raises:
        ValueError: If no machine with NN configuration is found in the test database
    """
    from ML import Machine
    from django.db import connection
    
    # Ensure Django connection is open
    try:
        _ = connection.settings_dict['NAME']
    except Exception:
        connections.close_all()
    
    # Ensure admin user and team exist
    admin_user = get_admin_user()
    if admin_user is None:
        raise ValueError(
            "Cannot create admin user. Database tables may not be initialized. "
            "Please ensure the test database has been properly set up with migrations."
        )
    
    get_admin_team()
    
    # Find an existing machine with NN configuration ready from test database
    import sqlite3
    db_path = test_database_with_verification
    try:
        with sqlite3.connect(db_path) as sqlite_conn:
            sqlite_cursor = sqlite_conn.cursor()
            # Check if machine table exists
            sqlite_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='machine'")
            if not sqlite_cursor.fetchone():
                raise ValueError(
                    f"Machine table does not exist in test database file: {db_path}. "
                    "The database file may be empty or corrupted."
                )
            
            # Get machine IDs with NN configuration
            sqlite_cursor.execute("""
                SELECT id FROM machine
                WHERE parameter_nn_shape IS NOT NULL
                AND parameter_nn_shape != '{}'
                AND parameter_nn_shape != ''
                AND mdc_columns_name_input IS NOT NULL
                AND enc_dec_columns_info_input_encode_count IS NOT NULL
                LIMIT 20
            """)
            machine_ids = [row[0] for row in sqlite_cursor.fetchall()]
            
            if not machine_ids:
                raise ValueError(
                    f"No machines with NN configuration found in test database file: {db_path}. "
                    "Please ensure the test database contains machines with NN configuration."
                )
    except sqlite3.Error as e:
        raise ValueError(
            f"Cannot access test database file {db_path}: {str(e)}. "
            "Please ensure the database file exists and is accessible."
        ) from e
    
    # Try to load machines using Django ORM or Machine class
    from models.machine import Machine as MachineModel
    
    try:
        existing_machines = MachineModel.objects.filter(id__in=machine_ids)
        
        for machine_db in existing_machines:
            try:
                # Try to load with admin user first
                try:
                    machine = Machine(machine_db.id, machine_access_check_with_user_id=admin_user.id)
                except Exception:
                    try:
                        machine = Machine(machine_db.id)
                    except Exception:
                        continue
                
                # Verify NN configuration is ready
                if machine.is_config_ready_nn_configuration():
                    return machine
            except Exception:
                continue
    except Exception:
        # If ORM fails, try loading machines directly by ID
        for machine_id in machine_ids:
            try:
                machine = Machine(machine_id, machine_access_check_with_user_id=admin_user.id)
                if machine.is_config_ready_nn_configuration():
                    return machine
            except Exception:
                continue
    
    # No suitable machine found
    raise ValueError(
        "No machine with NN configuration found in test database. "
        "Please ensure the test database contains at least one machine with NN configuration ready."
    )


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================

def pytest_configure(config):
    """Configure pytest with Django settings."""
    # Ensure Django is set up
    if not settings.configured:
        django.setup()


def pytest_collection_modifyitems(config, items):
    """
    Modify test items to add django_db marker to tests that need database access.
    
    This automatically adds @pytest.mark.django_db to tests that use database fixtures.
    """
    for item in items:
        # Check if test uses any database fixture
        if any('test_database' in str(arg) or 'use_test_database' in str(arg) 
               for arg in item.fixturenames):
            # Add django_db marker if not already present
            if 'django_db' not in [mark.name for mark in item.iter_markers()]:
                item.add_marker(pytest.mark.django_db)

