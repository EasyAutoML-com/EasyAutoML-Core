from pandas import DataFrame, read_sql_query
import sqlalchemy as sq
from django.db import connections
from django.apps import apps
from models.logger import EasyAutoMLLogger

# Lazy initialization of SQLAlchemy engine to allow patching in tests
_alchemy_engine = None

def _get_alchemy_engine():
    """Get or create SQLAlchemy engine, using lazy initialization to allow test patching"""
    global _alchemy_engine
    if _alchemy_engine is None:
        from django.conf import settings
        # Configure connect args based on database type
        if 'sqlite' in settings.DATABASE_SQL.lower():
            # SQLite-specific connect args to prevent database locking
            connect_args = {
                "check_same_thread": False,
                "timeout": 60,  # Wait up to 60 seconds for locks to clear
                "isolation_level": None  # Autocommit mode to reduce lock contention
            }
            # SQLite doesn't support pool_recycle, echo_pool, or max_overflow
            _alchemy_engine = sq.create_engine(
                settings.DATABASE_SQL, 
                connect_args=connect_args,
                # Use NullPool to avoid connection pooling issues with SQLite
                poolclass=sq.pool.NullPool,
                # Enable echo for debugging (can be disabled in production)
                echo=False
            )
        else:
            # MySQL-specific connect args
            connect_args = {"connect_timeout": 3600  }
            _alchemy_engine = sq.create_engine(
                settings.DATABASE_SQL, pool_recycle=3000, echo_pool=True, max_overflow=25, connect_args=connect_args
            )
    return _alchemy_engine

logger = EasyAutoMLLogger( )





class DynamicModel:

    @classmethod
    def append_dataframe(cls, dataframe):
        if not isinstance(dataframe, DataFrame):
            logger.error("Input parameter is not a DataFrame")

        name_of_table = cls._meta.db_table

        try:
            from django.db import connection as django_connection
            import pandas as pd
            import time
            import sqlite3
            
            engine_name = str(django_connection.settings_dict['ENGINE']).lower()
            
            # For SQLite: Try SQLAlchemy first (for proper type handling), fallback to Django connection
            # This is critical in test environments where Django holds transaction locks
            if 'sqlite' in engine_name:
                max_retries = 5
                retry_delay = 0.2
                use_django_connection = False
                
                # Strategy: Try SQLAlchemy first (better type handling), fallback to Django connection if locked
                for attempt in range(max_retries):
                    try:
                        if use_django_connection:
                            # Fallback: Use Django's connection (same connection = no lock conflict)
                            # But: pandas writes types as strings, so we need to handle this
                            dataframe.to_sql(
                                name=name_of_table,
                                con=django_connection.connection,
                                if_exists="append",
                                method="multi",
                                index=False
                            )
                        else:
                            # Primary: Use SQLAlchemy (proper type handling)
                            # Try to commit Django transaction first to release lock
                            try:
                                django_connection.commit()
                            except Exception:
                                pass  # Ignore if commit fails
                            
                            dataframe.to_sql(
                                name=name_of_table,
                                con=_get_alchemy_engine(),
                                if_exists="append",
                                method="multi"
                            )
                        # Success - break out of retry loop
                        break
                    except (sqlite3.OperationalError, Exception) as e:
                        error_msg = str(e).lower()
                        if "database is locked" in error_msg:
                            if attempt < max_retries - 1:
                                # Wait and retry
                                time.sleep(retry_delay * (attempt + 1))
                                # After 2 attempts with SQLAlchemy, switch to Django connection
                                if attempt >= 2 and not use_django_connection:
                                    use_django_connection = True
                                    logger.warning(f"Switching to Django connection for table '{name_of_table}' due to lock conflicts")
                                continue
                            else:
                                # Last attempt failed - log and re-raise
                                logger.error(f"There was a problem during dataframe.to_sql execution in table '{name_of_table}' after {max_retries} attempts: {e}")
                                raise
                        else:
                            # Different error - log and re-raise immediately
                            logger.error(f"There was a problem during dataframe.to_sql execution in table '{name_of_table}' : {e}")
                            raise
            else:
                # For MySQL/PostgreSQL: Use SQLAlchemy for proper type handling
                # These databases handle concurrent connections better
                dataframe.to_sql(
                    name=name_of_table,
                    con=_get_alchemy_engine(),
                    if_exists="append",
                    method="multi"
                )
        except Exception as error:
            logger.error(f"There was a problem during dataframe.to_sql execution in table '{name_of_table}' : {error}")
            # Re-raise the error so calling code knows it failed
            raise

    @classmethod
    def mark_lines(cls, field_name_to_set, field_value_to_set, where_clause=None, list_with_ids=None):
        if not isinstance(field_name_to_set, str):
            logger.error(f"field_name_to_set {field_name_to_set} parameter must be a string")

        if where_clause is None and list_with_ids is None:
            logger.error("Where_clause and list_of_line_ids_to_mark only one of them can be None")

        if list_with_ids is not None:
            logger.warning( f"using list_with_ids is deprecated because it is too slow and do not work on large tables")
            ids = ",".join([str(i) for i in list_with_ids])
            sql = f"UPDATE {cls._meta.db_table} SET {field_name_to_set} = {field_value_to_set} WHERE LINE_ID in ({ids})"
        else:
            sql = f"UPDATE {cls._meta.db_table} SET {field_name_to_set} = {field_value_to_set} {where_clause}"

        _sql_error = ""

        with connections["default"].cursor() as cursor:
            try:
                cursor.execute(sql)
            except Exception as error:
                logger.error(f"There was a problem during to_sql execution of '{sql}' : {error}" )

    @classmethod
    def truncate_table(cls):
        """
        This method is deleting all content of the table given as argument
        """
        with connections["default"].cursor() as cursor:
            try:
                cursor.execute(f"TRUNCATE TABLE {cls.__name__} ")
            except Exception as error:
                logger.error(f"There was a problem during sql TRUNCATE table {cls.__name__} execution : {error}")

    @classmethod
    def delete_table(cls):
        """
        This method is deleting all content of the table given as argument
        """
        with connections["default"].cursor() as cursor:
            try:
                cursor.execute(f"DROP TABLE {cls.__name__}")
            except Exception as error:
                logger.error(f"There was a problem during sql DROP table {cls.__name__} execution : {error}")


def create_model(base_classes, name, fields=None, app_label="", module="", options=None):
    """
    Create specified Machine Model
    """
    if not isinstance(base_classes, tuple):
        base_classes = (base_classes,)

    class Meta:
        # Using type('Meta', ...) gives a dictproxy error during MachineKerasModel creation
        pass

    if app_label:
        # app_label must be set using the Meta inner class
        setattr(Meta, "app_label", app_label)

    # Update Meta with any options that were provided
    if options is not None:
        for key, value in options.items():
            setattr(Meta, key, value)

    # Set up a dictionary to simulate declarations within a class
    attrs = {"__module__": module, "Meta": Meta}

    # Add in any fields that were provided
    if fields:
        attrs.update(fields)

    # Create the class, which automatically triggers ModelBase processing
    model = type(name, base_classes, attrs)

    return model


def create_dynamic_model(
    base_classes,
    class_name,
    table,
    fields,
    app_label,
    module_name,
    primary_key_column,
    db_name="default",
):
    """
    Dynamic Django Model factory
    - create MachineKerasModel
    - read table structure from DB
    - auto append fields to MachineKerasModel
    - fields names: c0, c1, c2, ..., cN
    example:
        class_name = "BatchInput{}".format(batch_id)
        from_table = "BATCH_INPUT_{}".format(batch_id)
        MachineKerasModel = get_dynamic_model(class_name, from_table, pk_field, 'dynamic', 'core.batchs')
    """

    # check exists already
    if (app_label in apps.all_models and class_name in apps.all_models[app_label]) or \
        (app_label.lower() in apps.all_models and class_name.lower() in apps.all_models[app_label.lower()]):
        model = apps.get_model(app_label, class_name)
    else:
        # else create the dynamic model
        model = create_model(
            base_classes,
            class_name,
            fields,
            app_label=app_label,
            module=module_name,
            options={"db_table": table},
        )
        model._meta._db_name = db_name

    return model
