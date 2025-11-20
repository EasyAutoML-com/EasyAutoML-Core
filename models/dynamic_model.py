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
                "timeout": 30  # Wait up to 30 seconds for locks to clear
            }
            # SQLite doesn't support pool_recycle, echo_pool, or max_overflow
            _alchemy_engine = sq.create_engine(
                settings.DATABASE_SQL, 
                connect_args=connect_args,
                # Use NullPool to avoid connection pooling issues with SQLite
                poolclass=sq.pool.NullPool
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
            # Use Django's database connection to avoid SQLite locking issues
            # pandas can work with raw DB-API connections
            from django.db import connection as django_connection
            import pandas as pd
            from django.db import models
            from sqlalchemy import types as sqltypes
            
            # Check if we're using SQLite
            engine_name = str(django_connection.settings_dict['ENGINE']).lower()
            
            if 'sqlite' in engine_name:
                # For SQLite: Use Django's raw connection directly with pandas
                # This avoids the "database is locked" error by sharing the same connection
                django_connection.ensure_connection()
                
                # ========== TYPE CONVERSION FIX ==========
                # Convert DataFrame column types to match Django model field types
                # This prevents string numerics from being saved incorrectly
                dataframe = dataframe.copy()  # Don't modify original
                
                # Get model fields to determine correct types
                try:
                    model_fields = cls._meta.get_fields()
                    
                    for field in model_fields:
                        field_name = field.name
                        
                        # Skip if column not in DataFrame
                        if field_name not in dataframe.columns:
                            continue
                        
                        # Convert based on Django field type
                        if isinstance(field, models.DecimalField) or isinstance(field, models.FloatField):
                            # Convert to numeric (handles strings like '-1.41230370133529' → float)
                            try:
                                dataframe[field_name] = pd.to_numeric(
                                    dataframe[field_name], 
                                    errors='coerce'  # Convert invalid values to NaN
                                )
                            except Exception as e:
                                logger.warning(f"Could not convert {field_name} to numeric: {e}")
                        
                        elif isinstance(field, models.BooleanField):
                            # Ensure boolean type
                            try:
                                dataframe[field_name] = dataframe[field_name].astype(bool)
                            except Exception as e:
                                logger.warning(f"Could not convert {field_name} to boolean: {e}")
                        
                        elif isinstance(field, (models.IntegerField, models.BigIntegerField, models.AutoField)):
                            # Ensure integer type
                            try:
                                dataframe[field_name] = pd.to_numeric(
                                    dataframe[field_name], 
                                    errors='coerce',
                                    downcast='integer'
                                )
                            except Exception as e:
                                logger.warning(f"Could not convert {field_name} to integer: {e}")
                
                except Exception as e:
                    logger.warning(f"Could not apply type conversions: {e}")
                # ========== END TYPE CONVERSION FIX ==========
                
                # Build dtype dict to ensure proper column types in SQLite schema
                # Use SQLAlchemy types instead of strings for better type preservation
                dtype_dict = {}
                
                # Ensure Line_ID is treated as INTEGER, not DECIMAL
                if 'Line_ID' in dataframe.columns:
                    dtype_dict['Line_ID'] = sqltypes.INTEGER()
                
                # Map DataFrame columns to SQLAlchemy types based on actual dtypes
                # This is critical - pandas to_sql() needs explicit SQLAlchemy type hints
                for col in dataframe.columns:
                    if col in dtype_dict:
                        continue  # Already set
                    
                    col_dtype = dataframe[col].dtype
                    
                    # Boolean columns
                    if col.startswith('IsFor') or col.startswith('Is') or col_dtype == 'bool':
                        dtype_dict[col] = sqltypes.Boolean()
                    # Numeric columns - Use SQLAlchemy REAL/Float type
                    elif pd.api.types.is_float_dtype(col_dtype):
                        dtype_dict[col] = sqltypes.Float()  # SQLAlchemy Float type
                    elif pd.api.types.is_integer_dtype(col_dtype):
                        dtype_dict[col] = sqltypes.INTEGER()
                    # String/object columns - Use TEXT type
                    elif pd.api.types.is_object_dtype(col_dtype) or pd.api.types.is_string_dtype(col_dtype):
                        dtype_dict[col] = sqltypes.TEXT()
                
                # Check if Line_ID is in the index
                if dataframe.index.name == 'Line_ID':
                    # Line_ID is the index - specify index_label
                    dataframe.to_sql(
                        name=name_of_table,
                        con=django_connection.connection,
                        if_exists="append",
                        method="multi",
                        index=True,
                        index_label='Line_ID',
                        dtype=dtype_dict if dtype_dict else None
                    )
                else:
                    # Regular append with explicit dtypes
                    dataframe.to_sql(
                        name=name_of_table,
                        con=django_connection.connection,
                        if_exists="append",
                        method="multi",
                        dtype=dtype_dict if dtype_dict else None
                    )
            else:
                # For MySQL and other databases: Use SQLAlchemy engine as before
                dataframe.to_sql(
                    name=name_of_table,
                    con=_get_alchemy_engine(),
                    if_exists="append",
                    method="multi"
                )
        except Exception as error:
            logger.error(f"There was a problem during dataframe.to_sql execution in table '{name_of_table}' : {error} ")

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
