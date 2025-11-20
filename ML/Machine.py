import random
import re
import pandas as pd
import numpy as np
from typing import Union, Optional, NoReturn, Literal
from django.utils import timezone
from django.core.exceptions import MultipleObjectsReturned
from decimal import Decimal
from django.conf import settings

from SharedConstants import *

from ML import EasyAutoMLDBModels

# Lazy initialization - don't instantiate on import
_db_models = None

def _get_db_models():
    """Lazily initialize database models on first use"""
    global _db_models
    if _db_models is None:
        _db_models = EasyAutoMLDBModels()
    return _db_models

# These will be populated when first accessed
machine_model = None
machinetablelockwrite_model = None
user_model = None
team_model = None
data_lines_operation_model = None
logger = None

def _init_models():
    """Initialize models on first use"""
    global machine_model, machinetablelockwrite_model, user_model, team_model, data_lines_operation_model, logger
    if machine_model is None:
        db = _get_db_models()
        machine_model = db.Machine
        machinetablelockwrite_model = db.MachineTableLockWrite
        user_model = db.User
        team_model = db.Team
        data_lines_operation_model = db.DataLinesOperation
        logger = db.logger


pd.options.mode.chained_assignment = None


class Machine:
    """
    machine is a class for managing the machines
    init will load the machine or
    init will create the machine if we provide a name and a dataset - it will create mdc with full configuration

    Machine have 3 attributes:
    self.db_machine
    self.db_data_input_lines
    self.db_data_output_lines
    """


    def __repr__(self):
        return str(self.db_machine)

    def __init__(
            self,
            machine_identifier_or_name: Union[int, str ],
            user_dataset_unformatted: Optional[pd.DataFrame ] = None,
            dfr: Optional = None,
            machine_level: Optional[int] = None,
            machine_access_check_with_user_id: Optional[int ] = None,
            machine_create_user_id: Optional[int ] = None,
            machine_create_team_id: Optional[int ] = None,
            force_create_with_this_inputs: Optional[dict] = None,
            force_create_with_this_outputs: Optional[dict] = None,
            force_create_with_this_descriptions: Optional[dict] = None,
            decimal_separator: Optional[Literal[".", ","]] = None,
            date_format: Optional[Literal["DMY", "MDY", "YMD"]] = None,
            disable_foreign_key_checking: bool = False,
            **kwargs,
    ):
        """
        init will load the machine if machine_access_user_id have access to it
        init will create the machine if we provide => a name and => a dataset (create a dfr) or a dfr
        it will create mdc with full configuration
        if machine_identifier_or_name is None we create empty machine_source
        if machine_identifier_or_name is int we load machine_source by id
        if machine_identifier_or_name is string and dataframe is None we load machine_source by machine_name_to_load

        if machine_identifier_or_name is string and dataframe is not None will create the machine if we provide a name and a dataset - it will create mdc with full configuration
        
        Args:
            disable_foreign_key_checking: If True, disables SQLite foreign key constraints during table creation (useful for tests)
        """
        
        # Initialize models on first use
        _init_models()

        # Apply defaults if not provided
        if decimal_separator is None:
            decimal_separator = DEFAULT_DECIMAL_SEPARATOR
        if date_format is None:
            date_format = DEFAULT_DATE_FORMAT

        # Validate decimal_separator
        if decimal_separator not in DECIMAL_SEPARATOR_CHOICES:
            logger.error(f"Invalid decimal_separator: '{decimal_separator}'. Must be one of {DECIMAL_SEPARATOR_CHOICES}")
        
        # Validate date_format
        if date_format not in DATE_FORMAT_CHOICES:
            logger.error(f"Invalid date_format: '{date_format}'. Must be one of {DATE_FORMAT_CHOICES}")

        self.id = None
        self.disable_foreign_key_checking = disable_foreign_key_checking

        self.db_machine = None
        self.db_data_input_lines = None
        self.db_data_output_lines = None

        if isinstance(machine_identifier_or_name, str ) and isinstance(user_dataset_unformatted, pd.DataFrame ):
            # date_format and decimal_separator now have defaults, so they are always available
            self._init_create_new_machine_by_machine_name_and_dataset(
                machine_name=machine_identifier_or_name,
                user_dataset_unformatted=user_dataset_unformatted,
                machine_level=machine_level,
                force_create_with_this_descriptions=force_create_with_this_descriptions,
                force_create_with_this_inputs=force_create_with_this_inputs,
                force_create_with_this_outputs=force_create_with_this_outputs,
                machine_owner_user_id=machine_create_user_id,
                machine_owner_team_id=machine_create_team_id,
                decimal_separator = decimal_separator,
                date_format = date_format,
                **kwargs,
            )

        elif isinstance(machine_identifier_or_name, str ) and dfr is not None:
            self._init_create_new_machine_by_machine_name_and_dfr(
                machine_name=machine_identifier_or_name,
                dfr=dfr,
                force_create_with_this_inputs=force_create_with_this_inputs,
                force_create_with_this_outputs=force_create_with_this_outputs,
                force_create_with_this_descriptions=force_create_with_this_descriptions,
                machine_level=machine_level,
                machine_owner_user_id=machine_create_user_id,
                machine_owner_team_id=machine_create_team_id,
                decimal_separator = decimal_separator,
                date_format = date_format,
                **kwargs,
            )

        elif isinstance(machine_identifier_or_name, int):
            self._init_load_machine_by_id( machine_id=machine_identifier_or_name , machine_access_check_with_user_id=machine_access_check_with_user_id )

        elif isinstance(machine_identifier_or_name, str) and user_dataset_unformatted is None:
            self._init_load_machine_by_name(machine_name_to_load=machine_identifier_or_name, machine_access_user_id=machine_access_check_with_user_id )

        else:
            error_msg = f"Combination of argument are not valid , machine_identifier_or_name:{machine_identifier_or_name} , machine_access_check_with_user_id:{machine_access_check_with_user_id} , "
            logger.error(error_msg)
            raise ValueError(error_msg)


    def _init_create_new_machine_by_machine_name_and_dfr(
                    self,
                    machine_name:str,
                    dfr,
                    force_create_with_this_inputs:dict,
                    force_create_with_this_outputs:dict,
                    force_create_with_this_descriptions:dict,
                    machine_level:int,
                    machine_owner_user_id:int,
                    machine_owner_team_id:int,
                    decimal_separator: Literal[".", ","],
                    date_format: Literal["DMY", "MDY", "YMD"],
                    **kwargs,
        ):
        # Validate decimal_separator
        if decimal_separator not in DECIMAL_SEPARATOR_CHOICES:
            logger.error(f"Invalid decimal_separator: '{decimal_separator}'. Must be one of {DECIMAL_SEPARATOR_CHOICES}")
        
        # Validate date_format
        if date_format not in DATE_FORMAT_CHOICES:
            logger.error(f"Invalid date_format: '{date_format}'. Must be one of {DATE_FORMAT_CHOICES}")
        
        if ENABLE_LOGGER_DEBUG_Machine: logger.debug( f"Creation {machine_name} by DFR " )

        from ML import MachineDataConfiguration
        mdc = MachineDataConfiguration(
                    machine=self,
                    user_dataframe_for_create_cfg=dfr.get_formatted_user_dataframe,
                    columns_type_user_df=dfr.get_user_columns_datatype,
                    columns_description_user_df=dfr.get_user_columns_description,
                    force_create_with_this_inputs=force_create_with_this_inputs,
                    force_create_with_this_outputs=force_create_with_this_outputs,
                    decimal_separator=decimal_separator,
                    date_format=date_format,
        )

        # checking in MDC if the columns inputs and outputs are presents
        list_columns_outputs_names = [ key for key, value in mdc.columns_name_output.items( ) if value ]
        if len( list_columns_outputs_names ) == 0:
            logger.error( " Impossible to create the machine because there is no outputs columns defined/found" )
        list_inputs_outputs_names = [ key for key, value in mdc.columns_name_input.items( ) if value ]
        if len( list_inputs_outputs_names ) == 0:
            logger.error( " Impossible to create the machine because there is no inputs columns defined/found" )

        # TODO : To avoid duplicates  , check if this user have not already a machine with this name (can be his own machine or available trough the team)

        machine_name_is_reserved = machine_name.startswith( "__" ) and machine_name.endswith( "__" )

        # define the machine level
        if machine_name_is_reserved:
            creation_with_machine_level = machine_level if machine_level else MACHINE_EASYAUTOML_RESERVED_DEFAULT_LEVEL
            if machine_owner_user_id != user_model.get_super_admin( ).id:
                logger.error( "machine name is reserved , machine_owner_user_id must be super user " )
            machine_owner_user_id = user_model.get_super_admin( ).id
        # machine_owner_team_id = team_model.get_super_admin_team().id   # TODO FIXME
        else:
            creation_with_machine_level = machine_level if machine_level else MACHINE_USER_DEFAULT_LEVEL
            if not machine_owner_user_id:
                logger.error( "machine name is not reserved , so machine_owner_user_id must be set " )
        machine_level_instance = MachineLevel( creation_with_machine_level )

        # creation of the machine in the table machine
        self.db_machine = machine_model(
                    machine_owner_user_id=machine_owner_user_id,
                    machine_owner_team_id=machine_owner_team_id,
                    machine_name=machine_name,
                    machine_level=int( creation_with_machine_level ),
                    **kwargs,
        )

        # saving information from dfr and mdc
        dfr.save_configuration_in_machine( self )
        mdc.save_configuration_in_machine( )

        # define some values in the table machine
        self.db_machine.fe_budget_total = (machine_level_instance.feature_engineering_budget( )[ 1 ])

        # save the Machine to be able to get his ID because it we need the id to create the 2 datalines tables at next step
        self.db_machine.save( )
        self.id = self.db_machine.id

        # in this step we create Machine_<ID>_DataInputLines/Machine_<ID>_DataOutputLines tables
        self.data_lines_create_both_tables( )

        self.data_lines_append(
                    user_dataframe_to_append=dfr.get_formatted_user_dataframe ,
                    split_lines_in_learning_and_evaluation=True ,
                    **kwargs  )

        self.db_machine.save( )

        if ENABLE_LOGGER_DEBUG_Machine:
            logger.debug( f"Machine created and saved : {self}" )


    def _init_create_new_machine_by_machine_name_and_dataset(
            self,
            machine_name:str,
            user_dataset_unformatted: pd.DataFrame,
            force_create_with_this_inputs: dict,
            force_create_with_this_outputs: dict,
            force_create_with_this_descriptions: dict,
            machine_level: int,
            machine_owner_user_id: int,
            machine_owner_team_id: int,
            decimal_separator: Literal[".", ","],
            date_format: Literal["DMY", "MDY", "YMD"],
            **kwargs,
    ):
        # Validate decimal_separator
        if decimal_separator not in DECIMAL_SEPARATOR_CHOICES:
            logger.error(f"Invalid decimal_separator: '{decimal_separator}'. Must be one of {DECIMAL_SEPARATOR_CHOICES}")
        
        # Validate date_format
        if date_format not in DATE_FORMAT_CHOICES:
            logger.error(f"Invalid date_format: '{date_format}'. Must be one of {DATE_FORMAT_CHOICES}")

        if ENABLE_LOGGER_DEBUG_Machine: logger.debug( f"Creation {machine_name} by Dataset " )

        from ML import DataFileReader
        dfr = DataFileReader(
                        user_dataset_unformatted,
                        decimal_separator=decimal_separator,
                        date_format=date_format,
                        force_create_with_this_descriptions=force_create_with_this_descriptions,
            )

        self._init_create_new_machine_by_machine_name_and_dfr(
                    machine_name=machine_name,
                    dfr=dfr,
                    force_create_with_this_inputs=force_create_with_this_inputs,
                    force_create_with_this_outputs=force_create_with_this_outputs,
                    force_create_with_this_descriptions=force_create_with_this_descriptions,
                    machine_level=machine_level,
                    machine_owner_user_id=machine_owner_user_id,
                    machine_owner_team_id=machine_owner_team_id,
                    decimal_separator=decimal_separator,
                    date_format=date_format,
                    **kwargs,
        )


    def _init_load_machine_by_name(
            self,
            machine_name_to_load: str,
            machine_access_user_id: int,
    ) -> NoReturn:
        if ENABLE_LOGGER_DEBUG_Machine:
            logger.debug(f"Loading <machine: name={machine_name_to_load}> was started")

        if machine_access_user_id is None:
            logger.warning("DEPRECATED - for security and because machine name can have duplicates, we need machine_access_user_id")
            # FIXME TODO search a machine by name ONLY from the list of machine owned by user.id or if user.id is in the machine.team

        try:
            # FIXME TODO search a machine by name ONLY from the list of machine owned by user.id or if user.id is in the machine.team - TODO TEAM
            self.db_machine = machine_model.objects.get(machine_name=machine_name_to_load, machine_owner_user__id=machine_access_user_id )
        except machine_model.DoesNotExist:
            raise Exception(f"Unable to load machine by name {machine_name_to_load} because do not exist!")
        except Exception as e:
            raise Exception(f"Unable to load machine with name {machine_name_to_load} because: {e}")

        if not self.db_machine:
            raise Exception(f"Unable to load machine with name {machine_name_to_load} !")

        self.id = self.db_machine.id
        self.db_data_input_lines = self.db_machine.get_machine_data_input_lines_model()
        self.db_data_output_lines = (self.db_machine.get_machine_data_output_lines_model())


    def _init_load_machine_by_id( self,
                machine_id: int,
                machine_access_check_with_user_id: int,
            ) -> NoReturn:
        if ENABLE_LOGGER_DEBUG_Machine: logger.debug(f"Loading <machine: id={machine_id}> ")

        if machine_access_check_with_user_id is None:
            logger.warning("DEPRECATED - for security and because machine name can have duplicates, we need machine_access_check_with_user_id")

        try:
            # FIXME TODO search a machine  ONLY from the list of machine owned by user.id or if user.id is in the machine.team
            self.db_machine = machine_model.objects.get(id=machine_id)
        except machine_model.DoesNotExist:
            pass
        except Exception as e:
            logger.error(f"Unable to load machine with ID because : {e}")

        if not self.db_machine:
            logger.error(f"Unable to load machine with ID <{machine_id}> !")

        self.id = machine_id
        self.db_data_input_lines = self.db_machine.get_machine_data_input_lines_model()
        self.db_data_output_lines = (
            self.db_machine.get_machine_data_output_lines_model()
        )


    def data_lines_get_last_id(self, column_mode: Union[str, ColumnDirectionType ] ):
        if isinstance(column_mode, str):
            logger.error("str is deprecated !")
            column_mode = ColumnDirectionType(column_mode.lower())

        if not isinstance(column_mode, ColumnDirectionType) or column_mode == ColumnDirectionType.IGNORED:
            logger.error("Column mode type is wrong")

        if column_mode == ColumnDirectionType.INPUT:
            return get_last_id_from_dynamic_model(self.db_data_input_lines)
        elif column_mode == ColumnDirectionType.OUTPUT:
            return get_last_id_from_dynamic_model(self.db_data_output_lines)


    def data_lines_create_both_tables(self ) -> NoReturn:
        from django.db import connections
        from django.conf import settings

        if ENABLE_LOGGER_DEBUG_Machine: logger.debug( f"Create_data_tables for {self} started" )

        if not self.id or not self.db_machine.id:
            logger.error("No ID ! Please save machine_source model before _create_data_tables!")

        # Handle SQLite foreign key constraint issue
        if self.disable_foreign_key_checking and settings.DATABASES['default']['ENGINE'] == 'django.db.backends.sqlite3':
            # For SQLite with disabled foreign key checking, use raw SQL to avoid schema editor issues
            with connections["default"].cursor() as cursor:
                # Disable foreign key constraints
                cursor.execute("PRAGMA foreign_keys=OFF;")
                
                try:
                    # Get the models
                    input_model = self.db_machine.get_machine_data_input_lines_model()
                    output_model = self.db_machine.get_machine_data_output_lines_model()
                    
                    # Create input table first using raw SQL
                    input_sql = self._get_create_table_sql(input_model)
                    cursor.execute(input_sql)
                    
                    # Create output table second using raw SQL
                    output_sql = self._get_create_table_sql(output_model)
                    cursor.execute(output_sql)
                    
                finally:
                    # Re-enable foreign key constraints
                    cursor.execute("PRAGMA foreign_keys=ON;")
        else:
            # For other databases or when foreign key checking is enabled, use normal schema editor
            with connections["default"].schema_editor() as schema_editor:
                # Create input table first (referenced table)
                schema_editor.create_model(
                    self.db_machine.get_machine_data_input_lines_model()
                )
                # Create output table second (table with foreign key)
                schema_editor.create_model(
                    self.db_machine.get_machine_data_output_lines_model()
                )

        self.db_data_input_lines = self.db_machine.get_machine_data_input_lines_model()
        self.db_data_output_lines = self.db_machine.get_machine_data_output_lines_model()

    def _get_create_table_sql(self, model):
        """Generate CREATE TABLE SQL for a Django model"""
        from django.db import connection
        from django.db import models
        
        # Get the table name
        table_name = model._meta.db_table
        
        # Get field definitions
        fields = []
        for field in model._meta.fields:
            if field.primary_key:
                fields.append(f'"{field.column}" INTEGER PRIMARY KEY')
            elif isinstance(field, models.ForeignKey):
                # For foreign keys, just create as INTEGER (no constraint)
                fields.append(f'"{field.column}" INTEGER')
            elif isinstance(field, models.BooleanField):
                fields.append(f'"{field.column}" BOOLEAN')
            elif isinstance(field, models.IntegerField):
                fields.append(f'"{field.column}" INTEGER')
            elif isinstance(field, models.FloatField):
                fields.append(f'"{field.column}" REAL')
            elif isinstance(field, models.TextField):
                fields.append(f'"{field.column}" TEXT')
            elif isinstance(field, models.CharField):
                max_length = field.max_length
                fields.append(f'"{field.column}" VARCHAR({max_length})')
            else:
                # Default to TEXT for unknown field types
                fields.append(f'"{field.column}" TEXT')
        
        # Create the SQL
        sql = f'CREATE TABLE "{table_name}" ({", ".join(fields)})'
        return sql


    def data_lines_get_last_id( self ) -> int:
        """
        provide the last row id of the table indicated in argument
        used to add data in both table input and output, so the row id mlust match in both tables
        Normally We need to lock the table before calling this function ,  to ensure that another server will not add rows during an append operation

        :params column_mode: indicate if we want the last_row_id from the input or from the output table
        :return: Give the last row id used , so to get the next row id free just add 1
        """
        # return the last used row id --- sometime data are appended only in INPUT table (but never only in OUTPUT table) , so we take the id from INPUT
        return get_last_id_from_dynamic_model( self.db_data_input_lines )


    def data_lines_read(
            self,
            sort_by: str = "",
            rows_count_limit: Optional[int ] = None,
            **kwargs,
    ) -> pd.DataFrame:
        """
        this method get the pandas dataframe which return inner join of two dataframe, one from data_input_lines_read
        and second from data_output_lines_read

        :params **kwargs: can be None or column_mode or status for lines
        :return: inner joined pandas dataframe
        """
        return self.db_machine.read_data_lines_from_db(
            sort_by=sort_by,
            rows_count_limit=rows_count_limit,
            where_clause_dict=kwargs
        )


    def data_lines_update(self, dataframe: pd.DataFrame) -> NoReturn:
        """
        this method will update lines in db_data_input_lines and/or db_data_output_lines

        :params dataframe: pandas dataframe which have index.name == Line_ID
        :return: None
        """

        input_columns_list = [
            column
            for column in self.get_list_of_columns_name(
                column_mode=ColumnDirectionType.INPUT
            )
            if column in dataframe
        ]
        output_columns_list = [
                                  column
                                  for column in self.get_list_of_columns_name(ColumnDirectionType.OUTPUT)
                                  if column in dataframe
                              ]

        input_dataframe = dataframe[input_columns_list]
        output_dataframe = dataframe[output_columns_list]
        self.data_input_lines_update(input_dataframe)
        self.data_output_lines_update(output_dataframe)
        if ENABLE_LOGGER_DEBUG_Machine: logger.debug( f"data_lines_updated" )


    def data_lines_delete_all(self) -> NoReturn:
        self.db_data_input_lines.truncate_table()
        self.db_data_output_lines.truncate_table()
        if ENABLE_LOGGER_DEBUG_Machine:
            logger.debug(f"machine data lines deleted")


    def data_lines_append(
            self,
            user_dataframe_to_append: pd.DataFrame,
            split_lines_in_learning_and_evaluation: bool = False,
            **kwargs
            ) -> NoReturn:
        """
        format all cells of dataframe according to the columns_type
        then will append lines to database
        we can append inputs or inputs+outputs

        :params dataframe_to_append: pandas dataframe with NO index name LineInput_ID
        :params IsForLearning_and_IsForEvaluation: when we want to create lines for learning and a percentage for evaluation we need this flag
        :params **kwargs: Should have one of [IsForLearning/IsForSolving/IsForEvaluation] which true
        :return: None
        """

        all_input_columns = self.get_list_of_columns_name(column_mode=ColumnDirectionType.INPUT)
        all_output_columns = self.get_list_of_columns_name(column_mode=ColumnDirectionType.OUTPUT)


        if not all( col in user_dataframe_to_append.columns for col in all_input_columns):
            logger.error( f"Dataframe to add must have all input columns, this input cols are missing : {set(all_input_columns) - user_dataframe_to_append.columns}")

        output_cols_data_to_append = [ col for col in all_output_columns if col in user_dataframe_to_append.columns ]
        if not output_cols_data_to_append:
            # no outputs to add
            pass
        elif not (set(output_cols_data_to_append) ^ set( all_output_columns )):
            # all outputs cols to add
            pass
        else:
            logger.error( f"Dataframe to add must have all outputs cols or none of them. to append={output_cols_data_to_append}, all={all_output_columns}, diff={set(output_cols_data_to_append) ^ set( all_output_columns )} " )

        dataframe_to_append_formatted = user_dataframe_to_append.rename_axis(None )
        input_dataframe = dataframe_to_append_formatted[all_input_columns ]

        with DoMachineLockTables( [self.db_data_input_lines._meta.db_table , self.db_data_output_lines._meta.db_table]):
            first_new_row_id = self.data_lines_get_last_id( ) + 1
            try:
                if split_lines_in_learning_and_evaluation:
                    self.data_input_lines_append(input_dataframe, split_lines_in_learning_and_evaluation , already_locked_skip_lock=True )
                else:
                    self.data_input_lines_append( input_dataframe , already_locked_skip_lock=True , **kwargs )

                if output_cols_data_to_append:
                    output_dataframe = dataframe_to_append_formatted[all_output_columns ]
                    output_dataframe.index = pd.Index(range(first_new_row_id , first_new_row_id  + len(output_dataframe)), name="Line_ID")
                    self.data_output_lines_append(output_dataframe , already_locked_skip_lock=True )

            except Exception as e:
                logger.error( f"Unable to write data in input+output tables for machine {self} because {e}")
                self.store_error( "*" , f"Unable to write data in input+output tables for machine {self} because {e}" )

        # it is impossible but we check anyway
        if get_last_id_from_dynamic_model( self.db_data_input_lines ) != get_last_id_from_dynamic_model( self.db_data_output_lines ):
            logger.error( f"Not same last id in input and output table for machine {self} !" )

        if ENABLE_LOGGER_DEBUG_Machine:
            logger.debug(f"data_lines_appended {user_dataframe_to_append} rows, with kwarg:{kwargs} ")


    def data_lines_mark(
            self,
            list_of_line_ids_to_mark: pd.Index = None,
            **kwargs,
    ):
        """
        this method will mark lines at db_data_input_lines (there is no flags in outputs lines)

        :params **kwargs: Should have one or more of ['IsSolved', 'IsLearned'] which true
        :return: None
        """

        if list_of_line_ids_to_mark is None:
            for mode, value in kwargs.items():
                self.data_input_lines_mark(**{mode: value})
        else:
            for mode, value in kwargs.items():
                self.data_input_lines_mark(list_with_ids=list_of_line_ids_to_mark, **{mode: value})

        if ENABLE_LOGGER_DEBUG_Machine:
            logger.debug(f"data_lines_mark with kwargs {kwargs}")


    @staticmethod
    def _data_lines_update_in_this_table( column_list, dataframe, db_table):
        if len(
                [column for column in dataframe.columns if column in column_list]
        ) != len(dataframe.columns):
            raise AttributeError(
                "unknown column found in dataframe, "
                "we cannot update because we don`t have this column in target db"
            )

        for line_id, row in dataframe.iterrows():

            try:
                line = db_table.objects.get(Line_ID=line_id)
            except db_table.DoesNotExist:
                logger.error(f"line with Line_ID={line_id} doest exist in db")
            except Exception as e:
                logger.error( f"Error unable to get the line {line_id} for table {db_table} , the error is: {e}")

            for column in dataframe.columns:
                setattr(line, column, row[column] if pd.notna(row[column]) else 0)
            line.save()


    def data_input_lines_read(
            self,
            sort_by: str = "",
            rows_count_limit: Optional[int ] = None,
            **kwargs,
    ) -> pd.DataFrame:
        """
        this method will get the user_dataframe from db_data_input_lines
        this is the Machine of table Machine_<ID>_DataInputLines

        :params **kwargs: can be None or column_mode or status for lines
        :return: pandas user_dataframe
        """

        # if ENABLE_LOGGER_DEBUG: logger.debug(f"machine data input lines reading with kwargs {kwargs}")

        # validate the input parameters | For IsForLearning/IsForSolving/IsForEvaluation/IsLearned/IsSolved
        # if sum([kwargs.get(mode, False) for mode in MACHINES_DATALINES_INPUT_RESERVED_FIELDS]) > 1:
        #     logger.error(f"only one of this value {MACHINES_DATALINES_INPUT_RESERVED_FIELDS} can be True")
        return self.db_machine.read_data_lines_from_db(
            input_columns=self.get_list_of_columns_name(ColumnDirectionType.INPUT),
            sort_by=sort_by,
            rows_count_limit=rows_count_limit,
            where_clause_dict=kwargs
            )


    def data_input_lines_append(
            self,
            dataframe_to_append: pd.DataFrame,
            split_lines_in_learning_and_evaluation: bool = True,
            already_locked_skip_lock: bool = False,
            **kwargs
    ) -> NoReturn:
        """
        this method will append lines to db_data_input_lines

        :params dataframe: pandas user_dataframe with no index name LineInput_ID
        :params IsForLearning_and_IsForEvaluation: when we want to create lines for learning and a percentage for evaluation we need this flag because kwarg put exactly same flag in all lines
        :params skip_do_lock: normally we lock the table during the operation , so there is no problems if another process do a data_line_append (working with input and output succesively)
        :params **kwargs: Should have one of [IsForLearning/IsForSolving/IsForEvaluation] which true apply this flag on all lines
        :return: None
        """
        dataframe = dataframe_to_append.copy()

        list_of_enabled_datalines_flags_IsFor = [
            mode
            for mode in ["IsForLearning", "IsForSolving", "IsForEvaluation"]
            if mode in kwargs and kwargs[mode] == True
        ]
        if split_lines_in_learning_and_evaluation and len(list_of_enabled_datalines_flags_IsFor ) != 0:
            logger.error( "IsForLearning_and_IsForEvaluation should not set a flag [IsForLearning/IsForSolving/IsForEvaluation]")
        elif not split_lines_in_learning_and_evaluation and len(list_of_enabled_datalines_flags_IsFor ) != 1:
            logger.error("One single flag must be enabled from [IsForLearning/IsForSolving/IsForEvaluation]")

        # Dataframe do not have Line_ID (if it have we trow a fatal error)
        if "Line_ID" == dataframe.index.name:
            logger.error("dataframe should not contain index Line_ID")
        input_columns_list = self.get_list_of_columns_name(ColumnDirectionType.INPUT)

        # Check if dataframe columns satisfy the columns in Machine_<ID>_DataInputLines table
        if not check_equality_two_columns_lists(list(dataframe.columns), input_columns_list):
            logger.error("the columns of DataFrame is not the same like in Machine_DataInputTable")

        # set default values FALSE for all MACHINES_DATALINES_INPUT_RESERVED_FIELDS
        for reserved_field_name in MACHINES_DATALINES_INPUT_RESERVED_FIELDS:
            dataframe[reserved_field_name] = False

        if split_lines_in_learning_and_evaluation:
            for index, row in dataframe.iterrows():
                if random.randint(0, 100) > DATALINES_PERCENTAGE_OF_VALIDATION_LINES:
                    dataframe.loc[ index, "IsForLearning"] = True
                else:
                    dataframe.loc[ index , "IsForEvaluation"] = True
        elif list_of_enabled_datalines_flags_IsFor[0] == "IsForLearning":
            for i in range(0, len(dataframe.index)):
                dataframe.at[i, "IsForLearning"] = True
        elif list_of_enabled_datalines_flags_IsFor[0] == "IsForEvaluation":
            for i in range(0, len(dataframe.index)):
                dataframe.at[i, "IsForEvaluation"] = True
        elif list_of_enabled_datalines_flags_IsFor[0] == "IsForSolving":
            for i in range(0, len(dataframe.index)):
                dataframe.at[i, "IsForSolving"] = True

        try:
            # add the dataframe in the table with generated Line_ID as consecutive numbers in the table
            if already_locked_skip_lock:
                first_new_row_id = self.data_lines_get_last_id( ) + 1
                dataframe.index = pd.Index(range(first_new_row_id, first_new_row_id + len(dataframe)), name="Line_ID")
                self.db_data_input_lines.append_dataframe(dataframe)
            else:
                with DoMachineLockTables( [self.db_data_input_lines._meta.db_table]):
                    first_new_row_id = self.data_lines_get_last_id( ) + 1
                    dataframe.index = pd.Index(range(first_new_row_id, first_new_row_id + len(dataframe)), name="Line_ID")
                    self.db_data_input_lines.append_dataframe(dataframe)
        except Exception as e:
            logger.error( f"Unable to write data in {self.db_data_input_lines} in {self} because {e}" )
            self.store_error( "*" , f"Unable to write data in {self.db_data_input_lines} in {self} because {e}" )

        # create a data_lines_operation_model row, this is used by WorkDispatcher to be very fast to dispatch works
        if split_lines_in_learning_and_evaluation or \
                "IsForLearning" in list_of_enabled_datalines_flags_IsFor or \
                "IsForSolving" in list_of_enabled_datalines_flags_IsFor:

            # is there already a line in data_lines_operation_model for this machine and this work
            data_line_operation_row_count = data_lines_operation_model.objects.filter(
                machine_id=self.db_machine.id,
                is_added_for_learning=("IsForLearning" in list_of_enabled_datalines_flags_IsFor) or split_lines_in_learning_and_evaluation,
                is_added_for_solving=("IsForSolving" in list_of_enabled_datalines_flags_IsFor),
            ).count()
            if data_line_operation_row_count > 0:
                new_dlo = data_lines_operation_model()
                new_dlo.machine_id = self.db_machine.id
                new_dlo.date_time = timezone.now()
                new_dlo.is_added_for_learning = ("IsForLearning" in list_of_enabled_datalines_flags_IsFor) or split_lines_in_learning_and_evaluation
                new_dlo.is_added_for_solving = ("IsForSolving" in list_of_enabled_datalines_flags_IsFor)
                new_dlo.save()

        if ENABLE_LOGGER_DEBUG_Machine:
            logger.debug(f"machine data input lines was appended. First_new_row_id:{first_new_row_id} with arguments:{kwargs}")


    def data_input_lines_mark(
            self,
            list_with_ids: pd.Index = None,
            **kwargs,
    ) -> NoReturn:
        """
        this method will mark lines at db_data_input_lines
        :params **kwargs: Should have one of ['IsSolved', 'IsLearned'] which true
        :return: None
        """

        # Validate for input parameters MACHINES_DATALINES_INPUT_RESERVED_FIELDS
        enabled_mode = [
            mode
            for mode, mode_value in kwargs.items()
            if mode in MACHINES_DATALINES_INPUT_RESERVED_FIELDS
        ]
        if len(enabled_mode) != 1:
            logger.error(f"only one column_mode {MACHINES_DATALINES_INPUT_RESERVED_FIELDS} can be true")

        elif list_with_ids is None:
            _, where_clause = generate_sql_fields_and_where_clause(
                fields=[],
                table_name=self.db_data_input_lines._meta.db_table,
            )
            self.db_data_input_lines.mark_lines(
                enabled_mode[0], kwargs[enabled_mode[0]], where_clause
            )
        else:
            self.db_data_input_lines.mark_lines(enabled_mode[0], kwargs[enabled_mode[0]], list_with_ids=list_with_ids)

        if ENABLE_LOGGER_DEBUG_Machine: logger.debug( f"machine data input lines was marked with kwargs {kwargs}" )


    def data_input_lines_mark_all_IsForLearning_as_IsLearned(self) -> NoReturn:
        """
        this method will mark lines at db_data_input_lines where IsForLearning=True as IsLearned=True

        :return: None
        """
        self.db_data_input_lines.mark_lines(
            field_name_to_set="IsLearned",
            field_value_to_set="True",
            where_clause=f" WHERE {self.db_data_input_lines._meta.db_table}.IsForLearning = True "
        )

        if ENABLE_LOGGER_DEBUG_Machine:
            logger.debug(f"Done")


    def data_input_lines_update(self, dataframe: pd.DataFrame) -> NoReturn:
        """
        this method will update lines in db_data_input_lines

        :params dataframe: pandas dataframe which have index.name == Line_ID
        :return: None
        """
        # Validate for Line_ID in dataframe
        if dataframe.index.name != "Line_ID":
            raise NameError("Line_ID index does not exist in dataframe")

        # Check if all columns in dataframe exist in mdc_columns_name_input_user_df
        input_columns_list = self.get_list_of_columns_name(ColumnDirectionType.INPUT)
        self._data_lines_update_in_this_table(input_columns_list, dataframe, self.db_data_input_lines)
        if ENABLE_LOGGER_DEBUG_Machine: logger.debug( f"machine data input lines was updated" )


    def data_input_lines_count(
            self,
            **kwargs,
    ) -> int:
        """
        count the rows depending of the arguments filters

        :return: how many rows found
        """
        return self.db_data_input_lines.objects.filter( **kwargs ).count()



    def data_output_lines_read(
            self,
            list_with_ids: pd.Index = None,
            sort_by: str = "",
            rows_count_limit: Optional[int ] = None,
    ) -> pd.DataFrame:
        """
        this method will get the pandas dataframe from db_data_output_lines
        :return: pandas dataframe
        """
        return self.db_machine.read_data_lines_from_db(
            output_columns=self.get_list_of_columns_name(ColumnDirectionType.OUTPUT),
            sort_by=sort_by,
            rows_count_limit=rows_count_limit
        )


    def data_output_lines_append(
                self,
                dataframe: pd.DataFrame ,
                already_locked_skip_lock: bool = False,
                ) -> NoReturn:
        """
        this method will append lines to db_data_output_lines
        :params dataframe: pandas dataframe with no index name LineInput_ID
        :params skip_do_lock: normally we lock the table during the operation , so there is no problems if another process do a data_line_append (working with input and output succesively)
        """

        # Check Line_ID exist in dataframe
        if dataframe.index.name != "Line_ID":
            raise NameError("dataframe not have index Line_ID")

        output_columns_list = self.get_list_of_columns_name(ColumnDirectionType.OUTPUT)

        # Check if dataframe columns satisfy the columns in Machine_<ID>_DataOutputLines table
        if not set(dataframe.columns) == set(output_columns_list):
            raise AttributeError("the columns of DataFrame to add is not the same as in Machine_DataOutputTable")

        try:
            # add the dataframe in the table with generated Line_ID as consecutive numbers in the table
            if already_locked_skip_lock:
                self.db_data_output_lines.append_dataframe(dataframe)
            else:
                with DoMachineLockTables( [self.db_data_output_lines._meta.db_table]):
                    self.db_data_output_lines.append_dataframe(dataframe)
        except Exception as e:
            logger.error( f"Unable to write data in {self.db_data_output_lines} in {self} because {e}" )

        if ENABLE_LOGGER_DEBUG_Machine: logger.debug( f"machine data output lines was appended" )


    @staticmethod
    def data_output_lines_mark() -> NoReturn:
        """
        for now there is no flags in outputs table
        """
        logger.error("THIS METHOD cannot be used because there is no flag in outputs table")


    def data_output_lines_update(self, dataframe: pd.DataFrame) -> NoReturn:
        """
        this method will update lines in db_data_output_lines

        :params dataframe: pandas dataframe must have index.name == Line_ID
        :return: None
        """

        # Validate for Line_ID in dataframe
        if dataframe.index.name != "Line_ID":
            raise NameError("Line_ID does not exist in dataframe")

        self._data_lines_update_in_this_table(
            self.get_list_of_columns_name(ColumnDirectionType.OUTPUT),
            dataframe, self.db_data_output_lines)

        if ENABLE_LOGGER_DEBUG_Machine: logger.debug( f"machine data input lines was updated" )


    def data_output_lines_count(
            self,
            **kwargs,
    ) -> int:
        """
        count the rows depending on the arguments filters

        :return: how many rows found
        """
        return self.db_data_output_lines.objects.filter( **kwargs ).count()


    def delete(self) -> NoReturn:
        if ENABLE_LOGGER_DEBUG_Machine:
            logger.debug(f"Deleting {self} starting")
        try:
            self.db_machine.delete()
        except Exception as e:
            logger.error( f"Unable to delete main record of machine {self} because {e}" )
        try:
            self.db_data_input_lines.delete_table()
            self.db_data_output_lines.delete_table()
        except Exception as e:
            logger.error( f"Unable to delete data table of machine {self} because {e}" )


    def copy(self, new_user: user_model) -> "Machine":
        """
        Duplicate a machine and give it to a new user

        :params: new_user the owner of the new machine

        :return: the new machine AI instance (already saved)
        """

        fields_to_clone = ['enc_dec_configuration', 'training_nn_model']

        _new_machine_model = machine_model.objects.get(pk=self.db_machine.id)

        _new_machine_model.pk = None
        _new_machine_model.machine_name = "Copy of " + self.db_machine.machine_name
        _new_machine_model.machine_original = self.db_machine.machine_original or self.db_machine
        _new_machine_model.machine_owner_user = new_user
        _new_machine_model.machine_owner_team = None

        for field_to_clone in fields_to_clone:
            setattr(_new_machine_model, field_to_clone, None)

        _new_machine_model.save()

        # load the new machine created
        new_ai_machine = Machine( _new_machine_model.id , machine_access_check_with_user_id=new_user )

        # load the dataframe of original machine and store it into the machine copy
        dataframe_machine_source = self.data_lines_read()
        dataframe_machine_source.reset_index(drop=True, inplace=True)
        new_ai_machine.data_lines_append(
                    user_dataframe_to_append=dataframe_machine_source,
                    split_lines_in_learning_and_evaluation=True,
                    )
        new_ai_machine.save_machine_to_db()

        return new_ai_machine


    def save_machine_to_db(self) -> NoReturn:
        if ENABLE_LOGGER_DEBUG_Machine: logger.debug( f"Saving {self} on disk" )

        # Check if any value is np.float32 type in the django model (django do not support this)
        for model_field_name, model_field_value in self.db_machine.__dict__.items():
            if isinstance(model_field_value, dict):
                for value_in_dict in model_field_value.values():
                    if isinstance(value_in_dict, np.float32):
                        logger.error(f"Unsupported (django) type np.float32 in machine field {model_field_name}")
                        _updated_value = {
                            key: str(value) for key, value in getattr(self.db_machine, model_field_name).items()
                        }
                        setattr(self.db_machine, model_field_name, _updated_value)
                        break
        self.db_machine.save()


    @staticmethod
    def is_this_machine_exist_and_authorized(
            machine_identifier_or_name,
            machine_check_access_user_id: int = None,
    )->bool:
        """
        check if the machine exist and is available for the user specified

        :params machine_identifier_or_name: the name of the machine or the id of the machine
        :params machine_access_user_id: the user id which want to access the machine
        :return: True if this user can access this machine
        """
        _init_models()
        if isinstance(machine_identifier_or_name, int) and isinstance(machine_check_access_user_id, int ):
            # FIXME TODO search a machine by name ONLY from the list of machine owned by the machine_access_user_id or if machine_access_user_id belong in the machine.team
            try:
                db_machine = machine_model.objects.get(id=machine_identifier_or_name , machine_owner_user__id=machine_check_access_user_id )
            except (machine_model.DoesNotExist, MultipleObjectsReturned):
                return False
            else:
                return True

        elif isinstance(machine_identifier_or_name, str) and isinstance(machine_check_access_user_id, int ):
            # FIXME TODO search a machine by name ONLY from the list of machine owned by the machine_access_user_id or if machine_access_user_id belong in the machine.team
            try:
                db_machine = machine_model.objects.get(machine_name=machine_identifier_or_name, machine_owner_user__id=machine_check_access_user_id )
            except (machine_model.DoesNotExist, MultipleObjectsReturned):
                return False
            else:
                return True

        else:
            raise Exception( "Combination of method arguments not valid")


    def store_error(self, column_name: str, error_message: str ) -> NoReturn:
        if column_name not in self.db_machine.machine_columns_errors:
            self.db_machine.machine_columns_errors[column_name] = error_message
        else:
            if len( str( self.db_machine.machine_columns_warnings ) ) > 50000:
                return
            self.db_machine.machine_columns_errors[column_name] += "\n\n" + error_message

        if ENABLE_LOGGER_DEBUG_Machine: logger.debug( f"Added error for '{column_name}' column with message: '{error_message}'" )


    def store_warning(self, column_name: str, warning_message: str ) -> NoReturn:
        if column_name not in self.db_machine.machine_columns_warnings:
            self.db_machine.machine_columns_warnings[column_name] = warning_message
        else:
            # we limit because some errors on large dataset can be too huge
            if len( str( self.db_machine.machine_columns_warnings ) ) > 55000:
                return
            elif len( str( self.db_machine.machine_columns_warnings ) ) > 54000:
                self.db_machine.machine_columns_warnings[column_name] += "."
            else:
                self.db_machine.machine_columns_warnings[column_name] += ("\n\n" + warning_message)

        if ENABLE_LOGGER_DEBUG_Machine:
            logger.debug(f"Added warning for '{column_name}' column with message: '{warning_message}'")


    def get_machine_overview_information(
            self,
            with_base_info: bool = False,
            with_fec_encdec_info: bool = False,
            with_nn_model_info: bool = False,
            with_training_infos: bool = False,
            with_training_cycle_result: bool = False,
            with_training_eval_result: bool = False,
    ) -> dict:
        """
        give a dict keys:string with all information about the machine_source - the dict keys will always be the same
        we use this dict as a context for training NN about a machine_source and for determining the FET to enable
        we do not put any information related to the NNModel because we use this machine_overview_information
        to predict what NNModel is best

        :params with_nn_model_info: When this dict will not be used by solutionfinder to find the model, we can include NNModel informations
        :return:
        """

        machine_overview_information = {}

        if with_base_info:
            machine_overview_information.update( {
                    "machine_name": self.db_machine.machine_name,
                    "machine_description": self.db_machine.machine_description,
                    "machine_level": self.db_machine.machine_level,
                    "mdc_columns_input_count": self.db_machine.mdc_columns_input_count,
                    "mdc_columns_output_count": self.db_machine.mdc_columns_output_count,
                    "mdc_columns_total_count": self.db_machine.mdc_columns_total_count,
                    "machine_data_lines_training_count_limit": self.db_machine.machine_data_lines_training_count_limit,
                    "machine_is_data_lines_training_count_limit_enabled": self.db_machine.machine_is_data_lines_training_count_limit_enabled,
                } )
            # we are storing datatypes count, but we need to have all empty datatypes because the get_machine_overview_information must return always the same dict keys
            _mdc_columns_count_of_datatypes = {}
            _mdc_columns_input_count_of_datatypes = {}
            _mdc_columns_output_count_of_datatypes = {}
            for one_datatype in DatasetColumnDataType:
                _mdc_columns_count_of_datatypes["mdc_columns_count_of_datatypes_" + one_datatype.name ] = 0
                _mdc_columns_input_count_of_datatypes["mdc_columns_input_count_of_datatypes_" + one_datatype.name ] = 0
                _mdc_columns_output_count_of_datatypes["mdc_columns_output_count_of_datatypes_" + one_datatype.name ] = 0
            for k, v in self.db_machine.mdc_columns_count_of_datatypes.items():
                _mdc_columns_count_of_datatypes["mdc_columns_count_of_datatypes_" + k] = v
            machine_overview_information.update(_mdc_columns_count_of_datatypes)
            for k, v in self.db_machine.mdc_columns_input_count_of_datatypes.items():
                _mdc_columns_input_count_of_datatypes["mdc_columns_input_count_of_datatypes_" + k] = v
            machine_overview_information.update(_mdc_columns_input_count_of_datatypes)
            for k, v in self.db_machine.mdc_columns_output_count_of_datatypes.items():
                _mdc_columns_output_count_of_datatypes["mdc_columns_output_count_of_datatypes_" + k] = v
            machine_overview_information.update(_mdc_columns_output_count_of_datatypes)

        if with_fec_encdec_info:
            machine_overview_information.update( self.db_machine.fe_count_of_fet )
            machine_overview_information.update( {
                    "fe_budget_total": int(self.db_machine.fe_budget_total),
                    "fe_columns_inputs_importance_evaluation_mean": np.mean( list(self.db_machine.fe_columns_inputs_importance_evaluation.values() )),
                    "fe_columns_inputs_importance_evaluation_std_dev": np.std( list(self.db_machine.fe_columns_inputs_importance_evaluation.values() )),
                    "enc_dec_columns_info_input_encode_count": self.db_machine.enc_dec_columns_info_input_encode_count,
                    "enc_dec_columns_info_output_encode_count": self.db_machine.enc_dec_columns_info_output_encode_count,
                } )

        if with_nn_model_info:
            # all information related to the NNModel configuration is optional
            from ML.NNConfiguration import NNShape
            this_neurons_total_count = NNShape(self.db_machine.parameter_nn_shape).neurons_total_count(
                self.db_machine.enc_dec_columns_info_input_encode_count,
                self.db_machine.enc_dec_columns_info_output_encode_count,
            )
            this_neurons_weight_total_count = NNShape(self.db_machine.parameter_nn_shape).weight_total_count(
                self.db_machine.enc_dec_columns_info_input_encode_count,
                self.db_machine.enc_dec_columns_info_output_encode_count,
            )
            machine_overview_information.update( {
                    "parameter_nn_loss": self.db_machine.parameter_nn_loss,
                    "parameter_nn_loss_scaler": self.db_machine.parameter_nn_loss_scaler,
                    "parameter_nn_optimizer": self.db_machine.parameter_nn_optimizer,
                    "neurons_total_count": this_neurons_total_count,
                    "neurons_weight_total_count": this_neurons_weight_total_count,
                } )
            machine_overview_information.update(self.db_machine.parameter_nn_shape)

        if with_training_infos:
            # all information related to the training result is optional
            machine_overview_information.update( {
                "training_date_time_machine_model": self.db_machine.training_date_time_machine_model,
                "training_type_machine_hardware": self.db_machine.training_type_machine_hardware,
                "training_total_training_line_count": self.db_machine.training_total_training_line_count,
                "training_training_batch_size": self.db_machine.training_training_batch_size,
                } )

        if with_training_cycle_result:
            # all information related to the training cycle result
            machine_overview_information.update( {
                    "training_training_epoch_count": self.db_machine.training_training_epoch_count,
                    "training_training_cell_delay_sec": self.db_machine.training_training_cell_delay_sec,
                    "training_training_total_delay_sec": self.db_machine.training_training_total_delay_sec,
                } )

        if with_training_eval_result:
            # all information related to the training evaluation
            machine_overview_information.update( {
                    "training_eval_loss_sample_training": self.db_machine.training_eval_loss_sample_training,
                    "training_eval_loss_sample_evaluation": self.db_machine.training_eval_loss_sample_evaluation,
                    "training_eval_loss_sample_training_noise": self.db_machine.training_eval_loss_sample_training_noise,
                    "training_eval_loss_sample_evaluation_noise": self.db_machine.training_eval_loss_sample_evaluation_noise,
                    "training_eval_accuracy_sample_training": self.db_machine.training_eval_accuracy_sample_training,
                    "training_eval_accuracy_sample_evaluation": self.db_machine.training_eval_accuracy_sample_evaluation,
                    "training_eval_accuracy_sample_training_noise": self.db_machine.training_eval_accuracy_sample_training_noise,
                    "training_eval_accuracy_sample_evaluation_noise": self.db_machine.training_eval_accuracy_sample_evaluation_noise,
                } )

        return machine_overview_information


    def get_random_user_dataframe_for_training_trial(
            self,
            is_for_learning: bool = False,
            is_for_evaluation: bool = False,
            force_rows_count: int = None,
            force_row_count_same_as_for_evaluation: bool = False,
            only_column_direction_type: Optional[ColumnDirectionType] = None,
    ) -> NoReturn:
        """
        Provide a random set of row from the training rows

        :params force_rows_count: define how many row to return
        :params force_row_count_same_as_for_evaluation: will return same count of rows as for evaluation dataset
        :params seed: to generate same random dataset
        :params with_reserved_columns: include the columns loss_prediction
        :params only_column_direction_type: if specified will include only inputs, or outputs columns
        :params **kwargs: can be None or column_mode or status for lines
        :return: user_dataframe(s)
        """

        _input_columns = []
        _output_columns = []
        _where_clause_dict = {}
        machine_level = MachineLevel( self )

        if force_row_count_same_as_for_evaluation:
            if force_rows_count:
                logger.error( "It is not possible to define force_rows_count when force_row_count_same_as_for_evaluation")
            force_rows_count = round(machine_level.evaluation_lines_count()[1])

        if is_for_evaluation:
            _where_clause_dict["IsForEvaluation"] = "=1"
            force_rows_count = round(machine_level.evaluation_lines_count()[1])

        if is_for_learning:
            _where_clause_dict["IsForLearning"] = "=1"
            if not force_rows_count:
                force_rows_count = round(machine_level.training_trial_lines_count()[1])

        if not only_column_direction_type:
            _input_columns = []
            _output_columns = []
        elif only_column_direction_type == ColumnDirectionType.OUTPUT:
            _input_columns = []
            _output_columns = self.get_list_of_columns_name(ColumnDirectionType.OUTPUT)
        elif only_column_direction_type == ColumnDirectionType.INPUT:
            _input_columns = self.get_list_of_columns_name(ColumnDirectionType.INPUT)
            _output_columns = []

        if DEBUG_TRAINING_ROWS_COUNT_LIMIT and force_rows_count:
            force_rows_count = min( force_rows_count , DEBUG_TRAINING_ROWS_COUNT_LIMIT )

        dataframe = self.db_machine.read_data_lines_from_db(
            input_columns=_input_columns,
            output_columns=_output_columns,
            where_clause_dict=_where_clause_dict,
            rows_count_limit=force_rows_count,
            is_random=True
        )

        if ENABLE_LOGGER_DEBUG_Machine:
            logger.debug(f"Dataframe readen with IsForLearning:{is_for_learning}, IsForEvaluation:{is_for_evaluation}, only_column_direction_type:{only_column_direction_type},... had {len(dataframe)} rows")

        return dataframe


    def user_dataframe_format_then_save_in_db(
            self,
            dataframe_to_append: pd.DataFrame,
            columns_type: dict,
            decimal_separator: Literal[".", ","],
            date_format: Literal["DMY", "MDY", "YMD"],
            split_lines_in_learning_and_evaluation: bool = False,
            **kwargs
    ) -> NoReturn:
        """
        format all cells of dataframe according to the columns_type
        then will append lines to database
        we can append inputs or inputs+outputs

        :params dataframe_to_append: pandas dataframe with NO index name LineInput_ID
        :params columns_type: dict of datatypes of the columns
        :params decimal_separator: the decimal separator for converting str to float
        :params date_format: the date format DMY, MDY, or YMD to convert str to date
        :params IsForLearning_and_IsForEvaluation: when we want to create lines for learning and a percentage for evaluation we need this flag
        :params **kwargs: Should have one of [IsForLearning/IsForSolving/IsForEvaluation] which true
        :return: None
        """
        # Validate decimal_separator
        if decimal_separator not in DECIMAL_SEPARATOR_CHOICES:
            logger.error(f"Invalid decimal_separator: '{decimal_separator}'. Must be one of {DECIMAL_SEPARATOR_CHOICES}")
        
        # Validate date_format
        if date_format not in DATE_FORMAT_CHOICES:
            logger.error(f"Invalid date_format: '{date_format}'. Must be one of {DATE_FORMAT_CHOICES}")

        # we can save IGNORED columns
        #ignored_columns = self.get_list_of_columns_name( ColumnDirectionType.IGNORED )
        #dataframe_to_append.drop( columns=ignored_columns, inplace=True )

        from ML import DataFileReader
        dataframe_to_append_formatted  = DataFileReader.reformat_pandas_cells_by_columns_datatypes(
                    dataframe_to_append,
                    columns_type,
                    decimal_separator = decimal_separator,
                    date_format = date_format,
        )
        self.data_lines_append(
                    user_dataframe_to_append=dataframe_to_append_formatted ,
                    split_lines_in_learning_and_evaluation=split_lines_in_learning_and_evaluation ,
                    **kwargs  )


    def get_all_encoded_for_ai_columns_names_by_the_pre_encoded_column_name(self, pre_encoded_column_name: str) -> list:
        """
        Returns a set of the encoded columns name created from the pre-encoded column
        """
        enc_dec_configuration = self.db_machine.enc_dec_configuration_extfield
        return [
            encoded_column
            for transform_parameter in enc_dec_configuration[pre_encoded_column_name]["fet_list"]
            for encoded_column in transform_parameter["list_encoded_columns_name"]
        ]


    def get_list_of_columns_name(
            self,
            column_mode: Union[ColumnDirectionType, str],
            dataframe_status: [DataframeEncodingType] = DataframeEncodingType.USER,
            # todo set default to none (and change all calling functions)
    ) -> list:
        """
        return the list of column (INPUT, OUTPUT, IGNORE) for a dataframe_status ( USER, PRE_ENCODED, ENCODED_FOR_AI )
        :param column_mode: the direction type : (INPUT, OUTPUT, IGNORE)
        :param dataframe_status: the type of dataframe : ( USER, PRE_ENCODED, ENCODED_FOR_AI )
        """

        if isinstance(column_mode, str):
            logger.warning( "Deprecated : get_list_of_columns_name prefer ColumnDirectionType rather than str !")
            column_mode = ColumnDirectionType(column_mode.lower())
        elif not isinstance(column_mode, ColumnDirectionType):
            logger.error("Argument Column_mode do not have the right type")

        if not isinstance(dataframe_status, DataframeEncodingType):
            logger.error("Argument Dataframe status have not the right type")

        if column_mode == ColumnDirectionType.INPUT and dataframe_status == DataframeEncodingType.USER:
            columns_dict = self.db_machine.mdc_columns_name_input_user_df

        elif column_mode == ColumnDirectionType.OUTPUT and dataframe_status == DataframeEncodingType.USER:
            columns_dict = self.db_machine.mdc_columns_name_output_user_df

        elif column_mode == ColumnDirectionType.IGNORED and dataframe_status == DataframeEncodingType.USER:
            columns_name = self.db_machine.mdc_columns_name_input_user_df.keys()
            columns_input_user_df = self.db_machine.mdc_columns_name_input_user_df.values()
            columns_output_user_df = self.db_machine.mdc_columns_name_output_user_df.values()

            columns_ignored = map(
                lambda is_input, is_output: not is_input and not is_output,
                columns_input_user_df,
                columns_output_user_df,
            )

            columns_dict = dict(zip(columns_name, columns_ignored))

        elif column_mode == ColumnDirectionType.INPUT and dataframe_status == DataframeEncodingType.PRE_ENCODED:
            columns_dict = self.db_machine.mdc_columns_name_input

        elif column_mode == ColumnDirectionType.OUTPUT and dataframe_status == DataframeEncodingType.PRE_ENCODED:
            columns_dict = self.db_machine.mdc_columns_name_output

        elif column_mode == ColumnDirectionType.IGNORED and dataframe_status == DataframeEncodingType.PRE_ENCODED:
            columns_name = self.db_machine.mdc_columns_name_input.keys()
            columns_input_user_df = self.db_machine.mdc_columns_name_input.values()
            columns_output_user_df = self.db_machine.mdc_columns_name_output.values()

            columns_ignored = map(
                lambda is_input, is_output: not is_input and not is_output,
                columns_input_user_df,
                columns_output_user_df,
            )

            columns_dict = dict(zip(columns_name, columns_ignored))

        elif dataframe_status == DataframeEncodingType.ENCODED_FOR_AI:
            columns_dict = dict()
            for pre_encoded_column_name in self.get_list_of_columns_name(column_mode, DataframeEncodingType.PRE_ENCODED):
                columns_dict.update(
                    dict.fromkeys(
                        self.get_all_encoded_for_ai_columns_names_by_the_pre_encoded_column_name(pre_encoded_column_name),
                        True
                    )
                )

        else:
            logger.error("Mode or dataframe status is wrong")

        return [column for column, column_value in columns_dict.items() if column_value]


    def is_config_ready_mdc(self) -> bool:
        return True if (
            self.db_machine.mdc_columns_data_type
        ) else False


    def is_config_ready_ici(self ) -> bool:
        return True if (
            self.db_machine.fe_columns_inputs_importance_evaluation
        ) else False


    def is_config_ready_fe(self) -> bool:
        return True if (
            self.db_machine.fe_columns_fet
        ) else False


    def is_config_ready_enc_dec(self) -> bool:
        return True if (
            self.db_machine.enc_dec_configuration_extfield is not None
        ) else False


    def is_config_ready_nn_configuration(self) -> bool:
        return True if (
            self.db_machine.parameter_nn_loss is not None and
            self.db_machine.parameter_nn_optimizer is not None and
            self.db_machine.parameter_nn_shape is not None and
            self.db_machine.parameter_nn_shape
        ) else False


    def is_config_ready_nn_model(self) -> bool:
        """
        indicate if the machine is ready for solving , when it have a model stored

        :return: True if the machine is ready to solve , otherwise False
        """
        return True if (
            self.db_machine.training_nn_model_extfield is not None
        ) else False


    def is_nn_training_pending(self) -> bool:
        """
        indicate if the machine will be trained again (or for the first time)

        :return: true if the machine will be trained again (or for the first time)
        """
        return (
                self.db_machine.machine_is_re_run_mdc or
                self.db_machine.machine_is_re_run_fe or
                self.db_machine.machine_is_re_run_ici or
                self.db_machine.machine_is_re_run_enc_dec or
                self.db_machine.machine_is_re_run_nn_config or
                self.db_machine.machine_is_re_run_model
                )


    def is_nn_solving_ready(self ) -> bool:
        """
        the NN is ready to solve when all configuration are present
        NOTE : there is a copy in WWW/machine/services/helpers.py

        :return: True if all configuration are present
        """
        return (
                    self.is_config_ready_mdc( ) and
                    self.is_config_ready_ici( ) and
                    self.is_config_ready_fe( ) and
                    self.is_config_ready_enc_dec( ) and
                    self.is_config_ready_nn_configuration( ) and
                    self.is_config_ready_nn_model( )
                )


    def clear_config_mdc( self ):
        """
        erase the configuration
        """
        self.db_machine.mdc_columns_data_type = None


    def clear_config_ici( self ):
        """
        erase the configuration
        """
        self.db_machine.fe_columns_inputs_importance_evaluation = None


    def clear_config_fe( self ):
        """
        erase the configuration
        """
        self.db_machine.fe_columns_fet = None


    def clear_config_enc_dec( self ):
        """
        erase the configuration
        """
        del self.db_machine.enc_dec_configuration_extfield
        self.db_machine.parameter_nn_loss_scaler = None
        self.db_machine.parameter_nn_loss = None


    def clear_config_nn_configuration( self ):
        """
        erase the configuration
        """
        self.db_machine.parameter_nn_optimizer = None
        self.db_machine.parameter_nn_shape = None


    def clear_config_nn_model( self ):
        """
        erase the configuration
        """
        del self.db_machine.training_nn_model_extfield


    def is_accuracy_available(self) -> bool:
        """
        indicate if the accuracy evaluations are available, they are only if all outputs are LABEL   (need to test this)
        """
        # TODO
        return True


    def scale_loss_to_user_loss(self, loss_to_scale: float) -> float:
        """
        will return scaler loss for user - user will see a loss between 0 and 1 always
        when we display loss to user we rescale it using this value, so the Loss_User is always from 0 to 1
        parameter_nn_loss_scaler is the loss between 2 random dataset

        :params loss_to_scale: the loss to rescale
        :return: the loss rescaled with a 0<value<1
        """


        if self.db_machine.parameter_nn_loss_scaler:
            return float( min(1, max(0.00001, loss_to_scale * float(self.db_machine.parameter_nn_loss_scaler))) )
        else:
            logger.error("Cannot rescale the loss because the parameter_nn_loss_scaler is not yet defined !")


    def get_count_of_rows_per_isforflags( self ):
        db_machine_model = self.db_machine.get_machine_data_input_lines_model()

        count_of_rows_per_isforflags = {
            "IsForEvaluation": db_machine_model.objects.filter(IsForEvaluation=True).count(),
            "IsForSolving": db_machine_model.objects.filter(IsForSolving=True).count(),
            "IsForLearning": db_machine_model.objects.filter(IsForLearning=True).count(),
        }
        return count_of_rows_per_isforflags



# ==================================================================================
# ==================================================================================



def generate_sql_fields_and_where_clause(
        fields: list,
        table_name: str,
        list_with_ids: list = None,
        **kwargs,
):
    """
    generate a string for selected fields and making where clause
    """
    fields.insert(0, "Line_ID")  # This filed will be index in dataframe
    fields = [f"`{field}`" for field in fields]
    fields = ", ".join(fields)

    if list_with_ids:
        list_condition_for_where_clause = [
            f"`{table_name}`.`Line_ID` in "
            f"{list_with_ids if len(list_with_ids) > 1 else '(' + str(list_with_ids[0]) + ')'}"
        ]
    else:
        list_condition_for_where_clause = []

    # TODO: check, error with offset
    list_condition_for_where_clause += [
        f"`{table_name}`.`{key}` = {bool(value)}" for key, value in kwargs.items() if key and key != "offset" and key != "with_reserved_columns"
    ]

    if list_condition_for_where_clause:
        where_clause = "WHERE(" + " AND ".join(list_condition_for_where_clause) + ")"
    else:
        where_clause = ""

    return fields, where_clause


def check_equality_two_columns_lists(
        columns_list_dataframe: list, columns_list_to_check: list
) -> bool:
    # Check if we have missed columns in dataframe
    if len(columns_list_dataframe) > len(columns_list_to_check):
        logger.warning("DataFrame has columns which is not in dynamic_model. SOME COLUMNS will be IGNORED.")
    return all(column in columns_list_dataframe for column in columns_list_to_check)


def get_last_id_from_dynamic_model(dynamic_model) -> int:
    if dynamic_model.objects.count()==0:
        return 0
    elif dynamic_model.__name__.lower().endswith("DataInputLines".lower()):
        return dynamic_model.objects.latest("Line_ID").Line_ID
    elif dynamic_model.__name__.lower().endswith("DataOutputLines".lower()):
        return dynamic_model.objects.latest("Line_ID").Line_ID_id


def get_indexes_lines_for_solving(
        dataframe: pd.DataFrame, output_columns: list
) -> pd.Index:
    count_of_output_columns = len(output_columns)

    filter_for_solving = (
            dataframe.loc[:, output_columns].isnull().sum(axis="columns")
            == count_of_output_columns
    )

    index_for_solving = dataframe[filter_for_solving].index.values
    return index_for_solving





# ==================================================================================
# ==================================================================================


class MachineLevel:
    def __init__(self, machine_or_machine_level: Union[Machine, int]):
        if isinstance(machine_or_machine_level, int):
            if not 1 <= machine_or_machine_level <= 3:
                logger.error( f"Expected MachineLevel in 1..3 but received '{machine_or_machine_level}'")
            self.level = machine_or_machine_level
            self.machine = None
        elif isinstance(machine_or_machine_level, Machine):
            self.machine = machine_or_machine_level
            self.level = self.machine.db_machine.machine_level
        else:
            logger.error(f"machine level machine_or_machine_level type must be int or machine, not {type( machine_or_machine_level )}")


    def feature_engineering_budget(self) -> tuple:
        users_columns_count = self.machine.db_machine.mdc_columns_total_count if self.machine is not None else 0
        if self.level == 1:              return "Basic", max(50, users_columns_count * 5)
        elif self.level == 2:            return "Medium", max(100, users_columns_count * 7)
        elif self.level == 3:            return "Advanced", max(250, users_columns_count * 10)


    def evaluation_lines_count(self) -> tuple:
        enc_dec_columns_count = (
                            0 if self.machine is None or self.machine.db_machine is None or self.machine.db_machine.enc_dec_columns_info_input_encode_count is None or self.machine.db_machine.enc_dec_columns_info_output_encode_count is None else
                              self.machine.db_machine.enc_dec_columns_info_input_encode_count +self.machine.db_machine.enc_dec_columns_info_output_encode_count
                              )
        if self.level == 1:              return "Basic", max( 150 , enc_dec_columns_count*2 )
        elif self.level == 2:            return "Medium", max( 500 , enc_dec_columns_count*2 )
        elif self.level == 3:            return "Advanced", max( 1000 , enc_dec_columns_count*2 )


    def training_trial_lines_count(self) -> tuple:
        enc_dec_columns_count = (
                            0 if self.machine is None else
                              self.machine.db_machine.enc_dec_columns_info_input_encode_count +self.machine.db_machine.enc_dec_columns_info_output_encode_count
                              )
        if self.level == 1:              return "Basic", max( 1000 , enc_dec_columns_count*10 )
        elif self.level == 2:            return "Medium", max( 2500 , enc_dec_columns_count*10 )
        elif self.level == 3:            return "Advanced", max( 5000 , enc_dec_columns_count*10 )


    def nn_shape_count_of_layer_max(self ) -> tuple:
        if self.level == 1:              return "Basic", 5
        elif self.level == 2:            return "Medium", 6
        elif self.level == 3:            return "Advanced", 7


    def nn_shape_count_of_neurons_max(self ) -> tuple:
        if self.level == 1:               return "Basic", 1000
        elif self.level == 2:            return "Medium", 5000
        elif self.level == 3:            return "Advanced", 10000    # tested limit on pc : 20 000 neurons ( 92M weights )






import threading
# machine_table_lock.py
from django.db import transaction, IntegrityError
from django.utils import timezone
import time

class DoMachineLockTables:
    """
    Context manager to acquire and release locks on specified tables.

    :param tables_to_lock: A list of table names to lock for write operations.
    :type tables_to_lock: list
    """
    def __init__(self, tables_to_lock):
        """
        Initialize the DoMachineLockTables class with a list of tables to lock.

        :param tables_to_lock: A list of table names to lock for write operations.
        :type tables_to_lock: list
        :raises TypeError: If the 'tables_to_lock' argument is not a list.
        """
        # Check if the input is a list
        if not isinstance(tables_to_lock, list):
            raise TypeError("The 'tables_to_lock' argument must be a list")

        # Store the list of tables to be locked for writing
        self.tables_locked_write = tables_to_lock
        self.locks = []

    def __enter__(self):
        """
        Acquire locks for the specified tables.

        This method is called when entering the context of the 'with' statement.
        It tries to acquire a lock for each table within a timeout of 180 seconds.

        If the lock cannot be acquired within the timeout, a warning is printed
        and the lock is acquired anyway.

        :returns: The instance of the class.
        :rtype: DoMachineLockTables
        """
        start_time = time.time()
        for table in self.tables_locked_write:
            while True:
                try:
                    with transaction.atomic():
                        # Try to create a lock entry for the table
                        lock = machinetablelockwrite_model.objects.create(table_name=table)
                        self.locks.append(lock)
                        #print(f"Acquired lock for {table}")
                        break
                except IntegrityError:
                    # If the lock already exists, check the timeout
                    if time.time() - start_time > 180:
                        logger.warning(f"Warning: Unable to lock table  '{table}' after 180 seconds - The locking process must have stop without releasing the lock So we will proced and use the current lock and release it after")
                        lock = machinetablelockwrite_model.objects.get(table_name=table)
                        self.locks.append(lock)
                        break
                    #print(f"Waiting for lock on {table}")
                    time.sleep(5)  # Sleep for a short period before retrying

        # Return self to allow usage with the 'with' statement
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Release locks for the specified tables.

        This method is called when exiting the context of the 'with' statement.
        It releases the lock for each table.

        :param exc_type: The exception type if an exception occurred.
        :type exc_type: type
        :param exc_val: The exception instance if an exception occurred.
        :type exc_val: Exception
        :param exc_tb: The traceback object if an exception occurred.
        :type exc_tb: traceback
        """
        for lock in self.locks:
            #print(f"Releasing lock for {lock.table_name}")
            lock.delete()
