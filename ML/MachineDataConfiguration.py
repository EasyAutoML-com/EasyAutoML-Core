from typing import NoReturn, Union, Optional, List, Dict, Iterable
import re
import json
import copy
import numpy as np
import pandas as pd
import ast
import datetime
from dateutil import parser as dateutilparser

from ML import EasyAutoMLDBModels, Machine, __getlogger

from SharedConstants import *

logger = __getlogger()


JSON_COLUMN_NAME_SEPARATOR = "_"

# in mdc_columns_most_frequent_values_count we will count how many there is values in the columns most frequent than this percentage
# example : if values occuring 15% 10% 5% 1% 1% and parameter  is 5% then we will return 3
# this count will be used for  FETMultiplexerMostFrequentsValues to know the cost without having the data of the column
MOST_FREQUENTS_VALUES_PERCENTAGE_OCCURRENCE_TO_KEEP_MIN = 5


class MachineDataConfiguration:
    """
    MachineDataConfiguration will Analyze the data and description of the columns of the dataframe:
        - split the columns of the dataframe into input, output and ignored
        - calculate main statistical properties for columns (mean, std, skew, kurt, ...)
        - pre-encode dataframe (extend json column to columns with simpler types)
        - post-decode dataframe (collapse columns created during pre-encoding)
    """

    def __init__(
            self,
            machine:Machine,
            user_dataframe_for_create_cfg: Optional[pd.DataFrame ] = None,
            columns_type_user_df: Optional[dict] = None,
            columns_description_user_df: Optional[dict ] = None,
            force_create_with_this_inputs: Optional[dict] = None,
            force_create_with_this_outputs: Optional[dict] = None,
            force_update_configuration_with_this_dataset: Optional[pd.DataFrame] = None,
            decimal_separator: Optional[ str ] = None,
            date_format: Optional[ str ] = None,
    ):
        """
        Create a configuration if a user_dataframe_for_create_cfg is provided
        Else Load a configuration from machine

        :param machine: the machine to load configuration from
        :param user_dataframe_for_create_cfg: the dataframe to analyse to create the configuration

        :param force_update_configuration_with_this_dataset: if there is a machine and a user_dataframe_for_create_cfg and this argument is True ,, we will load configuration  and update json configuration and data stats => usefull when the machine have additional new data
        """
        self._machine = machine
        self.columns_errors = dict()
        self.columns_warnings = dict()

        self.columns_name_input_user_df = dict()
        self.columns_name_output_user_df = dict()
        self.columns_type_user_df = dict()

        self.columns_name_input = dict()
        self.columns_name_output = dict()

        self.columns_input_count = 0
        self.columns_output_count = 0
        self.columns_total_count = 0

        self.columns_data_type = dict()

        self.columns_description_user_df = dict( )

        self.columns_json_structure = dict()

        self.columns_unique_values_count = dict()
        self.columns_missing_percentage = dict()
        self.columns_most_frequent_values_count = dict()

        self.columns_values_mean = dict()
        self.columns_values_std_dev = dict()
        self.columns_values_skewness = dict()
        self.columns_values_kurtosis = dict()
        self.columns_values_quantile02  = dict()
        self.columns_values_quantile03  = dict()
        self.columns_values_quantile07  = dict()
        self.columns_values_quantile08  = dict()
        self.columns_values_sem  = dict()
        self.columns_values_median  = dict()
        self.columns_values_mode  = dict()
        self.columns_values_min = dict()
        self.columns_values_max = dict()

        self.columns_values_str_min = dict()
        self.columns_values_str_max = dict()

        self.columns_values_str_percent_uppercase = dict()
        self.columns_values_str_percent_lowercase = dict()
        self.columns_values_str_percent_digit = dict()
        self.columns_values_str_percent_punctuation = dict()
        self.columns_values_str_percent_operators = dict()
        self.columns_values_str_percent_underscore = dict()
        self.columns_values_str_percent_space = dict()

        self.columns_values_str_language_en = dict()
        self.columns_values_str_language_fr = dict()
        self.columns_values_str_language_de = dict()
        self.columns_values_str_language_it = dict()
        self.columns_values_str_language_es = dict()
        self.columns_values_str_language_pt = dict()
        self.columns_values_str_language_others = dict()
        self.columns_values_str_language_none = dict()

        self.columns_count_of_datatypes = dict()
        self.columns_input_count_of_datatypes = dict()
        self.columns_output_count_of_datatypes = dict()

        if not isinstance(machine, Machine):
            logger.error( f"MDC expected an instance of the 'machine' class, but received '{type( machine )}'" )

        if user_dataframe_for_create_cfg is not None:
            if force_update_configuration_with_this_dataset:
                logger.error("unable to use parameters 'force_update_configuration_with_this_dataset' while generating configuration, only for updating an existing machine  ")
            if not columns_type_user_df:
                logger.error("columns_type_user_df is required if creating new MDC with dataset")
            if not columns_description_user_df:
                logger.error("columns_description_user_df is required if creating new MDC with dataset")
            if not decimal_separator:
                logger.error("decimal_separator is required if creating new MDC with dataset")
            if not date_format:
                logger.error("date_format is required if creating new MDC with dataset")
            self._init_generate_configuration(
                user_formatted_dataset=user_dataframe_for_create_cfg,
                columns_type_user_df=columns_type_user_df,
                columns_description_user_df=columns_description_user_df,
                force_create_with_this_inputs=force_create_with_this_inputs,
                force_create_with_this_outputs=force_create_with_this_outputs,
                decimal_separator = decimal_separator,
                date_format = date_format,
            )

        else:
            if force_create_with_this_inputs is not None or force_create_with_this_outputs is not None:
                logger.error("unable to use this parameters while loading => force_create_with_this_inputs: force_create_with_this_outputs: ")
            self._init_load_configuration( )
            # additional update
            if force_update_configuration_with_this_dataset:
                if not decimal_separator:
                    logger.error("decimal_separator is required if creating new MDC with dataset")
                if not date_format:
                    logger.error("date_format is required if creating new MDC with dataset")
                self._update_configuration_json_and_data_stats(   force_update_configuration_with_this_dataset ,
                                                                                        decimal_separator ,
                                                                                        date_format )



    def _init_generate_configuration(self,
                        user_formatted_dataset: pd.DataFrame,
                        columns_type_user_df: dict,
                        columns_description_user_df: dict,
                        decimal_separator: str,
                        date_format: str,
                        force_create_with_this_inputs: Optional[dict] = None,
                        force_create_with_this_outputs: Optional[dict] = None,
                         ) -> NoReturn:
        """
        Create a new instance of MDC
        :params user_formatted_dataset: the dataset formatted (datatypes of all cells) by DFR
        :params columns_type_user_df: the columns types detected by DFR
        :params columns_description_user_df: the descriptions detected by DFR
        :params force_create_with_this_inputs: set input if needed or automatic detection will set
        :params force_create_with_this_outputs: set output if needed or automatic detection will set
        """

        if ENABLE_LOGGER_DEBUG_MachineDataConfiguration: logger.debug( f"Creation MachineDataConfiguration from user dataset" )

        self.columns_type_user_df = columns_type_user_df
        self.columns_description_user_df = columns_description_user_df

        # we copy the inputs and outputs found from the data_file_reader
        (
            self.columns_name_input_user_df,
            self.columns_name_output_user_df,
        ) = self._determine_columns_input_output_from_dataframe( user_formatted_dataset, columns_description_user_df )

        # force inputs/outputs if provided
        if force_create_with_this_inputs is not None:
            for this_column_name, is_input in force_create_with_this_inputs.items():
                if not this_column_name in self.columns_name_input_user_df:
                    logger.warning( f"force_create_with_this_inputs argument contains columns '{this_column_name}' not from columns_name_input_user_df (maybe called MDC() with a list of column expanded from json ? ) ")
                elif is_input:
                    self.columns_name_input_user_df[this_column_name] = is_input
                    self.columns_name_output_user_df[this_column_name] = not( is_input )
        if force_create_with_this_outputs is not None:
            for this_column_name, is_output in force_create_with_this_outputs.items():
                if not this_column_name in self.columns_name_input_user_df:
                    logger.warning( f"force_create_with_this_inputs argument contains columns '{this_column_name}' not from columns_name_input_user_df (maybe called MDC() with a list of column expanded from json ? ) ")
                elif is_output:
                    self.columns_name_output_user_df[this_column_name] = is_output
                    self.columns_name_input_user_df[this_column_name] = not is_output

        if any( input_enabled for input_enabled in self.columns_name_input_user_df.values() ):
            pass
        else:
            logger.error( "There is no inputs columns identified in the dataframe !")

        if any( output_enabled for output_enabled in self.columns_name_output_user_df.values() ):
            pass
        else:
            logger.error( "There is no output columns identified in the dataframe !")

        self.columns_missing_percentage = (
            self._get_columns_missing_percentage_from_dataframe( user_formatted_dataset )
        )

        # if the machine have more than 50 rows
        if user_formatted_dataset.shape[0] >= 50:
            # We add warning in the machine for the rows where is more than 75% of missing values
            for (
                column_name,
                columns_missing_percentage,
            ) in self.columns_missing_percentage.items():
                if columns_missing_percentage > 75:
                    self._add_warning_for_column(
                        column_name,
                        f"Warning there is  {columns_missing_percentage} % of values missing in '{column_name}' column",
                    )


        # STEP 1/3 : Compute the datatypes for each USER_DF columns
        for column_name, column_type in self.columns_type_user_df.items():
            if not isinstance(column_type, DatasetColumnDataType):
                logger.error(f"columns_type_user_df should contains only DatasetColumnDataType but is '{type( column_type )}' ")

        self.columns_data_type  = {
            column_name: column_type
            for column_name, column_type in self.columns_type_user_df.items()
            if column_type is not DatasetColumnDataType.JSON
        }

        self._update_configuration_json_and_data_stats(
            user_formatted_dataset,
            decimal_separator = decimal_separator,
            date_format = date_format,)

        return self


    def _reformat_all_pandas_cells_to_numeric_for_computing_stats_columns_values( self,
            dataframe_to_format: pd.DataFrame,
            columns_datatypes: dict,
            decimal_separator: str,
            date_format: str,
    ) -> pd.DataFrame:
        """
        to compute columns_values_*** we need to convert to numeric => date, time, datetime : to stamp , Label : to len(str) , text = 0

        :param dataframe_to_format: this is user_dataframe we will convert to numeric
        :param columns_datatypes: the datatype of the column of the dataframe to reformat
        :param decimal_separator: . or , only
        :param date_format: DMY or MDY only

        :return: dataframe with only float
        """

        def _reformat_numeric_to_float( v ) -> Optional[float]:
                if pd.isnull( v ):
                    return None
                elif isinstance( v, (bool, int, float , np.bool_ , np.int_ , np.float_ ) ):
                    return float( v )
                elif isinstance( v , str):
                    if v.strip() == "":
                        return None
                    if decimal_separator != ".":
                        return(  float( v.replace(decimal_separator, ".", 1)) )
                    else:
                        return float( v )
                else:
                    logger.error(f"Can not convert data type {type(v)} , v={v}  ")


        def _reformat_time_to_float( t ) -> Optional[float]:
                if pd.isnull( t ):
                    return None
                elif isinstance( t , np.datetime64 ):
                    return ( t - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
                elif isinstance( t , ( datetime.time, datetime.datetime ) ):
                    return t.second + 60 * t.minute + 3600 * t.hour
                elif isinstance( t , str):
                    return _reformat_time_to_float( dateutilparser.parse(str( t )).time() )
                else:
                    logger.error(f"Can not convert data type {type(t)} , v={t}  ")


        def _reformat_date_to_float(d) -> Optional[float]:
            if pd.isnull(d):
                return None
            elif isinstance(d, np.datetime64):
                return (d - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
            elif isinstance(d, pd.Timestamp):
                return (d.to_pydatetime() - datetime.datetime(1970, 1, 1)).total_seconds()
            elif isinstance(d, datetime.date):
                return (datetime.datetime.combine(d, datetime.datetime.min.time()) - datetime.datetime(1970, 1, 1)).total_seconds()
            elif isinstance(d, str):
                # convert str to datetime then to float
                return _reformat_date_to_float(dateutilparser.parse(d, dayfirst=(date_format == "DMY"), yearfirst=(date_format == "YMD")))
            else:
                logger.error(f"Can not convert data type {type(d)} , v={d}")


        def _reformat_datetime_to_float( d ) -> Optional[float]:
                if pd.isnull( d ):
                    return None
                elif isinstance( d, np.datetime64 ):
                    return ( d - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
                elif isinstance( d, datetime.datetime ):
                    return ( d - datetime.datetime.combine( datetime.date(1970,1,1) , datetime.datetime.min.time() ) ).total_seconds()
                elif isinstance( d, str):
                    # convert str to datetime then to float
                    return _reformat_datetime_to_float( dateutilparser.parse(str( d ), dayfirst=(date_format == "DMY"), yearfirst=(date_format == "YMD")).replace(tzinfo=None) )
                else:
                    logger.error(f"Can not convert data type {type(d)} , v={d}  ")


        def _reformat_str_to_float( l ) -> Optional[float]:
                if pd.isnull( l ):
                    return None
                elif isinstance( l, str ):
                    return len( l.strip() )
                else:
                    logger.error(f"Can not convert data type {type(l)} , v={l} ")

        # START ==== def _reformat_all_pandas_cells_to_numeric_for_computing_stats_columns_values( self,
        dataframe_formatted = dataframe_to_format.copy( )
        for this_column_name in dataframe_formatted.columns:
            if not this_column_name in columns_datatypes:
                logger.error( f"the column {this_column_name} present in df=({dataframe_formatted.columns}) have no datatype in columns_datatypes={columns_datatypes.columns} ")
            this_column_datatype = columns_datatypes[this_column_name]

            if this_column_datatype is DatasetColumnDataType.FLOAT:
                dataframe_formatted[this_column_name] = dataframe_formatted[this_column_name].apply(  lambda cell: _reformat_numeric_to_float( cell )   )
            elif this_column_datatype is DatasetColumnDataType.DATETIME:
                dataframe_formatted[this_column_name] = pd.Series( [ _reformat_datetime_to_float( value ) for value in dataframe_formatted[this_column_name] ], index=dataframe_formatted[this_column_name].index )
                #dataframe_formatted[this_column_name] = dataframe_formatted[this_column_name].apply(  lambda cell: _reformat_datetime_to_float( cell )   ) # note : timezone make some bug with apply  https://github.com/pandas-dev/pandas/issues/33876
            elif this_column_datatype is DatasetColumnDataType.DATE:
                dataframe_formatted[this_column_name] = dataframe_formatted[this_column_name].apply(  lambda cell: _reformat_date_to_float( cell )   )
            elif this_column_datatype is DatasetColumnDataType.TIME:
                dataframe_formatted[this_column_name] = dataframe_formatted[this_column_name].apply(  lambda cell: _reformat_time_to_float( cell  )   )
            elif this_column_datatype is DatasetColumnDataType.LANGUAGE:
                dataframe_formatted[this_column_name] = dataframe_formatted[this_column_name].apply(  lambda cell: _reformat_str_to_float( cell )   )
            elif this_column_datatype is DatasetColumnDataType.LABEL:
                dataframe_formatted[this_column_name] = dataframe_formatted[this_column_name].apply(  lambda cell: _reformat_str_to_float( cell )   )
            elif this_column_datatype is DatasetColumnDataType.IGNORE:
                del dataframe_formatted[this_column_name]
            else:
                logger.error(f"Cannot process unknown column datatype '{this_column_datatype}' of column '{ this_column_name }' ! ")

        return dataframe_formatted


    def _recalculate_data_infos_stats( self , df_pre_encoded: pd.DataFrame, decimal_separator , date_format ):
        """
        Calculate all infos infos about datas, safe to call anytime, do not update data structure, only update infos/stats about data
        """

        def _compute_percents_columns_values_str_signs( data_serie: pd.Series ) -> (float,float,float,float,float,float,float,float,float):
            """
            compute 8 percentages of the pandas serie of string given as argument

            :params data_serie: the pandas serie to make statistiques of
            :return: 8 float of percentage (0..1)
            """
            columns_values_str_count_total = 0
            columns_values_str_count_uppercase = 0
            columns_values_str_count_lowercase = 0
            columns_values_str_count_digit = 0
            columns_values_str_count_punctuation = 0
            columns_values_str_count_operators = 0
            columns_values_str_count_underscore = 0
            columns_values_str_count_space = 0

            for indx, s in data_serie.items():
                columns_values_str_count_total += len( s )
                columns_values_str_count_uppercase += sum(c.isupper() for c in s)
                columns_values_str_count_lowercase += sum(c.islower() for c in s)
                columns_values_str_count_digit += sum(c.isdigit() for c in s)
                columns_values_str_count_punctuation += sum( c in ",?!." for c in s)
                columns_values_str_count_operators += sum( c in "<>=" for c in s)
                columns_values_str_count_underscore += sum( c=="_" for c in s)
                columns_values_str_count_space += sum(c.isspace() for c in s)

            if columns_values_str_count_total==0:
                return 0 , 0 , 0 , 0 , 0 , 0 , 0
            else:
                return (
                    columns_values_str_count_uppercase / columns_values_str_count_total ,
                    columns_values_str_count_lowercase / columns_values_str_count_total ,
                    columns_values_str_count_digit / columns_values_str_count_total ,
                    columns_values_str_count_punctuation / columns_values_str_count_total ,
                    columns_values_str_count_operators / columns_values_str_count_total ,
                    columns_values_str_count_underscore / columns_values_str_count_total ,
                    columns_values_str_count_space / columns_values_str_count_total
            )


        # --------- Start _calculate_values_infos_stats---------
        self.columns_unique_values_count=self._get_unique_values_count(df_pre_encoded,self.columns_data_type)

        self.columns_most_frequent_values_count = (
            self._get_columns_most_frequent_values_count(
                df_pre_encoded, self.columns_data_type
            )
        )

        # compute missing_percentage for all columns
        self.columns_missing_percentage=(self._get_columns_missing_percentage_from_dataframe( df_pre_encoded ))

        # Calculate the main data distribution properties of each column of the dataframe entirely converted to float
        dataframe_converted_all_float = self._reformat_all_pandas_cells_to_numeric_for_computing_stats_columns_values( df_pre_encoded , self.columns_data_type, decimal_separator , date_format )
        self.columns_values_std_dev = dataframe_converted_all_float.std( axis=0, numeric_only=True ).dropna( ).astype(float ).to_dict( )
        self.columns_values_skewness = dataframe_converted_all_float.skew( axis=0, numeric_only=True ).dropna( ).astype(float).to_dict( )
        self.columns_values_kurtosis = dataframe_converted_all_float.kurt( axis=0, numeric_only=True ).dropna( ).astype(float).to_dict( )
        # Calculate stats properties of each column of the dataframe entirely converted to float
        self.columns_values_quantile02 = dataframe_converted_all_float.quantile( q=0.2 , axis=0, numeric_only=True ).dropna( ).astype(float).to_dict( )
        self.columns_values_quantile03 = dataframe_converted_all_float.quantile( q=0.3 , axis=0, numeric_only=True ).dropna( ).astype(float).to_dict( )
        self.columns_values_quantile07 = dataframe_converted_all_float.quantile( q=0.7 , axis=0, numeric_only=True ).dropna( ).astype(float).to_dict( )
        self.columns_values_quantile08 = dataframe_converted_all_float.quantile( q=0.8 , axis=0, numeric_only=True ).dropna( ).astype(float).to_dict( )
        self.columns_values_sem = dataframe_converted_all_float.sem( axis=0, numeric_only=True ).dropna( ).astype(float).to_dict( )
        self.columns_values_median = dataframe_converted_all_float.median( axis=0, numeric_only=True ).dropna( ).astype(float).to_dict( )
        self.columns_values_mode = {} if dataframe_converted_all_float.empty else dataframe_converted_all_float.mode( axis=0, numeric_only=True ).iloc[0].dropna( ).astype(float).to_dict( )
        self.columns_values_mean = dataframe_converted_all_float.mean( axis=0, numeric_only=True ).dropna( ).astype(float).to_dict( )
        self.columns_values_min = dataframe_converted_all_float.min( axis=0, numeric_only=True).dropna( ).astype(float).to_dict( )
        self.columns_values_max = dataframe_converted_all_float.max( axis=0, numeric_only=True).dropna( ).astype(float).to_dict( )

        for column_name in self.columns_data_type:

            # Calculate stats properties of string columns with a maximum of 500 random values
            if self.columns_data_type[column_name] == DatasetColumnDataType.LABEL or self.columns_data_type[column_name] == DatasetColumnDataType.LANGUAGE:
                column_data = df_pre_encoded[column_name].dropna( )
                if len( column_data ) <= 150:
                    column_data_samples = column_data
                else:
                    column_data_samples = column_data.sample( n = 150 )
                (
                    self.columns_values_str_percent_uppercase[column_name],
                    self.columns_values_str_percent_lowercase[column_name],
                    self.columns_values_str_percent_digit[column_name],
                    self.columns_values_str_percent_punctuation[column_name],
                    self.columns_values_str_percent_operators[column_name],
                    self.columns_values_str_percent_underscore[column_name],
                    self.columns_values_str_percent_space[column_name]
                ) = _compute_percents_columns_values_str_signs( column_data_samples )

                from ML import DataFileReader
                languages_detected = DataFileReader.detect_6_languages_percentage_from_serie( column_data_samples )  # to speed up we check only uniques values of begining of sentences
                self.columns_values_str_language_en[column_name] = languages_detected["en"]
                self.columns_values_str_language_fr[column_name] = languages_detected["fr"]
                self.columns_values_str_language_de[column_name] = languages_detected["de"]
                self.columns_values_str_language_it[column_name] = languages_detected["it"]
                self.columns_values_str_language_es[column_name] = languages_detected["es"]
                self.columns_values_str_language_pt[column_name] = languages_detected["pt"]
                self.columns_values_str_language_others[column_name] = languages_detected["others"]
                self.columns_values_str_language_none[column_name] = languages_detected["none"]
            else:
                self.columns_values_str_percent_uppercase[column_name] = None
                self.columns_values_str_percent_lowercase[column_name] = None
                self.columns_values_str_percent_digit[column_name] = None
                self.columns_values_str_percent_punctuation[column_name] = None
                self.columns_values_str_percent_operators[column_name] = None
                self.columns_values_str_percent_underscore[column_name] = None
                self.columns_values_str_percent_space[column_name] = None
                self.columns_values_str_language_en[column_name] = None
                self.columns_values_str_language_fr[column_name] = None
                self.columns_values_str_language_de[column_name] = None
                self.columns_values_str_language_it[column_name] = None
                self.columns_values_str_language_es[column_name] = None
                self.columns_values_str_language_pt[column_name] = None
                self.columns_values_str_language_others[column_name] = None
                self.columns_values_str_language_none[column_name] = None


    def _update_configuration_json_and_data_stats( self , dataframe_to_extend: pd.DataFrame, decimal_separator , date_format ):
        """
        Define structure of data and do _calculate_values_infos_stats
        """

        # ===== start ========== def _update_configuration_json_and_df_stats( self , dataframe_to_extend: pd.DataFrame, decimal_separator , date_format ):
        full_dataframe_with_json_expanded, self.columns_json_structure, list_of_expanded_json_column = self._get_full_dataframe_with_json_columns_expanded(
            dataframe_to_extend,
            json_columns_names_to_extend=set(dataframe_to_extend.columns ) - self.columns_data_type.keys( ),
        )

        # Add column types of the expanded JSON column
        self.columns_data_type.update(
            self._get_columns_type_from_dataframe(
                full_dataframe_with_json_expanded[
                    [
                        column_name
                        for column_name in full_dataframe_with_json_expanded.columns
                        if column_name not in self.columns_data_type
                    ]
                ],
                decimal_separator = ".",
                date_format = "YMD",
            )
        )

        # start from user_df configuration - Split the columns into input, output
        columns_name_input = self.columns_name_input_user_df.copy()
        columns_name_output = self.columns_name_output_user_df.copy()

        # columns DatasetColumnDataType.IGNORE will not be input or output
        for column_name, data_type in self.columns_data_type.items():
            if data_type == DatasetColumnDataType.IGNORE:
                columns_name_input[ column_name ] = False
                columns_name_output[ column_name ] = False

        # Set the new json extended columns (input/output) mode in columns_name_input and columns_name_output
        # when the machine is already created the input and output are already defined by force_create_with_this_inputs, force_create_with_this_outputs
        for column_name in self.columns_type_user_df:
            if self.columns_type_user_df[column_name] == DatasetColumnDataType.JSON:
                for extended_column_name in self.get_children_of_json_column( column_name ):
                    columns_name_input[extended_column_name] = self.columns_name_input_user_df[column_name]
                    columns_name_output[extended_column_name] = self.columns_name_output_user_df[column_name]

                # We need to remove the json column we have expanded from configuration
                del columns_name_input[column_name], columns_name_output[column_name]

        # define the new input/output configuration for the dataframe now pre-encoded with json expanded
        self.columns_name_input = columns_name_input
        self.columns_name_output = columns_name_output

        # compute missing_percentage for all columns
        self.columns_missing_percentage=(self._get_columns_missing_percentage_from_dataframe(full_dataframe_with_json_expanded))

        # if a column contains more than COLUMN_MAXIMUM_MISSING_PERCENTAGE_OR_IGNORE of missing value, then it will be ignored
        for column_name in self.columns_missing_percentage:
            if self.columns_missing_percentage[ column_name ] > COLUMN_MAXIMUM_MISSING_PERCENTAGE_FOR_IGNORE:
                self.columns_name_input[ column_name ] = False
                self.columns_name_output[ column_name ] = False
                self.columns_type_user_df[column_name] = DatasetColumnDataType.IGNORE

        # count only the columns with a true in the value
        self.columns_input_count = sum(self.columns_name_input.values( ) )
        # count only the columns with a true in the value
        self.columns_output_count = sum(self.columns_name_output.values( ) )
        self.columns_total_count = self.columns_input_count + self.columns_output_count
        self.columns_count_of_datatypes = self._get_count_of_datatypes_per_columns( )
        self.columns_input_count_of_datatypes = self._get_count_of_datatypes_per_columns( only_columns_with_direction=ColumnDirectionType.INPUT )
        self.columns_output_count_of_datatypes = self._get_count_of_datatypes_per_columns( only_columns_with_direction=ColumnDirectionType.OUTPUT )

        self._recalculate_data_infos_stats( full_dataframe_with_json_expanded , decimal_separator , date_format )


    def _init_load_configuration(self) -> NoReturn:
        """
        Load the MachineDataConfiguration object using an instance of the Machine class

        :param machine: an instance of the Machine class from which we will take data
        """

        if ENABLE_LOGGER_DEBUG_MachineDataConfiguration: logger.debug( f"Loading MachineDataConfiguration for {self._machine} starting" )

        db_machine = self._machine.db_machine

        self.columns_errors = db_machine.machine_columns_errors
        self.columns_warnings = db_machine.machine_columns_warnings

        self.columns_name_input_user_df = db_machine.mdc_columns_name_input_user_df
        self.columns_name_output_user_df = db_machine.mdc_columns_name_output_user_df
        self.columns_type_user_df = db_machine.dfr_columns_type_user_df
        self.columns_name_input = db_machine.mdc_columns_name_input
        self.columns_name_output = db_machine.mdc_columns_name_output

        self.columns_data_type = db_machine.mdc_columns_data_type
        # {
        #     column_name: column_type
        #     for column_name, column_type, in db_machine.mdc_columns_data_type.items()
        # }

        self.columns_description_user_df = db_machine.dfr_columns_description_user_df

        self.columns_json_structure = db_machine.mdc_columns_json_structure

        self.columns_unique_values_count = db_machine.mdc_columns_unique_values_count
        self.columns_missing_percentage = db_machine.mdc_columns_missing_percentage
        self.columns_most_frequent_values_count = db_machine.mdc_columns_most_frequent_values_count

        self.columns_values_min = db_machine.mdc_columns_values_min
        self.columns_values_max = db_machine.mdc_columns_values_max
        self.columns_values_std_dev = db_machine.mdc_columns_values_std_dev
        self.columns_values_skewness = db_machine.mdc_columns_values_skewness
        self.columns_values_kurtosis = db_machine.mdc_columns_values_kurtosis
        self.columns_values_mean = db_machine.mdc_columns_values_mean
        self.columns_values_quantile02  = db_machine.mdc_columns_values_quantile02
        self.columns_values_quantile03  = db_machine.mdc_columns_values_quantile03
        self.columns_values_quantile07  = db_machine.mdc_columns_values_quantile07
        self.columns_values_quantile08  = db_machine.mdc_columns_values_quantile08
        self.columns_values_sem  = db_machine.mdc_columns_values_sem
        self.columns_values_median  = db_machine.mdc_columns_values_median
        self.columns_values_mode  = db_machine.mdc_columns_values_mode

        self.columns_values_str_percent_uppercase = db_machine.mdc_columns_values_str_percent_uppercase
        self.columns_values_str_percent_lowercase = db_machine.mdc_columns_values_str_percent_lowercase
        self.columns_values_str_percent_digit = db_machine.mdc_columns_values_str_percent_digit
        self.columns_values_str_percent_punctuation = db_machine.mdc_columns_values_str_percent_punctuation
        self.columns_values_str_percent_operators = db_machine.mdc_columns_values_str_percent_operators
        self.columns_values_str_percent_underscore = db_machine.mdc_columns_values_str_percent_underscore
        self.columns_values_str_percent_space = db_machine.mdc_columns_values_str_percent_space

        self.columns_values_str_language_en  = db_machine.mdc_columns_values_str_language_en
        self.columns_values_str_language_fr  = db_machine.mdc_columns_values_str_language_fr
        self.columns_values_str_language_de  = db_machine.mdc_columns_values_str_language_de
        self.columns_values_str_language_it  = db_machine.mdc_columns_values_str_language_it
        self.columns_values_str_language_es  = db_machine.mdc_columns_values_str_language_es
        self.columns_values_str_language_pt  = db_machine.mdc_columns_values_str_language_pt
        self.columns_values_str_language_others  = db_machine.mdc_columns_values_str_language_others
        self.columns_values_str_language_none  = db_machine.mdc_columns_values_str_language_none

        self.columns_count_of_datatypes = db_machine.mdc_columns_count_of_datatypes
        self.columns_input_count_of_datatypes = db_machine.mdc_columns_input_count_of_datatypes
        self.columns_output_count_of_datatypes = db_machine.mdc_columns_output_count_of_datatypes

        self.columns_total_count = db_machine.mdc_columns_total_count
        self.columns_input_count = db_machine.mdc_columns_input_count
        self.columns_output_count = db_machine.mdc_columns_output_count


    def save_configuration_in_machine(self) -> "MachineDataConfiguration":
        """Stores MachineDataConfiguration attributes inside the input machine_source"""

        if not self.columns_data_type:
            if ENABLE_LOGGER_DEBUG_MachineDataConfiguration: logger.debug( f"MDC saving configuration minimum base !!!" )
        else:
            if ENABLE_LOGGER_DEBUG_MachineDataConfiguration: logger.debug( f"MDC saving configuration full" )

        db_machine = self._machine.db_machine

        db_machine.machine_columns_errors = self.columns_errors
        db_machine.machine_columns_warnings = self.columns_warnings

        db_machine.mdc_columns_description_user_df = self.columns_description_user_df

        db_machine.mdc_columns_name_input_user_df = self.columns_name_input_user_df
        db_machine.mdc_columns_name_output_user_df = self.columns_name_output_user_df

        db_machine.mdc_columns_name_input = self.columns_name_input
        db_machine.mdc_columns_name_output = self.columns_name_output

        db_machine.mdc_columns_input_count = self.columns_input_count
        db_machine.mdc_columns_output_count = self.columns_output_count
        db_machine.mdc_columns_total_count = self.columns_total_count

        db_machine.mdc_columns_data_type = self.columns_data_type

        db_machine.mdc_columns_json_structure = self.columns_json_structure

        db_machine.mdc_columns_unique_values_count = self.columns_unique_values_count
        db_machine.mdc_columns_missing_percentage = self.columns_missing_percentage
        db_machine.mdc_columns_most_frequent_values_count=self.columns_most_frequent_values_count

        db_machine.mdc_columns_values_min = self.columns_values_min
        db_machine.mdc_columns_values_max = self.columns_values_max

        db_machine.mdc_columns_values_mean = self.columns_values_mean
        db_machine.mdc_columns_values_std_dev = self.columns_values_std_dev
        db_machine.mdc_columns_values_skewness = self.columns_values_skewness
        db_machine.mdc_columns_values_kurtosis = self.columns_values_kurtosis

        db_machine.mdc_columns_values_quantile02 = self.columns_values_quantile02
        db_machine.mdc_columns_values_quantile03 = self.columns_values_quantile03
        db_machine.mdc_columns_values_quantile07 = self.columns_values_quantile07
        db_machine.mdc_columns_values_quantile08 = self.columns_values_quantile08
        db_machine.mdc_columns_values_sem = self.columns_values_sem
        db_machine.mdc_columns_values_median = self.columns_values_median
        db_machine.mdc_columns_values_mode = self.columns_values_mode

        db_machine.mdc_columns_values_str_percent_uppercase = self.columns_values_str_percent_uppercase
        db_machine.mdc_columns_values_str_percent_lowercase = self.columns_values_str_percent_lowercase
        db_machine.mdc_columns_values_str_percent_digit = self.columns_values_str_percent_digit
        db_machine.mdc_columns_values_str_percent_punctuation = self.columns_values_str_percent_punctuation
        db_machine.mdc_columns_values_str_percent_operators = self.columns_values_str_percent_operators
        db_machine.mdc_columns_values_str_percent_underscore = self.columns_values_str_percent_underscore
        db_machine.mdc_columns_values_str_percent_space = self.columns_values_str_percent_space

        db_machine.mdc_columns_values_str_language_en = self.columns_values_str_language_en
        db_machine.mdc_columns_values_str_language_fr = self.columns_values_str_language_fr
        db_machine.mdc_columns_values_str_language_de = self.columns_values_str_language_de
        db_machine.mdc_columns_values_str_language_it = self.columns_values_str_language_it
        db_machine.mdc_columns_values_str_language_es = self.columns_values_str_language_es
        db_machine.mdc_columns_values_str_language_pt = self.columns_values_str_language_pt
        db_machine.mdc_columns_values_str_language_others = self.columns_values_str_language_others
        db_machine.mdc_columns_values_str_language_none = self.columns_values_str_language_none

        db_machine.mdc_columns_count_of_datatypes = self.columns_count_of_datatypes
        db_machine.mdc_columns_input_count_of_datatypes  = self.columns_input_count_of_datatypes
        db_machine.mdc_columns_output_count_of_datatypes  = self.columns_output_count_of_datatypes

        return self


    def dataframe_pre_encode(self,  user_dataframe: pd.DataFrame    ) -> pd.DataFrame:
        """
        format dataframe with good datatypes (according to datatypes defined in machine) and
        Extends json columns to columns with simpler types using the early-created json columns tree

        :param user_dataframe: dataframe to be pre-encoded
        :return: pre-encoded dataframe
        """

        if ENABLE_LOGGER_DEBUG_MachineDataConfiguration: logger.debug( f"pre_encoding user_dataframe of {user_dataframe.shape[0]} rows X {user_dataframe.shape[1]} cols" )

        if not isinstance(user_dataframe, pd.DataFrame):
            logger.error( f"MDC pre-encoding method expected DataFrame, but received '{type(user_dataframe)}'" )

        dataframe_pre_encoding_tmp = user_dataframe.copy()

        # Change the index values of the copy of the user dataframe to default
        dataframe_pre_encoding_tmp.reset_index(inplace=True, drop=True)

        for json_column_name in self.columns_json_structure:

            # Skip json columns that are not in the user dataframe
            if json_column_name not in dataframe_pre_encoding_tmp.columns:
                continue

            # Get a dataframe with columns expanded from the json column
            dataframe_from_json_column = self._extend_json_column_by_json_structure(
                dataframe_pre_encoding_tmp[json_column_name]
            )

            # Drop the json column expanded
            dataframe_pre_encoding_tmp.drop(
                columns=json_column_name, axis=1, inplace=True
            )

            # reformat datatypes of the new dataframe made from the json column
            from ML import DataFileReader
            dataframe_from_json_column = DataFileReader.reformat_pandas_cells_by_columns_datatypes(
                dataframe_from_json_column,
                self.columns_data_type,
                date_format = "YMD",
                decimal_separator = ".",
            )
            dataframe_pre_encoding_tmp = pd.concat(
                (
                    dataframe_pre_encoding_tmp,
                    dataframe_from_json_column,
                ),
                axis=1,
            )

        # Change pandas 'Na' and numpy 'NaN' to python 'None'
        formatted_dataframe = pd.DataFrame(
            np.where(
                pd.isna(dataframe_pre_encoding_tmp), [None], dataframe_pre_encoding_tmp
            ),
            columns=dataframe_pre_encoding_tmp.columns,
            # Return index values from user dataframe
            index=user_dataframe.index,
        )
        del dataframe_pre_encoding_tmp

        return formatted_dataframe


    def dataframe_post_decode(self, decoded_from_ai_dataframe: pd.DataFrame ) -> pd.DataFrame:
        """
        Collapses all columns extended from json columns using the early-created json columns tree

        :param decoded_from_ai_dataframe: dataframe to be post-decoded
        :return: post-decoded dataframe
        """

        if ENABLE_LOGGER_DEBUG_MachineDataConfiguration: logger.debug( f"Post_decoding decoded_dataframe of {decoded_from_ai_dataframe.shape[0 ]} rows X {decoded_from_ai_dataframe.shape[0 ]} cols" )

        if not isinstance(decoded_from_ai_dataframe, pd.DataFrame ):
            logger.error(f"MDC post-decoding method expected pandas dataframe, but received '{type(decoded_from_ai_dataframe )}'" )

        post_decoded_dataframe_tmp = decoded_from_ai_dataframe.copy( )

        # Change the index values of the copy of the decoded dataframe to default
        post_decoded_dataframe_tmp.reset_index(inplace=True, drop=True)

        for (
            json_column_name,
            column_json_structure,
        ) in self.columns_json_structure.items():

            if all(
                column_name not in self.get_children_of_json_column(json_column_name)
                for column_name in decoded_from_ai_dataframe.columns
            ):
                continue

            # Post-decode a group of columns expanded from the json column using the json columns structure
            post_decoded_dataframe_tmp = self._collapse_json_column(
                post_decoded_dataframe_tmp, column_json_structure, json_column_name
            )

        post_decoded_dataframe = pd.DataFrame(
            # Change pandas 'Na' and numpy 'NaN' to python 'None'
            np.where(
                pd.isna(post_decoded_dataframe_tmp), [None], post_decoded_dataframe_tmp
            ),
            columns=post_decoded_dataframe_tmp.columns,
            # Return index values from decoded dataframe
            index=decoded_from_ai_dataframe.index,
        )
        del post_decoded_dataframe_tmp

        return post_decoded_dataframe


    def verify_compatibility_additional_dataframe(
                self,
                additional_user_dataframe: pd.DataFrame,
                machine: Machine,
                decimal_separator : str,
                date_format : str,
    ) -> (bool, str):
        """
        Check if the additional dataframe have the same columns and types as defined in the properties

        :param additional_user_dataframe: another part of the dataframe
        :param machine: the machine to verify the compatibility with
        :param decimal_separator: . or , only
        :param date_format: DMY or MDY only

        :return: True,"" if addition dataframe have the same columns and types as defined in the MDC.configuration else return False, "error message"
        """
        if not isinstance(additional_user_dataframe, pd.DataFrame):
            logger.error(
                f"MDC 'check_data_validity' method expected pandas dataframe, "
                f"but received '{type(additional_user_dataframe)}'"
            )

        if not isinstance(machine, Machine):
            logger.error( f"The machine_source must be of type 'machine', but received '{type( machine )}'" )

        from ML import DataFileReader
        new_dfr = DataFileReader(
                additional_user_dataframe,
                decimal_separator = decimal_separator,
                date_format = date_format
            )

        mdc_from_additional_dataframe = MachineDataConfiguration(
            machine,
            new_dfr.get_formatted_user_dataframe,
            new_dfr.get_user_columns_datatype,
            new_dfr.get_user_columns_description,
            decimal_separator = decimal_separator,
            date_format = date_format
        )

        if not set( self.columns_name_input_user_df ).issubset( set( mdc_from_additional_dataframe.columns_name_input_user_df ) ):
            return False, f"The new dataframe is missing column(s) input(s) : { ( set( self.columns_name_input_user_df ) - set( mdc_from_additional_dataframe.columns_name_input_user_df) ) }"

        if not set( self.columns_name_output_user_df ).issubset( set( mdc_from_additional_dataframe.columns_name_output_user_df ) ):
            return False, f"The new dataframe is missing this column(s) output(s) in the machine : { ( set( self.columns_name_output_user_df ) - set( mdc_from_additional_dataframe.columns_name_output_user_df) ) }"

        if not set( mdc_from_additional_dataframe.columns_name_output_user_df).issubset( set( self.columns_name_output_user_df ) ):
            return False, f"The new dataframe have this additional column(s) output(s) : { ( set( mdc_from_additional_dataframe.columns_name_output_user_df ) - set( self.columns_name_output_user_df ) ) }"

        for (
            column_name,
            column_type,
        ) in mdc_from_additional_dataframe.columns_type_user_df.items():
            if not column_name in self.columns_type_user_df:
                return False, (f"New column '{column_name}' is not in machine inputs:{self.columns_name_input_user_df} or outputs:{self.columns_name_output_user_df} " )
            col_type_cfg = self.columns_type_user_df[column_name]
            if    ( not column_type == DatasetColumnDataType.IGNORE and
                    not column_type == DatasetColumnDataType.LANGUAGE and col_type_cfg == DatasetColumnDataType.LABEL and
                    not column_type == DatasetColumnDataType.LABEL and col_type_cfg == DatasetColumnDataType.LANGUAGE and
                    column_type != self.columns_type_user_df[column_name]
                    ):
                return False, (f"New data in column '{column_name}' is {column_type} but in machine it is {col_type_cfg}")

        return True, ""


    def get_parent_of_extended_column(
        self, child_column_name: str, sep: str = JSON_COLUMN_NAME_SEPARATOR
    ) -> str:
        """
        Search for the parent name of the json column by the child expanded column using the created json column tree.
        If the tree does not contain the name of the child column and it is a column name of the dataframe -
        the method will return the name of this column.

        :param child_column_name: name of an extended column (part of JSON column)
        :param sep: the SEPARATOR_IN_COLUMNS_NAMES used to collapse the json column
        :return: name of the parent json column of the child expanded column

        """

        def _parent_column_by_child_using_json_structure(
                _child_column_name: str, _columns_json_structure: dict
        ):
            if _child_column_name in _columns_json_structure:
                return _child_column_name

            for column_name in _columns_json_structure:
                if child_column_name.startswith(f"{column_name}{sep}"):
                    return _parent_column_by_child_using_json_structure(
                        _child_column_name=_child_column_name[len(column_name) + 1:],
                        _columns_json_structure=_columns_json_structure[column_name],
                    )

            logger.error(f"The json column tree does not contain a '{_child_column_name}' column")

        if child_column_name not in self.columns_name_input:
            logger.error(f"The dataframe does not contain a '{child_column_name}' column")

        try:
            json_parent_column_name = _parent_column_by_child_using_json_structure(
                child_column_name, self.columns_json_structure
            )
        except ValueError:
            return child_column_name

        return json_parent_column_name


    def get_children_of_json_column(
            self,
            parent_column_name: str,
            sep: str = JSON_COLUMN_NAME_SEPARATOR
    ) -> List[str]:
        """
        Search for the children names of the input json column using the created json column tree.
        If the parent column is not json and it's a column name of the dataframe - returns this column name in the list.

        :param parent_column_name: name of parent json column
        :param sep: the SEPARATOR_IN_COLUMNS_NAMES used when expanding the json column
        :return: list of children of the json column
        """

        def _children_columns_by_parent_using_json_structure(
            _parent_column_name: str, _columns_json_structure: dict
        ):
            if not _columns_json_structure:
                return [_parent_column_name]

            names_of_child_columns = list()
            for child_column_name in _columns_json_structure:
                names_of_child_columns.extend(
                    _children_columns_by_parent_using_json_structure(
                        child_column_name,
                        _columns_json_structure=_columns_json_structure[
                            child_column_name
                        ],
                    )
                )

            column_children = [
                f"{_parent_column_name}{sep}{child_column_name}"
                for child_column_name in names_of_child_columns
            ]

            return column_children

        # === start => get_children_of_json_column
        if parent_column_name not in self.columns_name_input_user_df:
            logger.error(f"The user dataframe does not contain a '{parent_column_name}' column")

        if parent_column_name not in self.columns_json_structure:
            return [parent_column_name]

        return _children_columns_by_parent_using_json_structure(
            parent_column_name, self.columns_json_structure[parent_column_name]
        )


    def _get_count_of_datatypes_per_columns(self , only_columns_with_direction: Optional[ColumnDirectionType] = None) -> Dict[str, int]:
        """
        return a dict indicating for each datatypes how many columns found
        :param only_columns_with_direction: if present will return dict only for inputs or for outputs columns
        """

        count_of_datatypes = {}
        for one_datatype in DatasetColumnDataType:
                count_of_datatypes[ one_datatype.name ] = 0

        for column_name, column_data_type in self.columns_data_type.items():
            if not isinstance(column_data_type, DatasetColumnDataType):
                logger.error( f"The column datatype must be of type 'DatasetColumnDataType', but is '{type( column_data_type )}'" )
            elif (
                    (only_columns_with_direction==None) or
                    (only_columns_with_direction==ColumnDirectionType.INPUT and column_name in self.columns_name_input and self.columns_name_input[column_name]==True) or
                    (only_columns_with_direction==ColumnDirectionType.OUTPUT and column_name in self.columns_name_output and self.columns_name_output[column_name]==True)
                    ):
                count_of_datatypes[column_data_type.name] = count_of_datatypes[column_data_type.name] + 1

        return count_of_datatypes


    def _get_full_dataframe_with_json_columns_expanded(
            self,
            user_dataframe: pd.DataFrame,
            json_columns_names_to_extend: Iterable
    ) -> (pd.DataFrame, dict):
        """
        Extends json columns, create and return json columns tree

        :param user_dataframe:
        :param json_columns_names_to_extend: a set of columns name with the JSON data type
        :returns: pre-encoded dataframe and json columns tree
        """


        def _expand_json_column( data_serie_json_column: pd.Series ):
            # Create empty dataframe for extended json column
            new_dataframe_from_json_column = pd.DataFrame()
            new_columns_json_structure = dict()

            # Convert json column to single dataframe
            for json_in_cell in data_serie_json_column:

                if pd.isna(json_in_cell):
                    # the value of the serie is NONE so we add a new line of NAN to the dataframe result
                    # Create a new DataFrame with a single row of nan values, matching the column structure of the existing DataFrame
                    new_row = pd.DataFrame(
                        [[np.nan for _ in range(new_dataframe_from_json_column.shape[1])]],
                                    columns=new_dataframe_from_json_column.columns
                            )
                    # Use pd.concat to append the new row to the existing DataFrame, ignoring the index to reset it
                    new_dataframe_from_json_column = pd.concat([new_dataframe_from_json_column, new_row], ignore_index=True)
                    continue

                if isinstance(json_in_cell, (dict,list) ):
                    list_or_dict_from_json_cell = json_in_cell
                else:
                    try:
                        list_or_dict_from_json_cell = json.loads( json_in_cell )
                    except:
                        try:
                            list_or_dict_from_json_cell = ast.literal_eval( json_in_cell )
                        except:
                            list_or_dict_from_json_cell = str( json_in_cell )

                if not list_or_dict_from_json_cell:
                    # the value of the serie is NONE so we add a new line of NAN to the dataframe result
                    # Create a new DataFrame with a single row of nan values, matching the column structure of the existing DataFrame
                    new_row = pd.DataFrame(
                        [[np.nan for _ in range(new_dataframe_from_json_column.shape[1])]],
                                    columns=new_dataframe_from_json_column.columns
                            )
                    # Use pd.concat to append the new row to the existing DataFrame, ignoring the index to reset it
                    new_dataframe_from_json_column = pd.concat([new_dataframe_from_json_column, new_row], ignore_index=True)
                    continue

                dict_for_creating_dataframe, json_cell_structure = self._extend_json_cell(
                    str(data_serie_json_column.name ), list_or_dict_from_json_cell
                )

                new_columns_json_structure = self._append_to_columns_json_structure(
                    new_columns_json_structure, json_cell_structure
                )

                dict_for_creating_dataframe = {
                    key: [value] for key, value in dict_for_creating_dataframe.items()
                }

                new_dataframe_from_json_column = pd.concat([new_dataframe_from_json_column, pd.DataFrame.from_dict(dict_for_creating_dataframe)], ignore_index=True)

            return new_dataframe_from_json_column, new_columns_json_structure



        # Create a empty temporary dataframe to store inside columns extended from json columns
        dataframe_extended_from_json_columns = pd.DataFrame(
            np.empty((user_dataframe.shape[0], 0))
        )

        dataframe_tmp = user_dataframe.copy()

        columns_json_structure = { }
        list_of_expanded_json_column = [ ]
        for column_name in self.columns_type_user_df:

            # In this loop we work only with json columns
            if column_name not in json_columns_names_to_extend:
                continue

            json_column_name = column_name
            list_of_expanded_json_column.append( json_column_name )

            (dataframe_from_json_column, column_json_structure) = _expand_json_column(
                user_dataframe[json_column_name]
            )

            # Update json columns tree
            columns_json_structure.update(column_json_structure)

            # Drop json column from the dataframe
            dataframe_tmp = dataframe_tmp.drop(
                json_column_name, axis=1
            )

            # Add to the end of the temporary dataframe columns derived from json column
            dataframe_extended_from_json_columns = pd.merge(
                dataframe_extended_from_json_columns,
                dataframe_from_json_column,
                left_index=True,
                right_index=True,
            )
        dataframe_extended_from_json_columns.index = dataframe_tmp.index

        # Add to the end of the user dataframe columns derived from json columns
        dataframe = pd.merge(
            dataframe_tmp,
            dataframe_extended_from_json_columns,
            left_index=True,
            right_index=True,
        )

        return dataframe, columns_json_structure, list_of_expanded_json_column


    def _add_warning_for_column(self, column_name: str, warning_message: str) -> NoReturn:
        """Adds a warning message to the column, but only if the message is unique"""
        if column_name not in self.columns_warnings:
            self.columns_warnings[column_name] = warning_message

        elif warning_message not in self.columns_warnings[column_name]:
            self.columns_warnings[column_name] += ", \n " + warning_message


    def _add_error_for_column(self, column_name: str, error_message: str) -> NoReturn:
        """Adds an error message to the column, but only if the message is unique"""
        if column_name not in self.columns_errors:
            self.columns_errors[column_name] = error_message

        elif error_message not in self.columns_errors[column_name]:
            self.columns_errors[column_name] += ", " + error_message


    def _extend_json_column_by_json_structure(self, json_column: pd.Series):
        """Creates a dataframe from the column containing an n-dimensional json"""
        # Create empty dataframe for extended json column
        dataframe_from_json_column = pd.DataFrame()
        json_columns_name = self.get_children_of_json_column( json_column.name )

        # Convert json column to dataframe
        for json_in_cell in json_column:

            if pd.isna(json_in_cell):
                # json cell is missing, add a row with all np.nan
                # Create a new DataFrame with a single row filled with np.nan, matching the specified column names
                new_row = pd.DataFrame(
                    np.full((1, len(json_columns_name)), np.nan),  # Creates a 1xN array of np.nan values
                    columns=json_columns_name
                )
                # Use pd.concat to append the new row to the existing DataFrame, and ignore the index to reset it
                dataframe_from_json_column = pd.concat([dataframe_from_json_column, new_row], ignore_index=True)
                continue

            if not isinstance(json_in_cell, dict):
                try:
                    json_in_cell = json.loads( json_in_cell )
                except:
                    try:
                        null = None
                        json_in_cell = ast.literal_eval( json_in_cell )
                    except:
                        continue

            json_column_name = json_column.name

            dict_for_creating_dataframe = self._extend_json_cell_by_structure(
                json_column_name=json_column_name,
                json_element=json_in_cell,
                json_structure=self.columns_json_structure[json_column_name],
            )

            # Wrap values in list
            dict_for_creating_dataframe = {
                key: [value] for key, value in dict_for_creating_dataframe.items()
            }

            # Add a row formed from the json record to the dataframe
            df_from_dict = pd.DataFrame.from_dict(dict_for_creating_dataframe)
            # Use pd.concat to append the new DataFrame created from the dictionary to the existing DataFrame
            dataframe_from_json_column = pd.concat([dataframe_from_json_column, df_from_dict], ignore_index=True)

        return dataframe_from_json_column


    def _determine_columns_input_output_from_dataframe(
            self,
            dataframe_to_analyse: pd,
            columns_description: dict,
            ) -> (Dict[str, bool], Dict[str, bool]):
        """
            STEP 1  empty columns are ignored (not input, not output)
            STEP 1 if there is missing value => set the columns as outputs
            STEP 2  if there is keyword in title,description set them
            STEP 3 if there is no outputs => assign last column as outputs
            STEP 4 make all others columns not detected as inputs

        :param dataframe_to_analyse:  the dataframe containing data and titles and description
        :return: columns_name_input, columns_name_output , both are dict with key=Columns_names and values (True,False)
        """

        def determine_input_output_ignore_columns_by_missing_values( dataframe_to_analyse: pd.DataFrame ) -> (Dict[str, bool ], Dict[str, bool ]):
            """
            Detect the columns with missing values as OUTPUT

            :param dataframe_to_analyse:
            :return: columns_name_input, columns_name_output , both are dict with key=Columns_names and values (True,False)
            """

            if not isinstance(dataframe_to_analyse, pd.DataFrame ):
                raise TypeError

            total_count_of_rows = dataframe_to_analyse.shape[0 ]

            # Create a dictionary to store how many total NaN count for each column
            columns_count_null_total = dataframe_to_analyse.isnull( ).sum( )

            # Create a dictionary to store the contiguous NaN count for each column from bottom
            columns_count_null_from_bottom = { }
            for column_name in dataframe_to_analyse.columns:
                count = 0
                for value in dataframe_to_analyse[ column_name ][ ::-1 ]:
                    if pd.isnull( value ):
                        count += 1
                    else:
                        break
                columns_count_null_from_bottom[ column_name ] = count


            # Will determine the columns input/outputs by missing values
            columns_name_input, columns_name_output = dict(), dict()

            for column_name, column_missing_values_count in columns_count_null_from_bottom.items():

                # If 100% rows of column with missing it's an ignored column
                if column_missing_values_count == total_count_of_rows:
                    columns_name_input[column_name] = False
                    columns_name_output[column_name] = False

                # If ONLY rows from bottom are null , it's an output column
                elif (
                    column_missing_values_count >= 1 and
                    column_missing_values_count < total_count_of_rows and
                    column_missing_values_count == columns_count_null_total[column_name]
                ):
                    columns_name_input[column_name] = False
                    columns_name_output[column_name] = True

            return columns_name_input, columns_name_output
            # end of _determine_input_output_columns_by_missing_values( self , dataframe: pd.DataFrame )


        def is_text_have_keyword(
                    text_to_check: str,
                    keywords_to_search: str,
                    ) -> (Dict[str, bool], Dict[str, bool]):
            """
            :params text_to_check: search the keyword(s) in the text_to_check
            ;params keywords_to_search: string with several keywords separated by separator /
            :return: True if at least one keyword have been found
            """
            try:
                for keyword in keywords_to_search.lower( ).split( "/" ):
                    if re.search(fr"(?:^|[^A-Za-z]){keyword}(?:$|[^A-Za-z])", text_to_check.lower()):
                        return True
            except Exception as e:
                logger.error( f"unable to check {keyword} because {e}")
                return False
            else:
                return False


        # ------- start of _determine_columns_input_output_from_dfr --------------------------------
        if not isinstance(dataframe_to_analyse, pd.DataFrame):
            raise TypeError

        # ------- STEP 1
        columns_name_input, columns_name_output = determine_input_output_ignore_columns_by_missing_values( dataframe_to_analyse )

        # ------- STEP 2
        for column_name in dataframe_to_analyse.columns:
                if is_text_have_keyword( column_name , "input" ) or is_text_have_keyword( columns_description[column_name] , "input" ):
                    columns_name_input[ column_name ] = True
                    columns_name_output[ column_name ] = False
                elif is_text_have_keyword( column_name , "output" ) or is_text_have_keyword( columns_description[column_name] , "output" ):
                    columns_name_input[ column_name ] = False
                    columns_name_output[ column_name ] = True

        # ------- STEP 3
        if not ( any( enabled for enabled in iter( columns_name_output.values())) ):
            # we search the first column from the right wich is not IGNORE
            for column_name, datatype in reversed( self.columns_type_user_df.items( ) ):
                if datatype != DatasetColumnDataType.IGNORE:
                    columns_name_output[ column_name ] = True
                    columns_name_input[ column_name ] = False
                    break
            else:
                logger.error( f"We was unable to find any column to set as output by default ! {self.columns_type_user_df}")

        # ------- STEP 4
        for column_name in dataframe_to_analyse.columns:
            if not column_name in columns_name_output and not column_name in columns_name_input:
                columns_name_input[ column_name ] = True
                columns_name_output[ column_name ] = False

        return columns_name_input, columns_name_output


    def _get_columns_missing_percentage_from_dataframe(
            self,
            dataframe: pd.DataFrame,
            ) -> Dict[str, float]:
        """
        Create a dictionary with columns name as keys and missing values percentage as values using a dataframe

        :params dataframe: the dataframe to check for missing values
        :return: a dictionary with columns name as keys and missing values percentage as values using a dataframe
        """

        number_of_rows = dataframe.shape[0]

        columns_missing_percentage = { }
        for column_name in dataframe.columns:
            missing_values = dataframe[ column_name ].isna( ).sum( )
            columns_missing_percentage[ column_name ] = 0 if number_of_rows==0 else (missing_values / number_of_rows * 100)

        return columns_missing_percentage


    def _get_columns_type_from_dataframe(
            self,
            dataframe: pd.DataFrame,
            decimal_separator : str,
            date_format : str,
            ) -> Dict[str, str]:
        """
        Create a dictionary with columns name as keys and columns type as values using a dataframe
        """
        from ML import DataFileReader
        return {
            column_name: DataFileReader.determine_column_datatype(
                dataframe[column_name] ,
                decimal_separator = decimal_separator,
                date_format = date_format,
            )
            for column_name in dataframe.columns
        }


    def _get_unique_values_count(
            self,
            dataframe: pd.DataFrame,
            columns_datatype: Dict[str, DatasetColumnDataType]
        ) -> Dict[str, Optional[int]]:
        """
        Create a dictionary with columns name as keys and number of unique values as values using a dataframe

        :param dataframe: a dataframe from which the number of unique values is calculated by columns
        :param columns_datatype: a dictionary with the name of the dataframe columns and their types
        :return: a dictionary with the name of the dataframe columns and the number of unique values in them

        """
        columns_unique_values_count = dict()
        for column_name in dataframe.columns:
            if not column_name in columns_datatype or columns_datatype[column_name] == DatasetColumnDataType.JSON:
                columns_unique_values_count[column_name] = None
            else:
                columns_unique_values_count[column_name] = dataframe[column_name].nunique()
        return columns_unique_values_count


    def _get_columns_most_frequent_values_count(
            self,
            dataframe: pd.DataFrame,
            columns_datatype: Dict[str, DatasetColumnDataType]
        ) -> Dict[str, Optional[int]]:
        """
        Create a dictionary with columns name as keys and count of most frequent values present more than MOST_FREQUENTS_VALUES_PERCENTAGE_OCCURRENCE_TO_KEEP_MIN %

        :param dataframe: a dataframe from which the number of most frequent values is calculated by columns
        :param columns_datatype: a dictionary with the name of the dataframe columns and their types
        :return: a dictionary with the name of the dataframe columns and number of their most frequent values

        """
        columns_most_frequent_values_count = dict()
        for column_name in dataframe.columns:

            if not column_name in columns_datatype:
                columns_most_frequent_values_count[column_name] = None
                continue

            if columns_datatype[column_name] in [DatasetColumnDataType.LANGUAGE, DatasetColumnDataType.IGNORE ]:
                columns_most_frequent_values_count[column_name] = None
                continue

            min_number_of_occurence = (
                        MOST_FREQUENTS_VALUES_PERCENTAGE_OCCURRENCE_TO_KEEP_MIN / 100
                        * dataframe.shape[ 0 ]
            )

            # Drop values whose frequency is less than the minimum
            count_most_frequent_column_values = dataframe[column_name].dropna().value_counts()
            count_most_frequent_column_values.drop(
                        count_most_frequent_column_values[ count_most_frequent_column_values < min_number_of_occurence ].index,
                        inplace=True, )

            columns_most_frequent_values_count[ column_name ] = count_most_frequent_column_values.shape[0]

        return columns_most_frequent_values_count


    def _append_to_columns_json_structure(
        self, columns_json_structure: dict, column_json_structure: dict
    ):
        """Add the json column tree to the main json column tree"""
        main_columns_json_structure = copy.deepcopy(columns_json_structure)

        for child_column_name, child_column_structure in column_json_structure.items():
            if child_column_name in main_columns_json_structure.keys():
                main_columns_json_structure[
                    child_column_name
                ] = self._append_to_columns_json_structure(
                    columns_json_structure=main_columns_json_structure[child_column_name],
                    column_json_structure=child_column_structure,
                )

            else:
                main_columns_json_structure[child_column_name] = child_column_structure

        return main_columns_json_structure


    def _collapse_json_column(
            self,
            input_dataframe,
            json_structure,
            json_column_name,
            prefix=None,
            sep=JSON_COLUMN_NAME_SEPARATOR,
        ):
        dataframe_with_collapse_json_column = input_dataframe

        if not json_structure:
            return dataframe_with_collapse_json_column

        expand_columns_name, expand_columns_full_name = list(), list()

        new_prefix = (
            json_column_name if prefix is None else f"{prefix}{sep}{json_column_name}"
        )

        for column_name, sub_json_structure in json_structure.items():
            dataframe_with_collapse_json_column = self._collapse_json_column(
                dataframe_with_collapse_json_column,
                sub_json_structure,
                column_name,
                new_prefix,
            )

            full_column_name = f"{new_prefix}{sep}{column_name}"

            if full_column_name in dataframe_with_collapse_json_column.columns:
                expand_columns_name.append(column_name)
                expand_columns_full_name.append(full_column_name)

        expand_dataframe = dataframe_with_collapse_json_column[expand_columns_full_name]

        expand_dataframe.columns = expand_columns_name

        dataframe_with_collapse_json_column = dataframe_with_collapse_json_column.drop(
            expand_columns_full_name, axis=1
        )

        json_column = expand_dataframe.apply(
            lambda row: re.sub(
                r"\"(?=[\[{])|\\(?=\")|(?<=[}\]])\"|\\(?=/)", "", row.dropna().to_json()
            ),
            axis=1,
        )

        dataframe_with_collapse_json_column[new_prefix] = [
            re.sub(r",\"\w+\":{}|\"\w+\":{},|\"\w+\":{}", "", cell) for cell in json_column
        ]

        return dataframe_with_collapse_json_column


    # ----------------------------------------------------------------------------------------------------------------------
    # Here we have functions to extend json columns and create json structure
    # ----------------------------------------------------------------------------------------------------------------------



    def _extend_json_cell(
            self,
            column_to_extend_name: str,
            value_to_extend: Union[list, dict, int, float, str ],
            optional_col_name_prefix: Optional[str ] = None,
            columns_name_separator: str = JSON_COLUMN_NAME_SEPARATOR,
        ):
        """
        Recursive function that extend json of any dimension
        """


        def extend_json_dict_element(  column_to_extend_name: str, json_dict_to_extend: dict, optional_col_name_prefix: str, columns_name_separator: str ):

            new_prefix = column_to_extend_name if optional_col_name_prefix is None else f"{optional_col_name_prefix}{columns_name_separator}{column_to_extend_name}"

            extended_json = dict()
            json_structure = {column_to_extend_name: dict( ) }

            for key, value in json_dict_to_extend.items( ):

                extended_json_element, element_json_structure = self._extend_json_cell(
                    key, value, optional_col_name_prefix=new_prefix, columns_name_separator=columns_name_separator
                )

                extended_json.update(extended_json_element)
                json_structure[column_to_extend_name ].update(element_json_structure )

            return extended_json, json_structure
            # end of extend_json_dict_element


        def extend_json_list_element( column_to_extend_name: str, json_list_to_extend: list, optional_col_name_prefix: str, columns_name_separator: str ):
            """
            Create a column for each value of the list
            """

            new_prefix = column_to_extend_name if optional_col_name_prefix is None else f"{optional_col_name_prefix}{columns_name_separator}{column_to_extend_name}"

            extended_json = dict()
            json_structure = {column_to_extend_name: dict( ) }

            for element_index, json_element in enumerate(json_list_to_extend ):
                this_new_element_column_name = str( element_index + 1 )
                extended_json_element, element_json_structure = self._extend_json_cell(
                            this_new_element_column_name,
                            json_element,
                            optional_col_name_prefix=new_prefix,
                            columns_name_separator=columns_name_separator
                )

                extended_json.update(extended_json_element)
                json_structure[column_to_extend_name ].update(element_json_structure )

            return extended_json, json_structure
            # end of extend_json_list_element


        # --- start of _extend_json_cell
        if isinstance(value_to_extend, list ):
            extended_json, json_structure = extend_json_list_element(
                column_to_extend_name,
                value_to_extend,
                columns_name_separator=columns_name_separator,
                optional_col_name_prefix=optional_col_name_prefix
            )

        elif isinstance(value_to_extend, dict ):
            extended_json, json_structure = extend_json_dict_element(
                column_to_extend_name,
                value_to_extend,
                columns_name_separator=columns_name_separator,
                optional_col_name_prefix=optional_col_name_prefix
            )

        elif isinstance(value_to_extend, (int, float, str) ) or pd.isna(value_to_extend ):
            column_full_name = (column_to_extend_name if optional_col_name_prefix is None else f"{optional_col_name_prefix}{columns_name_separator}{column_to_extend_name}")
            extended_json, json_structure = {column_full_name: value_to_extend }, {column_to_extend_name: dict( ) }
        else:
            logger.warning(f"MDC cannot extend json with {type(value_to_extend )} type , value={value_to_extend} => it will be STRING" )
            column_full_name = (column_to_extend_name if optional_col_name_prefix is None else f"{optional_col_name_prefix}{columns_name_separator}{column_to_extend_name}")
            extended_json, json_structure = {column_full_name: str(value_to_extend) }, {column_to_extend_name: dict( ) }

        return extended_json, json_structure
        # end of _extend_json_cell



    # ----------------------------------------------------------------------------------------------------------------------
    # Here we have functions to extend json columns into a set of simpler columns type
    # ----------------------------------------------------------------------------------------------------------------------

    def _replace_empty_dict_with_none(
            self,
            dictionary: dict,
            root: str,
            prefix: Optional[str],
            sep: str = JSON_COLUMN_NAME_SEPARATOR,
        ):

        if not dictionary:
            column_full_name = root if prefix is None else f"{root}{sep}{prefix}"
            return {column_full_name: None}

        result = dict()
        for key in dictionary:
            new_prefix = key if prefix is None else f"{prefix}{sep}{key}"
            result.update(
                self._replace_empty_dict_with_none(dictionary[key], root, new_prefix, sep)
            )

        return result


    def _extend_json_cell_by_structure(
            self,
            json_column_name,
            json_element: Union[dict, list],
            json_structure,
            prefix: Optional[str] = None,
            sep: str = JSON_COLUMN_NAME_SEPARATOR,
        ):


        def extend_json_list_element_by_structure(
                json_column_name: str,
                json_list: list,
                json_structure,
                prefix: Optional[str],
                sep: str,
            ):

            extended_json = dict()

            new_prefix = (
                json_column_name if prefix is None else f"{prefix}{sep}{json_column_name}"
            )

            for column_name in json_structure:

                if column_name.isdigit() and int(column_name) <= len(json_list):
                    index = int(column_name) - 1

                    extended_json.update(
                        self._extend_json_cell_by_structure(
                            json_column_name=column_name,
                            json_element=json_list[index],
                            json_structure=json_structure[column_name],
                            prefix=new_prefix,
                            sep=sep,
                        )
                    )

                else:
                    column_full_name = f"{new_prefix}{sep}{column_name}"
                    extended_json.update(
                        self._replace_empty_dict_with_none(
                            json_structure[column_name], column_full_name, None
                        )
                    )

            return extended_json
         # end of extend_json_list_element_by_structure


        def extend_json_dict_element_by_structure(
                json_column_name: str,
                json_dict: dict,
                json_structure,
                prefix: Optional[str],
                sep: str,
            ):

            extended_json = dict()

            new_prefix = (
                json_column_name if prefix is None else f"{prefix}{sep}{json_column_name}"
            )

            for column_name in json_structure:

                if column_name in json_dict:
                    extended_json.update(
                        self._extend_json_cell_by_structure(
                            json_column_name=column_name,
                            json_element=json_dict[column_name],
                            json_structure=json_structure[column_name],
                            prefix=new_prefix,
                            sep=sep,
                        )
                    )
                else:
                    column_full_name = f"{new_prefix}{sep}{column_name}"
                    extended_json.update(
                        self._replace_empty_dict_with_none(
                            json_structure[column_name], column_full_name, None
                        )
                    )

            return extended_json
            #  end of _extend_json_dict_element_by_structure


        if not json_structure:
            column_full_name = (
                json_column_name if prefix is None else f"{prefix}{sep}{json_column_name}"
            )
            return {column_full_name: json_element}

        extended_json = dict()

        if isinstance(json_element, dict):
            extended_json.update(
                extend_json_dict_element_by_structure(
                    json_column_name, json_element, json_structure, prefix, sep
                )
            )

        elif isinstance(json_element, list):
            extended_json.update(
                extend_json_list_element_by_structure(
                    json_column_name, json_element, json_structure, prefix, sep
                )
            )

        elif isinstance(json_element, (int, float, complex, str)):
            extended_json.update(
                {
                    json_column_name
                    if prefix is None
                    else f"{prefix}{sep}{json_column_name}": json_element
                }
            )

        else:
            logger.error(f"json element have unsupported type {type(json_element)}")

        return extended_json


