from typing import Optional
import os
import re
import ast
import json
import numpy as np
import pandas as pd
from pandas.errors import ParserError
from dateutil import parser as dateutilparser
import datetime
import langdetect
import chardet
from humanfriendly import parse_size, parse_timespan, parse_length
from forex_python.converter import CurrencyCodes, CurrencyRates

from ML import EasyAutoMLDBModels, __getlogger

from ML import Machine

logger = __getlogger()

from SharedConstants import *



# to determine datatype LABEL => when we detect a column of text, we check if there is not too many unique values or it is text not label
DETECTION_LABEL_MAXIMUM_UNIQUES_VALUES = 1000 # maximum of unique value in the column to be LABEL
DETECTION_LABEL_MAXIMUM_STR_LEN_VALUES = 100 # if a value in the column is higher than that it will be not a LABEL column

# to determine datatype LANGUAGE => we need to detect in the column-values a language (from EN FR DE IT ES PT ) at minimum the percentage , or we ignore the columns data  # A value of 70 seem detect all texts when it is all texts
DETECTION_LANGUAGE_MINIMUM_PERCENTAGE = 65
DETECTION_LANGUAGE_MINIMUM_ROWS = 5

# the numbers parser will remove the unit and convert the number of the following SUFFIX --- suffix must be lowercase and without s at the end
NUMBERS_PARSER_SUFFIX_UNITS_MULTIPLIERS = {
        'thousand': 10**3, 'mille':10**3, 'mille': 10**3, 'mila': 10**3, 'tausend': 10**3, 'mil': 10**3,
        'million': 10**6, 'million': 10**6, 'millionen': 10**6, 'millón': 10**6, 'milhão': 10**6, 'milione': 10**6, 'million': 10**6,
        'billion': 10**9, 'milliard': 10**9, 'billionen': 10**9, 'milliarden': 10**9, 'bilhão': 10**9, 'billón': 10**9, 'billione': 10**9 ,'bilione': 10**9 , 'miliardo': 10**9 , 'miliardi': 10**9, 'milliard': 10**9,
        'trillion': 10**12, 'trilliard': 10**12, 'trillionen': 10**12, 'trilhão': 10**12,
        'quadrillion': 10**15, 'quadrilliard': 10**15, 'quadrillionen': 10**15, 'quadrilhão': 10**15,
        'k': 10**3, 'm': 10**6    # , 'b': 10**9, 't': 10**12, 'q': 10**15   (not frequents)
    }
# used in regex to detect if the string will be converted : thousand|mille|mila|tausend|mil|million|millionen|millón|milhão...........
NUMBERS_PARSER_SUFFIX_UNITS_REGEX = "|".join(NUMBERS_PARSER_SUFFIX_UNITS_MULTIPLIERS.keys())

# we load here becasue it is slow to initialize
numbers_parser_currency_forex_currency_rates = CurrencyRates( )

# we need to convert symbols to code because forex_python.converter do not work with symbols
NUMBERS_PARSER_CURRENCY_SYMBOL_CONV = {
    "$": "USD",
    "€": "EUR",
    "¥": "JPY",
    "£": "GBP",
    "A$": "AUD",
    "C$": "CAD",
    "CHF": "CHF",
    "CN¥": "CNY",
    "HK$": "HKD",
    "NZ$": "NZD",
    "SEK": "SEK",
    "₩": "KRW",
    "S$": "SGD",
    "NOK": "NOK",
    "MXN": "MXN",
    "₹": "INR",
    "₽": "RUB",
    "R": "ZAR",
    "TRY": "TRY",
    "R$": "BRL",
    "NT$": "TWD",
    "DKK": "DKK",
    "PLN": "PLN",
    "฿": "THB"
}
# Define a list of popular currency codes and symbols
NUMBERS_PARSER_CURRENCY_LIST = [ "USD", "EUR", "JPY", "GBP", "AUD", "CAD", "CHF", "CNY", "HKD", "NZD", "SEK", "KRW", "SGD", "NOK", "MXN", "INR", "RUB", "ZAR", "TRY", "BRL", "TWD", "DKK", "PLN", "THB" ] + list( NUMBERS_PARSER_CURRENCY_SYMBOL_CONV.keys ( ) )
NUMBERS_PARSER_CURRENCY_REGEX = "|".join( NUMBERS_PARSER_CURRENCY_LIST ).replace( "$" , "\$" )

class DataFileReader:
    """
    Reads data from a file or pandas user_dataframe_to_extend
    Supported files with extension: txt, tsv, xls, xlsx, json, csv
    """

    def __init__(
            self,
            data_source: [str,pd.DataFrame],
            decimal_separator: str,
            date_format: str,
            force_create_with_this_datatypes: Optional[ dict ] = None,
            force_create_with_this_descriptions: Optional[ dict ] = None,

    ):
        """
        Creates a new DataFileReader object from the data file that will read by file path

        :param data_source: relative or absolute path to the data file OR pandas.dataframe (user df not formatted or pre-encoded)
        :param decimal_separator: . or , only
        :param date_format: DMY or MDY only
        :param force_create_with_this_descriptions: columns description (            if None: columns will not have a description )
        :param force_create_with_this_datatypes: columns datatype            if None: columns datatype will be determined automatically
        """

        def correct_dataset_columns_names( s ):
            # Replace consecutive spaces with a single space
            s = re.sub( r' +', ' ', s ).strip( )
            # Replace consecutive underscores with a single underscore
            s = re.sub( r'_+', '_', s ).strip( )
            # Replace matched characters with -
            s = re.sub( r'[&"#%$*!:;,?]', '-', s ).strip( )
            if len(s)<=64:
                return s
            else:
                return s[ :62 ] + ".." # truncate to 64 chars


        self.dfr_columns_descriptions = { }
        self.dfr_columns_datatypes = { }
        self.formatted_user_dataframe = None

        # --- load dataframe -----------
        if isinstance( data_source , str):
            if ENABLE_LOGGER_DEBUG_DataFileReader:
                logger.debug( f"Creating DFR from {data_source}" )
            # read also description in self.dfr_columns_descriptions
            unformated_user_dataframe = self._read_dataframe_title_desc_from_file_path(
                        file_path=data_source,
                        decimal_separator=decimal_separator,
                        date_format=date_format,
            )
        elif isinstance( data_source , pd.DataFrame ):
            unformated_user_dataframe = data_source
            # detect if line 1 is description => set self.dfr_columns_descriptions or set self.dfr_columns_descriptions as dict with ""
            # if the first data row do contains only strings or null => then it is a description row !
            if unformated_user_dataframe.iloc[ 0 ].apply( lambda x: isinstance( x, str ) or pd.isnull( x ) ).all( ):
                # there is no numbers/dates/objects in any cell in the first data row then it is a description row
                self.dfr_columns_descriptions , unformated_user_dataframe = self._extract_columns_descriptions_from_first_row( unformated_user_dataframe )
            else:
                # there is no description row in the user_dataframe_to_extend we set the dict with all columns names as empty string
                self.dfr_columns_descriptions = dict.fromkeys(unformated_user_dataframe.columns, "" )



            if ENABLE_LOGGER_DEBUG_DataFileReader:
                logger.debug( f"Creating DFR from dataframe , shape:{unformated_user_dataframe.shape}" )

        #---reformat columns names to remove forbidden chars --------------
        unformated_user_dataframe.columns = unformated_user_dataframe.columns.map( correct_dataset_columns_names )
        if force_create_with_this_descriptions:
            force_create_with_this_descriptions = { correct_dataset_columns_names( key ): value for key, value in force_create_with_this_descriptions.items( ) }
        if force_create_with_this_datatypes:
            force_create_with_this_datatypes = { correct_dataset_columns_names( key ): value for key, value in force_create_with_this_datatypes.items( ) }

        # --- define descriptions -----------
        if force_create_with_this_descriptions:
            # update columns description present
            for col_name, desc in force_create_with_this_descriptions.items():
                self.dfr_columns_descriptions[col_name ] = desc

        # --- determine datatypes -----------------
        if force_create_with_this_datatypes is not None and not isinstance(force_create_with_this_datatypes, dict ):
            logger.error( f"Columns datatype must be of type 'dict' or None, but received '{type( force_create_with_this_datatypes )}'" )

        self.dfr_columns_datatypes = {
            column_name: DataFileReader.determine_column_datatype(
                unformated_user_dataframe[column_name],
                decimal_separator=decimal_separator,
                date_format=date_format,
            )
            for column_name in unformated_user_dataframe.columns
            }

        if force_create_with_this_datatypes:
            for col_name, col_datatype in force_create_with_this_datatypes.items():
                self.dfr_columns_datatypes[col_name ] = \
                    col_datatype if isinstance(col_datatype, DatasetColumnDataType) else DatasetColumnDataType[col_datatype]


        # --- reformat cells from determined datatypes -----------------
        self.formatted_user_dataframe  = self.reformat_pandas_cells_by_columns_datatypes(
            unformated_user_dataframe,
            self.dfr_columns_datatypes,
            decimal_separator=decimal_separator,
            date_format=date_format,
        )


    def save_configuration_in_machine(self, machine: Machine) -> "DataFileReader":
        """
        Stores DataFileReader attributes inside the input machine_source
        """
        from ML.Machine import Machine

        if not isinstance(machine, Machine):
            logger.error(f"The machine_source must be of type 'Machine', but received '{type(machine)}'")

        machine.db_machine.dfr_columns_description_user_df = self.dfr_columns_descriptions
        machine.db_machine.dfr_columns_type_user_df = self.dfr_columns_datatypes
        machine.db_machine.dfr_columns_python_user_df = {}

        return self


    @property
    def get_user_columns_datatype(self ):
        return self.dfr_columns_datatypes


    @property
    def get_formatted_user_dataframe(self ):
        return self.formatted_user_dataframe


    @property
    def get_user_columns_description(self ):
        return self.dfr_columns_descriptions


    def _extract_columns_descriptions_from_first_row(
            self,
            dataframe: pd.DataFrame
            ) -> (dict, pd.DataFrame):
        """
        Extracts the description of the columns from the data frame first data row, and delete the data row
        """
        row_with_descriptions = dataframe.loc[0]
        dataframe.drop([0], inplace=True)
        dataframe.index -= 1

        row_with_descriptions[pd.isna(row_with_descriptions)] = ""

        return row_with_descriptions.to_dict() ,  dataframe

    def _read_dataframe_title_desc_from_file_path(
            self,
            file_path: str,
            decimal_separator: str,
            date_format: str,
    ) -> pd.DataFrame:
        """
        Reads user_dataframe_to_extend from the data file using file path,
        and if there are no column headers in the user_dataframe_to_extend, it create them

        :param file_path: path to the data file
        :param decimal_separator: . or , only
        :param date_format: DMY or MDY only

        :return: read user_dataframe_to_extend
        """


        def _move_back_inside_dataframe_the_column_names_and_create_columns_names(
            dataframe_to_add_titles: pd.DataFrame
        ) -> pd.DataFrame:
            """
            Adds columns title to dataframe_to_add_titles, also create first row of data because the current columns titles are first row data

            :params dataframe_to_add_titles: the dataframe where we add new columns titles
            """
            if not isinstance(dataframe_to_add_titles, pd.DataFrame ):
                logger.error(
                    f"The user_dataframe_to_extend must be of type 'pd.DataFrame', but received '{type(dataframe_to_add_titles )}'"
                )

            new_titles = [f"Column_{index + 1}" for index in range(dataframe_to_add_titles.shape[1 ] ) ]

            dataframe_tmp = dataframe_to_add_titles.copy( )

            # Insert the previous row of column titles before the user_dataframe_to_extend
            dataframe_tmp.index += 1
            dataframe_tmp.loc[0] = dataframe_tmp.columns
            dataframe_tmp.sort_index(inplace=True)

            dataframe_tmp.columns = list(new_titles)

            return dataframe_tmp
            # ----------------- end of _add_in_pandas_the_title_row

        # ----- start _read_dataframe_from_file_path --------------------------------
        if not os.path.exists(file_path):
            logger.error( f"DataFileReader can not find the file using this path: {file_path}" )

        if os.stat(file_path).st_size > 1024 * 1024 * 100:
            logger.warning( "file t {file_path}  size is bigger than 100 mb")

        dataframe = None
        if file_path.endswith(("csv", "tsv", "txt")):
            dataframe = self._read_txt_file(file_path, decimal_separator=decimal_separator, date_format=date_format)

        elif file_path.endswith("xlsx"):
            dataframe = pd.read_excel(file_path, engine="openpyxl")

        elif file_path.endswith("xls"):
            dataframe = pd.read_excel(file_path)

        else:
            logger.error(f"file extension should be: txt csv tsv, xls, xlsx ")

        if dataframe.shape[1] < 2:
            logger.error( f"Dataset {file_path} have less than 2 columns")

        # titles are already inside the pandas columns titles, but it can be wrong and we may need to move back titles to data !!!!!
        if dataframe.columns.to_series().dropna().apply(DataFileReader.is_number, decimal_separator=decimal_separator ).any( ):
            # number detected in the first row , so it is not titles
            dataframe = _move_back_inside_dataframe_the_column_names_and_create_columns_names( dataframe )
            if ENABLE_LOGGER_DEBUG_DataFileReader:
                logger.debug( "DataFileReader does not detect the header string, so it created the header with titles itself" )

        # if the first data row do contains only strings or null => then it is a description row !
        if dataframe.iloc[ 0 ].apply( lambda x: isinstance( x, str ) or pd.isnull( x ) ).all( ):
            # there is no numbers/dates/objects in any cell in the first data row then it is a description row
            self.dfr_columns_descriptions , dataframe = self._extract_columns_descriptions_from_first_row( dataframe )
        else:
            # there is no description row in the user_dataframe_to_extend
            self.dfr_columns_descriptions = dict.fromkeys(dataframe.columns, "" )

        return dataframe




    # =========================================================================================================
    # =========================================================================================================
    # =========================================================================================================

    @staticmethod
    def pd_series_apply( pd_series:pd.Series, the_function, *args ):
        """
        Apply have a bug https://github.com/pandas-dev/pandas/issues/33876 so we use this function instead
        """
        return pd.Series( [ the_function( value, *args ) for value in pd_series ], index=pd_series.index )
        # result = pd.Series( dtype=pd_series.dtype )
        # for index, value in pd_series.items( ):
        #     result.at[ index ] = the_function( value, *args )
        # return result


    @staticmethod
    def _convert_to_number(
            value,
            decimal_separator: str
    ) -> float:
        """
        Converts value from pandas cell to float

        :param value: cell of pandas to convert
        :param decimal_separator: . or , only

        :return: converted value otherwise None

        """

        if pd.isnull( value ):
            return None

        elif isinstance(value, float):
            return value

        elif isinstance(value, (bool, int, np.int64, float , np.float_, np.bool_ , np.int_ , np.float_ ) ):
            return float(value)

        elif isinstance(value, str):
            value = value.strip("'\"$£€").replace( " " , "" )
            if value == "":
                return None
            if decimal_separator != ".":
                value = value.replace( ".", "", 1 )
            try:
                value_float = float( value )
            except:
                try:
                    if re.match(
                                r'^\d+(\.\d+)? ?('+NUMBERS_PARSER_SUFFIX_UNITS_REGEX+')s?$',
                                value, re.IGNORECASE ):
                        pattern = re.compile( r'^\s*([\d,.]+)\s*([\w]+)\s*$' )
                        match = pattern.match( value.lower( ) )
                        number = float( match.group( 1 ) )
                        unit = match.group( 2 )
                        if unit.endswith( 's' ):
                            unit = unit[ :-1 ]
                        value_float = number * NUMBERS_PARSER_SUFFIX_UNITS_MULTIPLIERS[ unit.lower( ) ]
                    elif re.match(
                                r'^\d+(\.\d+)? ?(KB|kilobyte|KiB|kibibyte|MB|megabyte|MiB|mebibyte|GB|gigabyte|GiB|gibibyte|TB|terabyte|TiB|tebibyte|PB|petabyte|PiB|pebibyte|EB|exabyte|EiB|exbibyte|ZB|zettabyte|ZiB|zebibyte|YB|yottabyte|YiB|yobibyte)$',
                                value, re.IGNORECASE ):
                        value_float = parse_size( value )
                    elif re.match( r'^\d+(\.\d+)? ?(s|sec|second|seconds|m|min|minute|minutes|h|hour|hours|d|day|days|w|week|weeks|month|months|y|year|years)$', value, re.IGNORECASE ):
                        value_float = parse_timespan( value )
                    elif re.match(
                                r'^\d+(\.\d+)? ?(nanosecond|nanoseconds|ns|microsecond|microseconds|us|millisecond|milliseconds|ms|second|seconds|s|sec|secs|minute|minutes|m|min|mins|hour|hours|h|day|days|d|week|weeks|w|year|years|y)$',
                                value, re.IGNORECASE ):
                        value_float = parse_length( value )
                    elif re.match(
                                fr'^\d+(\.\d+)? ?({NUMBERS_PARSER_CURRENCY_REGEX})$',
                                value, re.IGNORECASE ):
                        value_float = convert_string_to_float_in_usd( value )
                    else:
                        raise ValueError(f"Unable to convert str value '{value}' to float , no method found")
                except Exception as e:
                    raise ValueError( f"Error Unable to convert str value '{value}' to float because {e}" )
                return value_float
            else:
                return value_float

        else:
            raise ValueError( f"Error Unable to convert str value '{value}' to float because type is : {type(value)}" )


    @staticmethod
    def _convert_to_text(
            value,
    ) -> str:
        """
        Converts value from pandas cell to text

        :param value: cell of pandas to convert

        :return: text converted value
        """
        if pd.isnull( value ):
            return None
        elif isinstance(value, str):
            return value.strip( )
        else:
            return str(value).strip( )


    @staticmethod
    def _convert_to_datetime(
            value,
            date_format: str
    ) -> datetime.datetime:
        """
        Converts value from pandas cell to date

        :param value: cell of pandas to convert
        :param date_format: DMY MDY YMD

        :return: datetime.datetime converted value otherwise None

        """
        if pd.isnull( value ):
            return None
        elif isinstance(value, datetime.datetime):
            return value.replace(tzinfo=None)
        elif isinstance(value, str):
            return dateutilparser.parse(
                        str(value),
                        dayfirst=(date_format == "DMY"),
                        yearfirst=(date_format == "YMD"),
                        ignoretz=True )
        else:
            raise ValueError(f"Unable to convert value ({value})  - {type(value)} to datetime")


    @staticmethod
    def _convert_to_date(
            value,
            date_format: str
    ) -> datetime.date:
        """
        Converts value from pandas cell to date

        :param value: cell of pandas to convert
        :param date_format: DMY MDY YMD

        :return: datetime.date converted value otherwise None

        """

        if pd.isnull( value ):
            return None
        elif isinstance(value, datetime.date):
            return value
        if isinstance(value, datetime.datetime):
            return value.date()
        elif isinstance(value, str):
            return dateutilparser.parse(value, dayfirst=(date_format == "DMY"), yearfirst=(date_format == "YMD"), ignoretz=True ).date()
        else:
            raise ValueError(f"Unable to convert value ({value}) - {type(value)} to date")


    @staticmethod
    def _convert_to_time(
            value,
    ) -> datetime.time:
        """
        Converts value from pandas cell to time

        :param value: cell of pandas to convert

        :return: datetime.time converted value otherwise None
        """

        if pd.isnull( value ):
            return None
        elif isinstance(value, datetime.time):
            return value
        elif isinstance(value, datetime.datetime):
            return value.time()
        elif isinstance(value, str):
            return dateutilparser.parse(str(value), ignoretz=True ).time()
        else:
            raise ValueError(f"Unable to convert value ({value}) to time because it is type {type(value)}")


    @staticmethod
    def _convert_to_json(
            value,
            date_format: str,
            decimal_separator: str,
    ) -> str:
        """
        Converts value from pandas cell to json
        json is text, but inside json , we need to format dates in standard pandas : YYYY-MM-DD , and float with a DOT as decimal separator

        :param value: cell of pandas to convert

        :return: converted value
        """
        if pd.isnull( value ):
            return None
        elif isinstance(value, str):
            jsonvalue = value
        else:
            jsonvalue = str(value)

        # TODO  inside json , we need to format dates in standard pandas : YYYY-MM-DD , and float with a DOT as decimal separator

        return jsonvalue


    @staticmethod
    def is_number(
            value,
            decimal_separator: str,
    ) -> bool:
        """Returns True if the input value is a number, otherwise return False"""

        if isinstance(value,  (bool, int, float , np.bool_ , np.int_ , np.float_ ) ):
            return True
        else:
            try:
                v = DataFileReader._convert_to_number(value , decimal_separator )
            except:
                return False
            else:
                return True


    @staticmethod
    def is_text( value: str ) -> bool:
        """
        returns True if the input is text
        """

        if isinstance(value, str):
            if value.strip() == "":
                raise ValueError(f"Value is '{value}' oups we should have drop the none and empty string) )")
            else:
                return True

        return False


    @staticmethod
    def is_label( value: str ) -> bool:
        """
        returns True if the input is label => do not contains comma or dot-comma or /
        """
        if isinstance(value, str):
            return True
        else:
            return False


    @staticmethod
    def is_time(
            value: str) -> bool:
        """Returns True if the input value is a string that has a time format, otherwise return False"""

        value_str = str(value).strip()

        if value_str.strip() == "":
            raise ValueError(f"Value is '{value}' oups we should have drop the none and empty string) )")

        # TODO we need to remove this , this is only to avoid the debugger stoping at each exception
        if ( not value_str[0].isdigit() and not value_str[-1].isdigit() ) or len(value_str) > 11:
            return False

        try:
            valuetime = dateutilparser.parse( value_str, default=datetime.datetime( 1, 1, 1, 0, 0 ), ignoretz=True )
        except ValueError:
            return False
        else:
            # No exception , so do we have the default date ?
            if not (valuetime.year == 1 and valuetime.month == 1 and valuetime.day == 1):
                return False
            # if we do not have default time, then it is a TIME
            if not (valuetime.hour == 0 and valuetime.minute == 0 and valuetime.second == 0 and valuetime.microsecond == 0):
                return True
            # we have detected the DEFAULT time or midnight ?
            valuetime = dateutilparser.parse( value_str, default=datetime.datetime( 1, 1, 1, 2, 2, 2, 2 ), ignoretz=True )
            if not ( valuetime.hour == 2 and valuetime.minute == 2 and valuetime.second == 2 and valuetime.microsecond == 2 ):
                return True

            return False


    @staticmethod
    def is_date(
            value,
            date_format: str) -> bool:
        """Returns True if the input value is a string that has a date format, otherwise return False"""

        value_str = str(value).strip()

        if value_str.strip( ) == "":
            raise ValueError(f"Value is '{value}' oups we should have drop the none and empty string) )")
        # TODO we need to remove this , this is only to avoid the debugger stoping at each exception
        if not value_str[0].isdigit() and not value_str[-1].isdigit() :
            return False
        try:
            valuedate = dateutilparser.parse(value_str, dayfirst=(date_format == "DMY"),
                                                yearfirst=(date_format == "YMD"),
                                                default=datetime.datetime(1, 1, 1, 0, 0 , 0 , 0),
                                                ignoretz=True )
        except ValueError:
            return False
        else:
            #do we have a time ? Yes => not a Date !
            if not (valuedate.hour == 0 and valuedate.minute == 0 and valuedate.second == 0 and valuedate.microsecond == 0):
                return False
            # if we have a DATE not == DEFAULT then it is a DATE
            if not ( valuedate.year == 1 and valuedate.month == 1 and valuedate.day == 1 ):
                return True
            # we may have a DateTime with MIDNIGHT !
            valuedate = dateutilparser.parse( value_str, dayfirst=(date_format == "DMY"),
                            yearfirst=(date_format == "YMD"),
                            default=datetime.datetime( 2, 2, 2, 3, 3, 3 , 3 ),
                            ignoretz=True )
            #do we have a time ? Yes => not a Date !
            if not (valuedate.hour == 3 and valuedate.minute == 3 and valuedate.second == 3 and valuedate.microsecond == 3):
                return False
            # do we have default date ? => not a date
            if ( valuedate.year == 2 and valuedate.month == 2 and valuedate.day == 2 ):
                # Date default values detected , so it is not a complete datetime string
                return False

            # it is not the default values, so it is a date
            return True


    @staticmethod
    def is_datetime(
            value,
            date_format: str) -> bool:
        """Returns True if the input value is a string that has a datetime format, otherwise return False"""

        value_str = str(value).strip()

        if value_str.strip( ) == "":
            raise ValueError(f"Value is '{value}' oups we should have drop the none and empty string) )")
        # TODO we need to remove this , this is only to avoid the debugger stoping at each exception
        if not value_str[0].isdigit() and not value_str[-1].isdigit() :
            return False
        try:
            valuedate = dateutilparser.parse(value_str,
                                            dayfirst=(date_format == "DMY"),
                                            yearfirst=(date_format == "YMD"),
                                            default=datetime.datetime(1, 1, 1, 0, 0, 0 , 0),
                                            ignoretz=True )
        except ValueError:
            return False
        else:
            # No exception so we will now detect if the result is the DEFAULT or the REAL parsed data
            if (not (valuedate.year == 1 and valuedate.month == 1 and valuedate.day == 1) and
                not ( valuedate.hour == 0 and valuedate.minute == 0 and valuedate.second == 0 and valuedate.microsecond == 0) ):
                return True
            # we have detected Date==default or Time=Default => We test more
            valuedate = dateutilparser.parse( value_str, dayfirst=(date_format == "DMY"),
                            yearfirst=(date_format == "YMD"),
                            default=datetime.datetime( 2, 2, 2, 3, 3, 3, 3 ),
                            ignoretz=True )
            if (not (valuedate.year == 2 and valuedate.month == 2 and valuedate.day == 2) and
                not ( valuedate.hour == 3 and valuedate.minute == 3 and valuedate.second == 3 and valuedate.microsecond == 3) ):
                return True

            # stil ldetected DEFAULT => not a datetime
            return False


    @staticmethod
    def is_json(
            value) -> bool:
        """Returns True if the input value is a JSON, otherwise return False"""

        if isinstance(value, (list, dict)):
            return True

        value_str = str(value).strip()

        if value_str.strip( ) == "":
            raise ValueError(f"Value is '{value}' oups we should have drop the none and empty string) )")

        if not "[" in value_str[:5] and not "{" in value_str[:5]:
            return False
        else:
            try:
                json.loads(value_str)
            except:
                try:
                    null = None
                    one_dict = ast.literal_eval(value_str)
                except:
                    return False
                else:
                    return True
            else:
                return True


    @staticmethod
    def detect_6_languages_percentage_from_serie( data_serie: pd.Series ) -> (float, float, float, float, float, float, float, float):
        """
        from all the texts in argument, will detect 6 languages and will return the percentage of this language

        :params data_serie: the list of text to analyse
        :return: 8 float percentage for this languages : { 'en':0 , 'fr':0 , 'de':0 , 'it':0, 'es':0 , 'pt':0 , 'others':0 , 'none':0 }
        """

        def detect_languages_text( text ):
            if len( text ) <= 2 or not (any(char.isalpha() for char in text)):
                return None
            try:
                languages_found = langdetect.detect_langs( text )
            except:
                return None
            if len( languages_found ) == 0:
                return None
            else:
                for detected_result in languages_found:
                    # a sentence is usually 0.99 but if there is some words unknown 0.95 will give some margin
                    if detected_result.prob > 0.95:
                        if detected_result.lang == "ca":
                            return "fr"  # they put ca instead of fr !
                        else:
                            return detected_result.lang
                return None

        all_detected_languages = { 'en': 0, 'fr': 0, 'de': 0, 'it': 0, 'es': 0, 'pt': 0, 'others': 0, 'none': 0 }
        # to speed up we use only 100 rows with a lenght of 40 chars max
        data_serie_sampled = pd.Series( data_serie ).sample( min( len(data_serie) , 100 ) ).str[ :40 ].unique()
        for one_sentence in data_serie_sampled:
            language_detected = detect_languages_text( one_sentence )  # to speed up we take only the first part of the sentence
            if language_detected is None:
                all_detected_languages[ 'none' ] += 1
            elif language_detected in all_detected_languages:
                all_detected_languages[ language_detected ] += 1
            else:
                all_detected_languages[ 'others' ] += 1

        for language_detected in all_detected_languages:
            all_detected_languages[ language_detected ] = 0 if len( data_serie_sampled )==0 else all_detected_languages[ language_detected ] / len( data_serie_sampled )

        return all_detected_languages


    @staticmethod
    def determine_column_datatype(
            column_data: pd.Series,
            decimal_separator: str,
            date_format: str,
    ) -> DatasetColumnDataType:
        """
        Determines the type of column datatype based on the column data by checking all cells of the column

        :param column_data: column data to analyze
        :param decimal_separator: . or , only
        :param date_format: DMY or MDY only

        :return: column data type

        """


        def strip_spaces( element ):
            if isinstance( element, str ):
                return element.replace( u'\xa0', ' ' ).strip( ' "\'()' )
            else:
                return element


        if not isinstance(column_data, pd.Series):
            logger.error(f"The column must be of type 'pd.Series', but received '{type(column_data)}'")

        # remove all empty string, string with only spaces and NaN
        column_data_without_none = column_data.apply( strip_spaces ).replace( '', np.nan ).dropna()

        if len(column_data_without_none) == 0:
            if ENABLE_LOGGER_DEBUG_DataFileReader:
                logger.debug( f"Analysing column : {column_data.name} -> DatasetColumnDataType.IGNORE" )
            return DatasetColumnDataType.IGNORE

        factor_repeated_unique_values_for_label = 0
        # optimization for big files, we will test only unique values
        try:
            # exception when the pandas series contains json/dict, so we just skip this step if we cannot do unique( )
            column_data_unique_values_to_test = pd.Series( column_data_without_none.unique() )
        except:
            # in case of error we will check all data
            column_data_unique_values_to_test = column_data_without_none
        finally:
            factor_repeated_unique_values_for_label = len( column_data_without_none ) / len( column_data_unique_values_to_test )

        for index, value in column_data_unique_values_to_test.items():
            if not DataFileReader.is_number( value, decimal_separator=decimal_separator ):
                break
        else:
            if ENABLE_LOGGER_DEBUG_DataFileReader:
                logger.debug( f"Analysing column : {column_data.name} -> DatasetColumnDataType.FLOAT" )
            return DatasetColumnDataType.FLOAT

        for index, value in column_data_unique_values_to_test.items():
            if not DataFileReader.is_time( value ):
                break
        else:
            if ENABLE_LOGGER_DEBUG_DataFileReader:
                logger.debug( f"Analysing column : {column_data.name} -> DatasetColumnDataType.TIME" )
            return DatasetColumnDataType.TIME

        for index, value in column_data_unique_values_to_test.items():
            if not DataFileReader.is_date( value, date_format=date_format ):
                break
        else:
            if ENABLE_LOGGER_DEBUG_DataFileReader:
                logger.debug( f"Analysing column : {column_data.name} -> DatasetColumnDataType.DATE" )
            return DatasetColumnDataType.DATE

        for index, value in column_data_unique_values_to_test.items():
            if not DataFileReader.is_datetime( value, date_format=date_format ):
                break
        else:
            if ENABLE_LOGGER_DEBUG_DataFileReader:
                logger.debug( f"Analysing column : {column_data.name} -> DatasetColumnDataType.DATETIME" )
            return DatasetColumnDataType.DATETIME

        for index, value in column_data_unique_values_to_test.items():
            if not DataFileReader.is_json( value ):
                break
        else:
            if ENABLE_LOGGER_DEBUG_DataFileReader:
                logger.debug( f"Analysing column : {column_data.name} -> DatasetColumnDataType.JSON" )
            return DatasetColumnDataType.JSON

        # here we check if the string columns is  labels
        uniques_values_with_count = column_data_unique_values_to_test.value_counts( )
        maximum_len_of_str_values = column_data_unique_values_to_test.str.len( ).max( )
        if (        len( uniques_values_with_count ) <= DETECTION_LABEL_MAXIMUM_UNIQUES_VALUES and
                    maximum_len_of_str_values <= DETECTION_LABEL_MAXIMUM_STR_LEN_VALUES
        ):
            for index, value in column_data_unique_values_to_test.items():
                if not DataFileReader.is_label( value ):
                    break
            else:
                if ENABLE_LOGGER_DEBUG_DataFileReader:
                    logger.debug( f"Analysing column : {column_data.name} -> DatasetColumnDataType.LABEL" )
                return DatasetColumnDataType.LABEL

        # here we check if the string columns is language
        for index, value in column_data_unique_values_to_test.items():
            if not DataFileReader.is_text( value ):
                break
        else:
            # on language we will test 100 random data maximum because it is slow and we do not break before the end
            samples_values_uniques = uniques_values_with_count.sample( min( len( uniques_values_with_count ), 100 ) ).index
            # detection of language is not reliable with less than a few rows
            if len( samples_values_uniques ) >= DETECTION_LANGUAGE_MINIMUM_ROWS:
                languages_detected = DataFileReader.detect_6_languages_percentage_from_serie( samples_values_uniques )
                if languages_detected:
                    if languages_detected["en"]+languages_detected["fr"]+languages_detected["de"]+languages_detected["it"]+languages_detected["es"]+languages_detected["pt"] >= (DETECTION_LANGUAGE_MINIMUM_PERCENTAGE/100):
                        if ENABLE_LOGGER_DEBUG_DataFileReader:
                            logger.debug( f"Analysing column : {column_data.name} -> DatasetColumnDataType.LANGUAGE" )
                        return DatasetColumnDataType.LANGUAGE

        logger.warning(f"Unable to detect datatype of the column  '{column_data.name}'  --- Analysed {len(column_data_unique_values_to_test)} values for column : {column_data.sample( min( len(column_data) , 3 ) ).tolist()} ---> column type is IGNORE ")
        return DatasetColumnDataType.IGNORE


    @staticmethod
    def reformat_pandas_cells_by_columns_datatypes(
            dataframe_to_format: pd.DataFrame,
            columns_datatypes: dict,
            decimal_separator: str,
            date_format: str,
    ) -> pd.DataFrame:
        """
        Normalizes dataframe  by unicodedata.normalize function
        with Normalization Form Compatibility Composition (NFKC) unicode equivalence

        :param dataframe_to_format: this is user_dataframe_to_extend to be formatted with columns_datatypes
        :param columns_datatypes: the datatype of the column of the dataframe to reformat
        :param decimal_separator: . or , only
        :param date_format: DMY or MDY only

        :return: normalized dataframe
        """
        if ENABLE_LOGGER_DEBUG_DataFileReader:
            logger.debug(f"Reformat DF : {dataframe_to_format.shape[ 0 ]} rows X {dataframe_to_format.shape[ 1 ]} cols " )

        for this_column_name in dataframe_to_format.columns:
            if not this_column_name in columns_datatypes:
                logger.error( f"The column '{this_column_name}' exist in the dataframe but not in columns_datatypes {columns_datatypes} ")
            this_column_datatype = columns_datatypes[this_column_name]

            if this_column_datatype is DatasetColumnDataType.FLOAT:
                try:
                    dataframe_to_format[ this_column_name ] = DataFileReader.pd_series_apply(
                                dataframe_to_format[this_column_name]  ,
                                DataFileReader._convert_to_number,
                                decimal_separator )
                    # dataframe_to_format[this_column_name] = dataframe_to_format[this_column_name].apply(
                    #     lambda cell: DataFileReader._convert_to_number(cell, decimal_separator) if pd.notnull(
                    #         cell) else None                    )
                except ValueError as err:
                    logger.error(f"Error '{err}' -- Unable to convert all the column '{ this_column_name }' in detected type {this_column_datatype}")

            elif this_column_datatype is DatasetColumnDataType.DATETIME:
                try:
                    dataframe_to_format[ this_column_name ] = DataFileReader.pd_series_apply(
                                dataframe_to_format[this_column_name]  ,
                                DataFileReader._convert_to_datetime ,
                                date_format )
                    # dataframe_to_format[this_column_name] = dataframe_to_format[this_column_name].apply(
                    #     lambda cell: DataFileReader._convert_to_datetime(cell, date_format)                    )
                except ValueError as err:
                    logger.error(f"Error '{err}' -- Unable to convert all the column '{ this_column_name }' in detected type {this_column_datatype}")

            elif this_column_datatype is DatasetColumnDataType.DATE:
                try:
                    dataframe_to_format[ this_column_name ] = DataFileReader.pd_series_apply(
                                dataframe_to_format[this_column_name]  ,
                                DataFileReader._convert_to_date ,
                                date_format )
                    # dataframe_to_format[this_column_name] = dataframe_to_format[this_column_name].apply(
                    #     lambda cell: DataFileReader._convert_to_date(cell, date_format)                    )
                except ValueError as err:
                    logger.error(f"Error '{err}' -- Unable to convert all the column '{ this_column_name }' in detected type {this_column_datatype}")

            elif this_column_datatype is DatasetColumnDataType.TIME:
                try:
                    dataframe_to_format[ this_column_name ] = DataFileReader.pd_series_apply(
                                dataframe_to_format[this_column_name],
                                DataFileReader._convert_to_time )
                    # dataframe_to_format[this_column_name] = dataframe_to_format[this_column_name].apply(
                    #     lambda cell: DataFileReader._convert_to_time(cell)                    )
                except ValueError as err:
                    logger.error(f"Error '{err}' -- Unable to convert all the column '{ this_column_name }' in detected type {this_column_datatype}")

            elif this_column_datatype is DatasetColumnDataType.LABEL or this_column_datatype is DatasetColumnDataType.LANGUAGE:
                try:
                    dataframe_to_format[ this_column_name ] = DataFileReader.pd_series_apply(
                                dataframe_to_format[this_column_name],
                                DataFileReader._convert_to_text )
                    # dataframe_to_format[this_column_name] = dataframe_to_format[this_column_name].str.strip()
                    # dataframe_to_format[this_column_name] = dataframe_to_format[this_column_name].apply(
                    #     lambda cell: DataFileReader._convert_to_text(cell)                   )
                except ValueError as err:
                    logger.error(f"Error '{err}' -- Unable to convert all the column '{ this_column_name }' in detected type {this_column_datatype}")

            elif this_column_datatype is DatasetColumnDataType.JSON:
                try:
                    dataframe_to_format[ this_column_name ] = DataFileReader.pd_series_apply(
                                dataframe_to_format[ this_column_name ],
                                DataFileReader._convert_to_json,
                                decimal_separator,
                                date_format
                                )
                    # dataframe_to_format[this_column_name] = dataframe_to_format[this_column_name].apply(
                    #     lambda cell: DataFileReader._convert_to_json(cell, decimal_separator,
                    #                                                  date_format)
                except ValueError as err:
                    logger.error(f"Error '{err}' -- Unable to convert all the column '{ this_column_name }' in detected type {this_column_datatype}")

            elif this_column_datatype is DatasetColumnDataType.IGNORE:
                pass

            else:
                logger.error(f"Cannot process unknown column datatype '{this_column_datatype}' of column '{ this_column_name }' ! ")

        return dataframe_to_format


    @staticmethod
    def _read_txt_file(
            file_path: str,
            decimal_separator: str,
            date_format: str,
    ) -> pd.DataFrame:
        """
        Reads user_dataframe_to_extend from the data file with unknown value separator using file path

        :param file_path: path to the data file
        :param decimal_separator: . or , only
        :param date_format: DMY, MDY, or YMD

        :return: read user_dataframe_to_extend

        """
        if not os.path.exists(file_path):
            raise FileExistsError(
                f"DataFileReader can not find the file using this path: {file_path}"
            )

        try:
            # Detect the encoding of the CSV file
            with open(file_path, 'rb' ) as f:
                encoding_detected = chardet.detect( f.read( ) )

            # For YMD format, pandas doesn't support yearfirst, so dates will be read as strings
            # and parsed later by conversion functions. The dayfirst parameter only affects
            # pandas' automatic date parsing, which is disabled by default.
            dataframe = pd.read_csv(
                        file_path,
                        skipinitialspace=True,
                        dayfirst=(date_format == "DMY"),
                        decimal=decimal_separator,
                        encoding=encoding_detected[ 'encoding' ]
            )
            return dataframe
        except ParserError as error:
            logger.error(f"Unable to read this csv file. because {error}. ")


@staticmethod
def convert_string_to_float_in_usd( text_to_convert ):

    # Split the input text into individual words
    words = re.split( r'((?:\d{1,3}(?:,\d{3})*(?:\.\d+)?|\.\d+)|\D+)' , text_to_convert )  # Split the string using digit/non-digit groups
    words = [s.strip().upper() for s in words if s]

    if len( words ) != 2:
        raise ValueError( f"Unable to convert '{text_to_convert}' to value in USD because there is not 2 parts but {len( words )}" )

    if words[0] in NUMBERS_PARSER_CURRENCY_LIST:
        c = 0
        n = 1
    elif words[1] in NUMBERS_PARSER_CURRENCY_LIST:
        c = 1
        n = 0
    else:
        raise ValueError( f"Unable to find a known currency in '{text_to_convert}' " )

    if words[ c ] in NUMBERS_PARSER_CURRENCY_SYMBOL_CONV.keys():
        currency_code = NUMBERS_PARSER_CURRENCY_SYMBOL_CONV[ words[ c ] ]
    else:
        currency_code = words[ c ]

    # Get the exchange rate and convert the amount to USD
    exchange_rate = numbers_parser_currency_forex_currency_rates.get_rate( currency_code, 'USD' )
    amount = float(words[ n ].replace(',', ''))
    return exchange_rate * amount





