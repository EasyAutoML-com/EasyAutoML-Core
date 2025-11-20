from abc import ABC, abstractmethod
from collections import namedtuple
import datetime
from typing import NoReturn
from typing import Union, Optional
import pickle
import base64
import numpy as np
import pandas as pd
from scipy.stats import truncnorm
from functools import lru_cache

from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    LabelEncoder,
    StandardScaler,
    RobustScaler,
    PowerTransformer,
    QuantileTransformer,
)

from ML import EasyAutoMLDBModels, __getlogger
from SharedConstants import DatasetColumnDataType


logger = __getlogger()


# ------sentence_transformer-----------to make faster , we load the model and the configuration one single time here---------------------------------
# IMPORTANT: Load sentence transformer on CPU to avoid GPU conflicts with TensorFlow
# 
# WHY THIS IS NECESSARY:
# - sentence-transformers uses PyTorch under the hood, which tries to manage GPU resources
# - Our neural network training (NNEngine) uses TensorFlow/Keras, which also needs GPU access
# - When both frameworks are loaded simultaneously and try to access the GPU, they conflict
# - This causes crashes with errors like "Unhandled exception caught in c10/util/AbortHandler.h"
# - The conflict occurs during GPU resource cleanup/destruction when both frameworks are active
#
# WHY CPU IS ACCEPTABLE:
# - SentenceTransformer is only used for text feature encoding (not training)
# - Text encoding is relatively fast on CPU and doesn't need GPU acceleration
# - Neural network training (which benefits most from GPU) can use the GPU exclusively
# - This separation ensures stable operation without performance degradation for training
#
#sentence_transformer_model = SentenceTransformer( 'sentence-transformers/LaBSE' )    # multilingual - 768 Vectors
#sentence_transformer_model_vectors_len = 768
sentence_transformer_model = SentenceTransformer( 
    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    device='cpu'  # Force CPU usage to prevent PyTorch/TensorFlow GPU conflicts
)    # multilingual - 384 Vectors
sentence_transformer_model_vectors_len = 384
embeddings = sentence_transformer_model.encode(
            [ "This", "is", "a", "sentence", "example", "to initialize the MinMax Scaler.", "Une autre phrase en Français",
                "Auf Deutsch", "rtf1ger nb2cvd", "wXCvf 123 bx4dgf G6bc", "center",
                "CENTER",
                "The curious cat cautiously crept closer to the cozy couch, then pounced on a playful piece of string, unraveling it into a tangled mess, while the mischievous dog barked loudly and the sleepy hamster snuggled in its cage, oblivious to the commotion." ] )
# we define min value, but we adjust it twice lower , to have a margin
sentence_transformer_model_vectors_value_min = np.min( embeddings )
if sentence_transformer_model_vectors_value_min < 0:
    sentence_transformer_model_vectors_value_min *= 2
else:
    sentence_transformer_model_vectors_value_min /= 2
# we define value max but we adjust it twice higher , to have a margin
sentence_transformer_model_vectors_value_max = np.max( embeddings )
if sentence_transformer_model_vectors_value_max > 0:
    sentence_transformer_model_vectors_value_max *= 2
else:
    sentence_transformer_model_vectors_value_max /= 2
# ------sentence_transformer----------------------------------------------------------------------------------------------------------------


# in ALL FET , all None or label unknown or out of range values are replaced by NAN
# then all FET call super and finaly reach FETNumericMinMaxFloat which replace all NAN by this value
# So any value other than 0..1 will be replaced by this value
FETNumericMinMaxFloat_ENCODE_UNKNOWN_NAN_WITH_THIS_VALUE = 0.5


# The multiplexer should not create less column than indicated below
FETMultiplexer_MIN_UNIQUE_VALUES_TO_ENABLE = 3
# The multiplexer should not create more column than indicated below
FETMultiplexer_MAX_COLUMNS_TO_CREATE = 20

# for FETMultiplexerMostFrequentsValues how many most frequent values maximum we will use
FETMultiplexerMostFrequentsValues_MAX_COLUMNS_TO_CREATE = 15
FETMultiplexerMostFrequentsValues_MIN_MOST_FREQUENT_VALUES_COUNT_TO_ENABLE = 1

# The multiplexer should not create less column than indicated below
FET3FrequencyLevel_MIN_UNIQUE_VALUES_TO_ENABLE = 10

# The power scaler will use these power coefficients to data processing
# Symbols '_' in the name will be replaced by dot (1_5 -> 1.5)
FET6Power_SCALER_COEFFICIENTS_POWER = ["0_33", "0_5", "0_66", "1_5", "2", "3"]

FETSentenceTransformer_MINIMUM_PERCENTAGE_6_LANGUAGE = 75
# FETSentenceTransformer will be actived if the percentage of rows having one of 6 language is more than this number


# when generating names of encoded columns we use this separator
SEPARATOR_IN_COLUMNS_NAMES = "-"

MDC_CLUSTERING_MINIMUM_UNIQUE_VALUES_TO_ENABLE = 30

# this namedtuple contains data about a column - it is used to pass all column data to some functions
# we use it to pass information in FEC , so FEC can use this information to decide to activate or not and more
Column_datas_infos = namedtuple(
    "Column_datas_infos",
    [
        "name",
        "is_input",
        "is_output",
        "datatype",
        "description_user_df",
        "unique_value_count",
        "missing_percentage",
        "most_frequent_values_count",
        "min",
        "max",
        "mean",
        "std_dev",
        "skewness",
        "kurtosis",
        "quantile02",
        "quantile03",
        "quantile07",
        "quantile08",
        "sem",
        "median",
        "mode",
        "str_percent_uppercase",
        "str_percent_lowercase",
        "str_percent_digit",
        "str_percent_punctuation",
        "str_percent_operators",
        "str_percent_underscore",
        "str_percent_space",
        "str_language_en",
        "str_language_fr",
        "str_language_de",
        "str_language_it",
        "str_language_es",
        "str_language_pt",
        "str_language_others",
        "str_language_none",
        "fet_list",
    ],
)

# complete list of fet that can be enabled or not by the FEC _set_configuration_best_having_column_budget
LIST_ALL_FET_NAMES_FEC_SELECTABLE = [
    "FETNumericMinMax",
    "FETNumericStandard",
    "FETNumericPowerTransformer",
    "FETNumericQuantileTransformer",
    "FETNumericQuantileTransformerNormal",
    "FETNumericRobustScaler",
    "FETMultiplexerAll",
    "FETMultiplexerMostFrequentsValues",
    "FET6Power",
    "FETSentenceTransformer",
]

# when using minimum simple FE configuration, we will enable all this FET for all columns ( FET listed will be enabled if they are compatible with INPUT or OUTPUT )
#FEC_CONFIGURATION_SIMPLE_MINIMUM_LIST_FET = [ "FETNumericMinMax", "FETNumericStandard", "FETNumericRobustScaler", ]
#FEC_CONFIGURATION_SIMPLE_MINIMUM_LIST_FET = LIST_ALL_FET_NAMES_FEC_SELECTABLE    # TODO remove this is only for debug
FEC_CONFIGURATION_SIMPLE_MINIMUM_LIST_FET = [ "FETNumericStandard", "FETSentenceTransformer" ]

# when the budget is equal or below this value, the column will have automatically the FEC_CONFIGURATION_SIMPLE_MINIMUM_LIST_FET
# should be the cost of the FEC_CONFIGURATION_SIMPLE_MINIMUM_LIST_FET
FEC_SIMPLE_MINIMUM_CONFIGURATION_WHEN_BUDGET_BELOW = ( 1 )

# when a column can have missing values it will always activate this FET
FET_TO_ENABLE_WHEN_NONE_PRESENT = "FETIsNone"

# this is all the FET can be enabled - but FETIsNone will always be enabled for all columns with missing values
LIST_ALL_FET_NAMES = LIST_ALL_FET_NAMES_FEC_SELECTABLE + [ FET_TO_ENABLE_WHEN_NONE_PRESENT ]



def get_all_fet_name_of_fet_having_lossless_encoder() -> list:
    # from the namespace , get the list of all FET
    # then use the classmethod fet_encoder_is_lossless to refine the list
    return [fet_name for fet_name in LIST_ALL_FET_NAMES_FEC_SELECTABLE if getattr(FeatureEngineeringTemplate, fet_name ).fet_encoder_is_lossless ]


class FeatureEngineeringTemplate(ABC):
    """
    FeatureEngineeringTemplate provide base class for encoding, decoding and creating configuration
    * Each FET works only with one datatype (Float, Label, etc)
    * Each FET inherit from base class
    * Depends on transformation method and data type FET can decode data lossless or not
    * Each FET can encode data
    * Each FET can be serialized or deserialized
    """

    # class attribute because we read them sometime with classmethod
    # they must be all 3 overwritten  one time in the children class
    fet_encoder_is_lossless = None
    fet_is_encoder = None
    fet_is_decoder = None

    def __init__(
        self,
        column_data_or_serialized_config: Union[list, np.ndarray, dict],
        column_datas_infos: Optional[Column_datas_infos] = None,
    ):
        """
        Constructor method
        Depends on data inn can create or load FET
        """
        self.warning_message = ""
        self.clear_warning_message( )

        if isinstance(column_data_or_serialized_config, np.ndarray) and column_datas_infos:
            self._create_configuration(column_data_or_serialized_config, column_datas_infos)

        elif isinstance(column_data_or_serialized_config, dict) and column_datas_infos is None:
            self.load_serialized_fet_configuration(column_data_or_serialized_config )

        else:
            logger.error( f"Encoder cannot support this type of input data : {type(column_data_or_serialized_config)}")


    def clear_warning_message(self) -> NoReturn:
        """
        if we use the same instance to encode different dataset we should clear the warning messages
        """
        self.warning_message = ""


    def add_warning(self, warning_msg: str) -> NoReturn:
        """
        Add messages in the WARNING FET
        the message can contains a CODE like : [is_re_run_enc_dec]
        this code will set the flag in the machine to True (to correct the error later)

        :params warning_msg: the warning message to add in the FET
        """
        if warning_msg not in self.warning_message and len( self.warning_message ) < 1500:
            self.warning_message += f"{self} : " + warning_msg + "\n"


    @abstractmethod
    def _create_configuration(self, column_data: np.ndarray, column_datas_infos: Column_datas_infos) -> NoReturn:
        """
        Creates FET
        Calculates all config for sklearn transformers(MinMaxScaler, OneHoEncoder, etc.)
        """
        logger.error("function not defined yet")
        self._numerator.classes_ = None

    def serialize_fet_configuration(self ) -> dict:
        """
        return the configuration as a dict
        """
        pass

    def load_serialized_fet_configuration(self, serialized_data: dict ) -> NoReturn:
        """
        Recreates FET instance from serialized data Dict
        """
        pass

    @classmethod
    def cls_is_possible_to_enable_this_fet_with_this_infos(cls, column_datas_infos: Column_datas_infos ) -> bool:
        """
        Verify that this column_datas_infos can be encoded by this FEC
        Must be overwritten if it is not like default :

        by comparing the class suffix and the datatype of the column to encode : if it is same it is ok
        """

        def _get_str_datatype_this_fet_name_encdec( cls_name: str ):
            for one_possible_column_type_name in ("Float", "Label", "Date", "Time", "Language"):
                if cls_name.endswith( one_possible_column_type_name ):
                    return one_possible_column_type_name
            raise NameError( f"FET name ({cls_name}) must end with 'Float', 'Label', 'Date', 'Time' or 'Language'" )


        str_datatype_this_fet_encdec = _get_str_datatype_this_fet_name_encdec( cls.__name__ )

        if column_datas_infos.datatype == DatasetColumnDataType.IGNORE:
            return False

        # if fet encode the same datatype as the column data it is ok
        if column_datas_infos.datatype is getattr(DatasetColumnDataType, str_datatype_this_fet_encdec.upper()):
            return True

        # if this fet encode date or time and column is datetime, it is ok
        if str_datatype_this_fet_encdec in ["Date", "Time"] and column_datas_infos.datatype is DatasetColumnDataType.DATETIME:
            return True

        return False

    @abstractmethod
    def cls_get_activation_cost(cls, column_datas_infos: Column_datas_infos) -> int:
        """
        Evaluates column cost with enabled FeatureEngineeringTemplate
        (the cost is how many new column will be created by this function if it is enabled
        """
        logger.error(f"function cls_get_activation_cost not defined yet for {cls} for column : {column_datas_infos}")
        return 0

    @abstractmethod
    def encode(self, column_data: np.ndarray) -> np.ndarray:
        """
        Encode the array of one column data
        and return another array with same quantity of rows but can have more columns
        """
        logger.error(f"function encode not defined yet for {self} ")
        return 0

    def decode(self, column_data: np.ndarray) -> np.ndarray:
        """
        Decoder of the FET , in argument the array to decode (one or more columns)
        in return the array of data of one single column
        """
        if not self.fet_is_decoder:
            logger.error(f"'{type(self).__name__}' encoder does not have a decode method")
        logger.error(f"function decode not defined yet for {self}")
        return 0

    # if the class create more than one column, it must be overwritten
    def get_list_encoded_columns_name(self, user_column_name: str):
        """
        Returns a list of encoded columns names
        """
        return [f"{user_column_name}{SEPARATOR_IN_COLUMNS_NAMES}{type(self).__name__}"]

    def __repr__(self) -> str:
        return type(self).__name__ + "()"

    def _update_preprocessing_for_labels_encoder( self , array_labels_to_update: np.array ) -> NoReturn:
        for i in range( len( array_labels_to_update ) ):
            if pd.isnull( array_labels_to_update[ i ] ):
                array_labels_to_update[ i ] = np.nan
            elif not isinstance( array_labels_to_update[ i ] , str):
                try:
                    array_labels_to_update[ i ] = str(array_labels_to_update[ i ]).lower( ).strip( )
                except Exception as e:
                    logger.error( f"errors {e} - trying to convert data to str for label - data is : {array_labels_to_update[ i ]} , data type is { type(array_labels_to_update[ i ]) } " )
            else:
                try:
                    array_labels_to_update[ i ] = array_labels_to_update[ i ].lower( ).strip( )
                except Exception as e:
                    logger.error( f"errors {e} - data str for label - data is : {array_labels_to_update[ i ]} , data type is { type(array_labels_to_update[ i ]) } " )


    def _update_unknown_labels_for_labels_encoder( self, array_to_update: np.array ) -> NoReturn:
        # we use LabelEncoder , but for multiplexers we use OneHotEncoder
        if hasattr( self._numerator , "classes_" ):
            Labels = self._numerator.classes_
        elif hasattr( self._numerator , "categories_" ):
            Labels = self._numerator.categories_[0]
        else:
            logger.error( f"Label configuration invalid for {self._numerator}")

        for i in range(len(array_to_update ) ):
            if pd.isnull( array_to_update[i] ):
                array_to_update[i] = np.nan
            elif array_to_update[i] not in Labels:
                self.add_warning(f"Detected unknown label {array_to_update[i ]} while encoding =>  [is_re_run_enc_dec];" )
                array_to_update[i] = np.nan

    def _update_overlimits_for_labels_decoder( self, array_to_update: np.array ) -> NoReturn:
        for i in range( 0 , len(array_to_update ) ) :
            array_to_update[i ] = np.rint( array_to_update[i ] )
        if array_to_update[np.where( array_to_update > len( self._numerator.classes_ ) - 1 ) ]:
            #self.add_warning(f"Detected overlimit '{array[np.where(array > len(self._numerator.classes_) - 1)]}', when decoding, replaced by '{UNKNOWN_NONE_FOR_FET_LABEL}' " )
            array_to_update[np.where( array_to_update > len(self._numerator.classes_ ) - 1 ) ] = self._numerator.transform([ np.nan ] )
        if array_to_update[np.where( array_to_update < 0 ) ]:
            #self.add_warning(f"Detected overlimit '{array[np.where(array < 0)]}', when decoding, replaced by '{UNKNOWN_NONE_FOR_FET_LABEL}' " )
            array_to_update[np.where( array_to_update < 0 ) ] = self._numerator.transform([ np.nan ] )


class FETImpossible:
    """
    used to inherit from to mark impossible decoder/encoder
    """
    fet_encoder_is_lossless = None
    fet_is_encoder = None
    fet_is_decoder = None

    def _create_configuration(self, column_data: np.ndarray, column_datas_infos: Column_datas_infos) -> NoReturn:
        logger.error("FETImpossible")

    def encode(self, column_data: np.ndarray) -> np.ndarray:
        logger.error("FETImpossible")
        return None

    def decode(self, column_data: np.ndarray) -> np.ndarray:
        logger.error("FETImpossible")
        return None

    def _get_list_columns_names_encoded(self, user_column_name: str):
        logger.error("FETImpossible")
        return None

    @classmethod
    def cls_is_possible_to_enable_this_fet_with_this_infos(cls, column_datas_infos: Column_datas_infos) -> bool:
        return False

class FETIsNone(FeatureEngineeringTemplate):
    """
    add a single column with a 1 if the data is None else 0
    """

    fet_encoder_is_lossless = False
    fet_is_encoder = True
    fet_is_decoder = False

    def _create_configuration(self, column_data: np.ndarray, column_datas_infos: Column_datas_infos) -> NoReturn:
        # no configuration to load or save
        pass

    def load_serialized_fet_configuration(self, serialized_data: dict ) -> NoReturn:
        # no configuration to load or save
        pass

    def serialize_fet_configuration(self ) -> dict:
        return {}

    def encode(self, column_data: np.ndarray) -> np.ndarray:
        # 1 if None else 0
        return np.array([1 if pd.isnull(value) else 0 for value in column_data])

    @classmethod
    def cls_get_activation_cost(cls, column_datas_infos: Column_datas_infos) -> int:
        return 1

    @classmethod
    def cls_is_possible_to_enable_this_fet_with_this_infos(cls, column_datas_infos: Column_datas_infos ) -> bool:
        # if the parent class say it cannot be activated then this is it
        if not super().cls_is_possible_to_enable_this_fet_with_this_infos(column_datas_infos ):
            return False

        return column_datas_infos.missing_percentage > 0

class FETIsNoneFloat(FETIsNone):
    ...

class FETIsNoneLabel(FETIsNone):
    ...

class FETIsNoneDate(FETIsNone):
    ...

class FETIsNoneTime(FETIsNone):
    ...

class FETIsNoneLanguage(FETIsNone):
    ...


class FETNumericMinMax(FeatureEngineeringTemplate):
    """
    Encoding process
    for x in column_data:
        x = (x - min(column_data)) / (max(column_data) - min(column_data))

    Decoding process:
    for x in column_data:
        x = x * (max(column_data) - min(column_data)) + min(column_data)
    """

    fet_encoder_is_lossless = True
    fet_is_encoder = True
    fet_is_decoder = True

    @abstractmethod
    def _create_configuration(self, column_data: np.ndarray, column_datas_infos: Column_datas_infos) -> NoReturn:
        pass


    def load_serialized_fet_configuration( self, serialized_data: dict ) -> NoReturn:
        self._scaler  = pickle_load_str( serialized_data[ "pickle.dumps.MinMaxScaler" ] )
        # scaler = MinMaxScaler( feature_range=(0.1, 0.9) )
        # scaler.n_samples_seen_ = serialized_data[ "n_samples" ]
        # scaler.scale_ = np.array( [ serialized_data[ "scale_number" ] ] )
        # scaler.min_ = np.array( [ serialized_data[ "min" ] ] )
        # scaler.data_min_ = np.array( [ serialized_data[ "data_min" ] ] )
        # scaler.data_max_ = np.array( [ serialized_data[ "data_max" ] ] )
        # scaler.data_range = scaler.data_max - scaler.data_min_
        # self._scaler = scaler


    def serialize_fet_configuration( self ) -> dict:
        return { "pickle.dumps.MinMaxScaler": pickle_dump_str( self._scaler ) }
        # return {
        #     "n_samples": self._scaler.n_samples_seen_,
        #     "scale_number": self._scaler.scale_[ 0 ],
        #     "min": self._scaler.min_[ 0 ],
        #     "data_min": self._scaler.data_min_[ 0 ],
        #     "data_max": self._scaler.data_max_[ 0 ],
        # }


    @abstractmethod
    def encode(self, column_data: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def decode(self, column_data: np.ndarray) -> np.ndarray:
        pass

    @classmethod
    def cls_get_activation_cost(cls, column_datas_infos: Column_datas_infos) -> int:
        return 1


class FETNumericMinMaxFloat(FETNumericMinMax):
    # Overview
    # Encoder - encoded data in range [0.1, 0.9]
    # Decoder - lossless
    # Serialization - tested

    def _create_configuration(self, column_data: np.ndarray, column_datas_infos: Column_datas_infos) -> NoReturn:
        self._scaler = MinMaxScaler(feature_range=(0.1, 0.9))

        # Create a boolean mask for ninf and None values
        mask = np.logical_or( np.isnan( column_data.astype( float ) ), np.isinf( column_data.astype( float ) ), np.isneginf( column_data.astype( float ) ) )
        # Replace all inf ninf None with nan
        column_data_without_inf_none = np.where( mask, np.nan, column_data )

        if len( column_data.shape ) == 1:
            self._scaler.fit( np.append( column_data_without_inf_none, np.nan ).reshape( -1, 1 ) )
        elif len( column_data.shape ) == 2:
            row_nan = np.full((1, column_data.shape[1]), np.nan)
            self._scaler.fit( np.vstack( [ column_data_without_inf_none, row_nan ] ) )
        else:
            logger.error( f"impossible to have an array with more than 2 dimensions : {column_data}")


    def encode(self, column_data: np.ndarray) -> np.ndarray:

        def __normalize_encoded_data( array: np.array) -> np.array:
            if len(np.where(array < 0)[0]) > 0:
                self.add_warning(f"Encoded data is out of range. Invalid values {array[np.where(array < 0)]}, replaced by 0. [is_re_run_enc_dec];")
                array[np.where(array < 0)] = 0
            if len(np.where(array > 1)[0]) > 0:
                self.add_warning(f"Encoded data is out of range. Invalid values {array[np.where(array > 1)]}, replaced by 1. [is_re_run_enc_dec];")
                array[np.where(array > 1)] = 1
            np.nan_to_num( array , copy=False, nan=FETNumericMinMaxFloat_ENCODE_UNKNOWN_NAN_WITH_THIS_VALUE )
            return array

        if len( column_data.shape ) == 1:
            encoded_column_data = self._scaler.transform( column_data.reshape( -1, 1 ) ).ravel( )
        elif len( column_data.shape ) == 2:
            encoded_column_data = self._scaler.transform( column_data )

        return __normalize_encoded_data(encoded_column_data)


    def decode(self, scaled_column_data: np.ndarray) -> np.ndarray:

        def __fix_nn_overlimit_values( array ):
            for i in range(len(array)):
                if pd.isnull( array[i] ):
                    logger.error( "found a None/NaN in the data to decode ! ")
                    pass
                elif not 0 <= array[i] <= 1:
                    self.add_warning(f"Detected out or range(0, 1) value {array[i]}  in the data to decode ! [is_re_run_enc_dec] ")
                    if array[i] < 0:
                        array[i] = 0
                    elif array[i] > 1:
                        array[i] = 1
            return array

        scaled_column_data = scaled_column_data
        scaled_column_data = __fix_nn_overlimit_values(scaled_column_data )
        decoded_column_data = self._scaler.inverse_transform(scaled_column_data.reshape(-1, 1)).ravel()
        return  ( decoded_column_data )


class FETNumericMinMaxLabel(FETNumericMinMaxFloat, FETNumericMinMax):
    # Overview
    # Encoder - encoded data in range [0.1, 0.9]
    # Decoder - losseless
    # Serialization - tested

    def load_serialized_fet_configuration(self, serialized_data: dict ) -> NoReturn:
        super(FETNumericMinMaxLabel, self).load_serialized_fet_configuration(serialized_data )
        self._numerator = pickle_load_str( serialized_data[ "pickle.dumps.LabelEncoder" ] )
        # numerator = LabelEncoder()
        # numerator.classes_ = np.array(serialized_data["classes"])
        # self._numerator = numerator

    def serialize_fet_configuration(self ) -> dict:
        # serialized_data.update({"classes": self._numerator.classes_.tolist()})
        return super().serialize_fet_configuration( ) | { "pickle.dumps.LabelEncoder": pickle_dump_str( self._numerator ) }

    def _create_configuration(self, column_data: np.ndarray, column_datas_infos: Column_datas_infos) -> NoReturn:
        self._numerator = LabelEncoder()
        self._update_preprocessing_for_labels_encoder( column_data )
        numeric_column_data = self._numerator.fit_transform(np.append(column_data, [ np.nan ] ) )
        super(FETNumericMinMaxLabel, self)._create_configuration(numeric_column_data, column_datas_infos)

    def encode(self, column_data: np.ndarray) -> np.ndarray:
        self._update_preprocessing_for_labels_encoder( column_data )
        self._update_unknown_labels_for_labels_encoder( column_data )
        numeric_column_data = self._numerator.transform( column_data )
        return super(FETNumericMinMaxLabel, self).encode(numeric_column_data)

    def decode(self, scaled_column_data: np.ndarray) -> np.ndarray:
        descaled_data = super(FETNumericMinMaxLabel, self).decode( scaled_column_data )
        self._update_overlimits_for_labels_decoder( descaled_data )
        return self._numerator.inverse_transform( descaled_data.astype( int ) )


class FETNumericMinMaxDate(FETNumericMinMaxFloat, FETNumericMinMax):
    # Overview
    # Encoder - encoded data in range [0.1, 0.9]
    # Decoder - losseless
    # Serialization - tested

    def _create_configuration(self, column_data: np.ndarray, column_datas_infos: Column_datas_infos) -> NoReturn:
        super()._create_configuration( encoder_date_to_float(column_data ) , column_datas_infos )

    def encode(self, column_data: np.ndarray) -> np.ndarray:
        return super().encode(encoder_date_to_float(column_data ) )

    def decode(self, column_data: np.ndarray) -> np.ndarray:
        return decoder_float_to_date(super( ).decode(column_data ) )


class FETNumericMinMaxTime(FETNumericMinMaxFloat, FETNumericMinMax):
    # Overview
    # Encoder - encoded data in range [0.1, 0.9]
    # Decoder - losseless
    # Serialization - tested

    def _create_configuration(self, column_data: np.ndarray, column_datas_infos: Column_datas_infos) -> NoReturn:
        super()._create_configuration(encoder_time_to_float( (column_data) ), column_datas_infos )

    def encode(self, column_data: np.ndarray) -> np.ndarray:
        return super().encode(encoder_time_to_float(column_data ) )

    def decode(self, column_data: np.ndarray) -> np.ndarray:
        return decoder_float_to_time(super( ).decode(column_data ) )


class FETNumericMinMaxLanguage( FETImpossible ):
    pass
    # to make it work same as label we can enable the following lines
    #class FETNumericMinMaxLanguage( FETNumericMinMax, FETImpossible ):
    # def _create_configuration(self, column_data: np.ndarray, column_datas_infos: Column_datas_infos) -> NoReturn:
    #     super( )._create_configuration( column_data , column_datas_infos )
    #
    # def encode(self, column_data: np.ndarray) -> np.ndarray:
    #     return super( ).encode( column_data )


class FETMultiplexerAll(FeatureEngineeringTemplate):
    """
    Encoding process:
    _categories = np.unique(column_data)
    for x in column_data:
        x = np.zeros(len(_categories))
        x[np.where(x, _categories)] = 1

    Decoding process:
    for x in column_data:
        x = _categories[np.where(1, x)]
    """

    fet_encoder_is_lossless = True
    fet_is_encoder = True
    fet_is_decoder = True

    @abstractmethod
    def _create_configuration(self, column_data: np.ndarray, column_datas_infos: Column_datas_infos) -> NoReturn:
        pass

    def load_serialized_fet_configuration(self, serialized_data: dict ) -> NoReturn:
        self._numerator = pickle_load_str( serialized_data[ "pickle.dumps.OneHotEncoder" ] )
        # multiplexer = OneHotEncoder(  handle_unknown='ignore'  )
        # multiplexer.categories_ = [np.array(serialized_data["categories"], dtype=object )]    # dtype=object is necessary to avoid nan to be converted in string 'nan'
        # #multiplexer.n_features_in_ = serialized_data["n_features_in_"]
        # multiplexer._n_features_outs = serialized_data[ "_n_features_outs" ]
        # multiplexer.drop = None
        # multiplexer.drop_idx_ = None
        # multiplexer._infrequent_enabled = serialized_data[ "_infrequent_enabled" ]
        # multiplexer._infrequent_enabled = serialized_data[ "_infrequent_enabled" ]
        # self._numerator = multiplexer

    def serialize_fet_configuration(self ) -> dict:
        return { "pickle.dumps.OneHotEncoder": pickle_dump_str( self._numerator ) }
        # return {
        #     "categories": self._numerator.categories_[0].tolist(),
        #     #"n_features_in_": self._numerator.n_features_in_,
        #     '_n_features_outs': self._numerator._n_features_outs,
        #     '_infrequent_enabled': self._numerator._infrequent_enabled,
        # }

    @abstractmethod
    def encode(self, column_data: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def decode(self, column_data: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_list_encoded_columns_name(self, user_column_name: str) -> list:
        pass

    @classmethod
    def cls_get_activation_cost(cls, column_datas_infos: Column_datas_infos) -> int:
        return column_datas_infos.unique_value_count + 1

    @classmethod
    def cls_is_possible_to_enable_this_fet_with_this_infos(cls, column_datas_infos: Column_datas_infos ) -> bool:
        # if the parent class say it cannot be activated then this is it
        if not super().cls_is_possible_to_enable_this_fet_with_this_infos(column_datas_infos ):
            return False

        return column_datas_infos.unique_value_count >= FETMultiplexer_MIN_UNIQUE_VALUES_TO_ENABLE


class FETMultiplexerAllFloat(FETMultiplexerAll):
    # Overview
    # Encoder - encoded data in range [0.1, 0.9]
    # Decoder - losseless
    # Serialization - tested

    def get_list_encoded_columns_name(self, user_column_name: str) -> list:
        return [
                f"{user_column_name}{SEPARATOR_IN_COLUMNS_NAMES}{type(self).__name__}{SEPARATOR_IN_COLUMNS_NAMES}{columns_names}"
                for columns_names in self._numerator.categories_[0].tolist()
            ]

    def __normalize_encoded_data(self, array: np.array) -> np.array:
        if len(np.where(array < 0)[0]) > 0:
            self.add_warning(f"Encoded data is out of range. Invalid value {array[np.where(array < 0)]}, replaced by 0. [is_re_run_enc_dec];")
            array[np.where(array < 0)] = 0
        if len(np.where(array > 1)[0]) > 0:
            self.add_warning(f"Encoded data is out of range. Invalid value {array[np.where(array > 1)]}, replaced by 1. [is_re_run_enc_dec];")
            array[np.where(array > 1)] = 1
        return array

    def _create_configuration(self, column_data: np.ndarray, column_datas_infos: Column_datas_infos) -> NoReturn:
        self._numerator = OneHotEncoder( handle_unknown='ignore' ).fit(
            np.append( column_data , [np.nan ] ).reshape(-1, 1 ).astype(float )
        )

    def encode(self, column_data: np.ndarray) -> np.ndarray:
        return self.__normalize_encoded_data(
            self._numerator.transform( (column_data ).reshape(-1, 1 ) ).toarray( )
        )

    def decode(self, column_data: np.ndarray) -> np.ndarray:
        return  (self._numerator.inverse_transform( column_data ).ravel( ) )

    @classmethod
    def cls_is_possible_to_enable_this_fet_with_this_infos(cls, column_datas_infos: Column_datas_infos ) -> bool:
        # if the parent class say it cannot be activated then this is it
        if not super().cls_is_possible_to_enable_this_fet_with_this_infos(column_datas_infos ):
            return False

        return FETMultiplexer_MAX_COLUMNS_TO_CREATE >= column_datas_infos.unique_value_count

    @classmethod
    def cls_get_activation_cost(cls, column_datas_infos: Column_datas_infos) -> int:
        return column_datas_infos.unique_value_count + 1


class FETMultiplexerAllLabel(FETMultiplexerAll):
    # Overview
    # Encoder - encoded data in range [0.1, 0.9]
    # Decoder - lossless
    # Serialization - tested

    def get_list_encoded_columns_name(self, user_column_name: str) -> list:
        return [
                f"{user_column_name}{SEPARATOR_IN_COLUMNS_NAMES}{type(self).__name__}{SEPARATOR_IN_COLUMNS_NAMES}{columns_names}"
                for columns_names in self._numerator.categories_[0].tolist()
            ]

    def __filter_config_data(self, array: np.array) -> np.array:
        return array[np.where(pd.notna(array))]

    def _create_configuration(self, column_data: np.ndarray, column_datas_infos: Column_datas_infos) -> NoReturn:
        self._update_preprocessing_for_labels_encoder( column_data )
        prepared_column_data = np.append( column_data , [ np.nan ] ).reshape(-1, 1 )
        self._numerator = OneHotEncoder( handle_unknown="ignore" ).fit(prepared_column_data)

    def encode(self, column_data: np.ndarray) -> np.ndarray:
        self._update_preprocessing_for_labels_encoder( column_data )
        # OneHotEncoder will convert all unknown label and nan to a list of 0
        return self._numerator.transform( column_data.reshape(-1, 1) ).toarray()

    def decode(self, column_data: np.ndarray) -> np.ndarray:
        decoded_data = self._numerator.inverse_transform( column_data.astype( int ) )
        return decoded_data

    @classmethod
    def cls_get_activation_cost(cls, column_datas_infos: Column_datas_infos) -> int:
        return column_datas_infos.unique_value_count + 1

    @classmethod
    def cls_is_possible_to_enable_this_fet_with_this_infos(cls, column_datas_infos: Column_datas_infos ) -> bool:
        # if the parent class say it cannot be activated then this is it
        if not super().cls_is_possible_to_enable_this_fet_with_this_infos(column_datas_infos ):
            return False

        return FETMultiplexer_MAX_COLUMNS_TO_CREATE >= column_datas_infos.unique_value_count


class FETMultiplexerAllDate(FETMultiplexerAll):
    # Overview
    # Encoder - encoded data in range [0.1, 0.9]
    # Decoder - unavailable   # todo add a decoder ( probably need to use argmax instead of inverse tranform to decode data)
    # Serialization - tested

    fet_is_decoder = True

    def get_list_encoded_columns_name(self, user_column_name: str) -> list:
        return (
            [
                f"{user_column_name}{SEPARATOR_IN_COLUMNS_NAMES}{type(self).__name__}{SEPARATOR_IN_COLUMNS_NAMES}YEAR"
            ] +
            [
                f"{user_column_name}{SEPARATOR_IN_COLUMNS_NAMES}{type(self).__name__}{SEPARATOR_IN_COLUMNS_NAMES}MONTH{m}"
                   for m in range( 1 , 13 )
            ] +
            [f"{user_column_name}{SEPARATOR_IN_COLUMNS_NAMES}{type( self ).__name__}{SEPARATOR_IN_COLUMNS_NAMES}MONTH-None"] +
            [
                f"{user_column_name}{SEPARATOR_IN_COLUMNS_NAMES}{type(self).__name__}{SEPARATOR_IN_COLUMNS_NAMES}DAY{d}"
                   for d in range( 1 , 32 )
            ] +
            [f"{user_column_name}{SEPARATOR_IN_COLUMNS_NAMES}{type( self ).__name__}{SEPARATOR_IN_COLUMNS_NAMES}Day-None"] +
            [
                f"{user_column_name}{SEPARATOR_IN_COLUMNS_NAMES}{type(self).__name__}{SEPARATOR_IN_COLUMNS_NAMES}WEEKDAY{w}"
                   for w in range( 0 , 7 )
            ] +
            [ f"{user_column_name}{SEPARATOR_IN_COLUMNS_NAMES}{type( self ).__name__}{SEPARATOR_IN_COLUMNS_NAMES}WEEKDAY-None" ]
        )

    def _create_configuration(self, column_data: np.ndarray, column_datas_infos: Column_datas_infos) -> NoReturn:
        column_data_years = np.array([x.year if not pd.isnull( x ) else column_datas_infos.mean for x in pd.to_datetime(column_data)  ] ).reshape(-1, 1)
        scaler = MinMaxScaler()
        scaler.fit(column_data_years)
        self._scaler = scaler
        multiplexer = OneHotEncoder( handle_unknown='ignore' )
        multiplexer_template = np.array([np.array([x, y, z]).astype(int) for x in range(1, 13) for y in range(1, 32) for z in range(0,7)])
        multiplexer_template = np.concatenate((multiplexer_template, [[np.nan, np.nan, np.nan]])).astype(float)
        multiplexer.fit( multiplexer_template )
        self._multiplexer = multiplexer

    def load_serialized_fet_configuration(self, serialized_data: dict ) -> NoReturn:
        self._scaler = pickle_load_str( serialized_data[ "pickle.dumps.MinMaxScaler" ] )
        # scaler = MinMaxScaler()
        # scaler.n_samples_seen_ = serialized_data["n_samples"]
        # scaler.scale_ = np.array([serialized_data["scale_number"]])
        # scaler.min_ = np.array([serialized_data["min"]])
        # scaler.data_min_ = np.array([serialized_data["data_min"]])
        # scaler.data_max_ = np.array([serialized_data["data_max"]])
        # scaler.data_range_ = scaler.data_max_ - scaler.data_min_
        # self._scaler = scaler
        self._multiplexer = pickle_load_str( serialized_data[ "pickle.dumps.OneHotEncoder" ] )
        # multiplexer = OneHotEncoder( handle_unknown='ignore' )
        # multiplexer_template = np.array([np.array([x, y, z]).astype(int) for x in range(1, 13) for y in range(1, 32) for z in range(7)])
        # multiplexer_template = np.concatenate((multiplexer_template, [[np.nan, np.nan, np.nan]])).astype(float)
        # multiplexer.fit( multiplexer_template )
        # self._multiplexer = multiplexer

    def serialize_fet_configuration(self ) -> dict:
        return {
                "pickle.dumps.MinMaxScaler": pickle_dump_str( self._scaler )  ,
                "pickle.dumps.OneHotEncoder": pickle_dump_str( self._multiplexer )
                }
        # return {
        #     "n_samples": self._scaler.n_samples_seen_,
        #     "scale_number": self._scaler.scale_[0],
        #     "min": self._scaler.min_[0],
        #     "data_min": self._scaler.data_min_[0],
        #     "data_max": self._scaler.data_max_[0],
        # }

    def encode(self, column_data: np.ndarray) -> np.ndarray:
        column_data_years = np.array([x.year if not pd.isnull(x) else 2000 for x in column_data]).reshape(-1, 1)
        column_data_years = self._scaler.transform(column_data_years)
        column_data_to_multiplex = np.array(
            [np.array([x.month, x.day, x.weekday()]).astype(int) if not pd.isnull(x) else [np.nan, np.nan, np.nan] for x in column_data]
        ).astype(float)
        multiplexed_column_data = self._multiplexer.transform(column_data_to_multiplex).toarray()
        encoded_column_data = np.column_stack((column_data_years, multiplexed_column_data))
        return encoded_column_data

    def decode(self, column_data: np.ndarray) -> np.ndarray:
        year_rows = column_data[ : , 0 ]
        year_rows_unscaled = self._scaler.inverse_transform(year_rows.reshape( -1 , 1 )).ravel().astype( dtype=int )
        mdw_rows = column_data[ : , 1: ]
        decoded_column_data_mdw = self._multiplexer.inverse_transform( mdw_rows ).astype( dtype=int )
        rows_count = decoded_column_data_mdw.shape[ 0 ]
        decoded_column_data_date = np.empty( rows_count , dtype=datetime.date )
        for idxrow in range( 0, rows_count ):
            if ( pd.isnull( decoded_column_data_mdw[idxrow][0] ) or pd.isnull( decoded_column_data_mdw[idxrow][1] ) or
                    decoded_column_data_mdw[idxrow][0]<1 or decoded_column_data_mdw[idxrow][0]>12 or
                    decoded_column_data_mdw[idxrow][1]<1 or decoded_column_data_mdw[idxrow][1]>31
            ):
                decoded_column_data_date[ idxrow ] = np.nan
            else:
                try:
                    decoded_column_data_date[idxrow] = datetime.date( year_rows_unscaled[idxrow] , decoded_column_data_mdw[idxrow][0] , decoded_column_data_mdw[idxrow][1] )
                except:
                    decoded_column_data_date[ idxrow ] = np.nan

        return decoded_column_data_date

    @classmethod
    def cls_is_possible_to_enable_this_fet_with_this_infos(cls, column_datas_infos: Column_datas_infos ) -> bool:
        # if the parent class say it cannot be activated then this is it
        if not super().cls_is_possible_to_enable_this_fet_with_this_infos(column_datas_infos ):
            return False

        return True

    @classmethod
    def cls_get_activation_cost(cls, column_datas_infos: Column_datas_infos) -> int:
        return 1 + 12+1 + 31+1 +7+1 # 54


class FETMultiplexerAllTime(FETMultiplexerAll):

    fet_is_decoder = True

    def get_list_encoded_columns_name(self, user_column_name: str) -> list:
        return (
            [
                f"{user_column_name}{SEPARATOR_IN_COLUMNS_NAMES}{type(self).__name__}{SEPARATOR_IN_COLUMNS_NAMES}HOUR{h}"
                   for h in range( 1 , 25 )
            ] +
            [f"{user_column_name}{SEPARATOR_IN_COLUMNS_NAMES}{type( self ).__name__}{SEPARATOR_IN_COLUMNS_NAMES}HOUR-None"] +
            [
                f"{user_column_name}{SEPARATOR_IN_COLUMNS_NAMES}{type(self).__name__}{SEPARATOR_IN_COLUMNS_NAMES}MINS{m}"
                   for m in range( 1 , 61 )
            ] +
            [ f"{user_column_name}{SEPARATOR_IN_COLUMNS_NAMES}{type( self ).__name__}{SEPARATOR_IN_COLUMNS_NAMES}MIN-None" ]
        )

    def serialize_fet_configuration(self ) -> dict:
        return { "pickle.dumps.OneHotEncoder": pickle_dump_str( self._multiplexer ) }

    def load_serialized_fet_configuration(self, serialized_data: dict ) -> NoReturn:
        self._multiplexer = pickle_load_str( serialized_data[ "pickle.dumps.OneHotEncoder" ] )
        # multiplexer = OneHotEncoder( handle_unknown='ignore' )
        # # TODO too large array , optimize with 2 smaller multiplexer
        # multiplexer_template = np.array([[x, y] for x in range(24) for y in range(60)])
        # multiplexer_template = np.concatenate((multiplexer_template, [[np.nan, np.nan]])).astype(float)
        # multiplexer.fit(multiplexer_template)
        # self._multiplexer = multiplexer

    def _create_configuration(self, column_data: np.ndarray, column_datas_infos: Column_datas_infos) -> NoReturn:
        multiplexer = OneHotEncoder( handle_unknown='ignore' )
        # TODO too large array , optimize with 2 smaller multiplexer
        multiplexer_template = np.array([[x, y] for x in range(24) for y in range(60)])
        multiplexer_template = np.concatenate((multiplexer_template, [[np.nan, np.nan]])).astype(float)
        multiplexer.fit(multiplexer_template)
        self._multiplexer = multiplexer

    def encode(self, column_data: np.ndarray) -> np.ndarray:
        encoded_column_data = np.array(
            [[x.hour, x.minute] if not pd.isnull(x) else [np.nan, np.nan] for x in column_data]
        )
        multiplexed_column_data = self._multiplexer.transform(encoded_column_data).toarray()
        return multiplexed_column_data

    def decode(self, column_data: np.ndarray) -> np.ndarray:
        decoded_column_data = self._multiplexer.inverse_transform(column_data)
        rows_count = decoded_column_data.shape[ 0 ]
        decoded_column_data_time = np.empty( rows_count , dtype=datetime.time )
        for idxrow in range( 0, rows_count ):
            if ( pd.isnull( decoded_column_data[idxrow][0] ) or pd.isnull( decoded_column_data[idxrow][1] ) or
                    decoded_column_data[idxrow][0]<0 or decoded_column_data[idxrow][0]>24 or
                    decoded_column_data[idxrow][1]<0 or decoded_column_data[idxrow][1]>60
            ):
                decoded_column_data_time[ idxrow ] = np.nan
            else:
                try:
                    decoded_column_data_time[idxrow] = datetime.time( int(decoded_column_data[idxrow][0]) , int(decoded_column_data[idxrow][1]) )
                except:
                    decoded_column_data_time[ idxrow ] = np.nan

        return decoded_column_data_time

    @classmethod
    def cls_is_possible_to_enable_this_fet_with_this_infos(cls, column_datas_infos: Column_datas_infos ) -> bool:
        # if the parent class say it cannot be activated then this is it
        if not super().cls_is_possible_to_enable_this_fet_with_this_infos(column_datas_infos ):
            return False

        return True

    @classmethod
    def cls_get_activation_cost(cls, column_datas_infos: Column_datas_infos) -> int:
        return  60+1  +  24+1


class FETMultiplexerAllLanguage( FETImpossible):
    pass


class FET3FrequencyLevel(FeatureEngineeringTemplate):
    fet_encoder_is_lossless = False
    fet_is_encoder = True
    fet_is_decoder = False

    @abstractmethod
    def _create_configuration(self, column_data: np.ndarray, column_datas_infos: Column_datas_infos) -> NoReturn:
        pass

    def load_serialized_fet_configuration(self, serialized_data: dict ) -> NoReturn:
        self._std = serialized_data["std"]
        self._mean = serialized_data["mean"]
        return serialized_data

    def serialize_fet_configuration(self ) -> dict:
        return {
                    "std": self._std,
                    "mean": self._mean
                    }

    @abstractmethod
    def encode(self, column_data: np.ndarray) -> np.ndarray:
        pass

    def get_list_encoded_columns_name(self, user_column_name: str) -> list:
        return [
            f"{user_column_name}{SEPARATOR_IN_COLUMNS_NAMES}{type(self).__name__}{SEPARATOR_IN_COLUMNS_NAMES}{class_}"
            for class_ in ["FREQUENT", "OCCASIONAL", "RARE"]
        ]

    @classmethod
    def cls_get_activation_cost(cls, column_datas_infos: Column_datas_infos) -> int:
        return 3

    @classmethod
    def cls_is_possible_to_enable_this_fet_with_this_infos(cls, column_datas_infos: Column_datas_infos ) -> bool:
        # if the parent class say it cannot be activated then this is it
        if not super().cls_is_possible_to_enable_this_fet_with_this_infos(column_datas_infos ):
            return False

        return column_datas_infos.unique_value_count >= FET3FrequencyLevel_MIN_UNIQUE_VALUES_TO_ENABLE


class FET3FrequencyLevelFloat(FET3FrequencyLevel):
    # Overview
    # Encoder - encoded data in range [0.1, 0.9]
    # Decoder - unavailable
    # Serialization - tested

    def _create_configuration(self, column_data: np.ndarray, column_datas_infos: Column_datas_infos) -> NoReturn:
        _column_data_without_none = column_data[np.where(~pd.isna(column_data.astype(float)))]
        new_truncated_float_normal_distribution = _truncated_normal_distribution(
            mean=float(np.mean(_column_data_without_none)),
            std=float(np.std(_column_data_without_none)),
            min_val=_column_data_without_none.min(),
            max_val=_column_data_without_none.max(),
            num_samples=1000,
        )
        self._mean = new_truncated_float_normal_distribution.mean()
        self._std = new_truncated_float_normal_distribution.std()

    def encode(self, column_data: np.ndarray) -> np.ndarray:
        encoded_data = list()
        abs_mean = abs(self._mean)
        for cell_value in column_data:
            if pd.isnull(cell_value):
                encoded_data.append([ np.nan, np.nan, np.nan ])
                continue

            value = abs(cell_value - abs_mean)
            if value <= self._std:
                encoded_data.append([0.9, 0.1, 0.1])
            elif value <= self._std * 2:
                encoded_data.append([0.1, 0.9, 0.1])
            else:
                encoded_data.append([0.1, 0.1, 0.9])

        return np.array(encoded_data)


class FET3FrequencyLevelLabel(FET3FrequencyLevelFloat, FET3FrequencyLevel):
    # Overview
    # Encoder - encoded data in range [0.1, 0.9]
    # Decoder - unavailable
    # Serialization - tested

    def serialize_fet_configuration(self ) -> dict:
        data_super = super().serialize_fet_configuration( )
        # data["classes"] = self._numerator.classes_.tolist()
        # return data
        data_super.update( { "pickle.dumps.LabelEncoder": pickle_dump_str( self._numerator ) } )
        return data_super

    def load_serialized_fet_configuration(self, serialized_data: dict ) -> NoReturn:
        # numerator = LabelEncoder()
        # numerator.classes_ = np.array(serialized_data["classes"])
        # self._numerator = numerator
        self._numerator = pickle_load_str( serialized_data[ "pickle.dumps.LabelEncoder" ] )
        super().load_serialized_fet_configuration(serialized_data )

    def _create_configuration(self, column_data: np.ndarray, column_datas_infos: Column_datas_infos) -> NoReturn:
        self._numerator = LabelEncoder()
        self._update_preprocessing_for_labels_encoder( column_data )
        numeric_column_data = self._numerator.fit_transform(np.append( column_data , [np.nan]))
        super()._create_configuration(numeric_column_data, column_datas_infos)

    def encode(self, column_data: np.ndarray) -> np.ndarray:
        self._update_preprocessing_for_labels_encoder( column_data )
        self._update_unknown_labels_for_labels_encoder( column_data )
        numeric_column_data = self._numerator.transform( column_data )
        encoded_data = super().encode(numeric_column_data)
        return encoded_data


class FET3FrequencyLevelDate(FET3FrequencyLevelFloat, FET3FrequencyLevel):
    # Overview
    # Encoder - encoded data in range [0.1, 0.9]
    # Decoder - unavailable
    # Serialization - tested

    def serialize_fet_configuration(self ) -> dict:
        return (
                    super().serialize_fet_configuration( ) |
                    { "pickle.dumps.MinMaxScaler": pickle_dump_str( self._scaler ) }
                )
        # data.update(
        #     {
        #         "n_samples": self._scaler.n_samples_seen_,
        #         "scale_number": self._scaler.scale_,
        #         "min": self._scaler.min_,
        #         "data_min": self._scaler.data_min_,
        #         "data_max": self._scaler.data_max_,
        #     }
        # )
        # return data

    def load_serialized_fet_configuration(self, serialized_data: dict ) -> NoReturn:
        super().load_serialized_fet_configuration(serialized_data )
        # scaler = MinMaxScaler(feature_range=(0.1, 0.9))
        # scaler.n_samples_seen_ = serialized_data["n_samples"]
        # scaler.scale_ = np.array([serialized_data["scale_number"]])
        # scaler.min_ = np.array([serialized_data["min"]])
        # scaler.data_min_ = np.array([serialized_data["data_min"]])
        # scaler.data_max_ = np.array([serialized_data["data_max"]])
        # scaler.data_range_ = scaler.data_max_ - scaler.data_min_
        # self._scaler = scaler
        self._scaler = pickle_load_str( serialized_data[ "pickle.dumps.MinMaxScaler" ] )

    def _create_configuration(self, column_data: np.ndarray, column_datas_infos: Column_datas_infos) -> NoReturn:
        numeric_column_data = encoder_date_to_float(column_data )
        super()._create_configuration(numeric_column_data, column_datas_infos)

    def encode(self, column_data: np.ndarray) -> np.ndarray:
        column_data = encoder_date_to_float(  (column_data) )
        encoded_data = super().encode(column_data)
        return encoded_data

class FET3FrequencyLevelTime(FET3FrequencyLevelFloat, FET3FrequencyLevel):
    # Overview
    # Encoder - encoded data in range [0.1, 0.9]
    # Decoder - unavailable
    # Serialization - tested

    def _create_configuration(self, column_data: np.ndarray, column_datas_infos: Column_datas_infos) -> NoReturn:
        super()._create_configuration(encoder_time_to_float( (column_data) ), column_datas_infos )

    def encode(self, column_data: np.ndarray) -> np.ndarray:
        return super().encode(encoder_time_to_float(column_data ) )

class FET3FrequencyLevelLanguage(FETImpossible):
    pass


class FETMultiplexerMostFrequentsValues(FETMultiplexerAll, FeatureEngineeringTemplate):
    fet_encoder_is_lossless = True
    fet_is_encoder = True
    fet_is_decoder = True

    def get_list_encoded_columns_name(self, user_column_name: str) -> list:
        return [
                f"{user_column_name}{SEPARATOR_IN_COLUMNS_NAMES}{type(self).__name__}{SEPARATOR_IN_COLUMNS_NAMES}{columns_names}"
                for columns_names in self._numerator.categories_[0].tolist()
            ]

    def _create_configuration(self, column_data: np.ndarray, column_datas_infos: Column_datas_infos) -> NoReturn:
        super()._create_configuration(column_data, column_datas_infos)
        from collections import Counter

        # limit of columns to be created
        most_frequent_values_limit = min( column_datas_infos.most_frequent_values_count , FETMultiplexerMostFrequentsValues_MAX_COLUMNS_TO_CREATE )

        # scikit  .transform do not work well, if  None was in the data of .fit
        column_data_wn = column_data[ column_data != None ]

        # keep only most_frequent_values_limit labels
        labels = Counter(column_data_wn).most_common()
        labels = (
            labels if len(labels) <= most_frequent_values_limit
            else labels[:most_frequent_values_limit]
        )
        labels = np.array([label[0] for label in labels]).reshape(-1, 1)
        self._numerator = OneHotEncoder(  handle_unknown='ignore'  ).fit( np.append( labels , [np.nan ] ) .reshape( -1 , 1 ))

    @abstractmethod
    def encode(self, column_data: np.ndarray) -> np.ndarray:
        pass

    @classmethod
    def cls_get_activation_cost(cls, column_datas_infos: Column_datas_infos) -> int:
        return min( column_datas_infos.most_frequent_values_count , FETMultiplexerMostFrequentsValues_MAX_COLUMNS_TO_CREATE )

    @classmethod
    def cls_is_possible_to_enable_this_fet_with_this_infos(cls, column_datas_infos: Column_datas_infos ) -> bool:
        # if the parent class say it cannot be activated then this is it
        if not super().cls_is_possible_to_enable_this_fet_with_this_infos(column_datas_infos ):
            return False
        elif column_datas_infos.datatype in (
            DatasetColumnDataType.IGNORE,
            DatasetColumnDataType.JSON,
            ):
            return False
        elif column_datas_infos.most_frequent_values_count < FETMultiplexerMostFrequentsValues_MIN_MOST_FREQUENT_VALUES_COUNT_TO_ENABLE :
            return False
        else:
            return True


class FETMultiplexerMostFrequentsValuesFloat(FETMultiplexerMostFrequentsValues):
    # Overview
    # Encoder - encoded data in range [0.1, 0.9]
    # Decoder - losseless
    # Serialization - tested

    def _create_configuration(self, column_data: np.ndarray, column_datas_infos: Column_datas_infos) -> NoReturn:
        super()._create_configuration( column_data , column_datas_infos)

    def encode(self, column_data: np.ndarray) -> np.ndarray:
        encoded_column_data = self._numerator.transform( column_data.reshape(-1, 1 ) )
        return encoded_column_data.toarray()

    def decode(self, column_data: np.ndarray) -> np.ndarray:
        return self._numerator.inverse_transform( column_data ).ravel( )


class FETMultiplexerMostFrequentsValuesLabel(FETMultiplexerMostFrequentsValuesFloat, FETMultiplexerMostFrequentsValues):
    # Overview
    # Encoder - encoded data in range [0.1, 0.9]
    # Decoder - losseless
    # Serialization - tested

    # def _serialize(self) -> dict:
    #     return super()._serialize()

    def _create_configuration(self, column_data: np.ndarray, column_datas_infos: Column_datas_infos) -> NoReturn:
        self._update_preprocessing_for_labels_encoder( column_data )
        datas_wn = [ v for v in column_data if not pd.isnull( v ) ]
        list_unique = np.asarray(np.unique( datas_wn , return_counts=True)).T
        prepared_column_data = list_unique[list_unique[:, 1].argsort()][
            : FETMultiplexerMostFrequentsValues_MAX_COLUMNS_TO_CREATE - 2
        ]
        prepared_column_data = np.array([x[0] for x in prepared_column_data] , dtype=object )   # need dtype=object to handle nan inside the array otherwise it is 'nan'
        prepared_column_data = np.append(prepared_column_data, [ np.nan ] ).reshape(-1, 1 )
        self._numerator = OneHotEncoder(  handle_unknown='ignore'  ).fit(prepared_column_data)

    def encode(self, column_data: np.ndarray) -> np.ndarray:
        self._update_preprocessing_for_labels_encoder( column_data )
        # OneHotEncoder will convert all unknown label and nan to a list of 0
        return self._numerator.transform( column_data.reshape(-1, 1)).toarray()

    def decode(self, column_data: np.ndarray) -> np.ndarray:
        return self._numerator.inverse_transform( column_data.astype( int ) )


class FETMultiplexerMostFrequentsValuesDate(FETMultiplexerMostFrequentsValuesFloat, FETMultiplexerMostFrequentsValues):
    # Overview
    # Encoder - encoded data in range [0.1, 0.9]
    # Decoder - error is greater than 1^(-10) if date is None
    # Serialization - tested

    def get_list_encoded_columns_name(self, user_column_name: str) -> list:
        return super(FETMultiplexerMostFrequentsValuesDate, self).get_list_encoded_columns_name( user_column_name )

    def _create_configuration(self, column_data: np.ndarray, column_datas_infos: Column_datas_infos) -> NoReturn:
        prepared_column_data = encoder_date_to_float(  (column_data) )
        super(FETMultiplexerMostFrequentsValuesDate, self)._create_configuration(prepared_column_data, column_datas_infos)

    def encode(self, column_data: np.ndarray) -> np.ndarray:
        return super(FETMultiplexerMostFrequentsValuesDate, self).encode(
            encoder_date_to_float(  (column_data) )
        )

    def decode(self, column_data: np.ndarray) -> np.ndarray:
        return decoder_float_to_date(super(FETMultiplexerMostFrequentsValuesDate, self ).decode(column_data ) )


class FETMultiplexerMostFrequentsValuesTime(FETMultiplexerMostFrequentsValuesFloat, FETMultiplexerMostFrequentsValues):
    # Overview
    # Encoder - encoded data in range [0.1, 0.9]
    # Decoder - error is greater than 1^(-10) if time is None
    # Serialization - tested

    def get_list_encoded_columns_name(self, user_column_name: str) -> list:
        return super(FETMultiplexerMostFrequentsValuesTime, self).get_list_encoded_columns_name( user_column_name )

    def _create_configuration(self, column_data: np.ndarray, column_datas_infos: Column_datas_infos) -> NoReturn:
        column_data = encoder_time_to_float( (column_data).ravel( ) )
        super(FETMultiplexerMostFrequentsValuesTime, self)._create_configuration(column_data, column_datas_infos)

    def encode(self, column_data: np.ndarray) -> np.ndarray:
        return super(FETMultiplexerMostFrequentsValuesTime, self).encode(encoder_time_to_float(column_data ) )

    def decode(self, column_data: np.ndarray) -> np.ndarray:
        return decoder_float_to_time(super(FETMultiplexerMostFrequentsValuesTime, self ).decode(column_data ) ).ravel( )


class FETMultiplexerMostFrequentsValuesLanguage( FETImpossible):
    pass


# ======================================================================================================================
# ===================================================| FET6Power |=======================================================
# ======================================================================================================================
class FET6Power(FeatureEngineeringTemplate):
    """ "0_33", "0_5", "0_66", "1_5", "2", "3"
    Encoding process:
    for x in column_data:
        x_0_33 = x^(0.33)
        x_0_5 = x^(0.5)
        x_0_66 = x^(0.66)
        x_1_5 = x^(1.5)
        x_2 = x^(2)
        x_3 = x^(3)
    MinMaxScaler after

    Decoding process:
    MinMaxScaler before
    for x in column_data:
        x_0_33 = x^(1/0.33)
        x_0_5 = x^(1/0.5)
        x_0_66 = x^1/(0.66)
        x_1_5 = x^(1/1.5)
        x_2 = x^(1/2)
        x_3 = x^(1/3)
        x = np.mean([x_0_33, x_0_5, x_0_66, x_1_5, x_2, x_3])
    """

    fet_encoder_is_lossless = True
    fet_is_encoder = True
    fet_is_decoder = True

    @abstractmethod
    def _create_configuration(self, column_data: np.ndarray, column_datas_infos: Column_datas_infos) -> NoReturn:
        pass

    def load_serialized_fet_configuration(self, serialized_data: dict ) -> NoReturn:
        for scaler_coefficient_str in FET6Power_SCALER_COEFFICIENTS_POWER:
            # scaler = MinMaxScaler(feature_range=(0.1, 0.9))
            # scaler.n_samples_seen_ = serialized_data[f"{scaler_coefficient_str}_n_samples"]
            # scaler.scale_ = np.array([serialized_data[f"{scaler_coefficient_str}_scale_number"]])
            # scaler.min_ = np.array([serialized_data[f"{scaler_coefficient_str}_min"]])
            # scaler.data_min_ = np.array([serialized_data[f"{scaler_coefficient_str}_data_min"]])
            # scaler.data_max_ = np.array([serialized_data[f"{scaler_coefficient_str}_data_max"]])
            # scaler.data_range_ = scaler.data_max_ - scaler.data_min_
            scaler = pickle_load_str( serialized_data[ f"pickle.dumps.MinMaxScaler{scaler_coefficient_str}" ] )
            setattr(self, f"_scaler_{scaler_coefficient_str}", scaler)


    def serialize_fet_configuration(self ) -> dict:
        serialized_data = dict()
        for scaler_coefficient_str in FET6Power_SCALER_COEFFICIENTS_POWER:
            scaler = getattr(self, f"_scaler_{scaler_coefficient_str}")
            # serialized_data[f"{scaler_coefficient_str}_n_samples"] = scaler.n_samples_seen_
            # serialized_data[f"{scaler_coefficient_str}_scale_number"] = scaler.scale_[0]
            # serialized_data[f"{scaler_coefficient_str}_min"] = scaler.min_[0]
            # serialized_data[f"{scaler_coefficient_str}_data_min"] = scaler.data_min_[0]
            # serialized_data[f"{scaler_coefficient_str}_data_max"] = scaler.data_max_[0]
            serialized_data.update(
                        { f"pickle.dumps.MinMaxScaler{scaler_coefficient_str}": pickle_dump_str( scaler ) }
            )
        return serialized_data

    @abstractmethod
    def encode(self, column_data: np.ndarray) -> np.ndarray:
        pass

    def get_list_encoded_columns_name( self, user_column_name: str ) -> list:
        return [
            f"{user_column_name}{SEPARATOR_IN_COLUMNS_NAMES}{type( self ).__name__}{SEPARATOR_IN_COLUMNS_NAMES}POWER{power_coefficient}"
            for power_coefficient in FET6Power_SCALER_COEFFICIENTS_POWER
        ]

    @classmethod
    def cls_get_activation_cost(cls, column_datas_infos: Column_datas_infos) -> int:
        return len(FET6Power_SCALER_COEFFICIENTS_POWER)

    @classmethod
    def cls_is_possible_to_enable_this_fet_with_this_infos(cls, column_datas_infos: Column_datas_infos ) -> bool:

        # if the parent class say it cannot be activated then this is it
        if not super().cls_is_possible_to_enable_this_fet_with_this_infos(column_datas_infos ):
            return False

        return column_datas_infos.datatype == DatasetColumnDataType.FLOAT and column_datas_infos.min > 0.25


class FET6PowerFloat(FET6Power):
    # can encode values >= 0 -- to support 0 we will add 0.001 to every values we encode
    # Encoder - encoded data in range [0.1, 0.9]
    # Decoder - losseless, but in some case error can be greater than 1^(-10)
    # Serialization - tested

    def __update_out_of_range_values(self, array: np.array ) -> np.array:
        for i in range(len(array)):
            if not pd.isnull(array[i]) and array[i] < 0:
                self.add_warning(f"Detected out of range value {array[i]} replaced by {FET6PowerFloat._min_possible_value} [is_re_run_enc_dec];")
                array[i] = FET6PowerFloat._min_possible_value
        return array

    def _create_configuration(self, column_data: np.ndarray, column_datas_infos: Column_datas_infos) -> NoReturn:
        for scaler_coefficient_str in FET6Power_SCALER_COEFFICIENTS_POWER:
            scaler_coefficient = float(scaler_coefficient_str.replace("_", "."))
            scaler = MinMaxScaler(feature_range=(0.1, 0.9))
            column_data_plus = column_data[np.where(~pd.isna(column_data))] + 0.001
            scaler.fit(np.power(column_data_plus[np.where(~pd.isna(column_data_plus))], scaler_coefficient).reshape(-1, 1))
            setattr(self, f"_scaler_{scaler_coefficient_str}", scaler)

    def encode(self, column_data: np.ndarray) -> np.ndarray:
        column_data_v = self.__update_out_of_range_values( column_data.copy( ) )
        column_data_v[ ~pd.isnull( column_data_v ) ] += 0.001
        encoded_data = np.array([])
        for scaler_coefficient_str in FET6Power_SCALER_COEFFICIENTS_POWER:
            scaler_coefficient = float(scaler_coefficient_str.replace("_", "."))
            scaler = getattr(self, f"_scaler_{scaler_coefficient_str}")
            _encoded_data = np.array([])
            for val in column_data_v:
                if pd.isna(val):
                    _encoded_data = np.concatenate((_encoded_data, [0.5]))   # TODO for more speed fill the array at begining with 0 and replace concatenate by =
                else:
                    _encoded_data = np.concatenate((_encoded_data, scaler.transform(np.power([val], scaler_coefficient).reshape(-1, 1))[0]))
            if len(encoded_data) == 0:
                encoded_data = _encoded_data
            else:
                encoded_data = np.column_stack((encoded_data, _encoded_data))
        return encoded_data

    def decode(self, column_data: np.ndarray) -> np.ndarray:
        decoded_data = np.array([])
        column_data = np.transpose(column_data)
        for i in range(len(column_data)):
            scaler_coefficient_str = FET6Power_SCALER_COEFFICIENTS_POWER[i]
            scaler_coefficient = float(scaler_coefficient_str.replace("_", "."))
            scaler = getattr(self, f"_scaler_{scaler_coefficient_str}")
            descaled_data = scaler.inverse_transform(column_data[i].reshape(-1, 1))
            decoded_column_data = np.power(descaled_data, 1 / scaler_coefficient)
            decoded_column_data = self.__update_out_of_range_values(decoded_column_data )
            if len(decoded_data) == 0:
                decoded_data = decoded_column_data
            else:
                decoded_data = np.column_stack((decoded_data, decoded_column_data))

        return  decoded_data.mean(axis=1 ) - 0.001


@classmethod
def cls_is_possible_to_enable_this_fet_with_this_infos( cls, column_datas_infos: Column_datas_infos ) -> bool:
    # if the parent class say it cannot be activated then this is it
    if not super( ).cls_is_possible_to_enable_this_fet_with_this_infos( column_datas_infos ):
        return False
    return column_datas_infos.min >= 0 and column_datas_infos.max <= 999000


class FET6PowerLabel( FETImpossible):
    pass


class FET6PowerTime( FETImpossible):
    pass


class FET6PowerDate( FETImpossible):
    pass


class FET6PowerLanguage( FETImpossible):
    pass


# ======================================================================================================================
# ============================================| FETNumericStandard |====================================================
# ======================================================================================================================
class FETNumericStandard(FeatureEngineeringTemplate):
    fet_encoder_is_lossless = True
    fet_is_encoder = True
    fet_is_decoder = True

    @abstractmethod
    def _create_configuration(self, column_data: np.ndarray, column_datas_infos: Column_datas_infos) -> NoReturn:
        pass

    def load_serialized_fet_configuration(self, serialized_data: dict ) -> NoReturn:
        scaler = StandardScaler()
        scaler.mean_ = np.array(serialized_data["mean"])
        scaler.scale_ = np.array(serialized_data["scale"])
        scaler.var_ = np.array(serialized_data["var"])
        self._standard_scaler = scaler

    def serialize_fet_configuration(self ) -> dict:
        return {
            "mean": self._standard_scaler.mean_.tolist(),
            "scale": self._standard_scaler.scale_.tolist(),
            "var": self._standard_scaler.var_.tolist(),
        }

    @abstractmethod
    def encode(self, column_data: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def decode(self, column_data: np.ndarray) -> np.ndarray:
        pass

    @classmethod
    def cls_get_activation_cost(cls, column_datas_infos: Column_datas_infos) -> int:
        return 1


class FETNumericStandardFloat(FETNumericStandard, FETNumericMinMaxFloat):
    # Overview
    # Encoder - encoded data in range [0.1, 0.9]
    # Decoder - losseless, but in some case error can be greater than 1^(-10)
    # Serialization - tested

    def _create_configuration(self, column_data: np.ndarray, column_datas_infos: Column_datas_infos) -> NoReturn:
        self._standard_scaler = StandardScaler()
        self._standard_scaler.fit(np.append(column_data, np.array([ np.nan ])).reshape(-1, 1))
        data_scaled = self._standard_scaler.transform( column_data.reshape( -1, 1 ) )
        # will run FETNumericMinMaxFloat._create_configuration
        super( FETNumericStandard, self )._create_configuration( data_scaled , column_datas_infos )

    def serialize_fet_configuration(self ) -> dict:
        data = super().serialize_fet_configuration( )
        # will run FETNumericMinMaxFloat._serialize
        data.update(super(FETNumericStandard, self).serialize_fet_configuration( ) )
        return data

    def load_serialized_fet_configuration(self, serialized_data: dict ) -> NoReturn:
        super().load_serialized_fet_configuration(serialized_data )
        # will run FETNumericMinMaxFloat._load_serialized_configuration
        super(FETNumericStandard, self).load_serialized_fet_configuration(serialized_data )

    def encode(self, column_data: np.ndarray) -> np.ndarray:
        encoded_column_data = self._standard_scaler.transform( column_data.reshape(-1, 1)).ravel( )
        # will run FETNumericMinMaxFloat.encode
        encoded_column_data_scaled = super(FETNumericStandard, self).encode( encoded_column_data )
        return encoded_column_data_scaled

    def decode(self, scaled_column_data: np.ndarray) -> np.ndarray:
        # will run FETNumericMinMaxFloat.decode
        scaled_column_data = super(FETNumericStandard, self).decode(scaled_column_data)
        decoded_data = self._standard_scaler.inverse_transform( scaled_column_data.reshape( -1, 1 ) )
        return decoded_data.ravel()


class FETNumericStandardLabel(FETNumericStandardFloat, FETNumericStandard):
    # Overview
    # Encoder - encoded data in range [0.1, 0.9]
    # Decoder - losseless
    # Serialization - tested


    def __filter_config_data(self, array: np.array) -> np.array:
        return array[np.where(pd.notna(array))]

    def _create_configuration(self, column_data: np.ndarray, column_datas_infos: Column_datas_infos) -> NoReturn:
        self._numerator = LabelEncoder()
        self._update_preprocessing_for_labels_encoder( column_data )
        numeric_column_data = self._numerator.fit_transform(np.append( column_data , [ np.nan ] ) )
        # super do FETNumericStandardFloat
        super(FETNumericStandardLabel, self)._create_configuration(numeric_column_data, column_datas_infos)

    def load_serialized_fet_configuration(self, serialized_data: dict ) -> NoReturn:
        # super do FETNumericStandardFloat
        super(FETNumericStandardLabel, self).load_serialized_fet_configuration(serialized_data )
        # numerator = LabelEncoder()
        # numerator.classes_ = np.array(serialized_data["classes"])
        # self._numerator = numerator
        self._numerator = pickle_load_str( serialized_data[ "pickle.dumps.LabelEncoder" ] )

    def serialize_fet_configuration(self ) -> dict:
        # super do FETNumericStandardFloat
        serialized_data = super( FETNumericStandardLabel, self ).serialize_fet_configuration( )
        serialized_data.update( {"pickle.dumps.LabelEncoder" : pickle_dump_str( self._numerator ) } )
        return serialized_data

    def encode(self, column_data: np.ndarray) -> np.ndarray:
        self._update_preprocessing_for_labels_encoder( column_data )
        self._update_unknown_labels_for_labels_encoder( column_data )
        numeric_column_data = self._numerator.transform( column_data )
        # super do FETNumericStandardFloat
        return super(FETNumericStandardLabel, self).encode(numeric_column_data)

    def decode(self, scaled_column_data: np.ndarray) -> np.ndarray:
        # super do FETNumericStandardFloat
        decoded_scaled_column_data_pp = super(FETNumericStandardLabel, self).decode(scaled_column_data)
        self._update_overlimits_for_labels_decoder(decoded_scaled_column_data_pp )
        decoded_numeric_column = self._numerator.inverse_transform( decoded_scaled_column_data_pp.astype( int ) )
        return decoded_numeric_column


class FETNumericStandardDate(FETNumericStandardFloat, FETNumericStandard):
    # Overview
    # Encoder - encoded data in range [0.1, 0.9]
    # Decoder - losseless
    # Serialization - tested

    def _create_configuration(self, column_data: np.ndarray, column_datas_infos: Column_datas_infos) -> NoReturn:
        e = encoder_date_to_float(column_data )
        super()._create_configuration( e, column_datas_infos )

    def encode(self, column_data: np.ndarray) -> np.ndarray:
        e = encoder_date_to_float(column_data )
        return super().encode( e )

    def decode(self, column_data: np.ndarray) -> np.ndarray:
        return decoder_float_to_date( super( ).decode(column_data ) )


class FETNumericStandardTime(FETNumericStandardFloat, FETNumericStandard):
    # Overview
    # Encoder - encoded data in range [0.1, 0.9]
    # Decoder - losseless
    # Serialization - tested

    def _create_configuration(self, column_data: np.ndarray, column_datas_infos: Column_datas_infos) -> NoReturn:
        super()._create_configuration(encoder_time_to_float( (column_data) ), column_datas_infos )

    def encode(self, column_data: np.ndarray) -> np.ndarray:
        return super().encode(encoder_time_to_float(column_data ) )

    def decode(self, column_data: np.ndarray) -> np.ndarray:
        return decoder_float_to_time(super( ).decode(column_data ) )


class FETNumericStandardLanguage( FETImpossible):
    pass


# ======================================================================================================================
# ==========================================| FETNumericPowerTransformer |==============================================
# ======================================================================================================================
class FETNumericPowerTransformer(FeatureEngineeringTemplate):
    fet_encoder_is_lossless = True
    fet_is_encoder = True
    fet_is_decoder = True

    @abstractmethod
    def _create_configuration(self, column_data: np.ndarray, column_datas_infos: Column_datas_infos) -> NoReturn:
        pass

    def load_serialized_fet_configuration(self, serialized_data: dict ) -> NoReturn:
        # scaler = PowerTransformer()
        # scaler.n_features_in_ = serialized_data["n_features_in"]
        # scaler.lambdas_ = np.array(serialized_data["lambdas"])
        # scaler.method = serialized_data["method"]
        # scaler.standardize = serialized_data["standardize"]
        # self._power_scaler = scaler
        self._power_scaler = pickle_load_str( serialized_data[ "pickle.dumps.PowerTransformer" ] )

    def serialize_fet_configuration(self ) -> dict:
        # return {
        #     "n_features_in": self._power_scaler.n_features_in_,
        #     "lambdas": self._power_scaler.lambdas_.tolist(),
        #     "method": self._power_scaler.method,
        #     "standardize": False,
        # }
        return { "pickle.dumps.PowerTransformer": pickle_dump_str( self._power_scaler ) }

    @abstractmethod
    def encode(self, column_data: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def decode(self, column_data: np.ndarray) -> np.ndarray:
        pass

    @classmethod
    def cls_get_activation_cost(cls, column_datas_infos: Column_datas_infos) -> int:
        return 1


class FETNumericPowerTransformerFloat(FETNumericPowerTransformer, FETNumericMinMaxFloat):
    # Overview
    # Encoder - encoded data in range [0.1, 0.9]
    # Decoder - losseless
    # Serialization - tested

    def _create_configuration(self, column_data: np.ndarray, column_datas_infos: Column_datas_infos) -> NoReturn:
        self._power_scaler = PowerTransformer( )
        self._power_scaler.fit(np.append(column_data, np.array([ np.nan ])).reshape(-1, 1))
        # super will run method of FETNumericMinMaxFloat
        super(FETNumericPowerTransformer, self)._create_configuration(
            self._power_scaler.transform(column_data.reshape(-1, 1)), column_datas_infos
        )

    def load_serialized_fet_configuration(self, serialized_data: dict ) -> NoReturn:
        super().load_serialized_fet_configuration(serialized_data )
        # super will run method of FETNumericMinMaxFloat
        super(FETNumericPowerTransformer, self).load_serialized_fet_configuration(serialized_data )

    def serialize_fet_configuration(self ) -> dict:
        data = super().serialize_fet_configuration( )
        # super will run method of FETNumericMinMaxFloat
        data.update(super(FETNumericPowerTransformer, self).serialize_fet_configuration( ) )
        return data

    def encode(self, column_data: np.ndarray) -> np.ndarray:
        encoded_column_data = self._power_scaler.transform( column_data.reshape(-1, 1)).ravel( )
        # will run FETNumericMinMaxFloat.encode
        encoded_column_data_scaled = super(FETNumericPowerTransformer, self).encode( encoded_column_data )
        return encoded_column_data_scaled

    def decode(self, scaled_column_data: np.ndarray) -> np.ndarray:
        # super will run method of FETNumericMinMaxFloat
        d= super(FETNumericPowerTransformer, self).decode(scaled_column_data).reshape(-1, 1)
        return  self._power_scaler.inverse_transform( d ).ravel()


class FETNumericPowerTransformerLabel(FETNumericPowerTransformerFloat, FETNumericPowerTransformer):
    # Overview
    # Encoder - encoded data in range [0.1, 0.9]
    # Decoder - losseless
    # Serialization - tested

    def _create_configuration(self, column_data: np.ndarray, column_datas_infos: Column_datas_infos) -> NoReturn:
        self._numerator = LabelEncoder()
        self._update_preprocessing_for_labels_encoder( column_data )
        numeric_column_data = self._numerator.fit_transform(np.append( column_data , [np.nan ]  ))
        super(FETNumericPowerTransformerLabel, self)._create_configuration(numeric_column_data, column_datas_infos)

    def load_serialized_fet_configuration(self, serialized_data: dict ) -> NoReturn:
        super(FETNumericPowerTransformerLabel, self).load_serialized_fet_configuration(serialized_data )
        # numerator = LabelEncoder()
        # numerator.classes_ = np.array(serialized_data["classes"])
        # self._numerator = numerator
        self._numerator = pickle_load_str( serialized_data[ "pickle.dumps.LabelEncoder" ] )

    def serialize_fet_configuration(self ) -> dict:
        # serialized_data = super()._serialize()
        # serialized_data.update({"classes": self._numerator.classes_.tolist()})
        # return serialized_data
        serialized_data = super( ).serialize_fet_configuration( )
        serialized_data.update( { "pickle.dumps.LabelEncoder": pickle_dump_str( self._numerator ) } )
        return serialized_data

    def encode(self, column_data: np.ndarray) -> np.ndarray:
        self._update_preprocessing_for_labels_encoder( column_data )
        self._update_unknown_labels_for_labels_encoder( column_data )
        numeric_column_data = self._numerator.transform( column_data )
        return super(FETNumericPowerTransformerLabel, self).encode(numeric_column_data)

    def decode(self, scaled_column_data: np.ndarray) -> np.ndarray:
        decoded_scaled_column_data = super(FETNumericPowerTransformerLabel, self).decode(scaled_column_data)
        self._update_overlimits_for_labels_decoder( decoded_scaled_column_data )
        decoded_numeric_column = self._numerator.inverse_transform( decoded_scaled_column_data.astype( int ) )
        return decoded_numeric_column


class FETNumericPowerTransformerDate(FETNumericPowerTransformerFloat, FETNumericPowerTransformer):
    # Overview
    # Encoder - encoded data in range [0.1, 0.9]
    # Decoder - losseless
    # Serialization - tested

    def _create_configuration(self, column_data: np.ndarray, column_datas_infos: Column_datas_infos) -> NoReturn:
        super()._create_configuration(encoder_date_to_float(column_data ).reshape(-1, 1 ), column_datas_infos )

    def encode(self, column_data: np.ndarray) -> np.ndarray:
        return super().encode(encoder_date_to_float(column_data ).reshape(-1, 1 ) )

    def decode(self, column_data: np.ndarray) -> np.ndarray:
        return decoder_float_to_date(super( ).decode(column_data ) ).ravel( )


class FETNumericPowerTransformerTime(FETNumericPowerTransformerFloat, FETNumericPowerTransformer):
    # Overview
    # Encoder - encoded data in range [0.1, 0.9]
    # Decoder - losseless
    # Serialization - tested

    def _create_configuration(self, column_data: np.ndarray, column_datas_infos: Column_datas_infos) -> NoReturn:
        super()._create_configuration(encoder_time_to_float(column_data ).reshape(-1, 1 ), column_datas_infos )

    def encode(self, column_data: np.ndarray) -> np.ndarray:
        return super().encode(encoder_time_to_float(column_data ).reshape(-1, 1 ) )

    def decode(self, column_data: np.ndarray) -> np.ndarray:
        return decoder_float_to_time(super( ).decode(column_data ) ).ravel( )


class FETNumericPowerTransformerLanguage( FETImpossible):
    pass


# ======================================================================================================================
# =========================================| FETScalerQuantileTransformer |=============================================
# ======================================================================================================================
class FETNumericQuantileTransformer(FeatureEngineeringTemplate):
    fet_encoder_is_lossless = True
    fet_is_encoder = True
    fet_is_decoder = True

    @abstractmethod
    def _create_configuration(self, column_data: np.ndarray, column_datas_infos: Column_datas_infos) -> NoReturn:
        pass

    def load_serialized_fet_configuration(self, serialized_data: dict ) -> NoReturn:
        # scaler = QuantileTransformer()
        # scaler.n_features_in_ = serialized_data["n_features_in_"]
        # scaler.output_distribution = serialized_data["output_distribution"]
        # scaler.ignore_implicit_zeros = serialized_data["ignore_implicit_zeros"]
        # scaler.subsample = serialized_data["subsample"]
        # scaler.random_state = serialized_data["random_state"]
        # scaler.copy = serialized_data["copy"]
        # scaler.quantiles_ = np.array(serialized_data["quantiles_"])
        # scaler.references_ = np.array(serialized_data["references_"])
        # self._quantile_scaler = scaler
        self._quantile_scaler = pickle_load_str( serialized_data[ "pickle.dumps.QuantileTransformer" ] )

    def serialize_fet_configuration(self ) -> dict:
        # return {
        #     "n_features_in_": self._quantile_scaler.n_features_in_,
        #     "output_distribution": self._quantile_scaler.output_distribution,
        #     "ignore_implicit_zeros": self._quantile_scaler.ignore_implicit_zeros,
        #     "subsample": self._quantile_scaler.subsample,
        #     "random_state": self._quantile_scaler.random_state,
        #     "copy": self._quantile_scaler.copy,
        #     "quantiles_": self._quantile_scaler.quantiles_.tolist(),
        #     "references_": self._quantile_scaler.references_.tolist(),
        # }
        return {"pickle.dumps.QuantileTransformer" : pickle_dump_str( self._quantile_scaler ) }

    @abstractmethod
    def encode(self, column_data: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def decode(self, column_data: np.ndarray) -> np.ndarray:
        pass

    @classmethod
    def cls_get_activation_cost(cls, column_datas_infos: Column_datas_infos) -> int:
        return 1


class FETNumericQuantileTransformerFloat(FETNumericQuantileTransformer, FETNumericMinMaxFloat):
    # Overview
    # Encoder - encoded data in range [0.1, 0.9]
    # Decoder - losseless, but in some case error can be greater than 1^(-10)
    # Serialization - tested

    def _create_configuration(self, column_data: np.ndarray, column_datas_infos: Column_datas_infos) -> NoReturn:
        self._quantile_scaler = QuantileTransformer()
        self._quantile_scaler.fit(np.append(column_data, np.array([np.nan])).reshape(-1, 1))
        d = self._quantile_scaler.transform(column_data.reshape(-1, 1))
        # super will run method of FETNumericMinMaxFloat
        super(FETNumericQuantileTransformer, self)._create_configuration( d , column_datas_infos )

    def load_serialized_fet_configuration(self, serialized_data: dict ) -> NoReturn:
        # super will run method of FETNumericQuantileTransformer
        super().load_serialized_fet_configuration(serialized_data )
        # super will run method of FETNumericMinMaxFloat ( FETNumericMinMax in fact because there is not this method in FETNumericMinMaxFloat)
        super(FETNumericQuantileTransformer, self).load_serialized_fet_configuration(serialized_data )

    def serialize_fet_configuration(self ) -> dict:
        # super will run method of FETNumericQuantileTransformer
        data = super().serialize_fet_configuration( )
        # super will run method of FETNumericMinMaxFloat ( FETNumericMinMax in fact because there is not this method in FETNumericMinMaxFloat)
        data.update(super(FETNumericQuantileTransformer, self).serialize_fet_configuration( ) )
        return data

    def encode(self, column_data: np.ndarray) -> np.ndarray:
        encoded_column_data = self._quantile_scaler.transform( column_data.reshape( -1, 1 ) ).ravel( )
        # will run FETNumericMinMaxFloat.encode
        encoded_column_data_scaled = super( FETNumericQuantileTransformer, self ).encode( encoded_column_data )
        return encoded_column_data_scaled

    def decode(self, scaled_column_data: np.ndarray) -> np.ndarray:
        # super will run method of FETNumericMinMaxFloat
        scaled_column_data = super(FETNumericQuantileTransformer, self).decode(scaled_column_data)
        decoded_column_data = self._quantile_scaler.inverse_transform(scaled_column_data.reshape(-1, 1))
        return  decoded_column_data.ravel()


class FETNumericQuantileTransformerLabel(FETNumericQuantileTransformerFloat, FETNumericQuantileTransformer):
    # Overview
    # Encoder - encoded data in range [0.1, 0.9]
    # Decoder - losseless
    # Serialization - tested

    def _create_configuration(self, column_data: np.ndarray, column_datas_infos: Column_datas_infos) -> NoReturn:
        self._numerator = LabelEncoder()
        self._update_preprocessing_for_labels_encoder( column_data )
        numeric_column_data = self._numerator.fit_transform(np.append( column_data , [np.nan ] ) )
        super(FETNumericQuantileTransformerLabel, self)._create_configuration(numeric_column_data, column_datas_infos)

    def load_serialized_fet_configuration(self, serialized_data: dict ) -> NoReturn:
        super(FETNumericQuantileTransformerLabel, self).load_serialized_fet_configuration(serialized_data )
        # numerator = LabelEncoder()
        # numerator.classes_ = np.array(serialized_data["classes"])
        # self._numerator = numerator
        self._numerator = pickle_load_str( serialized_data[ "pickle.dumps.LabelEncoder" ] )

    def serialize_fet_configuration(self ) -> dict:
        serialized_data = super().serialize_fet_configuration( )
        # serialized_data.update({"classes": self._numerator.classes_.tolist()})
        # return serialized_data
        serialized_data.update( {"pickle.dumps.LabelEncoder" : pickle_dump_str( self._numerator ) } )
        return serialized_data

    def encode(self, column_data: np.ndarray) -> np.ndarray:
        self._update_preprocessing_for_labels_encoder( column_data )
        self._update_unknown_labels_for_labels_encoder( column_data )
        numeric_column_data = self._numerator.transform( column_data )
        return super(FETNumericQuantileTransformerLabel, self).encode(numeric_column_data)

    def decode(self, scaled_column_data: np.ndarray) -> np.ndarray:
        decoded_scaled_column_data = super(FETNumericQuantileTransformerLabel, self).decode(scaled_column_data)
        self._update_overlimits_for_labels_decoder(decoded_scaled_column_data )
        decoded_numeric_column = self._numerator.inverse_transform( decoded_scaled_column_data.astype( int ) )
        return decoded_numeric_column


class FETNumericQuantileTransformerDate(FETNumericQuantileTransformerFloat, FETNumericQuantileTransformer):
    # Overview
    # Encoder - encoded data in range [0.1, 0.9]
    # Decoder - losseless
    # Serialization - tested

    def _create_configuration(self, column_data: np.ndarray, column_datas_infos: Column_datas_infos) -> NoReturn:
        super()._create_configuration(encoder_date_to_float( column_data ), column_datas_infos )

    def encode(self, column_data: np.ndarray) -> np.ndarray:
        encoded_data = encoder_date_to_float( column_data )
        return super().encode( encoded_data )

    def decode(self, column_data: np.ndarray) -> np.ndarray:
        decoded_data = super( ).decode(column_data )
        return decoder_float_to_date( decoded_data )


class FETNumericQuantileTransformerTime(FETNumericQuantileTransformerFloat, FETNumericQuantileTransformer):
    # Overview
    # Encoder - encoded data in range [0.1, 0.9]
    # Decoder - losseless
    # Serialization - tested

    def _create_configuration(self, column_data: np.ndarray, column_datas_infos: Column_datas_infos) -> NoReturn:
        super()._create_configuration(encoder_time_to_float(column_data ), column_datas_infos )

    def encode(self, column_data: np.ndarray) -> np.ndarray:
        return super().encode(encoder_time_to_float(column_data ) )

    def decode(self, column_data: np.ndarray) -> np.ndarray:
        return decoder_float_to_time(super( ).decode(column_data ) )


class FETNumericQuantileTransformerLanguage( FETImpossible):
    pass


# ======================================================================================================================
# ======================================| FETNumericQuantileTransformerNormal |=========================================
# ======================================================================================================================
class FETNumericQuantileTransformerNormal(FeatureEngineeringTemplate):
    """
    Encoding process:
        for x in column_data:
            x = (x - np.median(column_data)) / (column_data_75 - column_data_25)

        column_data_75 - quantile 75 of column_data
        column_data_25 - quantile 25 of column_data

        MinMaxScaler after
    """
    fet_encoder_is_lossless = True
    fet_is_encoder = True
    fet_is_decoder = True

    @abstractmethod
    def _create_configuration(self, column_data: np.ndarray, column_datas_infos: Column_datas_infos) -> NoReturn:
        pass

    def load_serialized_fet_configuration(self, serialized_data: dict ) -> NoReturn:
        # scaler = QuantileTransformer(output_distribution="normal")
        # scaler.n_quantiles = serialized_data["n_quantiles"]
        # scaler.n_features_in_ = serialized_data["n_features_in_"]
        # scaler.output_distribution = serialized_data["output_distribution"]
        # scaler.ignore_implicit_zeros = serialized_data["ignore_implicit_zeros"]
        # scaler.subsample = serialized_data["subsample"]
        # scaler.random_state = serialized_data["random_state"]
        # scaler.copy = serialized_data["copy"]
        # scaler.quantiles_ = np.array(serialized_data["quantiles_"])
        # scaler.references_ = np.array(serialized_data["references_"])
        # self._quantile_transformer = scaler
        self._quantile_transformer = pickle_load_str( serialized_data[ "pickle.dumps.QuantileTransformer" ] )

    def serialize_fet_configuration(self ) -> dict:
        # return {
        #     "n_features_in_": self._quantile_transformer.n_features_in_,
        #     "output_distribution": self._quantile_transformer.output_distribution,
        #     "ignore_implicit_zeros": self._quantile_transformer.ignore_implicit_zeros,
        #     "subsample": self._quantile_transformer.subsample,
        #     "random_state": self._quantile_transformer.random_state,
        #     "copy": self._quantile_transformer.copy,
        #     "quantiles_": self._quantile_transformer.quantiles_.tolist(),
        #     "references_": self._quantile_transformer.references_.tolist(),
        #     "n_quantiles": self._quantile_transformer.n_quantiles,
        # }
        return { "pickle.dumps.QuantileTransformer": pickle_dump_str( self._quantile_transformer ) }

    @abstractmethod
    def encode(self, column_data: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def decode(self, column_data: np.ndarray) -> np.ndarray:
        pass

    @classmethod
    def cls_get_activation_cost(cls, column_datas_infos: Column_datas_infos) -> int:
        return 1


class FETNumericQuantileTransformerNormalFloat(FETNumericQuantileTransformerNormal, FETNumericMinMaxFloat):
    # Overview
    # Encoder - encoded data in range [0.1, 0.9]
    # Decoder - losseless, but in some case error can be greater than 1^(-10)
    # Serialization - tested

    def serialize_fet_configuration(self ) -> dict:
        serialized_data = super().serialize_fet_configuration( )
        serialized_data.update(super(FETNumericQuantileTransformerNormal, self).serialize_fet_configuration( ) )
        return serialized_data

    def load_serialized_fet_configuration(self, serialized_data: dict ) -> NoReturn:
        super().load_serialized_fet_configuration(serialized_data )
        super(FETNumericQuantileTransformerNormal, self).load_serialized_fet_configuration(serialized_data )

    def _create_configuration(self, column_data: np.ndarray, column_datas_infos: Column_datas_infos) -> NoReturn:
        self._quantile_transformer = QuantileTransformer(output_distribution="normal")
        self._quantile_transformer.fit(np.append(column_data, np.array([np.nan])).reshape(-1, 1))
        super(FETNumericQuantileTransformerNormal, self)._create_configuration(
            self._quantile_transformer.transform(column_data.reshape(-1, 1)), column_datas_infos
        )

    def encode(self, column_data: np.ndarray) -> np.ndarray:
        encoded_column_data = self._quantile_transformer.transform( column_data.reshape( -1, 1 ) ).ravel( )
        # will run FETNumericMinMaxFloat.encode
        encoded_column_data_scaled = super( FETNumericQuantileTransformerNormal, self ).encode( encoded_column_data )
        return encoded_column_data_scaled

    def decode(self, scaled_column_data: np.ndarray) -> np.ndarray:
        return  (
            self._quantile_transformer.inverse_transform(
                super(FETNumericQuantileTransformerNormal, self)
                .decode( (scaled_column_data ).reshape(-1, 1 ) )
                .reshape(-1, 1)
            ).ravel()
        )


class FETNumericQuantileTransformerNormalLabel(FETNumericQuantileTransformerNormalFloat, FETNumericQuantileTransformerNormal):
    # Overview
    # Encoder - encoded data in range [0.1, 0.9]
    # Decoder - losseless
    # Serialization - tested

    def _create_configuration(self, column_data: np.ndarray, column_datas_infos: Column_datas_infos) -> NoReturn:
        self._numerator = LabelEncoder()
        self._update_preprocessing_for_labels_encoder( column_data )
        numeric_column_data = self._numerator.fit_transform(np.append( column_data , [np.nan ] ) )
        super(FETNumericQuantileTransformerNormalLabel, self)._create_configuration(numeric_column_data, column_datas_infos)

    def load_serialized_fet_configuration(self, serialized_data: dict ) -> NoReturn:
        super(FETNumericQuantileTransformerNormalLabel, self).load_serialized_fet_configuration(serialized_data )
        # numerator = LabelEncoder()
        # numerator.classes_ = np.array(serialized_data["classes"])
        # self._numerator = numerator
        self._numerator = pickle_load_str( serialized_data[ "pickle.dumps.LabelEncoder" ] )

    def serialize_fet_configuration(self ) -> dict:
        serialized_data = super().serialize_fet_configuration( )
        # serialized_data.update({"classes": self._numerator.classes_.tolist()})
        # return serialized_data
        serialized_data.update( {"pickle.dumps.LabelEncoder" : pickle_dump_str( self._numerator ) } )
        return serialized_data

    def encode(self, column_data: np.ndarray) -> np.ndarray:
        self._update_preprocessing_for_labels_encoder( column_data )
        self._update_unknown_labels_for_labels_encoder( column_data )
        numeric_column_data = self._numerator.transform( column_data )
        return super(FETNumericQuantileTransformerNormalLabel, self).encode(numeric_column_data)

    def decode(self, scaled_column_data: np.ndarray) -> np.ndarray:
        decoded_scaled_column_data = super(FETNumericQuantileTransformerNormalLabel, self).decode(scaled_column_data)
        self._update_overlimits_for_labels_decoder( decoded_scaled_column_data )
        decoded_numeric_column = self._numerator.inverse_transform( decoded_scaled_column_data.astype( int ) )
        return decoded_numeric_column


class FETNumericQuantileTransformerNormalDate(FETNumericQuantileTransformerNormalFloat, FETNumericQuantileTransformerNormal):
    # Overview
    # Encoder - encoded data in range [0.1, 0.9]
    # Decoder - losseless
    # Serialization - tested

    def _create_configuration(self, column_data: np.ndarray, column_datas_infos: Column_datas_infos) -> NoReturn:
        super()._create_configuration(encoder_date_to_float( column_data ), column_datas_infos )

    def encode(self, column_data: np.ndarray) -> np.ndarray:
        return super().encode(encoder_date_to_float( column_data ).reshape(-1, 1 ) )

    def decode(self, column_data: np.ndarray) -> np.ndarray:
        return decoder_float_to_date(super( ).decode(column_data ) ).ravel( )


class FETNumericQuantileTransformerNormalTime(FETNumericQuantileTransformerNormalFloat, FETNumericQuantileTransformerNormal):
    # Overview
    # Encoder - encoded data in range [0.1, 0.9]
    # Decoder - losseless
    # Serialization - tested

    def _create_configuration(self, column_data: np.ndarray, column_datas_infos: Column_datas_infos) -> NoReturn:
        super()._create_configuration(encoder_time_to_float(column_data ), column_datas_infos )

    def encode(self, column_data: np.ndarray) -> np.ndarray:
        return super().encode(encoder_time_to_float(column_data ) )

    def decode(self, column_data: np.ndarray) -> np.ndarray:
        return decoder_float_to_time(super( ).decode(column_data ) ).ravel( )


class FETNumericQuantileTransformerNormalLanguage( FETImpossible):
    pass


# ======================================================================================================================
# ===========================================| FETNumericRobustScaler |=================================================
# ======================================================================================================================
class FETNumericRobustScaler(FeatureEngineeringTemplate):
    """
    Encoding process:
        for x in column_data:
            x = (x - np.median(column_data)) / (column_data_75 - column_data_25)

        column_data_75 - quantile 75 of column_data
        column_data_25 - quantile 25 of column_data

        MinMaxScaler after
    """

    fet_encoder_is_lossless = True
    fet_is_encoder = True
    fet_is_decoder = True

    @abstractmethod
    def _create_configuration(self, column_data: np.ndarray, column_datas_infos: Column_datas_infos) -> NoReturn:
        pass

    def load_serialized_fet_configuration(self, serialized_data: dict ) -> NoReturn:
        # scaler = RobustScaler()
        # scaler.with_centering = serialized_data["with_centering"]
        # scaler.with_scaling = serialized_data["with_scaling"]
        # scaler.quantile_range = serialized_data["quantile_range"]
        # scaler.unit_variance = serialized_data["unit_variance"]
        # scaler.copy = serialized_data["copy"]
        # scaler.n_features_in_ = serialized_data["n_features_in_"]
        # scaler.center_ = np.array(serialized_data["center_"])
        # scaler.scale_ = np.array(serialized_data["scale_"])
        # self._robust_scaler = scaler
        self._robust_scaler = pickle_load_str( serialized_data[ "pickle.dumps.RobustScaler" ] )

    def serialize_fet_configuration(self ) -> dict:
        # return {
        #     "with_centering": self._robust_scaler.with_centering,
        #     "with_scaling": self._robust_scaler.with_scaling,
        #     "quantile_range": self._robust_scaler.quantile_range,
        #     "unit_variance": self._robust_scaler.unit_variance,
        #     "copy": self._robust_scaler.copy,
        #     "n_features_in_": self._robust_scaler.n_features_in_,
        #     "center_": self._robust_scaler.center_.tolist(),
        #     "scale_": self._robust_scaler.scale_.tolist(),
        # }
        return { "pickle.dumps.RobustScaler": pickle_dump_str( self._robust_scaler ) }

    @abstractmethod
    def encode(self, column_data: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def decode(self, column_data: np.ndarray) -> np.ndarray:
        pass

    @classmethod
    def cls_get_activation_cost(cls, column_datas_infos: Column_datas_infos) -> int:
        return 1


class FETNumericRobustScalerFloat(FETNumericRobustScaler, FETNumericMinMaxFloat):
    # Overview
    # Encoder - encoded data in range [0.1, 0.9]
    # Decoder - losseless, but in some case error can be greater than 1^(-10)
    # Serialization - tested

    def _create_configuration(self, column_data: np.ndarray, column_datas_infos: Column_datas_infos) -> NoReturn:
        column_data_r =  column_data.reshape(-1, 1 )
        self._robust_scaler = RobustScaler()
        self._robust_scaler.fit( column_data_r )
        super(FETNumericRobustScaler, self)._create_configuration(
            self._robust_scaler.transform( column_data_r ), column_datas_infos
        )

    def serialize_fet_configuration(self ) -> dict:
        serialized_data = super().serialize_fet_configuration( )
        serialized_data.update(super(FETNumericRobustScaler, self).serialize_fet_configuration( ) )
        return serialized_data

    def load_serialized_fet_configuration(self, serialized_data: dict ) -> NoReturn:
        super().load_serialized_fet_configuration(serialized_data )
        super(FETNumericRobustScaler, self).load_serialized_fet_configuration(serialized_data )

    def encode(self, column_data: np.ndarray) -> np.ndarray:
        encoded_column_data = self._robust_scaler.transform( column_data.reshape( -1, 1 ) ).ravel( )
        # will run FETNumericMinMaxFloat.encode
        encoded_column_data_scaled = super( FETNumericRobustScaler, self ).encode( encoded_column_data )
        return encoded_column_data_scaled

    def decode(self, scaled_column_data: np.ndarray) -> np.ndarray:
        return  (
            self._robust_scaler.inverse_transform(super(FETNumericRobustScaler, self).decode(scaled_column_data).reshape(-1, 1)).ravel()
        )


class FETNumericRobustScalerLabel(FETNumericRobustScalerFloat, FETNumericRobustScaler):
    # Overview
    # Encoder - encoded data in range [0.1, 0.9]
    # Decoder - losseless
    # Serialization - tested

    def _create_configuration(self, column_data: np.ndarray, column_datas_infos: Column_datas_infos) -> NoReturn:
        self._numerator = LabelEncoder()
        self._update_preprocessing_for_labels_encoder( column_data )
        numeric_column_data = self._numerator.fit_transform(np.append( column_data , [np.nan ] ) )
        super(FETNumericRobustScalerLabel, self)._create_configuration(numeric_column_data, column_datas_infos)

    def load_serialized_fet_configuration(self, serialized_data: dict ) -> NoReturn:
        super(FETNumericRobustScalerLabel, self).load_serialized_fet_configuration(serialized_data )
        # numerator = LabelEncoder()
        # numerator.classes_ = np.array(serialized_data["classes"])
        # self._numerator = numerator
        self._numerator = pickle_load_str( serialized_data[ "pickle.dumps.LabelEncoder" ] )

    def serialize_fet_configuration(self ) -> dict:
        serialized_data = super().serialize_fet_configuration( )
        # serialized_data.update({"classes": self._numerator.classes_.tolist()})
        # return serialized_data
        serialized_data.update( { "pickle.dumps.LabelEncoder": pickle_dump_str( self._numerator ) } )
        return serialized_data

    def encode(self, column_data: np.ndarray) -> np.ndarray:
        self._update_preprocessing_for_labels_encoder( column_data )
        self._update_unknown_labels_for_labels_encoder( column_data )
        numeric_column_data = self._numerator.transform( column_data )
        return super(FETNumericRobustScalerLabel, self).encode(numeric_column_data)

    def decode(self, scaled_column_data: np.ndarray) -> np.ndarray:
        decoded_scaled_column_data = super( FETNumericRobustScalerLabel , self ).decode( scaled_column_data )
        self._update_overlimits_for_labels_decoder( decoded_scaled_column_data )
        decoded_numeric_column = self._numerator.inverse_transform( decoded_scaled_column_data.astype( int ) )
        return decoded_numeric_column


class FETNumericRobustScalerDate(FETNumericRobustScalerFloat, FETNumericRobustScaler):
    # Overview
    # Encoder - encoded data in range [0.1, 0.9]
    # Decoder - lossless
    # Serialization - tested

    def _create_configuration(self, column_data: np.ndarray, column_datas_infos: Column_datas_infos) -> NoReturn:
        super()._create_configuration(encoder_date_to_float( (column_data) ), column_datas_infos )

    def encode(self, column_data: np.ndarray) -> np.ndarray:
        return super().encode(encoder_date_to_float( (column_data) ) )

    def decode(self, column_data: np.ndarray) -> np.ndarray:
        return decoder_float_to_date(super( ).decode(column_data ) )


class FETNumericRobustScalerTime(FETNumericRobustScalerFloat, FETNumericRobustScaler):
    # Overview
    # Encoder - encoded data in range [0.1, 0.9]
    # Decoder - lossless
    # Serialization - tested

    def _create_configuration(self, column_data: np.ndarray, column_datas_infos: Column_datas_infos) -> NoReturn:
        super()._create_configuration(encoder_time_to_float(column_data ), column_datas_infos )

    def encode(self, column_data: np.ndarray) -> np.ndarray:
        return super().encode(encoder_time_to_float(column_data ) )

    def decode(self, column_data: np.ndarray) -> np.ndarray:
        return decoder_float_to_time(super( ).decode(column_data ) )


class FETNumericRobustScalerLanguage( FETImpossible):
    pass


class FETSentenceTransformer( FETNumericMinMaxFloat ):
    fet_encoder_is_lossless = False
    fet_is_encoder = True
    fet_is_decoder = False


    def _create_configuration(self, column_data: np.ndarray, column_datas_infos: Column_datas_infos) -> NoReturn:
        # we generate two lines of data , first row with min , second with max , this is to make configuration of min max float scaler
        data_to_scale = np.array([ [sentence_transformer_model_vectors_value_min]*sentence_transformer_model_vectors_len, [sentence_transformer_model_vectors_value_max]*sentence_transformer_model_vectors_len ])

        # will run FETNumericMinMaxFloat.
        super( )._create_configuration( data_to_scale, column_datas_infos )

    def encode( self, column_data: np.ndarray ) -> np.ndarray:
        data_result = np.empty(( len(column_data), sentence_transformer_model_vectors_len), dtype=float)
        for i , row in enumerate( column_data ):
            if pd.isnull( column_data[ i ] ):
                data_result[ i ] = np.zeros( sentence_transformer_model_vectors_len , dtype=float )
            else:
                data_result[ i ] = sentence_transformer_model_encoder_cached( column_data[ i ]  )
        # will run FETNumericMinMaxFloat.
        encoded_column_data_scaled = super( ).encode( data_result )
        return encoded_column_data_scaled

    def decode(self, column_data: np.ndarray) -> np.ndarray:
        logger.error( "no decoder available => fet_is_decoder = False")

    def load_serialized_fet_configuration( self, serialized_data: dict ) -> NoReturn:
        # super will run method of FETNumericMinMaxFloat
        super( ).load_serialized_fet_configuration( serialized_data )

    def serialize_fet_configuration( self ) -> dict:
        # super will run method of FETNumericMinMaxFloat
        serialized_fet_configuration = super(  ).serialize_fet_configuration( )
        return serialized_fet_configuration

    @classmethod
    def cls_get_activation_cost(cls, column_datas_infos: Column_datas_infos) -> int:
        return sentence_transformer_model_vectors_len

    def get_list_encoded_columns_name(self, user_column_name: str) -> list:
        return [ f"{user_column_name}{SEPARATOR_IN_COLUMNS_NAMES}{type(self).__name__}{SEPARATOR_IN_COLUMNS_NAMES}{i}"
                for i in range(0, sentence_transformer_model_vectors_len) ]

    @classmethod
    def cls_is_possible_to_enable_this_fet_with_this_infos( cls, column_datas_infos: Column_datas_infos ) -> bool:
        if not (column_datas_infos.str_language_en +
                column_datas_infos.str_language_fr +
                column_datas_infos.str_language_de +
                column_datas_infos.str_language_it +
                column_datas_infos.str_language_es +
                column_datas_infos.str_language_pt >=
                FETSentenceTransformer_MINIMUM_PERCENTAGE_6_LANGUAGE/100):
            return False

        # if the parent class say it cannot be activated then this is it
        return super( ).cls_is_possible_to_enable_this_fet_with_this_infos( column_datas_infos )


class FETSentenceTransformerFloat( FETImpossible ):
    pass

class FETSentenceTransformerLabel( FETImpossible ):
    pass

class FETSentenceTransformerDate( FETImpossible ):
    pass

class FETSentenceTransformerTime( FETImpossible ):
    pass

class FETSentenceTransformerLanguage( FETSentenceTransformer ):
    # will be activated if there is enough languages detected
    pass




# ======================================================================================================================
# ===============================================| additional functions |===============================================
# ======================================================================================================================

def _truncated_normal_distribution(mean: float, std: float, min_val: float, max_val: float, num_samples: int = 1) -> np.ndarray:
    """create truncated normal distribution"""

    if std == 0:
        raise ZeroDivisionError("Standard deviation cannot be zero!")

    return truncnorm((min_val - mean) / std, (max_val - mean) / std, loc=mean, scale=std).rvs(num_samples)


def encoder_time_to_float( data_to_encode:np.array ) -> np.array:

    def encoder_time_to_float_one_value( v ) -> float:
        if pd.isnull( v ):
            return np.nan
        elif isinstance( v, np.datetime64 ):
            return (v - np.datetime64( '1970-01-01T00:00:00Z' )) / np.timedelta64( 1, 's' )
        elif isinstance( v, (datetime.time, datetime.datetime) ):
            return v.second + 60 * v.minute + 3600 * v.hour
        else:
            logger.error( f"_encoder_time_to_float Can not convert data type {type( v )} , v={v}  " )

    if data_to_encode.ndim == 1:
        # Create an empty array to hold the updated values
        updated_arr = np.empty_like( data_to_encode , dtype=float  )
        # Apply the  function to each value in the array one at a time
        for i in range( data_to_encode.size ):
            updated_arr[ i ] = encoder_time_to_float_one_value( data_to_encode[ i ] )
        return updated_arr
    elif data_to_encode.ndim==2:
        # Create an empty array to hold the updated values
        updated_arr = np.empty_like( data_to_encode )
        # Apply the double function to each value in the array one at a time
        for i in range( data_to_encode.shape[ 0 ] ):
            for j in range( data_to_encode.shape[ 1 ] ):
                updated_arr[ i ][ j ] = encoder_time_to_float_one_value( data_to_encode[ i ][ j ] )
        return updated_arr
    else:
        logger.error( "The array must be 1D or 2D")


def encoder_date_to_float( data_to_encode:np.array ) -> np.array:

    def encoder_date_to_float_one_value( v ) -> float:
            if pd.isnull(v):
                return np.nan
            elif isinstance( v, np.datetime64 ):
                return float( ( v - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's') )
            elif isinstance( v, datetime.date ):
                return float( (datetime.datetime.combine( v , datetime.datetime.min.time()) - datetime.datetime(1970,1,1)).total_seconds() )
            elif isinstance( v, datetime.datetime  ):
                return float( v - datetime.datetime(1970,1,1)).total_seconds()
            else:
                logger.error(f"_encoder_date_to_float Can not convert data type {type(v)} , v={v}")

    if data_to_encode.ndim == 1:
        # Create an empty array to hold the updated values
        updated_arr = np.empty_like( data_to_encode , dtype=float  )
        # Apply the  function to each value in the array one at a time
        for i in range( data_to_encode.size ):
            updated_arr[ i ] = encoder_date_to_float_one_value( data_to_encode[ i ] )
        return updated_arr
    elif data_to_encode.ndim==2:
        # Create an empty array to hold the updated values
        updated_arr = np.empty_like( data_to_encode )
        # Apply the double function to each value in the array one at a time
        for i in range( data_to_encode.shape[ 0 ] ):
            for j in range( data_to_encode.shape[ 1 ] ):
                updated_arr[ i ][ j ] = encoder_date_to_float_one_value( data_to_encode[ i ][ j ] )
        return updated_arr
    else:
        logger.error( "The array must be 1D or 2D")


def decoder_float_to_date( data_to_encode:np.array ) -> np.array:

    def _decoder_float_to_date_one_value( v ) -> datetime.date:
        if pd.isnull(v):
            return np.nan
        else:
            # we add half of a day in second (86400) to round , if we are more than 12:00 then this will be the next day
            return (datetime.datetime(1970,1,1)  + datetime.timedelta( seconds=(v+86400/2) )).date( )

    if data_to_encode.ndim == 1:
        # Create an empty array to hold the updated values
        updated_arr = np.empty_like( data_to_encode, dtype=datetime.date  )
        # Apply the  function to each value in the array one at a time
        for i in range( data_to_encode.size ):
            updated_arr[ i ] = _decoder_float_to_date_one_value( data_to_encode[ i ] )
        return updated_arr
    elif data_to_encode.ndim==2:
        # Create an empty array to hold the updated values
        updated_arr = np.empty_like( data_to_encode )
        # Apply the double function to each value in the array one at a time
        for i in range( data_to_encode.shape[ 0 ] ):
            for j in range( data_to_encode.shape[ 1 ] ):
                updated_arr[ i ][ j ] = _decoder_float_to_date_one_value( data_to_encode[ i ][ j ] )
        return updated_arr
    else:
        logger.error( "The array must be 1D or 2D")


def decoder_float_to_time( data_to_encode:np.array ) -> np.array:

    def decoder_float_to_time_one_value( v ) -> datetime.time:
        if pd.isnull(v):
            return np.nan
        else:
            return datetime.datetime.utcfromtimestamp( float(v) ).time()

    if data_to_encode.ndim == 1:
        # Create an empty array to hold the updated values
        updated_arr = np.empty_like( data_to_encode , dtype=datetime.time )
        # Apply the  function to each value in the array one at a time
        for i in range( data_to_encode.size ):
            updated_arr[ i ] = decoder_float_to_time_one_value( data_to_encode[ i ] )
        return updated_arr
    elif data_to_encode.ndim==2:
        # Create an empty array to hold the updated values
        updated_arr = np.empty_like( data_to_encode )
        # Apply the double function to each value in the array one at a time
        for i in range( data_to_encode.shape[ 0 ] ):
            for j in range( data_to_encode.shape[ 1 ] ):
                updated_arr[ i ][ j ] = decoder_float_to_time_one_value( data_to_encode[ i ][ j ] )
        return updated_arr
    else:
        logger.error( "The array must be 1D or 2D")


def decoder_merger_combine_1D_nparray_date_time( d: np.array , t: np.array ) -> np.array:
    dt_array = np.empty( d.shape , dtype=datetime.datetime )
    for idx in range( 0 , len(d)):
        if pd.isnull(d[idx]) or pd.isnull(t[idx]):
            dt_array[idx] = None
        else:
            dt_array[idx] = datetime.datetime.combine( d[idx] , t[idx] )

    return dt_array


def decoder_merger_average_2d_nparray_similar_values_per_rows( array_to_average:np.array ) -> np.array:
    """
    used to merge in enc_dec all result of FET_DECODER for FLOAT
    :params array_to_average: the 2D array to get average for each rows
    :return: the average for every rows
    """

    # numpy operation do not work with None, this convert the none into nan
    array_to_average_wn = np.copy( array_to_average.astype(float) )

    # Calculate the row average for each row   (  .astype(float) will convert None to NaN because numpy is not compatible with Nan
    row_avg = np.nanmean( array_to_average_wn, axis=1 )

    # Calculate the distances between each element in each row and the row average
    distances = np.abs( array_to_average_wn.T - row_avg ).T

    # Sort the distances for each row
    sorted_distances = np.argsort( distances, axis=1 )

    # Choose the k most similar elements to the row average for each row
    k = int( np.rint( array_to_average_wn.shape[1] * 0.75 ) )
    most_similar = sorted_distances[ :, :k ]

    # Calculate the average of the k values closest to the row average for each row
    row_k_values_avg = np.mean( array_to_average_wn[ np.arange( len( array_to_average_wn ) )[ :, None ], most_similar ], axis=1 )

    # return 1D array , all rows averaged
    return row_k_values_avg


def decoder_merger_merge_2d_array_labels_similar_per_rows( array_label_to_merge:np.array ) -> np.array:
    """
    used to merge in enc_dec all result of FET_DECODER for LABEL
    :params array_to_average: the 2D array to get merge for each rows , by selecting the most used label in the row (or any not none if there is only one)
    :return: the label for every rows
    """

    # Initialize an empty list to hold the results
    results = [ ]

    # Iterate over the rows of the array
    for row in array_label_to_merge:
        # Get a list of non-None elements in the row
        non_none_elems = [ elem for elem in row if elem is not None ]

        # If there are no non-None elements, append None to the results list
        if not non_none_elems:
            results.append( None )
        else:
            # Find the most frequent non-None element in the row
            most_frequent_elem = max( set( non_none_elems ), key=non_none_elems.count )

            # Append the most frequent element to the results list
            results.append( most_frequent_elem )

    # Return the list of results
    return results




def pickle_load_str( data_pickle_string ):
    data_bytes = base64.b64decode( data_pickle_string )
    object = pickle.loads( data_bytes )
    return object


def pickle_dump_str( data_object ):
    data_bytes = pickle.dumps( data_object )
    return base64.b64encode( data_bytes ).decode( 'utf-8' )


@lru_cache( maxsize=999 )
def sentence_transformer_model_encoder_cached( string_to_vectorize ) -> np.array:
    """
    will convert the string in vector using the model
    :params string_to_vectorize: the sentence to vectorize
    :return: an array of vector for the sentence
    """
    return sentence_transformer_model.encode( string_to_vectorize )


