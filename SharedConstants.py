from enum import Enum, auto as enum_auto
import sys


#=====================================================
#   to disable some try/catch error management
#=====================================================
IS_RUNNING_IN_DEBUG_MODE = True
#=====================================================




#=====================================================
# Each flag enable detailed operation logging for one single module
#=====================================================
ENABLE_LOGGER_DEBUG_Machine = True
ENABLE_LOGGER_DEBUG_DataFileReader = True
ENABLE_LOGGER_DEBUG_MachineDataConfiguration = True
ENABLE_LOGGER_DEBUG_InputsColumnsImportance = True
ENABLE_LOGGER_DEBUG_FeatureEngineeringConfiguration = True
ENABLE_LOGGER_DEBUG_FeatureEngineeringTemplate = True
ENABLE_LOGGER_DEBUG_EncDec = True
ENABLE_LOGGER_DEBUG_NNConfiguration = True
ENABLE_LOGGER_DEBUG_NNEngine = True
ENABLE_LOGGER_DEBUG_NNEngine_DETAILED = False
ENABLE_LOGGER_DEBUG_MachineEasyAutoML = True
ENABLE_LOGGER_DEBUG_Experimenter = True
ENABLE_LOGGER_DEBUG_SolutionFinder = True
ENABLE_LOGGER_DEBUG_SolutionFinder_RESULT = True
#=====================================================




#=====================================================
# in machine.column_error we will write error exception TRUE : ONLY FOR DEBUG
#=====================================================
DEBUG_WRITE_ERROR_CLEARLY_INSIDE_MACHINES = True
#=====================================================



#=====================================================
# NNEngine : Set all to false for normal operations TRUE : ONLY FOR DEBUG
#=====================================================
DEBUG_DISABLE_FET_ENCODER_CACHE = False    # disable the cache for the fet encoder, and always check the cache with real encoding
DEBUG_DISABLE_find_machine_best_nn_configuration = False    # disable solution-finder
DEBUG_DISABLE_find_machine_best_FE_configuration = False    # disable solution-finder
DEBUG_FORCE_FASTER_SHORTER_DIFFERENTIAL_EVOLUTION = False # evaluate less solutions if it is true
DEBUG_MachineEasyAutoML_FORCE_EXPERIMENTER = False # force using experimenter even if prediction are available
DEBUG_MachineEasyAutoML_DISABLE_RANDOM_EXPERIMENTER = False # will never do experimenter if prediction are possible
DEBUG_TRAINING_ROWS_COUNT_LIMIT = None  # If not None then it will limit the rows for training and for training_trial
#=====================================================




#=====================================================
# this is the 4 columns of the user_dataframe_to_extend NNShape
# NNShape define the structure of the Neural Network of the machine
USER_NN_SHAPE_COLUMNS_NAME = ["NeuronPercentage", "LayerTypeActivation", "DropOut", "BatchNormalization"]

# all layers including input and output
NNCONFIGURATION_MAX_POSSIBLE_LAYER_COUNT = 7

# this is all the possible layer type supported in a layer of the neural network - each machine have one NN and several layers (1 to 10) each layer have one of this LayerTypes
NNCONFIGURATION_ALL_POSSIBLE_LAYER_TYPE_ACTIVATION = (
    "dense_relu",
    "dense_elu",
    "dense_softmax",
    "dense_sigmoid",
)

# this is all the possible optimizer function the neural network support - each machine have one NN and one function enabled
NNCONFIGURATION_ALL_POSSIBLE_OPTIMIZER_FUNCTION = (
    "Adam",
    "SGD",
    "RMSprop",
    "Adadelta")

# Now We do not vary the loss function
NNCONFIGURATION_LOSS_FUNCTION = "mean_absolute_error"
#=====================================================



#=====================================================
# Machine Learning have a machine_level, the machine_level start with 1 and is the less efficient, the machine_level increase the accuracy of the neural network but also increase the fees
POSSIBLE_MACHINES_LEVELS = list(range(1, 6))

# if the machine name start and end with double underscore it will have this machine_level if it is not indicated
MACHINE_EASYAUTOML_RESERVED_DEFAULT_LEVEL = 1

# if the machine name do not start and end with double underscore it will have this machine_level if it is not indicated
MACHINE_USER_DEFAULT_LEVEL = 1
#=====================================================



#=====================================================
# inside the experim
EXPERIMENTER_FET_PREFIX_FET_NAME = "Experiment_FET_"
#=====================================================



#=====================================================
"""
To reset database : python manage.py flush
To do migration : cd WWW / python manage.py makemigrations / python manage.py migrate  
Then at post-migration signal we generate the Admin user and the Admin team
To get SUPER ADMIN  : user_model.get_super_admin().id
To get SUPER ADMIN ID  :  team_model.get_super_admin_team().first().ID
Super Admin must  always ID=1 but better to use : EasyAutoMLDBModels().User.get_super_admin( ).id
Super Admin team must always TEAM=1 but better to use : EasyAutoMLDBModels().Team.get_super_admin_team.id
"""

SUPER_ADMIN_EASYAUTOML_EMAIL = "SuperAdmin@easyautoml.com"
SUPER_ADMIN_EASYAUTOML_TEAM_NAME = "__Team-EasyAutoML.com__"   # super Admin Team = EasyAutoMLDBModels().Team.get_super_admin_team
SUPER_ADMIN_TEAM_MEMBERS_EASYAUTOML_PASSWORD = "*Easy*Auto*ML*" # password for all super users
# super user to create first - will be in team SUPER_ADMIN_EASYAUTOML_TEAM_NAME and password = SUPER_ADMIN_TEAM_MEMBERS_EASYAUTOML_PASSWORD
SUPER_ADMIN_EASYAUTOML_TEAM_MEMBERS = (
    (SUPER_ADMIN_EASYAUTOML_EMAIL, "SuperAdmin", "EasyAutoML"),
    ("Laurent@EasyAutoML.com", "Laurent", "Bruère")
)


#=====================================================


DEFAULT_SITE_DOMAIN = "127.0.0.1:8000" # default site address when using locally without server ( local computer debug )

# this is the DNS address to display in the pages API
EASYAUTOML_API_DNS_ADDRESS_SERVER = "https://EasyAutoML.com"



# define how many percent of training lines will be validation (0 to 100) when we save training lines in the db
DATALINES_PERCENTAGE_OF_VALIDATION_LINES = 20


# if a column have more than this percentage of missing value, it will be totally ignored
COLUMN_MAXIMUM_MISSING_PERCENTAGE_FOR_IGNORE = 75

# if a column have more than this percentage of missing value, it cannot be skipped , then it will be filled or predicted
COLUMN_MAXIMUM_MISSING_PERCENTAGE_FOR_SKIPPING_VALUES = 20


# for bad models not well fitting the dataset,  keras training do not start at all and early_patience stop the training at the beginning because the loss is not evolving, we will retry several times
NNENGINE_TRAINING_ATTEMPT_TRAINING_START_CORRECTLY_MAX = 5



UsersCoupons = ("UserIsTester", "CouponTest")


# TODO I dont think we need to share this , it should be moved in django module
DISPLAYED_FIELDS_FOR_FE_PAGE = (
    "dfr_columns_type_user_df",
    "mdc_columns_name_input_user_df",
    "mdc_columns_name_output_user_df",
    "dfr_columns_description_user_df",
    "fe_columns_inputs_importance_evaluation",
    "fe_columns_fet",
    "=>LEN;fe_columns_fet",
)

# This fields are in every datainputs tables, they are used to manage datalines
MACHINES_DATALINES_INPUT_RESERVED_FIELDS = [
    "IsForLearning",
    "IsForSolving",
    "IsForEvaluation",
    "IsLearned",
    "IsSolved",
]



class DatasetColumnDataType(Enum):
    """
    all machines receive a dataset to be trained and dataset to solve - the columns in the dataset will match one of this columns_data_type
    """

    IGNORE = enum_auto( )
    FLOAT = enum_auto()
    LABEL = enum_auto()
    DATE = enum_auto()
    TIME = enum_auto()
    DATETIME = enum_auto()
    LANGUAGE = enum_auto()
    JSON = enum_auto()


    def __str__(self):
        """
        return the name of the columns_data_type
        """
        return self.name

    @property
    def is_numeric(self):
        """
        true if columns_data_type is FLOAT, DATE, TIME, DATETIME
        """
        return self in (DatasetColumnDataType.FLOAT, DatasetColumnDataType.DATE, DatasetColumnDataType.TIME, DatasetColumnDataType.DATETIME)

    @property
    def is_datetime(self):
        """
        true if columns_data_type is DATE, TIME, DATETIME
        """
        return self in (DatasetColumnDataType.DATE, DatasetColumnDataType.TIME, DatasetColumnDataType.DATETIME)

    @property
    def is_text(self):
        """
        true if columns_data_type is LABEL, LANGUAGE
        """
        return self in (DatasetColumnDataType.LABEL, DatasetColumnDataType.LANGUAGE)


class DataframeEncodingType(Enum):
    """
    Dataframe containing solving or training data can have 3 status :

    USER : the user_dataframe_to_extend is sent by the user, some columns can contains JSON
    PRE_ENCODED : the user_dataframe_to_extend have been MDC.pre_encoded and the json have been converted into simple new columns
     ENCODED_FOR_AI : the user_dataframe_to_extend have been EncDec.encoded_for_ai for the neural network, all columns values are now between 0 and 1 , all labels, dates, etc.. have been converted by FE and EncDec
    """
    USER = enum_auto()
    PRE_ENCODED = enum_auto()
    ENCODED_FOR_AI = enum_auto()

    def __str__(self):
        return self.name


class ColumnDirectionType(Enum):
    """
    all columns of Dataset user (or encoded for neural network) have columns inputs(to solve) outputs(solved or for training) and ignored (will not be processed by the neural network)
    """
    INPUT = "input" 
    OUTPUT = "output" 
    IGNORED = "ignore" 


#=====================================================
# Decimal separator and date format configuration
#=====================================================
DECIMAL_SEPARATOR_CHOICES = (".", ",")
DEFAULT_DECIMAL_SEPARATOR = "."

DATE_FORMAT_CHOICES = ("DMY", "MDY", "YMD")
DEFAULT_DATE_FORMAT = "YMD"
#=====================================================


