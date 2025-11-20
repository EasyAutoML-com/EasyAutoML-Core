import os
import warnings

warnings.filterwarnings("ignore")

# --------------------------------------------- DISABLE TENSORFLOW IN TRANSFORMERS -----------------------------------------------------
# sentence-transformers uses PyTorch, so we don't need TensorFlow backend in transformers
# This prevents tf_keras compatibility issues with TensorFlow 2.17
os.environ.setdefault('TRANSFORMERS_NO_TF', '1')
os.environ.setdefault('USE_TF', '0')
# --------------------------------------------- END DISABLE TENSORFLOW IN TRANSFORMERS ----------------------------------------------------

# --------------------------------------------- SET LOGGERS POSSIBLE_MACHINES_LEVELS -----------------------------------------------------

# 0 - all messages are logged (default behavior)
# 1 - INFO messages are not printed
# 2 - INFO and WARNING messages are not printed
# 3 - INFO, WARNING, and ERROR messages are not printed
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# DEBUG    - all messages are logged (default behavior)
# INFO     - DEBUG messages are not printed
# WARNING  - DEBUG and INFO messages are not printed (recommended after production)
# ERROR    - ERROR and CRITICAL messages are printed
# CRITICAL - ONLY CRITICAL messages are printed
os.environ["GLOBAL_LOGGER_CONSOLE_LEVEL"] = "DEBUG"

# DEBUG    - all messages are saved
# INFO     - DEBUG messages are not saved
# WARNING  - DEBUG and INFO messages are not saved (default behavior)
# ERROR    - ERROR and CRITICAL messages are saved
# CRITICAL - ONLY CRITICAL messages are saved
os.environ["GLOBAL_LOGGER_DB_LEVEL"] = "ERROR"

# --------------------------------------------- IMPORT MAIN CLASSES ----------------------------------------------------
from models.EasyAutoMLDBModels import EasyAutoMLDBModels

def __getlogger():
    """Get logger instance - centralized logger initialization"""
    return EasyAutoMLDBModels().logger
from ML.Machine import Machine, MachineLevel
from ML.MachineDataConfiguration import MachineDataConfiguration
from ML.FeatureEngineeringConfiguration import FeatureEngineeringConfiguration, FeatureEngineeringColumn
from ML.EncDec import EncDec
from ML.NNConfiguration import NNConfiguration, NNShape
from ML.NNEngine import NNEngine
from ML.SolutionScore import SolutionScore
from ML.MachineEasyAutoML import MachineEasyAutoML
from ML.SolutionFinder import SolutionFinder
from ML.Experimenter import ExperimenterNNConfiguration, ExperimenterColumnFETSelector
from ML.MachineEasyAutoMLAPI import MachineEasyAutoMLAPI
from ML.InputsColumnsImportance import InputsColumnsImportance
from ML.DataFileReader import DataFileReader


__all__ = [
    "EasyAutoMLDBModels",
    "__getlogger",
    "Machine",
    "MachineLevel",
    "MachineDataConfiguration",
    "FeatureEngineeringConfiguration",
    "FeatureEngineeringColumn",
    "EncDec",
    "NNConfiguration",
    "NNShape",
    "ExperimenterNNConfiguration",
    "ExperimenterColumnFETSelector",
    "SolutionScore",
    "SolutionFinder",
    "NNEngine",
    "MachineEasyAutoML",
    "MachineEasyAutoMLAPI",
    "InputsColumnsImportance",
    "DataFileReader",
]

