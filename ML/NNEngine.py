from typing import NoReturn
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import random
from timeit import default_timer
import json

from ML import EasyAutoMLDBModels

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as k
from tensorflow.keras.models import Sequential


from tensorflow.keras.callbacks import EarlyStopping


from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.losses import get as get_loss_function
from tensorflow.python.client import device_lib

from ML import Machine, MachineLevel, MachineDataConfiguration, EncDec
from ML.NNConfiguration import NNConfiguration, NNShape

# Optional import for Billing (may not be available after repo split)
try:
    from Servers import Billing
    BILLING_AVAILABLE = True
except ImportError:
    Billing = None
    BILLING_AVAILABLE = False

from SharedConstants import *


from ML import __getlogger

logger = __getlogger()


# real maximum count of weight the NNEngine is able to handle
NNENGINE_WEIGHT_COUNT_MIN = 25
NNENGINE_WEIGHT_COUNT_MAX = 50 * 1000 * 1000

# anti recursive configuration best - activated if a EasyAutoML internal machine is doing configuration
# during activation all others machines configuration will be minimum instead of best
# to be used only with methods : NNEngine.machine_nn_engine_configuration_***
global_nnengine_list_id_machines_doing_configuration = { }


class NNEngine:
    """
    Class NNEngine bring methods to perform machines training, predictions and model-performance-evaluation

    When we create NNEngine the first time, it create a minimum configuration
    Then when doing Do_Training it will improve and create best configuration if there is no all configuration,
    but if an element of the  configuration exist in Machine it will load the configuration-elements

    There is no argument in init to control if we create or load the configuration
    but there is several flag and methods in Machine to control each elements of the configuration:
            db_machine.machine_is_re_run_mdc = True
            db_machine.machine_is_re_run_ici = True
            db_machine.machine_is_re_run_fe = True
            db_machine.machine_is_re_run_enc_dec = True
            db_machine.machine_is_re_run_nn_config = True
            db_machine.machine_is_re_run_model = True
            _machine.is_config_ready_ci
            _machine.is_config_ready_fe
            _machine.is_config_ready_enc_dec
            _machine.is_config_ready_nn_configuration
            _machine.is_config_ready_nn_model
    """

    def __init__( self, machine: Machine, allow_re_run_configuration: bool = False ):
        """
        Create NNEngine or load it from Machine
        If configuration are available inside Machine we use it else we will generate it

        :param machine: the machine to work on
        """
        if not isinstance(machine, Machine):
            logger.error( "The constructor argument must have an instance of the machine" )

        self._machine = machine
        self._mdc = None
        self._ici = None
        self._fe = None
        self._enc_dec = None
        self._nn_configuration = None
        self._nn_model = None

        if IS_RUNNING_IN_DEBUG_MODE:
            self._init_load_or_create_configuration( allow_re_run_configuration=allow_re_run_configuration )
        else:
            try:
                self._init_load_or_create_configuration( allow_re_run_configuration=allow_re_run_configuration )
            except Exception as e:
                machine.db_machine.log_work_status.update( { str( datetime.now( ) ): False } )
                machine.db_machine.log_work_message.update( { str( datetime.now( ) ): f"Error Unable to load NNEngine for machine {machine} because {e}" } )
                machine.save_machine_to_db( )
                logger.error( f"Unable to load NNEngine for machine {machine} because {e}" )


    @classmethod
    def machine_nn_engine_configuration_set_starting( cls, machine: Machine ) -> NoReturn:
        # we avoid do have one machine using herself in a loop (can happen with EasyAutoML MACHINES) and both doing configuration because it can create an infinite loop
        global global_nnengine_list_id_machines_doing_configuration
        global_nnengine_list_id_machines_doing_configuration[ machine.db_machine.id ] = datetime.now()


    @classmethod
    def machine_nn_engine_configuration_set_finished( cls, machine: Machine ) -> NoReturn:
        # we avoid do have one machine using herself in a loop (can happen with EasyAutoML MACHINES) and both doing configuration because it can create an infinite loop
        global global_nnengine_list_id_machines_doing_configuration
        del global_nnengine_list_id_machines_doing_configuration[ machine.db_machine.id ]


    @classmethod
    def machine_nn_engine_configuration_is_configurating( cls, machine: Machine ) -> bool:
        # we avoid do have one machine using herself in a loop (can happen with EasyAutoML MACHINES) and both doing configuration because it can create an infinite loop
        global global_nnengine_list_id_machines_doing_configuration
        if machine.db_machine.id in global_nnengine_list_id_machines_doing_configuration:
            #logger.debug(f"Yes {machine} is now doing nn_configuration")
            return True
        else:
            return False


    def _init_load_or_create_configuration( self , allow_re_run_configuration: bool = False , update_data_infos_stats:bool=False ):
        """
        Create NNEngine or load it from Machine
        If configuration are available inside Machine we use it else we will generate it

        All configurations to run NNEngine will be prepared here
        NNEngine need to have all this configurations built : MDC, CI, FE, EncDec, NNConfiguration, NNModel
        We try to use the configuration stored into machine, if not available we will generate the configuration, using best if possible or else doing minimum_default

        at each step we load the configuration or we create the configuration and save it into machine object
        only last step : training nn model is not performed here (sometime we need nnengine for performing trial learning)

        Some flag can trigger some configuration to be rebuilt : re_run - If one Config need to be rebuilt then all following Config will be cleared and rebuilt
        """

        if ENABLE_LOGGER_DEBUG_NNEngine: logger.debug( f"NNEngine initializing (loading and/or creating) all configuration for {self._machine}" )

        db_machine = self._machine.db_machine

        # MDC and EncDec may load and use the dataset - we cache it for faster
        _c_full_df_user = None
        _c_full_df_pre_encoded = None

        if ENABLE_LOGGER_DEBUG_NNEngine: logger.debug( f"NNEngine Starting preparing all configurations. " )
        self.machine_nn_engine_configuration_set_starting( self._machine )

        # -----------------------------------------
        #  MDC
        # -----------------------------------------
        if not self._machine.is_config_ready_mdc() or (allow_re_run_configuration and db_machine.machine_is_re_run_mdc):
            _c_full_df_user = self._machine.data_lines_read( ) # we have to use the full dataset because with a subset we could miss some labels or out of bound values
            # We do not indicate what user choose because anyway the dataset is formatted and this 2 parameters are useless : decimal_separator  date_format
            self._mdc = MachineDataConfiguration(
                                                    self._machine,
                                                    _c_full_df_user,
                                                    force_create_with_this_inputs=self._machine.db_machine.mdc_columns_name_input,    # we cannot change this after creation
                                                    force_create_with_this_outputs=self._machine.db_machine.mdc_columns_name_output, # we cannot change this after creation
                                                    columns_type_user_df=self._machine.db_machine.dfr_columns_type_user_df,
                                                    columns_description_user_df= self._machine.db_machine.dfr_columns_description_user_df,
                                                    decimal_separator = ".",
                                                    date_format = "MDY" )
            self._mdc.save_configuration_in_machine( )
            # rerun configuration have been done, we can clear the flag
            db_machine.machine_is_re_run_mdc = False

            # configuration regenerated need to force following configurations to be rebuilt too
            self._machine.clear_config_ici( )
            self._machine.clear_config_fe( )
            self._machine.clear_config_enc_dec( )
            self._machine.clear_config_nn_configuration( )
            self._machine.clear_config_nn_model( )

        else:
            self._mdc = MachineDataConfiguration(self._machine)
            if update_data_infos_stats:
                _c_full_df_user = self._machine.data_lines_read( )  # we have to use the full dataset because with a subset we could miss some labels or out of bound values
                _c_full_df_pre_encoded = self._mdc.dataframe_pre_encode( _c_full_df_user )
                self._mdc._recalculate_data_infos_stats( _c_full_df_pre_encoded, decimal_separator = ".", date_format = "MDY" )

        # -----------------------------------------
        #  Column Importance
        # -----------------------------------------
        from ML.InputsColumnsImportance import InputsColumnsImportance
        if not self._machine.is_config_ready_ici( ) or (allow_re_run_configuration and db_machine.machine_is_re_run_ici):
            if (
                    not self._machine.is_config_ready_fe() or
                    not self._machine.is_config_ready_enc_dec() or
                    not self._machine.is_config_ready_nn_configuration() or
                    not self.is_nn_trained_and_ready()
                ):
                if ENABLE_LOGGER_DEBUG_NNEngine: logger.debug( "unable to do ICI best configuration because fe or encdec or nnconfig are not ready => doing minimum ICI" )
                self._ici = InputsColumnsImportance( self._machine , create_configuration_simple_minimum=True).save_configuration_in_machine()
                db_machine.machine_is_re_run_ici = True
            else:
                if ENABLE_LOGGER_DEBUG_NNEngine: logger.debug( "do ICI best configuration because fe or encdec or nnconfig are ready" )
                self._ici = InputsColumnsImportance( self._machine , create_configuration_best=True, nnengine_for_best_config=self ).save_configuration_in_machine()
                # rerun configuration have been done, we can clear the flag
                db_machine.machine_is_re_run_ici = False

            # configuration regenerated need to force following configurations to be rebuilt too
            self._machine.clear_config_fe( )
            self._machine.clear_config_enc_dec( )
            self._machine.clear_config_nn_configuration( )
            self._machine.clear_config_nn_model( )

        else:
            # load the configuration ICI by default if no force flag and if not needed by rerun to make configuration
             self._ici = InputsColumnsImportance( self._machine )

        # -----------------------------------------
        #  FE
        # -----------------------------------------
        from ML.FeatureEngineeringConfiguration import FeatureEngineeringConfiguration
        if not self._machine.is_config_ready_fe() or (allow_re_run_configuration and db_machine.machine_is_re_run_fe):
            if (
                    not self._machine.is_config_ready_enc_dec() or
                    not self._machine.is_config_ready_nn_configuration()
                    ):
                # generate the configuration FE MINIMUM because we cannot do FE_BEST now
                if ENABLE_LOGGER_DEBUG_NNEngine: logger.debug( f"generate the configuration FE MINIMUM because we cannot do FE_BEST now" )
                self._fe = FeatureEngineeringConfiguration(
                    machine=self._machine,
                    force_configuration_simple_minimum=True
                    ).save_configuration_in_machine()
                # we will do best FE later
                db_machine.machine_is_re_run_fe = True
            else:
                # generate the configuration FE BEST
                if ENABLE_LOGGER_DEBUG_NNEngine: logger.debug( f"generate the configuration FE BEST  " )
                db_machine.fe_budget_total = MachineLevel(self._machine).feature_engineering_budget()[1]
                # load (never do best config of course) the nnconfiguration that we will use to evaluate the FEC
                self._nn_configuration = NNConfiguration(self._machine)
                self._fe = FeatureEngineeringConfiguration(
                    machine=self._machine,
                    nn_engine_for_searching_best_config=self,
                    global_dataset_budget=self._machine.db_machine.fe_budget_total,
                    force_configuration_simple_minimum=False,
                    ).save_configuration_in_machine()
                # rerun configuration have been done, we can clear the flag
                db_machine.machine_is_re_run_fe = False

            #  configuration regenerated need to force following configurations to be rebuilt too
            self._machine.clear_config_enc_dec( )
            self._machine.clear_config_nn_configuration( )
            self._machine.clear_config_nn_model( )

        else:
            # load the configuration FE
            self._fe = FeatureEngineeringConfiguration(self._machine)


        # -----------------------------------------
        #  EncDec
        # -----------------------------------------
        if not self._machine.is_config_ready_enc_dec( ) or (allow_re_run_configuration and db_machine.machine_is_re_run_enc_dec):
            # generate configuration encdec
            if _c_full_df_pre_encoded is None:
                if _c_full_df_user is None:
                    _c_full_df_user = self._machine.data_lines_read( ) # we have to use the full dataset because with a subset we could miss some labels or out of bound values
                _c_full_df_pre_encoded = self._mdc.dataframe_pre_encode( _c_full_df_user )
            self._enc_dec = EncDec( self._machine, _c_full_df_pre_encoded ).save_configuration_in_machine(  )

            # when the EncDec created/updated we need to update the nn_loss_scaler (NNConfiguration or NNEngine will not change the scaler)
            db_machine.parameter_nn_loss = NNCONFIGURATION_LOSS_FUNCTION
            db_machine.parameter_nn_loss_scaler = 1 / self._compute_loss_of_random_dataset( db_machine.parameter_nn_loss )

            # rerun configuration have been done, we can clear the flag
            db_machine.machine_is_re_run_enc_dec = False
            # configuration regenerated need to force following configurations to be rebuilt too
            self._machine.clear_config_nn_configuration( )
            self._machine.clear_config_nn_model( )






        else:
            # load configuration encdec
            self._enc_dec = EncDec(self._machine)


        # -----------------------------------------
        #  NN_Configuration
        # -----------------------------------------
        if not self._machine.is_config_ready_nn_configuration() or (allow_re_run_configuration and db_machine.machine_is_re_run_nn_config):
            # Generate configuration for NN_configuration
            if ENABLE_LOGGER_DEBUG_NNEngine: logger.debug( f"Generate best configuration for NN_configuration" )
            self._nn_configuration = NNConfiguration(self._machine, machine_nnengine_for_searching_best_nnconfig=self, force_find_new_best_configuration=True ).save_configuration_in_machine( )
            # rerun configuration have been done, we can clear the flag
            db_machine.machine_is_re_run_nn_config = False
            # configuration regenerated need to force following configurations to be rebuilt too
            self._machine.clear_config_nn_model( )

        else:
            # Load configuration of NN_configuration
            self._nn_configuration = NNConfiguration(self._machine)


        # -----------------------------------------
        #  NNModel
        # -----------------------------------------
        if not self._machine.is_config_ready_nn_model() or (allow_re_run_configuration and db_machine.machine_is_re_run_model):
            # training will be done by work_processor later
            pass
        else:
            # the model is loaded
            self._nn_model = self._get_nn_model_from_db()


        # we avoid do have same machine doing nnconfiguration recursively
        self.machine_nn_engine_configuration_set_finished( self._machine )
        if ENABLE_LOGGER_DEBUG_NNEngine: logger.debug( "All configurations have been done" )


    def is_nn_trained_and_ready(self):
        if self._nn_model:
            return True
        else:
            return False


    def dataframe_full_encode(self, user_dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        do pre-encode and do encode on the user_dataframe provided in argument

        :param user_dataframe: user_dataframe not encoded to encode
        :return: return dataframe encoded for ai
        """

        if not isinstance(user_dataframe, pd.DataFrame):
            logger.error("user_dataframe must have a pandas DataFrame type")
        if len(user_dataframe) == 0:
            logger.error("user_dataframe is empty")
        if self._mdc is None:
            logger.error("Unable to pre-encode because mdc is not ready")
        if self._enc_dec is None:
            logger.error("Unable to encode because encdec is not ready")

        pre_encoded_dataframe = self._mdc.dataframe_pre_encode(user_dataframe)

        encoded_dataframe = self._enc_dec.encode_for_ai(pre_encoded_dataframe)

        #if ENABLE_LOGGER_DEBUG: logger.debug(f"user_dataframe have been full encoded for AI")
        return encoded_dataframe


    def dataframe_full_decode( self, dataframe_encoded_for_ai: pd.DataFrame ) -> pd.DataFrame:
        """
        decode_from_ai  and post-decode the argument dataframe_encoded_for_ai
        :param dataframe_encoded_for_ai: the dataframe to decode
        :return: the dataframe (full decoded)
        """

        if not isinstance(dataframe_encoded_for_ai, pd.DataFrame):
            logger.error("dataframe_encoded_for_ai must have a pandas DataFrame type")
        if len(dataframe_encoded_for_ai) == 0:
            logger.error("dataframe_encoded_for_ai is empty")
        if self._mdc is None:
            logger.error("Unable to pre-encode because mdc is not ready")
        if self._enc_dec is None:
            logger.error("Unable to encode because encdec is not ready")

        decoded_dataframe = self._enc_dec.decode_from_ai(dataframe_encoded_for_ai)
        post_decoded_dataframe = self._mdc.dataframe_post_decode(decoded_dataframe)

        #if ENABLE_LOGGER_DEBUG: logger.debug(f"dataframe_encoded_for_ai have been full decoded")
        return post_decoded_dataframe


    def do_training_and_save(self) -> NoReturn:
        """
        Compile the model from the NNconfiguration
        load the training dataframe from machine_source.dataLine table
        encode_for_ai the dataframe
        do the training (1 to 3 cycles)
        do evaluation of the training and store it
        if machine is with loss prediction then do also the training of the nn_loss prediction machine_source
        """

        def do_machine_training_single_cycle( ) -> bool:
            """
            Do one training cycle
            """
            if IS_RUNNING_IN_DEBUG_MODE:
                training_result = self._do_one_training_cycle( )
            else:
                try:
                    training_result = self._do_one_training_cycle( )
                except Exception as e:
                    if DEBUG_WRITE_ERROR_CLEARLY_INSIDE_MACHINES:
                        self._machine.store_error( "*" , f"Error while doing training cycle for {self._machine}. Error is : {e} \n We have set machine_is_re_run_mdc"  )
                    training_result = None

            if not training_result:
                # the training have failed , we mark the result of the training as failed
                self._machine.store_error( "*" , "We apologize the training failed" )
                self._machine.db_machine.machine_is_re_run_mdc = True
                self._machine.db_machine.log_work_status.update( { str(datetime.now()) : False } )
                self._machine.db_machine.log_work_message.update( { str(datetime.now()) :f"Error while doing training cycle for {self._machine}. We have set machine_is_re_run_mdc. " } )
                self._machine.save_machine_to_db()

            if ENABLE_LOGGER_DEBUG_NNEngine: print( f"Result: {training_result} , re_run_mdc: {self._machine.db_machine.machine_is_re_run_mdc} , re_run_ici: {self._machine.db_machine.machine_is_re_run_ici} ,re_run_fe: {self._machine.db_machine.machine_is_re_run_fe} ,  re_run_enc_dec: {self._machine.db_machine.machine_is_re_run_enc_dec} " )
            return training_result


        # ----------------------------------------------------------------------------------------------------------------------------------
        if ENABLE_LOGGER_DEBUG_NNEngine: logger.debug( f"Starting full training for {self._machine} " )

        self._machine.db_machine.machine_columns_errors = {}
        self._machine.db_machine.machine_columns_warnings = {}

        # on desktop windows computer it is not possible to go to sleep - in Windows, prevent the OS from sleeping
        if os.name == 'nt':
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState( 0x80000000 | 0x00000001 )

        # we will do 4 cycles because we need to have ICI best then FE best , etc..
        if not do_machine_training_single_cycle( ):
            return # a failed cycle break the training (WP will restart if there is not consecutives 3 errors)

        self._machine.db_machine.machine_is_re_run_ici = True
        self._init_load_or_create_configuration( allow_re_run_configuration=True, update_data_infos_stats=True )

        if not do_machine_training_single_cycle( ):
            return # a failed cycle break the training (WP will restart if there is not consecutives 3 errors)

        self._machine.db_machine.machine_is_re_run_fe = True
        self._init_load_or_create_configuration( allow_re_run_configuration=True )
        if not do_machine_training_single_cycle( ):
            return # a failed cycle break the training (WP will restart if there is not consecutives 3 errors)

        self._init_load_or_create_configuration( allow_re_run_configuration=True )
        if not do_machine_training_single_cycle( ):
            return # a failed cycle break the training (WP will restart if there is not consecutives 3 errors)

        if ( self._machine.db_machine.machine_is_re_run_mdc or
            self._machine.db_machine.machine_is_re_run_ici or
            self._machine.db_machine.machine_is_re_run_fe or
            self._machine.db_machine.machine_is_re_run_enc_dec or
            self._machine.db_machine.machine_is_re_run_nn_config ):
            # something failed there is ReRun so we will try one more last cycle to try to clear the ReRun
            if ENABLE_LOGGER_DEBUG_NNEngine: logger.debug( f"for {self._machine} we did several cycle but still some rerun, so we will do one last cycle. re_run_mdc={ self._machine.db_machine.machine_is_re_run_mdc } , re_run_ici={ self._machine.db_machine.machine_is_re_run_ici  } , re_run_fe={ self._machine.db_machine.machine_is_re_run_fe  } , re_run_enc_dec={ self._machine.db_machine.machine_is_re_run_enc_dec  } , re_run_nn_config={ self._machine.db_machine.machine_is_re_run_nn_config }  " )
            self._init_load_or_create_configuration( allow_re_run_configuration=True )
            if not do_machine_training_single_cycle( ):
                return  # a failed cycle break the training (WP will restart if there is not consecutives 3 errors)

        if ( self._machine.db_machine.machine_is_re_run_mdc or
            self._machine.db_machine.machine_is_re_run_ici or
            self._machine.db_machine.machine_is_re_run_fe or
            self._machine.db_machine.machine_is_re_run_enc_dec or
            self._machine.db_machine.machine_is_re_run_nn_config ):
            # this is not normal we need to mark the training as failed
            # the training have failed , we mark the result of the training as failed
            message_error_detailed = f"Training done with one extra cycle but still some ReRun for {self._machine}. Training is marked as incomplete. re_run_mdc={ self._machine.db_machine.machine_is_re_run_mdc } , re_run_ici={ self._machine.db_machine.machine_is_re_run_ici  } , re_run_fe={ self._machine.db_machine.machine_is_re_run_fe  } , re_run_enc_dec={ self._machine.db_machine.machine_is_re_run_enc_dec  } , re_run_nn_config={ self._machine.db_machine.machine_is_re_run_nn_config }  "
            if DEBUG_WRITE_ERROR_CLEARLY_INSIDE_MACHINES:
                self._machine.store_error( "*" , message_error_detailed )
            else:
                self._machine.store_error( "*" , "We apologize the training is incomplete and will resume shortly" )
            self._machine.db_machine.log_work_status.update( { str(datetime.now()) : False } )
            self._machine.db_machine.log_work_message.update( { str(datetime.now()) : message_error_detailed } )
            self._machine.save_machine_to_db()
            return


        message_success = f"Do_training have completed all cycles for machine {self._machine} (and all ReRun are cleared)"
        if ENABLE_LOGGER_DEBUG_NNEngine: logger.debug(message_success )
        self._machine.db_machine.log_work_status.update({str(datetime.now()): True})
        self._machine.db_machine.log_work_message.update({str(datetime.now()): message_success})

        # we save the trained model in DB
        self._save_nn_model_to_db()

        # we mark model as done and clear flag rerun
        self._machine.db_machine.machine_is_re_run_model = False

        self._machine.save_machine_to_db()

        # we mark the lines used for doing the training
        self._machine.data_input_lines_mark_all_IsForLearning_as_IsLearned()

        # Verification - this cannot happen never
        if not self._machine.is_config_ready_mdc( ):
            logger.error( f"The machine have do training cycles but still machine.is_config_ready_mdc=false !")
        if not self._machine.is_config_ready_ici( ):
            logger.error( f"The machine have do training cycles but still machine.is_config_ready_ci=false !")
        if not self._machine.is_config_ready_fe():
            logger.error( f"The machine have do training cycles but still machine.is_config_ready_fe=false !")
        if not self._machine.is_config_ready_enc_dec():
            logger.error( f"The machine have do training cycles but still machine.is_config_ready_enc_dec=false !")
        if not self._machine.is_config_ready_nn_configuration():
            logger.error( f"The machine have do training cycles but still machine.is_config_ready_nn_configuration=false !")
        if not self._machine.is_config_ready_nn_model():
            logger.error( f"The machine have do training cycles but still machine.is_config_ready_nn_model=false !")
        if not self._machine.is_nn_solving_ready( ):
            logger.error( f"The machine have do training cycles but still machine.is_nn_solving_ready=false !")
        if self._machine.db_machine.machine_is_re_run_model:
            logger.error( f"The machine have do training cycles but still the machine_is_re_run_model=True !")

        # #----------------------------------------------------------------------------------------------------
        # # if the machine_source have Machine_Loss_prediction then we train the Loss_Prediction machine_source
        # if self._machine.db_machine.multimachines_Is_confidence_enabled:
        #     if ENABLE_LOGGER_DEBUG_NNEngine:
        #         logger.debug(f"{self._machine} multimachines_Is_confidence_enabled")
        #
        #     self.machine_source_evaluate_loss_of_validation_lines(self._machine)
        #     machine_loss_prediction = Machine(
        #         Machine(
        #             self._machine.db_machine.machine_nn_loss_prediction_machine_id
        #         ).db_machine.machine_name,
        #         self._machine.data_lines_read(IsForEvaluation=True),
        #     )
        #     NNEngine(machine_loss_prediction , allow_re_run_configuration=True ).do_training_and_save( )
        #     machine_loss_prediction.save_machine_to_db()
        # #----------------------------------------------------------------------------------------------------

        # we do evaluation of the model
        if IS_RUNNING_IN_DEBUG_MODE:
            self.do_evaluation()
        else:
            try:
                self.do_evaluation()
            except Exception as e:
                logger.error(f"Unable to do training evaluation of {self._machine} because {e}")
                self._machine.training_eval_loss_sample_training = None
                self._machine.training_eval_loss_sample_evaluation = None
                self._machine.training_eval_loss_sample_training_noise = None
                self._machine.training_eval_loss_sample_evaluation_noise = None
                self._machine.training_eval_accuracy_sample_training = None
                self._machine.training_eval_accuracy_sample_evaluation = None
                self._machine.training_eval_accuracy_sample_training_noise = None
                self._machine.training_eval_accuracy_sample_evaluation_noise = None
                self._machine.training_eval_outputs_cols_loss_sample_evaluation = {}
                self._machine.training_eval_outputs_cols_accuracy_sample_evaluation = {}
                return

        self._machine.save_machine_to_db()

        # manage the billing for the training (if Billing module is available)
        if BILLING_AVAILABLE:
            rows_count = self._machine.data_input_lines_count(IsForLearning=True) + self._machine.data_input_lines_count(IsForEvaluation=True)
            Billing.bill_training(self._machine.db_machine, rows_count )

        # ==================================================
        # we learn to predict the delay a machine_source need for training_cycle
        # experience_inputs are Machine_context + input_data_for_train.count, output is training_training_total_delay_sec
        from ML import MachineEasyAutoML
        MachineEasyAutoML_inputs_as_machine_overview_with_nn_model = self._machine.get_machine_overview_information(
            with_base_info = True,
            with_fec_encdec_info=True,
            with_nn_model_info = True,
            with_training_infos = True,
            with_training_cycle_result= False,
            with_training_eval_result=False
        )
        MachineEasyAutoML_outputs_are_training_result = self._machine.get_machine_overview_information(
            with_base_info = False,
            with_fec_encdec_info=False,
            with_nn_model_info = False,
            with_training_infos = False,
            with_training_cycle_result= True,
            with_training_eval_result= True
        )
        MachineEasyAutoML( "__Results_NNEngine_Training_Full__" ).learn_this_inputs_outputs(
            inputsOnly_or_Both_inputsOutputs=MachineEasyAutoML_inputs_as_machine_overview_with_nn_model,
            outputs_optional=MachineEasyAutoML_outputs_are_training_result,
        )


    def _do_one_training_cycle(self ) -> bool:
        """
        Load and encode data
        Do training with patience stopping
        """

        if ENABLE_LOGGER_DEBUG_NNEngine: logger.debug( f"Starting training cycle for {self._machine} " )

        if not self._machine.is_config_ready_nn_configuration():
            logger.error("Unable to do the Training because the machine do not have a NNConfiguration ready")

        training_start_time = default_timer()
        # check the model is not too large
        total_weights_count = self._nn_configuration.nn_shape_instance.weight_total_count(
            self._nn_configuration.num_of_input_neurons,
            self._nn_configuration.num_of_output_neurons,
        )

        if NNENGINE_WEIGHT_COUNT_MIN > total_weights_count > NNENGINE_WEIGHT_COUNT_MAX:
            logger.warning(
                f"Total neurons weights:{total_weights_count} in this model is not "
                f"between {NNENGINE_WEIGHT_COUNT_MAX} and {NNENGINE_WEIGHT_COUNT_MIN}"
            )

        # load the dataframe encoded_for_ai to do the training with
        (input_data_for_training_encoded_for_ai, output_data_for_training_encoded_for_ai) = self._split_dataframe_into_input_and_output(
            self.dataframe_full_encode(self._machine.data_lines_read(IsForLearning=True , rows_count_limit=DEBUG_TRAINING_ROWS_COUNT_LIMIT)),
            dataframe_status=DataframeEncodingType.ENCODED_FOR_AI,
        )

        # load the dataframe encoded_for_ai to do the validation data
        (input_data_for_validation_encoded_for_ai, output_data_for_validation_encoded_for_ai) = self._split_dataframe_into_input_and_output(
            self.dataframe_full_encode(self._machine.data_lines_read( IsForEvaluation=True , rows_count_limit=DEBUG_TRAINING_ROWS_COUNT_LIMIT ) ),
            dataframe_status=DataframeEncodingType.ENCODED_FOR_AI,
        )

        if len( input_data_for_training_encoded_for_ai ) < 50:
            logger.error( f"Impossible to do a training because only { len( input_data_for_training_encoded_for_ai ) } training rows (minimum 50)")
        if len( input_data_for_validation_encoded_for_ai ) < 10:
            logger.error( f"Impossible to do a training because only { len( input_data_for_validation_encoded_for_ai ) } validation rows (minimum 10)")

        # the keras model do not work if there is all NONE in any row of output
        # the system should skip empty rows
        if input_data_for_training_encoded_for_ai.isnull().all(axis=1).any( ):
            logger.error( f"There is row full of nan/none in input_data_for_training_encoded_for_ai")
        if output_data_for_training_encoded_for_ai.isnull().all(axis=1).any( ):
            logger.error( f"There is row full of nan/none in output_data_for_training_encoded_for_ai")
        if input_data_for_validation_encoded_for_ai.isnull().all(axis=1).any( ):
            logger.error( f"There is row full of nan/none in input_data_for_validation_encoded_for_ai")
        if output_data_for_validation_encoded_for_ai.isnull().all(axis=1).any( ):
            logger.error( f"There is row full of nan/none in output_data_for_validation_encoded_for_ai")

        # prepare parameters for the training
        first_training_epoch_count_max = self._compute_optimal_epoch_count(mode_slow=True)
        training_batch_size = self._compute_optimal_batch_size(mode_slow=True)

        epoch_patience_end_training = 50
        # create callbacks list and add TensorBoard callback
        keras_earlystopping_fit_callback = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=epoch_patience_end_training,
            min_delta=0.0001,
            restore_best_weights=True
        )

        # sometime if the mode is not very good the training do not start and loss stay same value, then training stop with early-stopping-patience
        attempt_training_start_correctly = 1
        training_successfully_completed = False
        while attempt_training_start_correctly <= NNENGINE_TRAINING_ATTEMPT_TRAINING_START_CORRECTLY_MAX:

            #if ENABLE_LOGGER_DEBUG: logger.debug(f"evaluate_memory_required_gpu_training = {self.evaluate_memory_required_gpu_training(neural_network_model_to_train,batch_size=first_training_batch_size)}")
            # compile the neural_network_model_to_train
            #neural_network_model_to_train = self._build_keras_nn_model_from_nn_configuration( self._nn_configuration)
            seed = None
            keras_weights_initializer = None
            if attempt_training_start_correctly == 2:
                keras_weights_initializer = 'random_normal'
            elif attempt_training_start_correctly == 3:
                keras_weights_initializer = 'glorot_normal'
            elif attempt_training_start_correctly == 4:
                keras_weights_initializer = 'random_uniform'
            elif attempt_training_start_correctly == 5:
                keras_weights_initializer = 'truncated_normal'
            elif attempt_training_start_correctly == 6:
                keras_weights_initializer = 'glorot_uniform'

            from tensorflow.errors import InternalError, ResourceExhaustedError
            try:
                neural_network_model_to_train = self._nn_configuration.build_keras_nn_model( self._machine.db_machine.parameter_nn_loss , force_weight_initializer=keras_weights_initializer )
                history_first_fit = neural_network_model_to_train.fit(
                    input_data_for_training_encoded_for_ai,
                    output_data_for_training_encoded_for_ai,
                    validation_data=(input_data_for_validation_encoded_for_ai, output_data_for_validation_encoded_for_ai),
                    batch_size=training_batch_size,
                    epochs=first_training_epoch_count_max,
                    callbacks=[keras_earlystopping_fit_callback],
                    verbose= 0,
                )
            except (InternalError, ResourceExhaustedError) as e:
                logger.warning( f"Error tensorflow InternalError or ResourceExhaustedError : Unable to do training because : {e}, total_weights_count:{total_weights_count}, neural_network_model_to_train.summary: {neural_network_model_to_train.summary()}")
                # we stop the loop because it will not work better in next loops
                training_successfully_completed = False
                break
            except Exception as e:
                logger.error(f"ERROR - Unable to do model training cycle because : {e}")
            else:
                training_successfully_completed = True

            if training_successfully_completed:
                # check if results are valid
                if np.isnan( history_first_fit.history['loss'] ).any() or np.isnan( history_first_fit.history['val_loss'] ).any():
                    logger.error( f"NNModel {self._machine} has not trained because it is not possible to evaluate the loss or val_loss" )
                training_total_epoch_done_count = first_training_epoch_count_max if keras_earlystopping_fit_callback.stopped_epoch == 0 else keras_earlystopping_fit_callback.stopped_epoch
                # check if the training have do a minimum of PATIENCE epoch
                if training_total_epoch_done_count <= epoch_patience_end_training:
                    logger.warning( f"NNModel {self._machine} has not trained well, because he did only {training_total_epoch_done_count} epoch (patience={epoch_patience_end_training}), it was attempt {attempt_training_start_correctly} on {NNENGINE_TRAINING_ATTEMPT_TRAINING_START_CORRECTLY_MAX}" )
                    attempt_training_start_correctly += 1
                    training_successfully_completed = False
                else:
                    # Results are Valid, epoch done more than patience, so training is ok, so we stop the loop
                    break

        # WHILE END - end of the loop of training


        if not training_successfully_completed:
            # despite several attempt, the training is not ok
            self._nn_model = None
            self._machine.db_machine.training_type_machine_hardware=str(device_lib.list_local_devices())
            self._machine.db_machine.training_total_training_line_count = len( output_data_for_training_encoded_for_ai )
            self._machine.db_machine.training_training_epoch_count = training_total_epoch_done_count
            self._machine.db_machine.training_training_batch_size = training_batch_size
            self._machine.db_machine.machine_is_re_run_fe = True
            self._machine.db_machine.training_date_time_machine_model = datetime.now()
            logger.warning(f"Training Cycle Failed, it made {attempt_training_start_correctly} attempt ! Training for {self._machine} failed - The NNModel is not good or the data are not good or the FE is not good, or out of memory -- we have enabled ReRunFE to restart configuration and training later")
            return False

        # The training cycle have completed successfully - we store the model in self
        self._nn_model = neural_network_model_to_train
        self._machine.db_machine.training_type_machine_hardware=str(device_lib.list_local_devices())

        count_of_training_lines = len(input_data_for_training_encoded_for_ai)
        self._machine.db_machine.training_total_training_line_count = count_of_training_lines
        self._machine.db_machine.training_training_epoch_count = training_total_epoch_done_count
        self._machine.db_machine.training_training_batch_size = training_batch_size
        total_training_time_seconds = default_timer( ) - training_start_time
        self._machine.db_machine.training_training_total_delay_sec = total_training_time_seconds
        self._machine.db_machine.training_training_cell_delay_sec = ( total_training_time_seconds /
                    ( count_of_training_lines * (
                            self._machine.db_machine.enc_dec_columns_info_input_encode_count
                            + self._machine.db_machine.enc_dec_columns_info_output_encode_count
                            )
                    ) )
        self._machine.db_machine.training_date_time_machine_model = datetime.now()

        logger.info(f"Training Cycle done for machine {self._machine} in {total_training_time_seconds:0f} seconds, batch size={training_batch_size}, epoch={training_total_epoch_done_count}, rows={count_of_training_lines} ")

        # ==================================================
        # we learn to predict the delay a machine_source need for training_cycle
        # experience_inputs are Machine_context + input_data_for_train.count, output is training_training_total_delay_sec
        from ML import MachineEasyAutoML
        MachineEasyAutoML_inputs_as_machine_overview_with_nn_model = self._machine.get_machine_overview_information(
            with_base_info=True,
            with_fec_encdec_info=True,
            with_nn_model_info=True,
            with_training_infos=True,
            with_training_cycle_result=False,
            with_training_eval_result=False
        )
        MachineEasyAutoML_outputs_are_training_result = self._machine.get_machine_overview_information(
            with_base_info=False,
            with_fec_encdec_info=False,
            with_nn_model_info=False,
            with_training_infos=False,
            with_training_cycle_result=True,
            with_training_eval_result=False
        )
        MachineEasyAutoML( "__Results_NNEngine_Training_Cycle__" ).learn_this_inputs_outputs(
            inputsOnly_or_Both_inputsOutputs=MachineEasyAutoML_inputs_as_machine_overview_with_nn_model,
            outputs_optional=MachineEasyAutoML_outputs_are_training_result,
        )
        # ==================================================

        return True


    def do_training_trial(
            self,
            pre_encoded_dataframe: pd.DataFrame = None,
            pre_encoded_validation_dataframe: pd.DataFrame=None,
            encoded_for_ai_dataframe: pd.DataFrame = None,
            encoded_for_ai_validation_dataframe: pd.DataFrame = None
    ):
        """
        do a fast trial training using self._nn_configuration and return model evaluation
        it can use dataframe encoded or not encoded - we can use encoded because solutionfinder will run many trial and we do not want to do the encoding each time

        we use self._nn_configuration self._mdc self._enc_dec and NOTHING from machine.db_machine (the configuration stored in the machine)
              because when doing trial we do not save configurations in db_machine

        we do not use (read or write ) self._nn_model at all - the model trained is discarded, only accuracy and loss and returned

        USAGE: usually to test NNConfigurations or FEConfigurations
        USAGE: Never use for customer as the training is not good quality

        :param pre_encoded_dataframe: the pre_encoded_dataframe not encoded to train with
        :param pre_encoded_validation_dataframe: the pre_encoded_dataframe to use for evaluation, if none is provided then we will use the training pre_encoded_dataframe
        :param encoded_for_ai_dataframe: the dataframe encoded to train with
        :param encoded_for_ai_validation_dataframe: the dataframe encoded to train with
        :returns: loss of the model evaluated after the training, accuracy of the model evaluated after the training, epoch_done in percentage of the epoch max to do
        """

        # we check parameters are provided correctly
        if pre_encoded_dataframe is not None and encoded_for_ai_dataframe is not None:
            logger.error("It is not possible to provide simultaneously encoded and not encoded dataframe")
        if pre_encoded_validation_dataframe is not None and encoded_for_ai_validation_dataframe is not None:
            logger.error("It is not possible to provide simultaneously encoded and not encoded dataframe")

        if not self._nn_configuration:
            logger.error( "Unable to do the Training trial because the machine do not have a self._nn_configuration")
        if not self._mdc:
            logger.error( "Unable to do the Training trial because the machine do not have a self._mdc ")
        if not self._enc_dec:
            logger.error( "Unable to do the Training trial because the machine do not have a self._enc_dec ")

        # check the model is not too large or too small
        total_weights_count = self._nn_configuration.nn_shape_instance.weight_total_count(
                self._nn_configuration.num_of_input_neurons,
                self._nn_configuration.num_of_output_neurons,
            )
        if NNENGINE_WEIGHT_COUNT_MIN > total_weights_count > NNENGINE_WEIGHT_COUNT_MAX:
            logger.warning( f"Total neurons weights:{total_weights_count} in this model is not between NNENGINE_WEIGHT_COUNT_MAX and NNENGINE_WEIGHT_COUNT_MIN")
            return None, None , (None , None )

        if pre_encoded_dataframe is not None and encoded_for_ai_dataframe is None:
            encoded_for_ai_dataframe = self._enc_dec.encode_for_ai(pre_encoded_dataframe)
            if pre_encoded_validation_dataframe is None:
                logger.error( "Missing the pre_encoded_validation_dataframe")
            else:
                encoded_for_ai_validation_dataframe = self._enc_dec.encode_for_ai(pre_encoded_validation_dataframe )

        # check count of dataset rows
        if len(encoded_for_ai_dataframe) < 100:
            logger.warning(f"the training dataframe have only {len(encoded_for_ai_dataframe)} rows, less than 100 rows")
        if len(encoded_for_ai_validation_dataframe) < 25:
            logger.warning(f"the validation dataframe has only {len(encoded_for_ai_validation_dataframe)} rows, less than 25 rows")
        elif len(encoded_for_ai_validation_dataframe) > 500:
            logger.warning(f"the validation dataframe have {len(encoded_for_ai_validation_dataframe)} rows, more than 500 rows")

        if ENABLE_LOGGER_DEBUG_NNEngine: logger.debug( f"Starting trial training on {len(encoded_for_ai_dataframe)} rows by  {len(encoded_for_ai_dataframe.columns)} columns and with {total_weights_count} total NN weight...." )

        # split data to train with, in inputs/outputs
        (
            input_training_encoded_for_ai_dataframe,
            output_training_encoded_for_ai_dataframe,
        ) = self._split_dataframe_into_input_and_output(
            encoded_for_ai_dataframe,
            dataframe_status=DataframeEncodingType.ENCODED_FOR_AI,
        )

        # split data to validate, in inputs/outputs
        (
            input_validation_encoded_for_ai_dataframe,
            output_validation_encoded_for_ai_dataframe,
        ) = self._split_dataframe_into_input_and_output(
            encoded_for_ai_validation_dataframe,
            dataframe_status=DataframeEncodingType.ENCODED_FOR_AI,
        )

        # the keras model do not work if any rows have all None/NaN
        # the system should skip this rows before
        if input_training_encoded_for_ai_dataframe.isnull().all(axis=1).any( ):
            logger.error( f"There is row full of nan/none in input_training_encoded_for_ai_dataframe")
        if output_training_encoded_for_ai_dataframe.isnull().all(axis=1).any( ):
            logger.error( f"There is row full of nan/none in output_training_encoded_for_ai_dataframe")
        if input_validation_encoded_for_ai_dataframe.isnull().all(axis=1).any( ):
            logger.error( f"There is row full of nan/none in input_validation_encoded_for_ai_dataframe")
        if output_validation_encoded_for_ai_dataframe.isnull().all(axis=1).any( ):
            logger.error( f"There is row full of nan/none in output_validation_encoded_for_ai_dataframe")

        batch_size = self._compute_optimal_batch_size(mode_fast=True)
        epochs_count_max = self._compute_optimal_epoch_count(mode_fast=True)

        epoch_patience_end_training = 25
        keras_earlystopping_fit_callback = EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=epoch_patience_end_training,
                min_delta=0.0001,
            )

        from tensorflow.errors import InternalError, ResourceExhaustedError
        neural_network_model = self._nn_configuration.build_keras_nn_model( self._machine.db_machine.parameter_nn_loss )

        for fit_trial_attempt in range( 1, NNENGINE_TRAINING_ATTEMPT_TRAINING_START_CORRECTLY_MAX ):
            try:
                neural_network_model.fit(
                    input_training_encoded_for_ai_dataframe,
                    output_training_encoded_for_ai_dataframe,
                    epochs=epochs_count_max,
                    batch_size=batch_size,
                    verbose=1 if ENABLE_LOGGER_DEBUG_NNEngine_DETAILED else 0,
                    callbacks=[keras_earlystopping_fit_callback],
                    validation_data=(input_validation_encoded_for_ai_dataframe, output_validation_encoded_for_ai_dataframe)
                )
            except (InternalError, ResourceExhaustedError) as e:
                logger.error( f"Error tensorflow InternalError or ResourceExhaustedError : Unable to do trial training because : {e}, total_weights_count:{total_weights_count}, neural_network_model.summary: {neural_network_model.summary()}" )
                return None, None , ( None )
            except Exception as e:
                logger.error(f"Unable to do trial training because : {e}")

            epoch_done = keras_earlystopping_fit_callback.stopped_epoch if keras_earlystopping_fit_callback.stopped_epoch else epochs_count_max
            if epoch_done > epoch_patience_end_training:
                # fit successfull , we evaluate the model and return loss and accuracy
                (loss, accuracy) = neural_network_model.evaluate(
                        input_validation_encoded_for_ai_dataframe,
                        output_validation_encoded_for_ai_dataframe,
                        verbose=0
                    )
                loss = float( loss ) if not np.isnan( loss ) else None
                accuracy = float( accuracy ) if not np.isnan( accuracy ) else None
                if ENABLE_LOGGER_DEBUG_NNEngine: logger.debug( f"Trial Training #{fit_trial_attempt} successfull --- Hidden layers={self._nn_configuration.nn_shape_instance.get_hidden_layers_count( )}, batchsize={batch_size}, epoch={epoch_done}/{ epochs_count_max}, scaled_loss={ self._machine.scale_loss_to_user_loss( loss ) }, accuracy={accuracy} " )
                return loss, accuracy, (epoch_done/epochs_count_max)

        # All attempt of fit have failed
        if ENABLE_LOGGER_DEBUG_NNEngine: logger.debug( f"All attempt of fit have failed --- Hidden layers={self._nn_configuration.nn_shape_instance.get_hidden_layers_count( )}, batchsize={batch_size} , total_weights_count:{total_weights_count}" )
        return None, None, (None)


    def _compute_loss_of_random_dataset( self, parameter_nn_loss ) -> float:
        """
         compute parameter_nn_loss_scaler using MDC and ENCDEC
         this is the maximum loss a machine can have

        :params parameter_nn_loss: the loss function used to compute the RANDOM loss
         when we display loss to the user we will always rescale it using this value, so the Loss_User is always from 0 to 1
         parameter_nn_loss_scaler is the loss between 2 dataset with the outputs  randomized, so it will be the worst loss possible to have
        """

        loss_fn = get_loss_function( parameter_nn_loss )

        dataframe_evaluation_loss_scaler_encoded_for_ai = \
            self._enc_dec.encode_for_ai(
                self._mdc.dataframe_pre_encode(
                self._machine.get_random_user_dataframe_for_training_trial( is_for_evaluation=True ) ) )

        (
            input_loss_scaler_dataframe_encoded_for_ai,
            output_loss_scaler_dataframe_encoded_for_ai,
        ) = self._split_dataframe_into_input_and_output(
            dataframe_evaluation_loss_scaler_encoded_for_ai,
            dataframe_status=DataframeEncodingType.ENCODED_FOR_AI,
        )

        # randomize the outputs columns values (swap them with another row)
        output_loss_scaler_dataframe_encoded_for_ai_randomized = output_loss_scaler_dataframe_encoded_for_ai.copy( )
        for row_idx in range( 0, output_loss_scaler_dataframe_encoded_for_ai_randomized.shape[0]):
            for col_idx in range(0 , output_loss_scaler_dataframe_encoded_for_ai_randomized.shape[1]):
                output_loss_scaler_dataframe_encoded_for_ai_randomized.iloc[row_idx,col_idx] = output_loss_scaler_dataframe_encoded_for_ai.iloc[ int( random.uniform(0,output_loss_scaler_dataframe_encoded_for_ai_randomized.shape[0]) ) ,col_idx]


        # we compare normal evaluation dataset with a dataset with randomized outputs values
        # the loss of all lines are evaluated, the highest loss of all lines is the result
        df_rows_count = output_loss_scaler_dataframe_encoded_for_ai_randomized.shape[0]
        loss_on_random_dataset = np.nanmax(
            loss_fn(
                output_loss_scaler_dataframe_encoded_for_ai,
                output_loss_scaler_dataframe_encoded_for_ai_randomized
            ).numpy() )
        if np.isnan(loss_on_random_dataset):
            logger.error( "strange cannot compute loss_on_random_dataset - this can happen when outputs do not vary enough")
        if loss_on_random_dataset == 0:
            logger.error( "strange cannot compute loss_on_random_dataset - this can happen when outputs do not vary at all")

        return float(loss_on_random_dataset)


    def do_solving_all_data_lines(self ):
        """
        will solve all lines currently waiting to be solved in DataInputLines table
        will write the outputs in DataOutputsLines table
        will mark in DataInputLines table that the lines are solved
        """

        # Read lines to solve from database
        dataframe_user_to_solve = self._machine.data_input_lines_read(isForSolving=True)
        if dataframe_user_to_solve.empty:
            logger.warning("there is currently no rows to solve in DataInputsLines")
            return

        decoded_user_dataframe_solved_data = self.do_solving_direct_dataframe_user( dataframe_user_to_solve, )

        self._machine.data_output_lines_update(decoded_user_dataframe_solved_data)

        if ENABLE_LOGGER_DEBUG_NNEngine: logger.debug( f"do_solving_data_lines  {self._machine} have processed {len(dataframe_user_to_solve )} rows" )


    def do_solving_direct_encoded_for_ai(
            self,
            dataframe_to_solve_encoded_for_ai: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        will solve the dataframe dataframe_to_solve_encoded_for_ai and return the solved data encoded_for_ai

        :param dataframe_to_solve_encoded_for_ai: the dataframe to solve (the experience_inputs of the model)
        :return: the dataframe solved encoded_for_ai
        """

        # verify that the column count of the encoded_for_ai_dataframe is the same as machine.enc_dec_columns_info_input_encode_count
        if len(dataframe_to_solve_encoded_for_ai.columns) != self._machine.db_machine.enc_dec_columns_info_input_encode_count:
            logger.error( f"The dataframe encoded for ai do not have the expected columns count : {self._machine.db_machine.enc_dec_columns_info_input_encode_count} , it have {len(dataframe_to_solve_encoded_for_ai.columns) }")

        neuron_network_model = self._get_nn_model_from_db()

        solved_data = neuron_network_model.predict(dataframe_to_solve_encoded_for_ai)
        solved_dataframe_encoded_for_ai = pd.DataFrame(
            solved_data,
            columns=self._machine.get_list_of_columns_name(
                ColumnDirectionType.OUTPUT,
                dataframe_status=DataframeEncodingType.ENCODED_FOR_AI,
            ),
            index=dataframe_to_solve_encoded_for_ai.index,
        )

        if ENABLE_LOGGER_DEBUG_NNEngine: logger.debug( f"do_solving_direct_encoded_for_ai {self._machine} solved {len(solved_dataframe_encoded_for_ai )} rows " )

        return solved_dataframe_encoded_for_ai


    def do_solving_direct_dataframe_user(
            self,
            dataframe_to_solve_user: pd.DataFrame,
            disable_evaluation_loss_solving=False
    ) -> pd.DataFrame:
        """
        will solve the dataframe (experience_inputs) and return the solved data (outputs) user_decoded
        Note : we can provide only one single dataframe

        :param dataframe_to_solve_user: the dataframe to solve (the experience_inputs of the model)
        :param disable_evaluation_loss_solving: by default if self._machine have loss_prediction enabled we will add to the returned solved dataframe the prediction of the nn_loss

        :return: the dataframe solved (the outputs) predicted , user_decoded
        """

        dataframe_to_solve_encoded_for_ai = self.dataframe_full_encode( dataframe_to_solve_user )

        solved_dataframe = self.do_solving_direct_encoded_for_ai( dataframe_to_solve_encoded_for_ai )

        decoded_solved_data_of_machine_source = self.dataframe_full_decode( solved_dataframe )

        # # ------------- for machines having LOSS-PREDICTION --------------
        # if (not disable_evaluation_loss_solving and self._machine.db_machine.multimachines_Is_confidence_enabled):
        #     machine_loss_prediction = Machine( self._machine.db_machine.multimachines_Id_loss_predictor_machine )
        #     loss_prediction_dataframe_to_solve = dataframe_to_solve_user.join( decoded_solved_data_of_machine_source, how="inner" )
        #     nn_engine = NNEngine( machine_loss_prediction , allow_re_run_configuration=False )
        #     solved_loss = nn_engine.do_solving_direct_dataframe_user( loss_prediction_dataframe_to_solve, disable_evaluation_loss_solving=True )
        #     decoded_solved_data_of_machine_source = (decoded_solved_data_of_machine_source.join( solved_loss, how="inner" ))

        if ENABLE_LOGGER_DEBUG_NNEngine: logger.debug( f"do_solving_direct {self._machine} solved {len(decoded_solved_data_of_machine_source )} rows " )

        return decoded_solved_data_of_machine_source


    def do_evaluation(self):
        """
        evaluate the current model with the dataLines
        generate 8 values and store them in machine.Training_Eval_*
        """

        if ENABLE_LOGGER_DEBUG_NNEngine: logger.debug( f"NNModel evaluation starting for {self._machine}" )
        neural_network_model = self._get_nn_model_from_db()

        dataframe_pre_encoded_training = \
            self._mdc.dataframe_pre_encode(
                self._machine.get_random_user_dataframe_for_training_trial( is_for_learning=True, force_row_count_same_as_for_evaluation=True ) )
        if dataframe_pre_encoded_training.empty:
            logger.error(" dataframe_training is empty ! ")

        dataframe_pre_encoded_evaluation = \
                self._mdc.dataframe_pre_encode(
            self._machine.get_random_user_dataframe_for_training_trial( is_for_evaluation=True ) )
        if dataframe_pre_encoded_evaluation.empty:
            logger.error(" dataframe_evaluation is empty ! ")


        # -------------------------------------------------------------------
        # DO EVALUATION FOR -- Loss/Accuracy for all columns --- Between Data-Evaluation and prediction
        # -------------------------------------------------------------------
        losses_for_each_output_columns , accuracy_for_each_output_columns = self.compute_loss_accuracy_of_pre_encoded_dataframe_for_each_columns(dataframe_pre_encoded_evaluation )

        # store the loss and accuracy for all user outputs columns
        self._machine.db_machine.training_eval_outputs_cols_loss_sample_evaluation = losses_for_each_output_columns
        self._machine.db_machine.training_eval_outputs_cols_accuracy_sample_evaluation = accuracy_for_each_output_columns


        # -------------------------------------------------------------------
        # DO EVALUATION FOR -- Average Loss/Accuracy --- Between Data-Training  and prediction AND Between Data-Training with noise and prediction
        # -------------------------------------------------------------------
        dataframe_training_encoded_for_ai = self._enc_dec.encode_for_ai(dataframe_pre_encoded_training)

        (
            input_training_dataframe_encoded_for_ai,
            output_training_dataframe_encoded_for_ai,
        ) = self._split_dataframe_into_input_and_output(
            dataframe_training_encoded_for_ai,
            dataframe_status=DataframeEncodingType.ENCODED_FOR_AI,
        )

        dataframe_training_input_encoded_for_ai_with_noise=self._add_noise_to_dataframe_encoded_for_ai(input_training_dataframe_encoded_for_ai )

        loss_training, accuracy_training = neural_network_model.evaluate(
            input_training_dataframe_encoded_for_ai,
            output_training_dataframe_encoded_for_ai,
            verbose=0
        )
        if np.isnan(loss_training):
            loss_training = 1
            logger.warning( "No result for loss_training ")

        loss_training_with_noise, accuracy_training_with_noise = neural_network_model.evaluate(
            dataframe_training_input_encoded_for_ai_with_noise,
            output_training_dataframe_encoded_for_ai,
            verbose=0
        )
        if np.isnan(loss_training_with_noise):
            loss_training_with_noise = 1
            logger.warning( "No result for loss_training_with_noise ")

        self._machine.db_machine.training_eval_loss_sample_training = self._machine.scale_loss_to_user_loss( loss_training)
        self._machine.db_machine.training_eval_loss_sample_training_noise = self._machine.scale_loss_to_user_loss( loss_training_with_noise)
        self._machine.db_machine.training_eval_accuracy_sample_training = accuracy_training
        self._machine.db_machine.training_eval_accuracy_sample_training_noise = accuracy_training_with_noise

        if ENABLE_LOGGER_DEBUG_NNEngine: logger.debug(f"Start Training Evaluation -- loss_training: {loss_training:.3f} loss_training_with_noise: {loss_training_with_noise:.3f} accuracy_training: {accuracy_training:.3f} accuracy_training_with_noise: {accuracy_training_with_noise:.3f}")

        # -------------------------------------------------------------------
        # DO EVALUATION FOR  -- Average Loss/Accuracy --- Between Data-Evaluation  and prediction --- Between Data-Evaluation with noise and prediction
        # -------------------------------------------------------------------
        if dataframe_pre_encoded_evaluation.empty:
            logger.error(" dataframe_evaluation is empty ! ")
        encoded__for_ai_evaluation_dataframe=self._enc_dec.encode_for_ai(dataframe_pre_encoded_evaluation)

        (
            input_encoded_for_ai_evaluation_dataframe,
            output_encoded_for_ai_evaluation_dataframe,
        ) = self._split_dataframe_into_input_and_output(
            encoded__for_ai_evaluation_dataframe,
            dataframe_status=DataframeEncodingType.ENCODED_FOR_AI,
        )

        input_encoded_for_ai_evaluation_dataframe_with_noise=self._add_noise_to_dataframe_encoded_for_ai(input_encoded_for_ai_evaluation_dataframe )

        (
            loss_evaluation,
            accuracy_evaluation
        ) = neural_network_model.evaluate(
            input_encoded_for_ai_evaluation_dataframe,
            output_encoded_for_ai_evaluation_dataframe,
            verbose=0)
        if np.isnan(loss_evaluation):
            loss_evaluation = 1
            logger.warning( "No result for loss_evaluation ")
        loss_evaluation = self._machine.scale_loss_to_user_loss( loss_evaluation )

        (
            loss_evaluation_with_noise,
            accuracy_evaluation_with_noise,
        ) = neural_network_model.evaluate(
            input_encoded_for_ai_evaluation_dataframe_with_noise,
            output_encoded_for_ai_evaluation_dataframe,
            verbose=0,
        )
        if np.isnan(loss_evaluation_with_noise):
            loss_evaluation_with_noise = 1
            logger.warning( "No result for loss_evaluation_with_noise ")
        loss_evaluation_with_noise = self._machine.scale_loss_to_user_loss( loss_evaluation_with_noise )

        self._machine.db_machine.training_eval_loss_sample_evaluation=float(loss_evaluation)
        self._machine.db_machine.training_eval_loss_sample_evaluation_noise=float(loss_evaluation_with_noise)
        self._machine.db_machine.training_eval_accuracy_sample_evaluation=float(accuracy_evaluation)
        self._machine.db_machine.training_eval_accuracy_sample_evaluation_noise=float(accuracy_evaluation_with_noise)

        if ENABLE_LOGGER_DEBUG_NNEngine: logger.debug(f"Training Evaluation Done -- loss_evaluation: {loss_evaluation:.3f} loss_evaluation_with_noise: {loss_evaluation_with_noise:.3f} accuracy_evaluation: {accuracy_evaluation:.3f} accuracy_evaluation_with_noise: {accuracy_evaluation_with_noise:.3f}")


    def compute_loss_accuracy_of_pre_encoded_dataframe_for_each_columns(self, dataframe_pre_encoded ):
        """
        From the provided dataframe , we will compare the output predicted with the outputs provided and we will return the loss, accuracy for each outputs
        The provided dataframe should be from evaluation data

        return loss , accuracy of each output columns
        """

        # validation dataframe encoded (encoded for ai)
        output_pre_encoded_columns = self._machine.get_list_of_columns_name(
            column_mode=ColumnDirectionType.OUTPUT,
            dataframe_status = DataframeEncodingType.PRE_ENCODED
        )
        dataframe_outputs_encoded_for_ai = self._enc_dec.encode_for_ai(
            dataframe_pre_encoded[output_pre_encoded_columns ]
        )

        # predicted validation dataframe (encoded for ai)
        input_pre_encoded_columns = self._machine.get_list_of_columns_name(
            column_mode=ColumnDirectionType.INPUT,
            dataframe_status = DataframeEncodingType.PRE_ENCODED
        )
        predicted_outputs_encoded_for_ai = self.do_solving_direct_encoded_for_ai(
            self._enc_dec.encode_for_ai(
                dataframe_pre_encoded[input_pre_encoded_columns ] ) )

        # evaluate the loss between validation_dataframe_outputs_encoded_for_ai and predicted_outputs_encoded_for_ai
        # for each user column SO we need to average the loss of columns encoded for ai for each user columns
        losses_for_each_output_columns = dict()
        accuracy_for_each_output_columns = dict()

        output_columns_pre_encoded = self._machine.get_list_of_columns_name(
            column_mode=ColumnDirectionType.OUTPUT,
            dataframe_status = DataframeEncodingType.PRE_ENCODED
        )
        for current_output_column_pre_encoded in output_columns_pre_encoded:
            all_columns_encoded_for_ai_for_current_column_pre_encoded = \
                self._enc_dec.get_encoded_for_ai_columns_names_by_the_pre_encoded_column_name(
                    current_output_column_pre_encoded
                )

            # column output can be without any FET if it is type IGNORE
            if all_columns_encoded_for_ai_for_current_column_pre_encoded:
                # this is the loss function activated in the NNConfiguration for this machine
                loss_fn = get_loss_function(self._machine.db_machine.parameter_nn_loss)

                # this compute the mean of all loss of all children output columns of the user column
                current_pre_encoded_column_loss = np.nanmean(
                    loss_fn(
                        dataframe_outputs_encoded_for_ai[all_columns_encoded_for_ai_for_current_column_pre_encoded],
                        predicted_outputs_encoded_for_ai[all_columns_encoded_for_ai_for_current_column_pre_encoded],
                    ).numpy()
                )
                if np.isnan(current_pre_encoded_column_loss):
                    current_pre_encoded_column_loss = 0
                    logger.warning( f"Rare! : Loss==0 for column : {current_output_column_pre_encoded} in {self} (mean the column data do not vary)" )
                # store the computed loss for the user column
                losses_for_each_output_columns[current_output_column_pre_encoded] = self._machine.scale_loss_to_user_loss( float(current_pre_encoded_column_loss) )

                # compute the accuracy of the user output column
                acc = Accuracy()
                acc.update_state(
                    dataframe_outputs_encoded_for_ai[all_columns_encoded_for_ai_for_current_column_pre_encoded],
                    predicted_outputs_encoded_for_ai[all_columns_encoded_for_ai_for_current_column_pre_encoded],
                )
                accuracy_for_each_output_columns[current_output_column_pre_encoded] = float( np.nanmean(acc.result().numpy()))

        if ENABLE_LOGGER_DEBUG_NNEngine: logger.debug( f"evaluation done - training_eval_outputs_cols_loss_sample_evaluation={losses_for_each_output_columns}" )
        return losses_for_each_output_columns , accuracy_for_each_output_columns


    def compute_loss_all_cells_between_two_pre_encoded_dataframe(self, dataframe_source_pre_encoded , dataframe_solved_pre_encoded):
        """
        for each pair of values in both dataframe in argument , evaluate the loss - (will calculate only outputs)

        :params dataframe_source_pre_encoded: the dataframe source
        :params dataframe_solved_pre_encoded: the dataframe predicted

        return dict ( by columns ) of the loss of each pairs of values
        """

        output_pre_encoded_columns = self._machine.get_list_of_columns_name(
            column_mode=ColumnDirectionType.OUTPUT,
            dataframe_status = DataframeEncodingType.PRE_ENCODED
        )

        # get source dataframe encoded for ai
        dataframe_source_outputs_encoded_for_ai = self._enc_dec.encode_for_ai(
            dataframe_source_pre_encoded[output_pre_encoded_columns ]
        )

        # get solved dataframe encoded for ai
        dataframe_solved_outputs_encoded_for_ai = self._enc_dec.encode_for_ai(
            dataframe_solved_pre_encoded[output_pre_encoded_columns ]
        )

        # evaluate the loss between dataframe_source_outputs_encoded_for_ai and dataframe_solved_outputs_encoded_for_ai
        losses_all_cells_per_user_pre_encoded_columns = dict()

        output_user_columns = self._machine.get_list_of_columns_name(
            column_mode=ColumnDirectionType.OUTPUT,
            dataframe_status = DataframeEncodingType.USER
        )
        for current_output_column_pre_encoded in output_pre_encoded_columns:
            all_columns_encoded_for_ai_for_current_column_pre_encoded = \
                self._enc_dec.get_encoded_for_ai_columns_names_by_the_pre_encoded_column_name(
                    current_output_column_pre_encoded
                )

            # column output can be without any FET if it is type IGNORE
            if all_columns_encoded_for_ai_for_current_column_pre_encoded:
                # this is the loss function activated in the NNConfiguration for this machine
                loss_fn = get_loss_function(self._machine.db_machine.parameter_nn_loss)

                # this compute the mean of all loss of all children output columns of the user column
                current_pre_encoded_column_loss = (
                    loss_fn(
                        dataframe_source_outputs_encoded_for_ai[all_columns_encoded_for_ai_for_current_column_pre_encoded],
                        dataframe_solved_outputs_encoded_for_ai[all_columns_encoded_for_ai_for_current_column_pre_encoded],
                    ).numpy()
                )
                # store the computed loss for the user column
                losses_all_cells_per_user_pre_encoded_columns[current_output_column_pre_encoded] = current_pre_encoded_column_loss

        if ENABLE_LOGGER_DEBUG_NNEngine: logger.debug( f"evaluation done of loss for each cells" )
        return losses_all_cells_per_user_pre_encoded_columns


    def _split_dataframe_into_input_and_output(
        self,
        dataframe_: pd.DataFrame,
        dataframe_status: DataframeEncodingType,
    ) -> (pd.DataFrame, pd.DataFrame):
        """
        split the dataframe (encoded or not) in 2 dataframe : input and output and return both dataframe

        :param dataframe_: dataframe to split
        :param dataframe_status: the type of encoding in the dataframe (user, encoded_for_ai, pre_decoded.)
        """
        if not isinstance(dataframe_, pd.DataFrame):
            logger.error(f"_split_dataframe_into_input_and_output need one pandas-dataframe but get {type(dataframe_)} instead")

        input_columns = self._machine.get_list_of_columns_name(ColumnDirectionType.INPUT, dataframe_status)
        output_columns = self._machine.get_list_of_columns_name(ColumnDirectionType.OUTPUT, dataframe_status)

        if not all(colname in dataframe_.columns for colname in input_columns):
            logger.error(f"Cannot find all the configured inputs columns  {input_columns} in dataframe {dataframe_.columns}")
        if not all(colname in dataframe_.columns for colname in output_columns):
            logger.error(f"Cannot find all the configured outputs columns {output_columns} in dataframe {dataframe_.columns}")

        return dataframe_[input_columns], dataframe_[output_columns]


    def _get_data_output_lines_encoded_for_ai(self) -> pd.DataFrame:
        """
        read DataOutputLines and do full encode
        :return: the dataframe encoded_for_AI
        """
        dataframe_ = self._machine.data_output_lines_read()
        return self.dataframe_full_encode(dataframe_)


    def _delete_nn_model_from_db(self) -> NoReturn:
        """
        Will delete the model in table NNModel
        set none in table machine_source
        set none in self._nn_model
        """
        self._nn_model = None
        if self._machine.db_machine.training_nn_model_extfield is not None:
            del self._machine.db_machine.training_nn_model_extfield


    def _get_nn_model_from_db(self) -> Sequential:
        """
        we create an empty model with self._nn_configuration.build_keras_nn_model
        we load the model weights from self._machine.db_machine.training_nn_model_extfield
        and we return the model
        """

        # if already initialized we return the model
        if self._nn_model is not None:
            return self._nn_model

        if ENABLE_LOGGER_DEBUG_NNEngine: logger.debug( f"_load_nn_model_from_db for {self._machine} " )

        if self._machine.db_machine.training_nn_model_extfield is None:
            logger.error(f"Unable to load NNModel from db - the machine {self._machine} do not have a stored NNModel")

        try:
            nn_model_pickles_bytes = ( self._machine.db_machine.training_nn_model_extfield )
            nn_model_np_array = pickle.loads(nn_model_pickles_bytes)
        except Exception as e:
            logger.error(f"Unable to load NNModel {self._machine} from db. the error is : {e}")

        neural_network_model = self._nn_configuration.build_keras_nn_model( self._machine.db_machine.parameter_nn_loss )
        try:
            neural_network_model.set_weights( nn_model_np_array )
        except Exception as e:
            logger.error(f"Unable to set weight of the NNModel {self._machine} from disk - It is probably because the NNShape have changed after we saved NNModel and now the saved NNModel do not fit the current NNShape. the error is {e}")

        return neural_network_model


    def _save_nn_model_to_db(self) -> NoReturn:
        """
        We save the model self._nn_model into the db
        If a previous NNModel was saved we will delete it
        """

        nn_model_pickle_bytes = pickle.dumps( self._nn_model.get_weights() )
        self._machine.db_machine.training_nn_model_extfield = nn_model_pickle_bytes

        if ENABLE_LOGGER_DEBUG_NNEngine: logger.debug( f"_saved nn_model_to_db for {self._machine} " )


    def _compute_optimal_batch_size(self, mode_slow=None, mode_fast=None) -> int:
        """
        we compute the largest batch size possible for the training which can fit in standard memory of GPU (GTX1060 = 3Gb)
        Warning : changing value here make necessary to update value in _compute_optimal_epoch_count

        :param mode_slow: If true we will compute for slow mode training
        :param mode_fast: If true we will compute for fast mode training
        :return: optimal_batch_size for training
        """

        if mode_fast and mode_slow:
            logger.error(f"the mode can only be 'fast' or 'slow', not both'")
        elif mode_fast:
            batch_size = 64  # fail a little more often than batch=32 ---- batch 128 fail too often
        elif mode_slow:
            batch_size = 32
        else:
            logger.error(f"the mode can only be 'fast' or 'slow', not none")

        return int(batch_size)


    def _compute_optimal_epoch_count(self, mode_slow=None, mode_fast=None) -> int:
        """
        we define the best maximum epoch count for the training (it depend of value in _compute_optimal_batch_size):
        for fast mode it is the epoch count
        for slow mode it is the maximum epoch count because the slow mode have early stopping patience

        :param mode_slow: If true we will compute for slow mode training
        :param mode_fast: If true we will compute for fast mode training
        :return: optimal_epoch_count for training
        """

        if mode_fast and mode_slow:
            logger.error(f"the mode can only be 'fast' or 'slow', not both'")
        elif mode_fast:
            # in mode fast we want to see regular decrease of loss, but we will not use fast models
            # we need to see more epoch if the inputs are large because the problem is more complex
            # this setting depend of batch size - batchsize decrease need for epoch
            optimal_epoch_count = 100
        elif mode_slow:
            # in mode slow we will make no limit - the limit is made by early_stopping_patience only
            optimal_epoch_count = 100*100
        else:
            logger.error(f"the mode can only be 'fast' or 'slow', not 'None")

        #if ENABLE_LOGGER_DEBUG: logger.debug(f"optimal epoch count selected: {optimal_epoch_count}")
        return int( optimal_epoch_count )


    def _add_noise_to_dataframe_encoded_for_ai(
            self,
            dataframe_encoded_for_ai: pd.DataFrame,
            percentage_of_noise: float = 25.0
    ) -> pd.DataFrame:
        """
        add noise to the dataframe_encoded_for_ai and return the updated dataframe
        this is used to compute the evaluation with noise
        as it is encoded_for_ai , it do not expect nan or inf

        :param dataframe_encoded_for_ai: the dataframe_encoded_for_ai arg_source from where will will compute
        :param percentage_of_noise: We will add or substract this value / 2 / 100
        :return: the dataframe_encoded_for_ai with noise
        """

        dataframe_encoded_for_ai_with_noise = dataframe_encoded_for_ai.copy()

        dataframe_encoded_for_ai_with_noise = np.clip( dataframe_encoded_for_ai_with_noise +  np.random.rand(  *dataframe_encoded_for_ai_with_noise.shape )* percentage_of_noise/100 - percentage_of_noise/200 , 0, 1 )
        return dataframe_encoded_for_ai_with_noise


    def machine_source_evaluate_loss_of_validation_lines( self , machine_source):
        """
        Solve the validation dataset_LossSolving (only lines without values in _LossSolving_) of arg_source machine_source,
        evaluate the nn_loss for each line,
        store the nn_loss computed in the field _LossSolving_
        """

        validation_input_dataset = machine_source.data_input_lines_read(
            IsForEvaluation=True
        )
        validation_dataset = machine_source.data_lines_read(IsForEvaluation=True)
        # Filter only not evaluated lines
        validation_input_dataset = validation_input_dataset[
            validation_dataset._LossSolving_.isna()
        ]

        nn_engine = NNEngine(machine_source , allow_re_run_configuration=False )
        nn_model = nn_engine._get_nn_model_from_db()

        mdc = MachineDataConfiguration(machine_source)
        enc_dec = EncDec(machine_source)

        encoded_validation_dataframe = enc_dec.encode_for_ai(
            mdc.dataframe_pre_encode(validation_input_dataset)
        )

        predicted_output_dataframe = nn_engine.do_solving_direct_dataframe_user(
            validation_input_dataset, disable_evaluation_loss_solving=True
        )
        predicted_output_dataframe[pd.isna(predicted_output_dataframe)] = 0

        encoded_predicted_dataframe = enc_dec.encode_for_ai(
            mdc.dataframe_pre_encode(predicted_output_dataframe)
        )

        calculated_loss = np.zeros(shape=(validation_input_dataset.shape[0], 1))

        for index, row in enumerate(encoded_validation_dataframe.itertuples()):
            actual_row = encoded_validation_dataframe.iloc[[index]]
            predicted_row = encoded_predicted_dataframe.iloc[[index]]
            evaluated_loss = nn_model.evaluate(actual_row, predicted_row, verbose=0)[0]
            calculated_loss[index] = self._machine.scale_loss_to_user_loss( evaluated_loss )

        dataset_LossSolving = pd.DataFrame(
            calculated_loss,
            columns=["_LossSolving_"],
            index=validation_input_dataset.index.get_level_values("Line_ID"),
        )

        machine_source.data_output_lines_update(dataset_LossSolving)


    def evaluate_memory_required_gpu_training(self, neural_network_model, batch_size: int):
        """
        try to evaluate how much memory the training require in the GPU

        :param neural_network_model: the keras model we will use
        :param batch_size: the batch_size we will use for training
        :return: the quantity of byte necessary to make the training
        """
        shapes_mem_count = 0
        internal_model_mem_count = 0
        for l in neural_network_model.layers:
            layer_type = l.__class__.__name__
            if layer_type == "Model":
                internal_model_mem_count += self.evaluate_memory_required_gpu_training(
                    l, batch_size=batch_size
                )
            single_layer_mem = 1
            out_shape = l.output_shape
            if type(out_shape) is list:
                out_shape = out_shape[0]
            for s in out_shape:
                if s is None:
                    continue
                single_layer_mem *= s
            shapes_mem_count += single_layer_mem

        trainable_count = np.sum(
            [k.count_params(p) for p in neural_network_model.trainable_weights]
        )
        non_trainable_count = np.sum(
            [k.count_params(p) for p in neural_network_model.non_trainable_weights]
        )

        number_size = 4.0
        if k.floatx() == "float16":
            number_size = 2.0
        if k.floatx() == "float64":
            number_size = 8.0

        return (
            number_size
            * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
            + internal_model_mem_count
        )

