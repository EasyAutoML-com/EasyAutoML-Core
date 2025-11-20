import copy
from typing import Union, Optional
from abc import ABC, abstractmethod
import pandas as pd

from ML import EasyAutoMLDBModels, __getlogger

from ML import Machine, NNEngine, MachineDataConfiguration
from ML.NNConfiguration import NNConfiguration, NNShape
from ML.FeatureEngineeringTemplate import Column_datas_infos

from SharedConstants import *

logger = __getlogger()

class Experimenter(ABC):
    """
    Experimenter run the experience_inputs to evaluate the outputs
    the experimenter work always with the same experience_inputs and outputs
    We use the base class to build new experimenters for each experience_inputs/outputs problems

    :usage: used in SolutionFinder to evaluate a NN-configuration and return accuracy-nn_loss
    :usage: used in SolutionFinder to evaluate a FEC-configuration and return accuracy-delta
    """

    def do(self, user_inputs: Union[dict, pd.DataFrame]) -> Union[dict, pd.DataFrame]:
        """
        using experience_inputs , do the experiment and return the outputs
        the experimenter work always with the same experience_inputs and outputs columns

        :param user_inputs: read the experience_inputs to return the outputs, experience_inputs can be a dict (single row) or a
                            dataframe (multiple rows)

        :return: return the outputs as a dataframe
                            Dataframe will have same count of rows as inputsOnly_or_Both_inputsOutputs dataframe.
                            If the function fail for some rows it will return a dataframe with a none in rows not processed
        """
        if isinstance(user_inputs, dict):
            user_inputs_updated = (pd.DataFrame( [ user_inputs ] ).apply( self._do_single, axis=1 ).to_dict( ))
            return user_inputs_updated
        elif isinstance(user_inputs, pd.DataFrame):
            user_inputs_updated = user_inputs.apply( self._do_single, axis=1 )
            return user_inputs_updated
        else:
            logger.error( f"wrong datatype for user_inputs { type( user_inputs ) } ")


    @abstractmethod
    def _do_single(self, df_user_data_input: pd.Series) -> pd.Series:
        """
        using experience_inputs , evaluate and return the outputs
        the experimenter work always with the same experience_inputs and outputs

        :param df_user_data_input: read the experience_inputs to return the outputs, experience_inputs are always a SINGLE row
        :return: return the outputs as a dataframe , the outputs should be all scaled 0 to 1 (because we will do score on them)
                        the return dataframe is always a single row.
                        If the function fail it will return None or a dataframe with a none in the unique row not processed
        """
        pass


class ExperimenterNNConfiguration(Experimenter ):
    """
    Implements Experimenter abstract class
    to experiment a given NNConfigurations
    it runs trial training with this NNConfiguration and returns loss and accuracy

    :usage: used in SolutionFinder to evaluate a NN-configuration and return accuracy
    """

    def __init__(self, nnengine_trial: NNEngine ):
        """
        Creates new ExperimenterNNConfiguration object with given machine_source

        :param nnengine_trial: nnegine on which the experiments will be performed (it contains all objects necessary : machine, encdec, mdc , etc..)
        """
        if ENABLE_LOGGER_DEBUG_Experimenter: logger.debug( f"ExperimenterNNConfiguration init " )

        self._nnengine_trial = nnengine_trial

        self.cached_encoded_for_ai_dataframe = self._nnengine_trial._enc_dec.encode_for_ai(
            self._nnengine_trial._mdc.dataframe_pre_encode(
                self._nnengine_trial._machine.get_random_user_dataframe_for_training_trial( is_for_learning=True ) ) )
        self.cached_encoded_for_ai_validation_dataframe = self._nnengine_trial._enc_dec.encode_for_ai(
            self._nnengine_trial._mdc.dataframe_pre_encode(
                self._nnengine_trial._machine.get_random_user_dataframe_for_training_trial( is_for_evaluation=True ) ) )

        # now we will use parameter_nn_loss_scaler
        if not self._nnengine_trial._machine.db_machine.parameter_nn_loss_scaler:
           logger.error("parameter_nn_loss_scaler undefined !")

        # set the NNEngine to default configuration for the initial evaluation
        #self._nnengine_trial._nn_configuration = NNConfiguration( self._machine_for_trial, generate_configuration_minimum=True )
        #self.initial_loss, self.initial_accuracy = self._nnengine_trial.do_training_trial(
        #        encoded_for_ai_dataframe=self.cached_encoded_for_ai_dataframe,
        #        encoded_for_ai_validation_dataframe=self.cached_encoded_for_ai_validation_dataframe,
        #        )
        #if self.initial_loss is None or self.initial_accuracy is None:
        #        logger.error("Impossible to do the initial training_trial")

        # store the limit of cost to scale the cost to a float >0 and < 1
        from ML.Machine import MachineLevel
        self._cost_neurons_max = MachineLevel( self._nnengine_trial._machine ).nn_shape_count_of_neurons_max( )[ 1 ]
        self._cost_layers_max = MachineLevel( self._nnengine_trial._machine ).nn_shape_count_of_layer_max( )[ 1 ]



    def _do_single(self, nnshape_type_machine_to_evaluate: pd.Series ) -> Optional[pd.Series ]:
        """
        Implements Experimenter _do_single abstract method

        using NN-Configuration parameters given in experience_inputs runs a trial training and returns evaluated nn_loss and accuracy
        the experimenter work always with the same experience_inputs and outputs

        :return: return the outputs as a pandas.series , always a single row. If the function fail it will return
               a dataframe with a none in the unique row not processed

               OUTPUTS ARE:
                nn_loss
                accuracy
                cost
        """

        try:
            # read from arguments the configuration to experiment
            self._nnengine_trial._nn_configuration = NNConfiguration(
                {
                    "nn_optimizer": nnshape_type_machine_to_evaluate[ "parameter_nn_optimizer" ],
                    "nn_shape": NNShape(nnshape_type_machine_to_evaluate ),
                    "num_of_input_neurons": self._nnengine_trial._machine.db_machine.enc_dec_columns_info_input_encode_count,
                    "num_of_output_neurons": self._nnengine_trial._machine.db_machine.enc_dec_columns_info_output_encode_count,
                }
            )
        except Exception:
            logger.error("Unable to instantiate the NNconfiguration")

        if ENABLE_LOGGER_DEBUG_Experimenter: logger.debug( f"ExperimenterNNConfiguration, evaluate NNShape user df:  {NNShape(nnshape_type_machine_to_evaluate )} " )

        # do the experiment ( fast training )
        try:
            result_test_loss, result_test_accuracy, epoch_done_percent = self._nnengine_trial.do_training_trial(
                encoded_for_ai_dataframe=self.cached_encoded_for_ai_dataframe,
                encoded_for_ai_validation_dataframe=self.cached_encoded_for_ai_validation_dataframe,
                )
        except Exception as e:
            logger.error(f"Unable to perform NN experiment : {self._nnengine_trial._nn_configuration} because {e}" )

        if result_test_loss is None or result_test_accuracy is None:
            if ENABLE_LOGGER_DEBUG_Experimenter: logger.debug( f"ExperimenterNNConfiguration failed")
            return None

        # we return values scaled from 0 to 1
        loss_scaled =self._nnengine_trial._machine.scale_loss_to_user_loss( result_test_loss )
        # we compute the accuracy and loss but scaled by the initial loss and initial accuracy
        #loss_scaled = (result_test_loss / self.initial_loss) if self.initial_loss > 0 else result_test_loss

        if ENABLE_LOGGER_DEBUG_Experimenter: logger.debug( f"ExperimenterNNConfiguration done , loss = {loss_scaled:.2} ")

        # we compute the cost, the cost is between 0 and 1 (more than 1 mean over the limit)
        hidden_layers_count = self._nnengine_trial._nn_configuration.nn_shape_instance.get_hidden_layers_count( )
        cost_layers_percent_budget = hidden_layers_count / self._cost_layers_max
        this_neurons_total_count = self._nnengine_trial._nn_configuration.nn_shape_instance.neurons_total_count(
            self._nnengine_trial._nn_configuration.num_of_input_neurons,
            self._nnengine_trial._nn_configuration.num_of_output_neurons,
        )
        cost_neurons_percent_budget = this_neurons_total_count / self._cost_neurons_max

        return pd.Series( {
                "Result_loss_scaled": loss_scaled,
                "Result_epoch_done_percent": epoch_done_percent,
                "Result_cost_neurons_percent_budget": cost_neurons_percent_budget,
                "Result_cost_layers_percent_budget": cost_layers_percent_budget,
        })


class ExperimenterColumnFETSelector(Experimenter):
    """
    evaluate the FET selection by evaluating the loss by doing a trial training
    store the initial_loss for computing the relative loss later when do experiment
    """
    def __init__(self,
            nn_engine_to_use_in_trial: NNEngine,
            column_datas_infos : Column_datas_infos,
            fec_budget_max: int,
            ):

        if ENABLE_LOGGER_DEBUG_Experimenter: logger.debug( f"Initializing, computing nn_loss_scaler with minimum configuration" )

        self._column_datas_infos = column_datas_infos
        self._nnengine_trial = nn_engine_to_use_in_trial
        self._fec_budget_max = fec_budget_max
        self._minimum_fec = None

        from ML import FeatureEngineeringConfiguration, EncDec, InputsColumnsImportance

        # store dataset pre_encoded in self_cache
        self._nnengine_trial._mdc = MachineDataConfiguration( self._nnengine_trial._machine )
        self.cached_pre_encoded_training_dataframe = self._nnengine_trial._mdc.dataframe_pre_encode(
            self._nnengine_trial._machine.get_random_user_dataframe_for_training_trial( is_for_learning=True ) )
        self.cached_pre_encoded_validation_dataframe = self._nnengine_trial._mdc.dataframe_pre_encode(
            self._nnengine_trial._machine.get_random_user_dataframe_for_training_trial( is_for_evaluation=True ) )

        # the nn_engine_machine_to_use_in_trial is not complete it contains only MDC ICI
        # computing the loss scaler with a minimum FEC configuration
        self._minimum_fec = FeatureEngineeringConfiguration( self._nnengine_trial._machine, force_configuration_simple_minimum=True ).save_configuration_in_machine( )
        self._nnengine_trial._fe = copy.copy( self._minimum_fec )
        # we need to save encdec in machine, because some function use encdec configuration from machine
        self._nnengine_trial._enc_dec = EncDec( self._nnengine_trial._machine, self.cached_pre_encoded_training_dataframe ).save_configuration_in_machine( )
        self._nnengine_trial._nn_configuration.adapt_config_to_new_enc_dec( self._nnengine_trial._enc_dec )

        # Before to use the experimenter for the column-Fets we will initialize the loss-scaler
        if not self._nnengine_trial._machine.db_machine.parameter_nn_loss:
            logger.error("parameter_nn_loss undefined in the machine!")
        self._nn_loss_scaler_experimenter = 1 / self._nnengine_trial._compute_loss_of_random_dataset( self._nnengine_trial._machine.db_machine.parameter_nn_loss )
        if ENABLE_LOGGER_DEBUG_Experimenter: logger.debug( f"Initialization done, nn_loss_scaler: {self._nn_loss_scaler_experimenter}" )


    def _do_single(self, df_user_data_input: pd.Series) -> Optional[pd.Series]:

        # extract the list of FET from the solution-inputs - keys of fet start all with EXPERIMENTER_FET_PREFIX_FET_NAME
        solution_list_fet_names = {}
        for solution_input_name, value in df_user_data_input.to_dict().items():
            if solution_input_name.startswith( EXPERIMENTER_FET_PREFIX_FET_NAME ):
                # add the fet name to the list but without the double underscore
                fet_name = solution_input_name[ len(EXPERIMENTER_FET_PREFIX_FET_NAME):]
                solution_list_fet_names[ fet_name ] = value

        if ENABLE_LOGGER_DEBUG_Experimenter: logger.debug( f"ExperimenterColumnFETSelector, doing evaluation one FEC for column '{self._column_datas_infos}' " )

        from ML import FeatureEngineeringConfiguration, EncDec, FeatureEngineeringColumn

        # fec_to_test contains the configuration FEC to test
        fec_to_test = FeatureEngineeringColumn(
            this_column_datas_infos= self._column_datas_infos,
            dict_force_load_this_fet_names= solution_list_fet_names,
        )

        # apply the configuration minimum to all columns, then we update the column to test with the FEC to test : fec_to_test
        try:
            #self._nnengine_trial._fe = FeatureEngineeringConfiguration( self._machine_for_trial, force_configuration_simple_minimum=True )
            # maybe faster
            self._nnengine_trial._fe = copy.copy( self._minimum_fec )
            self._nnengine_trial._fe.store_this_fec_to_fet_list_configuration( fec_to_test , df_user_data_input[ "name" ] )
        except Exception:
            logger.error("unable to build the new FEC configuration to test ")

        # generate new encdec for the fet selected to try and save it in the machine (some method encoding read params here)
        self._nnengine_trial._fe.save_configuration_in_machine()
        self._nnengine_trial._enc_dec = EncDec( self._nnengine_trial._machine, self.cached_pre_encoded_training_dataframe ).save_configuration_in_machine( )
        self._nnengine_trial._nn_configuration.adapt_config_to_new_enc_dec( self._nnengine_trial._enc_dec )

        # do the experiment with fast training
        try:
            result_test_loss, result_test_accuracy, epoch_done_percent = self._nnengine_trial.do_training_trial(
                pre_encoded_dataframe=self.cached_pre_encoded_training_dataframe,
                pre_encoded_validation_dataframe=self.cached_pre_encoded_validation_dataframe
            )
        except Exception as e:
            logger.error(f"unable to do FEC experiment: {df_user_data_input} \n\nfor column: {self._column_datas_infos} \n\nbecause {e}")

        if result_test_loss is None or result_test_accuracy is None:
            if ENABLE_LOGGER_DEBUG_Experimenter: logger.debug( f"ExperimenterColumnFETSelector failed")
            return None

        # we return values scaled from 0 to 1
        loss_scaled = result_test_loss / self._nn_loss_scaler_experimenter
        if ENABLE_LOGGER_DEBUG_Experimenter: logger.debug( f"ExperimenterColumnFETSelector done , loss:{loss_scaled:.2} ")



        # cost is how many columns created , we divide by budget_max to make sure the result is usually between 0 and 1
        fec_cost_percent_budget = fec_to_test._fec_cost / self._fec_budget_max

        return pd.Series({
            "Result_loss_scaled": loss_scaled,
            "Result_epoch_done_percent": epoch_done_percent,
            "Result_fec_cost_percent_budget": fec_cost_percent_budget,
            })
