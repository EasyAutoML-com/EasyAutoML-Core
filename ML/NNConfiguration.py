from typing import Union, Optional, NoReturn
import numpy as np
import pandas as pd
from statistics import mean  # used in EVAL of computed formulas - so keep it even if pycharm say it is not needed
import io
from timeit import default_timer
import tensorflow as tf

from ML import EasyAutoMLDBModels

from ML import Machine, MachineLevel

from SharedConstants import *


from ML import __getlogger

logger = __getlogger()


# ========================================================================
# this is the possible configurations parameters we can use while searching a NNconfiguration with SolutionFinder
NNCONFIGURATION_FINDER_POSSIBLE_NEURON_PERCENTAGE = [ round(i ** 1.75 * 3) for i in range( 1 , 30 ) ]   # [3, 10, 21, 34, 50, 69, 90, 114, 140, 169, 199, 232, 267, 304, 343, 384, 427, 472, 519, 567, 618, 670, 725, 781, 839, 898, 959, 1022, 1087]
NNCONFIGURATION_FINDER_POSSIBLE_DROPOUT_VALUE = [0, 0.1, 0.2, 0.3]
NNCONFIGURATION_FINDER_POSSIBLE_BATCHNORMALIZATION_VALUE = [0, 1]
# ========================================================================




# Note : updating this list make necessary to delete all machine "__SolutionFinder_Experiments_of_NNConfiguration --- " and "__Results__Find_Best_NN_Configuration__"
NNCONFIGURATION_FINDER_INITIAL_EVALUATIONS_SHAPE = [
    {
        "hidden_1": {
            "NeuronPercentage": 500,
            "LayerTypeActivation": "dense_relu",
            "DropOut": 0,
            "BatchNormalization": 0,
        },
        "output": {
            "NeuronPercentage": None,
            "LayerTypeActivation": "dense_relu",
            "DropOut": None,
            "BatchNormalization": None,
        }
    },
    {
        "hidden_1": {
            "NeuronPercentage": 300,
            "LayerTypeActivation": "dense_relu",
            "DropOut": 0,
            "BatchNormalization": 0,
        },
        "hidden_2": {
            "NeuronPercentage": 200,
            "LayerTypeActivation": "dense_relu",
            "DropOut": 0,
            "BatchNormalization": 0,
        },
        "output": {
            "NeuronPercentage": None,
            "LayerTypeActivation": "dense_relu",
            "DropOut": None,
            "BatchNormalization": None,
        }
    },
    {
        "hidden_1": {
            "NeuronPercentage": 300,
            "LayerTypeActivation": "dense_relu",
            "DropOut": 0,
            "BatchNormalization": 0,
        },
        "hidden_2": {
            "NeuronPercentage": 200,
            "LayerTypeActivation": "dense_relu",
            "DropOut": 0,
            "BatchNormalization": 0,
        },
        "hidden_3": {
            "NeuronPercentage": 100,
            "LayerTypeActivation": "dense_relu",
            "DropOut": 0,
            "BatchNormalization": 0,
        },
        "output": {
            "NeuronPercentage": None,
            "LayerTypeActivation": "dense_relu",
            "DropOut": None,
            "BatchNormalization": None,
        }
    },
    {
        "hidden_1": {
            "NeuronPercentage": 400,
            "LayerTypeActivation": "dense_relu",
            "DropOut": 0,
            "BatchNormalization": 0,
        },
        "hidden_2": {
            "NeuronPercentage": 300,
            "LayerTypeActivation": "dense_relu",
            "DropOut": 0,
            "BatchNormalization": 0,
        },
        "hidden_3": {
            "NeuronPercentage": 200,
            "LayerTypeActivation": "dense_relu",
            "DropOut": 0,
            "BatchNormalization": 0,
        },
        "hidden_4": {
            "NeuronPercentage": 100,
            "LayerTypeActivation": "dense_relu",
            "DropOut": 0,
            "BatchNormalization": 0,
        },
        "output": {
            "NeuronPercentage": None,
            "LayerTypeActivation": "dense_relu",
            "DropOut": None,
            "BatchNormalization": None,
        },
    },
    {
        "hidden_1": {
            "NeuronPercentage": 500,
            "LayerTypeActivation": "dense_relu",
            "DropOut": 0,
            "BatchNormalization": 0,
        },
        "hidden_2": {
            "NeuronPercentage": 400,
            "LayerTypeActivation": "dense_relu",
            "DropOut": 0,
            "BatchNormalization": 0,
        },
        "hidden_3": {
            "NeuronPercentage": 300,
            "LayerTypeActivation": "dense_relu",
            "DropOut": 0,
            "BatchNormalization": 0,
        },
        "hidden_4": {
            "NeuronPercentage": 200,
            "LayerTypeActivation": "dense_relu",
            "DropOut": 0,
            "BatchNormalization": 0,
        },
        "hidden_5": {
            "NeuronPercentage": 100,
            "LayerTypeActivation": "dense_relu",
            "DropOut": 0,
            "BatchNormalization": 0,
        },
        "output": {
            "NeuronPercentage": None,
            "LayerTypeActivation": "dense_relu",
            "DropOut": None,
            "BatchNormalization": None,
        },
    },
    {
        "hidden_1": {
            "NeuronPercentage": 600,
            "LayerTypeActivation": "dense_relu",
            "DropOut": 0,
            "BatchNormalization": 0,
        },
        "hidden_2": {
            "NeuronPercentage": 500,
            "LayerTypeActivation": "dense_relu",
            "DropOut": 0,
            "BatchNormalization": 0,
        },
        "hidden_3": {
            "NeuronPercentage": 400,
            "LayerTypeActivation": "dense_relu",
            "DropOut": 0,
            "BatchNormalization": 0,
        },
        "hidden_4": {
            "NeuronPercentage": 300,
            "LayerTypeActivation": "dense_relu",
            "DropOut": 0,
            "BatchNormalization": 0,
        },
        "hidden_5": {
            "NeuronPercentage": 200,
            "LayerTypeActivation": "dense_relu",
            "DropOut": 0,
            "BatchNormalization": 0,
        },
        "hidden_6": {
            "NeuronPercentage": 100,
            "LayerTypeActivation": "dense_relu",
            "DropOut": 0,
            "BatchNormalization": 0,
        },
        "output": {
            "NeuronPercentage": None,
            "LayerTypeActivation": "dense_relu",
            "DropOut": None,
            "BatchNormalization": None,
        },
    },
    {
        "hidden_1": {
            "NeuronPercentage": 700,
            "LayerTypeActivation": "dense_relu",
            "DropOut": 0,
            "BatchNormalization": 0,
        },
        "hidden_2": {
            "NeuronPercentage": 600,
            "LayerTypeActivation": "dense_relu",
            "DropOut": 0,
            "BatchNormalization": 0,
        },
        "hidden_3": {
            "NeuronPercentage": 500,
            "LayerTypeActivation": "dense_relu",
            "DropOut": 0,
            "BatchNormalization": 0,
        },
        "hidden_4": {
            "NeuronPercentage": 400,
            "LayerTypeActivation": "dense_relu",
            "DropOut": 0,
            "BatchNormalization": 0,
        },
        "hidden_5": {
            "NeuronPercentage": 300,
            "LayerTypeActivation": "dense_relu",
            "DropOut": 0,
            "BatchNormalization": 0,
        },
        "hidden_6": {
            "NeuronPercentage": 200,
            "LayerTypeActivation": "dense_relu",
            "DropOut": 0,
            "BatchNormalization": 0,
        },
        "hidden_7": {
            "NeuronPercentage": 100,
            "LayerTypeActivation": "dense_relu",
            "DropOut": 0,
            "BatchNormalization": 0,
        },
        "output": {
            "NeuronPercentage": None,
            "LayerTypeActivation": "dense_relu",
            "DropOut": None,
            "BatchNormalization": None,
        },
    },
]

NNCONFIGURATION_FINDER_INITIAL_EVALUATIONS_LOSS = [
"mean_absolute_error",
"mean_absolute_error",
"mean_absolute_error",
"mean_absolute_error",
"mean_absolute_error",
"mean_absolute_error",
"mean_absolute_error",
]

NNCONFIGURATION_FINDER_INITIAL_EVALUATIONS_OPTIMIZER = [
"Adam",
"Adam",
"Adam",
"Adam",
"Adam",
"Adam",
"Adam",
]





class NNConfiguration:
    """
    Manage the configuration of the Keras NN
    Can find the best NNConfiguration for a dataset
    A NNConfiguration contains the information to build the Keras NN model
    """

    def __init__(
            self,
            machine_or_nnconfiguration: Union[ Machine, dict ],
            machine_nnengine_for_searching_best_nnconfig = None,
            force_find_new_best_configuration: bool = False,
    ):
        """
        Creates or loads a NNConfiguration object

        :param machine_or_nnconfiguration:
            if dict:            load nn configuration from this dict
            if Machine:     create or load nn configuration object using this machine_source object

        :param machine_nnengine_for_searching_best_nnconfig: this is the nnengine we will use to evaluate the nnconfiguration to find the best one
       """

        self.machine = None
        self._machine = None  # Set both machine and _machine for compatibility
        self.nn_shape_instance = None
        self.nn_optimizer = None
        self.num_of_input_neurons = None
        self.num_of_output_neurons = None
        self.nn_find_delay_sec = None

        if isinstance(machine_or_nnconfiguration, dict):
            self._init_set_configuration_from_dict(machine_or_nnconfiguration)

        elif not isinstance(machine_or_nnconfiguration, Machine):
            logger.error("machine_or_nnconfiguration must be dict or Machine ! ")

        else:
            self.machine = machine_or_nnconfiguration
            self._machine = machine_or_nnconfiguration  # Set both machine and _machine for compatibility
            if force_find_new_best_configuration or not self.machine.is_config_ready_nn_configuration( ):
                # Only try to find best configuration if nnengine is provided
                if machine_nnengine_for_searching_best_nnconfig is not None:
                    self._init_find_configuration_best(self.machine , machine_nnengine_for_searching_best_nnconfig )
                else:
                    # If no nnengine is provided, we can't generate a configuration
                    logger.warning("No nnengine provided for NNConfiguration. Configuration will remain uninitialized.")
                    # Set default values to allow object to be created but configuration will be empty
                    self.num_of_input_neurons = self.machine.db_machine.enc_dec_columns_info_input_encode_count if self.machine.db_machine.enc_dec_columns_info_input_encode_count else 0
                    self.num_of_output_neurons = self.machine.db_machine.enc_dec_columns_info_output_encode_count if self.machine.db_machine.enc_dec_columns_info_output_encode_count else 0
                    self.nn_optimizer = "Adam"  # Default optimizer
            else:
                self._init_load_configuration_from_machine( self.machine )



    def _init_set_configuration_from_dict(self, dict_NNConfiguration):
        """
        We set the NNConfiguration from the parameters in the argument

        :param dict_NNConfiguration: the dict values to set in current self
        """
        self.nn_shape_instance = dict_NNConfiguration["nn_shape"]
        self.nn_optimizer = dict_NNConfiguration["nn_optimizer"]
        self.num_of_input_neurons = dict_NNConfiguration["num_of_input_neurons"]
        self.num_of_output_neurons = dict_NNConfiguration["num_of_output_neurons"]


    def _init_load_configuration_from_machine(self, machine: Machine):
        """
        We load the self.configuration from the machine specified in argument

        :param machine: load the configuration of the machine in argument into self
        """

        if not machine.is_config_ready_nn_configuration():
            logger.error( "Unable to load NNConfiguration because there is no configuration saved in machine" )

        self.num_of_input_neurons = machine.db_machine.enc_dec_columns_info_input_encode_count
        self.num_of_output_neurons = machine.db_machine.enc_dec_columns_info_output_encode_count

        self.nn_optimizer = machine.db_machine.parameter_nn_optimizer

        self.nn_shape_instance = NNShape( nnshape_type_machine=machine.db_machine.parameter_nn_shape )


    def _init_find_configuration_best(self, machine: Machine, machine_nnengine ):
        """
        Create in self a new configuration, will try and test several configuration and keep the configuration providing best accuracy/nn_loss

        :param machine: the machine to find the best configuration for
        """

        delay_total_started_at = default_timer()
        if ENABLE_LOGGER_DEBUG_NNConfiguration: logger.debug( f"_init_create_configuration_best starting for : {machine}" )

        if not machine.is_config_ready_enc_dec():
            logger.error("You need to run EncDec before NNConfiguration")

        self.num_of_input_neurons = machine.db_machine.enc_dec_columns_info_input_encode_count
        self.num_of_output_neurons = machine.db_machine.enc_dec_columns_info_output_encode_count

        # use solution-finder to find the best configuration
        best_nn_configuration_found = self._find_machine_best_nn_configuration( machine , machine_nnengine )
        if not best_nn_configuration_found:
            logger.error(f"_find_machine_best_nn_configuration have not find a single good solution ")

        # set the configuration found in self
        self.nn_shape_instance = NNShape( nnshape_type_machine= best_nn_configuration_found )
        self.nn_optimizer = best_nn_configuration_found["parameter_nn_optimizer"]

        machine.db_machine.parameter_nn_find_delay_sec = default_timer() - delay_total_started_at

        logger.info(f"NNConfiguration found for machine : {machine} in {default_timer() - delay_total_started_at} seconds")


    def save_configuration_in_machine(self, save_config_in_machine: Optional[Machine ] = None ):
        """
        We write the configuration in self to the machine table
        """

        if save_config_in_machine:
            db_model_machine = save_config_in_machine.db_machine
        elif self.machine:
            db_model_machine = self.machine.db_machine
        else:
            db_model_machine = None
            logger.error("this NNConfiguration have not been instanced with a machine so need to provide it here")

        db_model_machine.parameter_nn_shape = (
            self.nn_shape_instance.get_machine_nn_shape().to_dict()
        )
        db_model_machine.parameter_nn_optimizer = self.nn_optimizer

        db_model_machine.parameter_nn_find_delay_sec = self.nn_find_delay_sec

        if ENABLE_LOGGER_DEBUG_NNConfiguration: logger.debug( f"NNConfiguration attributes saved in db_model" )

        return self


    def adapt_config_to_new_enc_dec( self , new_enc_dec: "EncDec" ) -> NoReturn:
        """
        update the current self.NNConfiguration to a new EncDec because we have changed the FE
        :params new_enc_dec: the new enc_dec to connect to
        """

        if not self.machine:
            logger.error("This function update NNConfiguration to a new EncDec in the SAME machine - But there is no machine defined" )
        if not self.nn_shape_instance:
            logger.error("This function update NNConfiguration to a new EncDec in the SAME machine - but there is no shape defined" )
        else:
            # the shape is in neurons percentage so there is no update needed
            pass
        self.num_of_input_neurons = new_enc_dec._columns_input_encoded_count
        self.num_of_output_neurons = new_enc_dec._columns_output_encoded_count


    def get_configuration_as_dict____( self ):
        """
        We write the configuration in self to the machine table
        """
        dict_NNConfiguration = {}

        dict_NNConfiguration["num_of_input_neurons"] = self.num_of_input_neurons
        dict_NNConfiguration["num_of_output_neurons"] = self.num_of_output_neurons
        dict_NNConfiguration["nn_shape"] = (
            self.nn_shape_instance.get_machine_nn_shape().to_dict()
        )
        dict_NNConfiguration["nn_optimizer"] = self.nn_optimizer

        return dict_NNConfiguration


    def _evaluate_all_initial_shapes(self, machine: Machine, nnengine ):
        """
        Evaluate on the given machine,nnengine all the NNCONFIGURATION_FINDER_INITIAL_EVALUATIONS

        :params machine: The machine on which we will evaluate all the NNConfiguration defined in NNCONFIGURATION_FINDER_INITIAL_EVALUATIONS
        :params nnengine: The nnengine which we will use to configure (with NNCONFIGURATION_FINDER_INITIAL_EVALUATIONS) and evaluate model

        :return: result_loss_scaled_dict, result_epoch_done_percent_dict : The NN performance result for each NNCONFIGURATION_FINDER_INITIAL_EVALUATIONS models evaluated
        """

        # If nnengine is None, we can't evaluate configurations
        if nnengine is None:
            logger.error("Cannot evaluate initial shapes: nnengine is None")
            return {}, {}

        result_loss_scaled_dict = {}
        result_epoch_done_percent_dict = {}
        random_user_dataframe_for_training_trial_evaluation = machine.get_random_user_dataframe_for_training_trial(is_for_evaluation=True)
        random_user_dataframe_for_training_trial_learning = machine.get_random_user_dataframe_for_training_trial(is_for_learning=True)
        # Evaluate all configurations from NNCONFIGURATION_FINDER_INITIAL_EVALUATIONS_*
        for configuration_number, configuration in enumerate(NNCONFIGURATION_FINDER_INITIAL_EVALUATIONS_SHAPE):
            #if ENABLE_LOGGER_DEBUG_NNConfiguration: logger.debug( f"Evaluation loss of NNCONFIGURATION_FINDER_INITIAL_EVALUATIONS_SHAPE #{configuration_number} Starting... (Encoding dataframe)")
            pandas_shape = pd.DataFrame(
                [list(val.values()) for val in configuration.values()],
                columns=[column_name for column_name in configuration["output"].keys()],
                index=list(configuration.keys())
            )

            # create the configuration from the CONSTANT list of config
            nnengine._nn_configuration = NNConfiguration(
                {
                    "nn_optimizer": NNCONFIGURATION_FINDER_INITIAL_EVALUATIONS_OPTIMIZER[configuration_number],
                    "nn_shape": NNShape(nnshape_type_user=pandas_shape),
                    "num_of_input_neurons": self.machine.db_machine.enc_dec_columns_info_input_encode_count,
                    "num_of_output_neurons": self.machine.db_machine.enc_dec_columns_info_output_encode_count,
                }
            )

            # create the encoded for ai dataframe to do the training
            encoded_for_ai_dataframe = nnengine._enc_dec.encode_for_ai(
                                                            nnengine._mdc.dataframe_pre_encode(
                                                                random_user_dataframe_for_training_trial_learning))
            encoded_for_ai_validation_dataframe = nnengine._enc_dec.encode_for_ai(
                                                            nnengine._mdc.dataframe_pre_encode(
                                                                random_user_dataframe_for_training_trial_evaluation))

            #if ENABLE_LOGGER_DEBUG_NNConfiguration: logger.debug( f"Evaluation loss of NNCONFIGURATION_FINDER_INITIAL_EVALUATIONS_SHAPE #{configuration_number} - encoded_for_ai_dataframe: {encoded_for_ai_dataframe.shape[ 0 ]} rows X {encoded_for_ai_dataframe.shape[ 1 ]} cols , Starting training.... ")

            # do the fast training
            try:
                result_test_loss, result_test_accuracy, epoch_done_percent = nnengine.do_training_trial(
                    encoded_for_ai_dataframe=encoded_for_ai_dataframe,
                    encoded_for_ai_validation_dataframe=encoded_for_ai_validation_dataframe,
                )
            except Exception as e:
                logger.error( f"Unable to Evaluate configurations {configuration_number} from NNCONFIGURATION_FINDER_INITIAL_EVALUATIONS_ : {nnengine._nn_configuration} because {e}" )
            else:
                if result_test_loss is None or result_test_accuracy is None:
                    if ENABLE_LOGGER_DEBUG_NNConfiguration: logger.debug( f"NNCONFIGURATION_FINDER_INITIAL_EVALUATIONS_LOSS #{configuration_number} failed")
                    result_loss_scaled_dict.update({"NNCONFIGURATION_FINDER_INITIAL_EVALUATIONS_LOSS_".lower() + str(configuration_number): None})
                    result_epoch_done_percent_dict.update({"NNCONFIGURATION_FINDER_INITIAL_EVALUATIONS_EPOCH_".lower() + str(configuration_number): None})
                else:
                    if ENABLE_LOGGER_DEBUG_NNConfiguration: logger.debug( f"NNCONFIGURATION_FINDER_INITIAL_EVALUATIONS_LOSS #{configuration_number} -- found : LOSS: {machine.scale_loss_to_user_loss( result_test_loss ):.2} (scaled) --- EPOCH: {epoch_done_percent*100:03} %" )
                    result_loss_scaled_dict.update( { "NNCONFIGURATION_FINDER_INITIAL_EVALUATIONS_LOSS_".lower()+str(configuration_number) : machine.scale_loss_to_user_loss( result_test_loss ) } )
                    result_epoch_done_percent_dict.update( { "NNCONFIGURATION_FINDER_INITIAL_EVALUATIONS_EPOCH_".lower()+str(configuration_number): epoch_done_percent } )

        return result_loss_scaled_dict, result_epoch_done_percent_dict


    def _find_machine_best_nn_configuration(self, machine: Machine, nnengine ):
        """
        Find the best possible NNConfiguration for the machine in argument
        We will build a SolutionScore with all possible values of all parameters of NNConfiguration
        We will use an experimenter, the experimenter is in charge to try a configuration and evaluate the model accuracy/loss

        :param machine: the machine to find the configuration for
        :param nnengine: the nnengine will be used for encoding/decoding , we will vary its nnconfiguration to find the best one
        :param faster_shorter_mode: if true we will evaluate less nnconfiguration, to make it faster

        :return: we return the best solution found (type of NNShape Machine)
        """

        # evaluate the NNCONFIGURATION_FINDER_INITIAL_EVALUATIONS
        initial_shapes_result_loss_scaled_dict, initial_shapes_result_epoch_done_percent_dict = self._evaluate_all_initial_shapes( machine , nnengine )

        dict_possible_values_constant_machine_overview = machine.get_machine_overview_information(
            with_base_info=True,
            with_fec_encdec_info=True,
            )

        dict_possible_values_constant_machine_overview.update( initial_shapes_result_loss_scaled_dict )
        dict_possible_values_constant_machine_overview.update( initial_shapes_result_epoch_done_percent_dict )

        dict_possible_values_varying_nn_config = {
                "parameter_nn_optimizer": NNCONFIGURATION_ALL_POSSIBLE_OPTIMIZER_FUNCTION,
                "LayerTypeActivationOutput": NNCONFIGURATION_ALL_POSSIBLE_LAYER_TYPE_ACTIVATION,
                "HiddenLayerCount": [ i for i in range( 1 , MachineLevel( machine ).nn_shape_count_of_layer_max( )[ 1 ]+1 ) ],
            }

        # NNCONFIGURATION_MAX_POSSIBLE_LAYER_COUNT include hidden output , so the range is correct
        for i in range( 1, NNCONFIGURATION_MAX_POSSIBLE_LAYER_COUNT ):
                dict_possible_values_varying_nn_config.update( {
                    "NeuronPercentage"+str(i) : NNCONFIGURATION_FINDER_POSSIBLE_NEURON_PERCENTAGE,
                    "LayerTypeActivation"+str(i) : NNCONFIGURATION_ALL_POSSIBLE_LAYER_TYPE_ACTIVATION,
                    "DropOut"+str(i) : NNCONFIGURATION_FINDER_POSSIBLE_DROPOUT_VALUE,
                    "BatchNormalization"+str(i) : NNCONFIGURATION_FINDER_POSSIBLE_BATCHNORMALIZATION_VALUE
                } )

        dict_solution_score_score_evaluation = {
            "Result_loss_scaled": "---(70%)",
            "Result_epoch_done_percent" : "+++(20%)",
            "Result_cost_neurons_percent_budget": "---(5%)",
            "Result_cost_layers_percent_budget": "---(5%)",
            }

        # build 9 names based on size of input/outputs
        if machine.db_machine.enc_dec_columns_info_input_encode_count > 1000:
            approximation_input_size = ">1000"
        elif machine.db_machine.enc_dec_columns_info_input_encode_count > 100:
            approximation_input_size = "100-1000"
        else:
            approximation_input_size = "<100"
        if machine.db_machine.enc_dec_columns_info_output_encode_count > 100:
            approximation_output_size = ">100"
        else:
            approximation_output_size = "<100"

        from ML import SolutionScore, ExperimenterNNConfiguration, SolutionFinder
        solution_finder = SolutionFinder(f"NNConfiguration--Level={machine.db_machine.machine_level}--Inputs="+approximation_input_size+"--Outputs="+approximation_output_size )
        solution_found = solution_finder.find_solution(
            dict_possible_values_constant_machine_overview,
            dict_possible_values_varying_nn_config,
            SolutionScore(dict_solution_score_score_evaluation),
            ExperimenterNNConfiguration( nnengine ),
        )


        # ==================================================
        # record -the best nnconfig
        from ML import MachineEasyAutoML
        MachineEasyAutoML_NNConfig = MachineEasyAutoML( "__Results_Find_Best_NN_Configuration__" )
        MachineEasyAutoML_NNConfig.learn_this_inputs_outputs(
            inputsOnly_or_Both_inputsOutputs = dict_possible_values_constant_machine_overview,
            outputs_optional = {
                "Result_NNConfig" : solution_found,
                "Result_delay sec":  solution_finder.result_delay_sec,
                "Result_evaluate_count_better_score" : solution_finder.result_evaluate_count_better_score,
                "Result_best_solution_final_score" : solution_finder.result_best_solution_final_score,
                "Result_evaluate_count_run" : solution_finder.result_evaluate_count_run,
                "Result_shorter_cycles_enabled" : solution_finder.result_shorter_cycles_enabled
            } )

        # end of _find_machine_best_nn_configuration
        return solution_found


    def build_keras_nn_model( self , nn_loss:str, force_weight_initializer=None ):
        """
        generate the keras model from self.nn_shape_instance
        :params nn_loss: the loss function name of the machine

        :return: a keras model Sequential
        """

        if self.num_of_input_neurons == 0 or self.num_of_output_neurons==0:
            logger.error("Unable to generate the shape because num_of_input_neurons:{self.num_of_input_neurons}, num_of_output_neurons:{self.num_of_output_neurons} ")

        # start by creating an empty keras model
        from tensorflow import keras
        from tensorflow.errors import ResourceExhaustedError
        neural_network_model = keras.Sequential()

        nn_shape = self.nn_shape_instance
        user_nn_shape = self.nn_shape_instance.get_user_nn_shape()

        # add first input layer
        neural_network_model.add(
            keras.layers.InputLayer(
                input_shape=(self.num_of_input_neurons,)
            )
        )

        for layer in user_nn_shape.index:
            if nn_shape.layer_get_type(layer) == "dense":
                layer_neurons_count = nn_shape.layer_neuron_count(
                            layer,
                            self.num_of_input_neurons,
                            self.num_of_output_neurons,
                        )
                if layer_neurons_count == 0:
                    logger.error("The count of neuron in a keras layer cannot be 0")
                try:
                    neural_network_model.add(
                        keras.layers.Dense(
                            layer_neurons_count,
                            name=layer,
                            activation=nn_shape.layer_get_activation_function(layer),
                            kernel_initializer=force_weight_initializer,
                        )
                    )
                except ResourceExhaustedError as e:
                    neurons_info = nn_shape.neurons_total_count(self.num_of_input_neurons, self.num_of_output_neurons)
                    weights_info = nn_shape.weight_total_count(self.num_of_input_neurons, self.num_of_output_neurons)
                    gpu_memory_info = tf.config.experimental.get_memory_info('GPU:0')
                    available_memory = gpu_memory_info['current'] / (1024 ** 3)  # Convert to GB
                    total_memory = gpu_memory_info['peak'] / (1024 ** 3)  # Convert to GB
                    raise ValueError(f"the GPU memory is not large enough to load the NN model - GPU_available:{available_memory:.2f} - GPU_total:{total_memory:.2f} - NN_total_weight:{weights_info} - NN_total_neurons:{neurons_info} - GPU:{gpu_memory_info}")
            else:
                raise ValueError("the nn MachineKerasModel can only have dense layers")

            if layer == "output":
                break

            if nn_shape.layer_get_dropout_level(layer):
                neural_network_model.add(
                    keras.layers.Dropout(nn_shape.layer_get_dropout_level(layer))
                )

            if nn_shape.layer_have_batch_normalization(layer):
                neural_network_model.add(keras.layers.BatchNormalization())

        # generate the keras model from the nn_configuration_to_compile
        neural_network_model.compile(
            optimizer=self.nn_optimizer,
            loss=nn_loss,
            metrics=['accuracy'],
        )

        if ENABLE_LOGGER_DEBUG_NNEngine_DETAILED:
            logger.debug( "NNModel have been created for keras from NNconfiguration" )
            with io.StringIO() as stream:
                neural_network_model.summary(print_fn=lambda x: stream.write(x + "\n"))
                logger.debug(stream.getvalue( ) )

        return neural_network_model


class NNShape:
    """
    Works with the shape of the neural network model:
        - store machine_source and user nn shapes
        - convert machine_source nn shape to user nn shape and vice versa
        - count total neurons and weights of the neural network model
        - count total neurons of the selected layer
        - return layer type, activation function, dropout machine_level, batch normalization of the selected layer

        NNShape have 2 dataframe with 4 columns : user(N rows-layers 4 cols pd.DataFrame) and machine(3*4 layers*col flattened = 12 values in a pd.Series)+hiddenlayercount
                the number of the layer can be -1 or 0 or 1 to 10 :
                    layer identifier : 0 / "input"
                    layer identifier : 1-10
                    layer identifier : -1 / "output"

    """

    def __init__(
            self,
            nnshape_type_machine: pd.Series = None,
            nnshape_type_user: pd.DataFrame = None,
    ):
        """
        Creates this NNShape object from a nnshape_machine or from a nnshape_type_user
        Usually we use

        :param nnshape_type_user: dataframe of nnshape_type_user (variable layers size)
        :param nnshape_type_machine: series of machine NNShape (Fixed 3 layers size)
        """
        if not nnshape_type_user is None and not nnshape_type_machine is None:
            logger.error( "Please do not specify nnshape_type_user and nnshape_machine simultaneously" )

        if isinstance(nnshape_type_user, pd.DataFrame ):
            self._user_nn_shape = nnshape_type_user.copy( )
            self._machine_nn_shape = None

        elif isinstance(nnshape_type_machine, pd.Series ):
            self._machine_nn_shape = nnshape_type_machine.copy( )
            self._user_nn_shape = None

        elif isinstance(nnshape_type_machine, dict ):
            self._machine_nn_shape = pd.Series( nnshape_type_machine )
            self._user_nn_shape = None

        else:
            logger.error( "nnshape_type_user must be of type DataFrame or nnshape_machine must be of type Series" )


    def __repr__(self):
        return f"NNShape with {self.get_hidden_layers_count()} hidden layers"


    def get_user_nn_shape(self) -> pd.DataFrame:
        """
        Returns user nn shape, if it's exists, otherwise convert from the machine_source nn shape and return them

        :return: dataframe of user_nn_shape
        """
        if self._user_nn_shape is None:
            self._user_nn_shape = self._convert_machine_nn_shape_to_user_nn_shape( self._machine_nn_shape )

        return self._user_nn_shape


    def get_machine_nn_shape(self) -> pd.Series:
        """
        Returns machine_source nn shape, if it's exists, otherwise convert from the user nn shape and return them

        :return: dataframe of machine_nn_shape
        """
        if self._machine_nn_shape is None:
            self._machine_nn_shape = self._convert_user_nn_shape_to_machine_nn_shape( self._user_nn_shape )

        return self._machine_nn_shape


    def get_hidden_layers_count(self) -> int:
        """
        Return how many hidden layer are present in the NNShape

        :return: how many hidden layer are present in the NNShape
        """
        return int(self.get_machine_nn_shape()["HiddenLayerCount"])


    def get_neurons_percentage(self, layer_identifier: Union[str, int]) -> float:
        """
        Returns the neuron percentage of the specified layer

        :param layer_identifier: the number of the layer (0-10) or string identifier of the layer
        :return: neuron percentage of the specified layer
        """
        if not isinstance( layer_identifier, (str, int) ):
            logger.error( f"Layer identifier must be of type 'str' or 'int', but received '{type( layer_identifier )}'" )

        if layer_identifier in [ "input", "output", 0, -1 ]:
            logger.error( "The input  and output layer cannot have a neurons percentage" )

        if isinstance( layer_identifier, int ):
            layer_identifier = self._convert_int_layer_identifier_to_layer_name( layer_identifier )

        return self.get_user_nn_shape().loc[layer_identifier, "NeuronPercentage"]


    def layer_get_type(self, layer_identifier: Union[str, int]) -> str:
        """
        Returns the type of the specified layer

        :param layer_identifier: the number of the layer (0-10) or string identifier of the layer
        :return: the type of the specified layer (string)
        """
        if not isinstance( layer_identifier, (str, int) ):
            logger.error( f"Layer identifier must be of type 'str' or 'int', but received '{type( layer_identifier )}'" )

        if layer_identifier in [ "input", 0 ]:
            logger.error( "The input layer cannot have an layer type" )

        if isinstance( layer_identifier, int ):
            layer_identifier = self._convert_int_layer_identifier_to_layer_name( layer_identifier )

        return (
            self.get_user_nn_shape()
            .loc[layer_identifier, "LayerTypeActivation"]
            .split("_")[0]
            .lower()
        )


    def layer_get_activation_function(self, layer_identifier: Union[str, int]) -> str:
        """
        Returns activation function of the specified layer

        :param layer_identifier: the number of the layer (0-10) or string identifier of the layer
        :return: activation function name of the specified layer
        """
        if not isinstance(layer_identifier, (str, int)):
            logger.error( f"Layer identifier must be of type 'str' or 'int', but received '{type( layer_identifier )}'" )

        if layer_identifier in [ "input", 0 ]:
            logger.error( "The input layer cannot have an activation function" )

        if isinstance( layer_identifier, int ):
            layer_identifier = self._convert_int_layer_identifier_to_layer_name( layer_identifier )

        return (
            self.get_user_nn_shape()
            .loc[layer_identifier, "LayerTypeActivation"]
            .split("_")[-1]
            .lower()
        )


    def layer_get_dropout_level(self, layer_identifier: Union[str, int]) -> float:
        """
        Returns dropout value of the specified layer. If layer without dropout returns 0

        :param layer_identifier: the number of the layer (0-10) or string identifier of the layer
        :return: dropout value of the specified layer. If layer without dropout returns 0
        """
        if not isinstance(layer_identifier, (str, int)):
            logger.error( f"Layer identifier must be of type 'str' or 'int', but received '{type( layer_identifier )}'" )

        if layer_identifier in ["input", "output", 0, -1]:
            logger.error("The input and output layer cannot have a dropout machine_level")

        if isinstance(layer_identifier, int):
            layer_identifier = self._convert_int_layer_identifier_to_layer_name( layer_identifier )

        return self.get_user_nn_shape().loc[layer_identifier, "DropOut"]


    def layer_have_batch_normalization(self, layer_identifier: Union[str, int]) -> int:
        """
        Returns True if specified layer has batch normalization, otherwise return False

        :param layer_identifier: the number of the layer (0-10) or string identifier of the layer
        :return: True if specified layer has batch normalization
        """
        if not isinstance(layer_identifier, (str, int) ):
            logger.error( f"Layer identifier must be of type 'str' or 'int', but received '{type( layer_identifier )}'" )

        if layer_identifier in [ "input", "output", 0, -1 ]:
            logger.error( "The input and output layer cannot have a batch normalization" )

        if isinstance( layer_identifier, int ):
            layer_identifier = self._convert_int_layer_identifier_to_layer_name( layer_identifier )

        return self.get_user_nn_shape().loc[layer_identifier, "BatchNormalization"]


    def neurons_total_count( self, input_neuron_count: int, output_neuron_count: int ) -> int:
        """
        Returns the total number of all neurons in the neural network model

        :param input_neuron_count: the number of neurons in the input layer
        :param output_neuron_count: the number of neurons in the output layer

        :return: total number of all neurons in the neural network model
        """
        return sum(
            self.layer_neuron_count( layer_identifier, input_neuron_count, output_neuron_count )
            for layer_identifier in range(1, self.get_hidden_layers_count() + 1)
        )


    def weight_total_count(
            self,
            input_neuron_count: int,
            output_neuron_count: int
    ) -> int:
        """
        Returns the total number of connections between all neurons (connections are called also weights)

        :param input_neuron_count: the number of neurons in the input layer
        :param output_neuron_count: the number of neurons in the output layer
        :return: the total number of connections (weights) between all neurons
        """
        layers_neuron_count = [
            self.layer_neuron_count( layer_identifier, input_neuron_count, output_neuron_count )
            for layer_identifier in range(0, self.get_hidden_layers_count() + 1)
        ]
        layers_neuron_count.append(
            self.layer_neuron_count(-1, input_neuron_count, output_neuron_count )
        )

        return sum(
            layer_1_neuron_count * layer_2_neuron_count
            for layer_1_neuron_count, layer_2_neuron_count in zip(
                layers_neuron_count[:-1], layers_neuron_count[1:]
            )
        )


    def layer_neuron_count(
        self,
        layer_identifier: Union[str, int],
        input_neuron_count: int,
        output_neuron_count: int,
    ):
        """Returns the total number of neurons in the selected layer"""

        if input_neuron_count==0 or output_neuron_count==0:
            logger.error("input_neuron_count or output_neuron_count cannot be 0")

        if layer_identifier in ["input", 0]:
            return int(input_neuron_count)

        elif layer_identifier in ["output", -1]:
            return int(output_neuron_count)

        layer_neurons_percentage = self.get_neurons_percentage(layer_identifier)

        if isinstance(layer_identifier, str):
            layer_identifier = self._convert_identifier_layer_name_to_layer_number( layer_identifier )

        if self.get_hidden_layers_count() == 1:
            # special case when there is only one single layer
            part_input = (
            layer_neurons_percentage
            * input_neuron_count
            / 100 / 2
            )
            part_output = (
                layer_neurons_percentage
                * output_neuron_count
                / 100 / 2
            )
        else:
            # from excel file G:\My Drive\EasyAutoML.com\EasyAutoML.com-Documentation\AI\NNConfiguration\DOC 4 - NNShape Update.xlsx
            part_from_input = self.get_hidden_layers_count() - layer_identifier
            part_input = (
                layer_neurons_percentage
                * part_from_input
                / (self.get_hidden_layers_count() - 1)
                * input_neuron_count
                / 100
            )
            part_from_output = layer_identifier - 1
            part_output = (
                layer_neurons_percentage
                * part_from_output
                / (self.get_hidden_layers_count() - 1)
                * output_neuron_count
                / 100
            )

        return max(1, int(part_input + part_output))


    def _convert_machine_nn_shape_to_user_nn_shape( self, machine_nn_shape: pd.Series ) -> pd.DataFrame:
        """
        this function will convert  dataframe machine_nn_shape to dataframe user_nn_shape
        so it will mostly change the 3 layers table NNShape into a table NNShape with exactly HiddenLayerCount rows

        the dataframe machine_nn_shape have always 3 hidden layers plus the information : HiddenLayerCount
        the dataframe user_nn_shape can have a variable hidden layer count from 1 to 10

         We store in machine the NNShape with a format machine_nn_shape with always 3 layers
         Everywhere we use only the machine_nn_shape , Only users can see/edit the user_nn_shape
        """

        # this is the list of parameters used to change the rows count of NNShape (enlarge or reduce)
        MACHINE_NN_SHAPE_TO_USER_NN_SHAPE_CONVERT_INFO = [
            ["[]", []],
            ["mean(({}, {}, {}))", [2]],
            ["{0} * .66 + {1} * .34, {1} * .66 + {2} *.34", [1, 3]],
            ["{}, {}, {}", [1, 2, 3]],
            ["{0}, {0} * .25 + {1} * .5, {1} * .5 + {2} * .25, {2}", [1, 2, 2, 3]],
            ["{0}, mean(({0}, {1})), {1}, mean(({1}, {2})), {2}", [1, 1, 2, 3, 3]],
            ["{0}, {0}, {1}, {1}, {2}, {2}", [1, 1, 2, 2, 3, 3]],
            ["{0}, {0}, {1}, {1}, {1}, {2}, {2}", [1, 1, 2, 2, 2, 3, 3]],
            ["{0}, {0}, {0}, {1}, {1}, {2}, {2}, {2}", [1, 1, 1, 2, 2, 3, 3, 3]],
            [
                "{0}, {0}, {0}, {1}, {1}, {1}, {2}, {2}, {2}",
                [1, 1, 1, 2, 2, 2, 3, 3, 3],
            ],
        ]


        machine_nn_shape = machine_nn_shape.copy()

        if not 0 <= int( machine_nn_shape[ "HiddenLayerCount" ] ) < 10:
            logger.error( "HiddenLayerCount must be in the range [0, 9]" )

        formula, layer_type_index = MACHINE_NN_SHAPE_TO_USER_NN_SHAPE_CONVERT_INFO[ int( machine_nn_shape[ "HiddenLayerCount" ] ) ]

        layer_type_activation = pd.Series( [ machine_nn_shape[ f"LayerTypeActivation{i}" ] for i in layer_type_index ], dtype=object, )
        neuron_percentage, drop_out, batch_normalization = (
            pd.Series(
                self._calculate_formula_in_all_cells_of_dataset(
                    formula,
                    machine_nn_shape,
                    [column_name + str(i) for i in range(1, 4)],
                ),
                dtype=float,
            )
            for column_name in ["NeuronPercentage", "DropOut", "BatchNormalization"]
        )

        user_nn_shape = pd.DataFrame(
            {
                "NeuronPercentage": neuron_percentage,
                "LayerTypeActivation": layer_type_activation,
                "DropOut": drop_out.round(1),
                "BatchNormalization": np.rint(batch_normalization + 0.001).astype(bool),
            },
            columns=USER_NN_SHAPE_COLUMNS_NAME,
        )

        user_nn_shape.index = [f"hidden_{i}" for i in range(1, 1 + len(user_nn_shape))]
        user_nn_shape.loc["output"] = [
            None,
            machine_nn_shape["LayerTypeActivationOutput"],
            None,
            None,
        ]

        return user_nn_shape


    def _convert_user_nn_shape_to_machine_nn_shape( self, user_nn_shape: pd.DataFrame ) -> pd.Series:
        """
        this function will convert dataframe user_nn_shape to dataframe machine_nn_shape
        so it will mostly change the 3 layers table NNShape into a table NNShape with exactly HiddenLayerCount rows

        the dataframe machine_nn_shape have always 3 hidden layers plus the information : HiddenLayerCount
        the dataframe user_nn_shape can have a variable hidden layer count from 1 to 10

         We store in machine the NNShape with a format machine_nn_shape with always 3 layers
         Everywhere we use only the machine_nn_shape , Only users can see/edit the user_nn_shape
        """

        # this is the list of parameters used to change the rows count of NNShape (enlarge or reduce)
        USER_NN_SHAPE_TO_MACHINE_NN_SHAPE_CONVERT_INFO = [
            [],
            ["{0}, {0}, {0}", [1, 1, 1]],
            ["{0}, mean(({0}, {1})), {1}", [1, 2, 2]],
            ["{}, {}, {}", [1, 2, 3]],
            ["{0}, mean(({0}, {1}, {2}, {3})), {3}", [1, 2, 4]],
            ["mean(({}, {})), {}, mean(({}, {}))", [1, 3, 5]],
            ["mean(({}, {})), mean(({}, {})), mean(({}, {}))", [1, 3, 5]],
            ["mean(({}, {})), mean(({}, {}, {})), mean(({}, {}))", [1, 3, 6]],
            ["mean(({}, {}, {})), mean(({}, {})), mean(({}, {}, {}))", [1, 4, 8]],
            ["mean(({}, {}, {})), mean(({}, {}, {})), mean(({}, {}, {}))", [1, 4, 8]],
        ]

        user_nn_shape = user_nn_shape.copy()

        hidden_layer_count = user_nn_shape.shape[0] - 1

        if not 0 < hidden_layer_count < 10:
            logger.error("HiddenLayerCount must be in the range [1, 9]")

        formula, layer_type_index = USER_NN_SHAPE_TO_MACHINE_NN_SHAPE_CONVERT_INFO[
            hidden_layer_count
        ]

        neuron_percentage, drop_out, batch_normalization = (
            np.array(
                self._calculate_formula_in_all_cells_of_dataset(
                    formula,
                    user_nn_shape,
                    [
                        (f"hidden_{i}", column_name)
                        for i in range(1, 1 + hidden_layer_count)
                    ],
                ),
                dtype=object,
            )
            for column_name in ["NeuronPercentage", "DropOut", "BatchNormalization"]
        )

        layer_type_activation = [
            user_nn_shape.loc[f"hidden_{i}", "LayerTypeActivation"]
            for i in layer_type_index
        ]

        machine_nn_shape = pd.Series(
            np.concatenate(
                (
                    np.column_stack(
                        (
                            neuron_percentage,
                            layer_type_activation,
                            drop_out,
                            batch_normalization,
                        )
                    ).ravel(),
                    [user_nn_shape.loc["output", "LayerTypeActivation"]],
                    [hidden_layer_count],
                )
            ),
            index=NNShape.get_list_of_nn_shape_columns_names(),
            dtype=object,
        )
        return machine_nn_shape


    def _convert_int_layer_identifier_to_layer_name(self, layer_identifier: int) -> str:
        """
        convert the number of the layer to the name of the layer
        the number of the layer can be -1 or 0 or 1 to 10

        :param layer_identifier: layer number
        :return: the name of the layer : layer name will be 'input', 'output' " "or consist of 'hidden_' and number of hidden layer
        """
        if not isinstance(layer_identifier, int):
            logger.error(
                f"The layer identifier must have 'int' type, but received {type(layer_identifier)}"
            )

        if not -1 <= layer_identifier <= self.get_hidden_layers_count():
            logger.error( f"Layer identifier must be in range [-1, {self.get_hidden_layers_count( )}], "
                                         f"but received {layer_identifier}" )

        if layer_identifier == 0:
            return "input"

        if layer_identifier == -1:
            return "output"

        return f"hidden_{layer_identifier}"


    def _convert_identifier_layer_name_to_layer_number(self, layer_name: str) -> int:
        """
        convert the name of the layer to the layer number
        The layer name can be 'input', 'output' " "or consist of 'hidden_' and number of hidden layer

        :param layer_name: layer name
        :return: the number of the layer : -1 or 0 or 1 to 10
        """
        if not isinstance(layer_name, str):
            logger.error(
                f"The layer name must have 'str' type, but received {type(layer_name)}"
            )

        if layer_name == "input":
            return 0

        elif layer_name == "output":
            return -1

        elif layer_name.startswith("hidden_"):
            return int(layer_name.split("_")[-1])

        else:
            logger.error(
                "The layer name can be 'input', 'output' "
                "or consist of 'hidden_' and number of hidden layer"
            )


    def _calculate_formula_in_all_cells_of_dataset(
            self,
            formula: str,
            data: Union[pd.Series, pd.DataFrame],
            index: list
    ):
        try:
            if isinstance(data, pd.Series):
                return eval(formula.format(*(data[index_name] for index_name in index)))
            elif isinstance(data, pd.DataFrame):
                return eval(formula.format(*(data.loc[i, j] for i, j in index)))
            else:
                logger.error( "data must be pd.Series or pd.DataFrame ")
        except Exception as e:
            logger.error("unable to compute formulas")


    @staticmethod
    def get_list_of_nn_shape_columns_names(  ) -> list:
        # this is the structure of a NNShape - it define the neurons model structure, it have 18 keys (or columns names)
        return [
                index_name
                for index in range(1, 4)
                        for index_name in [
                            f"NeuronPercentage{index}",
                            f"LayerTypeActivation{index}",
                            f"DropOut{index}",
                            f"BatchNormalization{index}",
                        ]
               ] + \
               ["LayerTypeActivationOutput",
               "HiddenLayerCount"]

