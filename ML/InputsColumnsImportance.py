from timeit import default_timer

import numpy as np
import pandas as pd
from typing import NoReturn

from ML import EasyAutoMLDBModels, __getlogger
from ML import Machine, NNEngine

from SharedConstants import *


logger = __getlogger()



class InputsColumnsImportance:
    """
    This class is using for calculating percentage of inputs column importance in dataset
    each inputs columns have a percentage, higher mean the input column is important to get a good prediction
    the sum of all input importance is always 1

    :param machine: machine, for which dataset is needed to calculate importance percentage of each column
    """

    def __init__(self,
                machine: Machine,
                nnengine_for_best_config: NNEngine = None,
                create_configuration_best: bool = False,
                create_configuration_simple_minimum: bool = False,
                load_configuration: bool = False):
        """
        will do one of the 3 operations indicated in the params

        :param create_configuration_best: If true init will calculate the ICI (it need NNEngine ready to do predictions)
        :param create_configuration_simple_minimum: If true init will calculate default configuration with all importance same for all columns
        :param load_configuration: If true init will read the configuration from the machine
        """

        # we save in self the machine on which evaluating the Columns Importance
        self._machine = machine
        self._column_importance_evaluation = None
        self._nnengine = nnengine_for_best_config
        self._fe_columns_inputs_importance_find_delay_sec = None

        if create_configuration_best and not nnengine_for_best_config:
            logger.error( "we need to provide NNEngine to compute the best configuration")
        if not create_configuration_best and nnengine_for_best_config:
            logger.error( "we do not need to provide NNEngine because we do not compute the best configuration")
        if create_configuration_best and not nnengine_for_best_config and not nnengine_for_best_config.is_nn_trained_and_ready( ):
            logger.error("unable to compute best ICI because the nnengine is not ready")

        self._columns_data_type = self._machine.db_machine.mdc_columns_data_type

        self._input_columns_names = [
                column_name
                for column_name, column_is_input in self._machine.db_machine.mdc_columns_name_input.items()
                if column_is_input
            ]
        self._output_columns_names = [
                column_name
                for column_name, column_is_output in self._machine.db_machine.mdc_columns_name_output.items()
                if column_is_output
            ]

        if create_configuration_simple_minimum:
            # we create minimum configuration and store in self
            self._generate_configuration_default_minimum()

        elif create_configuration_best:
            # create configuration using evaluation of LOSS between minimal/maximal and default loss
            # The machine must be trained because this evaluation process need to perform predictions
            if nnengine_for_best_config is None or not nnengine_for_best_config.is_nn_trained_and_ready():
                logger.error( "to do best ICI we need nnengine trained and ready to do predictions")
            # we create configuration , we will compute the InputsColumnsImportance and store in self
            self._generate_configuration_best()
        else:
            # we load previous configuration stored in machine in self
            # we load _column_importance_evaluation from the machine
            #  it is a dict , the keys are the inputs columns names and the values are the importance of the columns
            if nnengine_for_best_config:
                logger.error(f"this parameter is not necessary when loading the configuration : nnengine_for_best_config" )
            self._column_importance_evaluation=self._machine.db_machine.fe_columns_inputs_importance_evaluation
            self._fe_columns_inputs_importance_find_delay_sec = self._machine.db_machine.fe_columns_inputs_importance_find_delay_sec
            if ENABLE_LOGGER_DEBUG_InputsColumnsImportance: logger.debug( f"Loaded column_importance_evaluation from machine_source {self._machine}" )


    def _calculate_prediction_loss_between_df_inputs_solved_and_df_output_ref(
        self,
        dataframe_input_to_compare_user: pd.DataFrame = None,
        dataframe_input_to_compare_pre_encoded: pd.DataFrame = None,
        dataframe_input_to_compare_encoded_for_ai: pd.DataFrame = None,
        output_dataframe_reference_to_compare_with_encoded_for_ai: pd.DataFrame = None,
    ) -> float:
        """
        Predicting the outputs from dataframe_input_to_compare and then calculating loss between the result and output_dataframe_to_compare_with_encoded_for_ai
        Note : we can provide only one single dataframe_to_compare

        :param dataframe_input_to_compare_user: the dataframe to compare with  (will be solved)
        :param dataframe_input_to_compare_pre_encoded: the dataframe to compare with (will be solved)
        :param dataframe_input_to_compare_encoded_for_ai: the dataframe to compare with (will be solved)
        :param output_dataframe_reference_to_compare_with_encoded_for_ai: second dataframe to compare with the prediction of the first one

        :return: loss value = mean of all columns-loss
        """

        if not dataframe_input_to_compare_user is None:
            if dataframe_input_to_compare_pre_encoded:
                logger.error("only one dataframe can be specified in the argument")
            dataframe_input_to_compare_pre_encoded = self._nnengine._mdc.dataframe_pre_encode( dataframe_input_to_compare_user )

        if not dataframe_input_to_compare_pre_encoded is None:
            if dataframe_input_to_compare_encoded_for_ai:
                logger.error("only one dataframe can be specified in the argument")
            dataframe_input_to_compare_encoded_for_ai = self._nnengine._enc_dec.encode_for_ai( dataframe_input_to_compare_pre_encoded )

        # predict the outputs from the argument user_dataframe_inputs_to_evaluate
        dataframe_output_predicted_to_compare_encoded_for_ai = self._nnengine.do_solving_direct_encoded_for_ai(
            dataframe_input_to_compare_encoded_for_ai,
        )

        from tensorflow import keras
        mean_outputs_loss = np.nanmean( keras.losses.get( self._machine.db_machine.parameter_nn_loss )(
                    output_dataframe_reference_to_compare_with_encoded_for_ai,
                    dataframe_output_predicted_to_compare_encoded_for_ai,
            ).numpy() , axis=0 )

        if np.isnan(mean_outputs_loss):
            logger.error("Unable to calculate loss between the 2 dataframes")

        return mean_outputs_loss


    def _generate_configuration_best(self) -> NoReturn:
        """
        Calculating importance percentage of each user input column
        the sum of all importance will always be 1
        the result will later be stored in machine.fe_columns_inputs_importance_evaluation by set_machine_properties

        :return: NoReturn
        """

        if ENABLE_LOGGER_DEBUG_InputsColumnsImportance: logger.debug( f"Calculating best ICI for {self._machine}" )
        delay_total_started_at = default_timer()

        # we load the validation pre_encoded_dataframe to evaluate global loss
        user_pre_encoded_evaluation_dataframe = self._nnengine._mdc.dataframe_pre_encode(
            self._machine.get_random_user_dataframe_for_training_trial( is_for_learning=True ) )
        if user_pre_encoded_evaluation_dataframe.empty:
            logger.error(" user_evaluation_dataframe is empty ! ")

        user_pre_encoded_evaluation_dataframe_inputs = user_pre_encoded_evaluation_dataframe[self._input_columns_names]
        user_pre_encoded_evaluation_dataframe_outputs = user_pre_encoded_evaluation_dataframe[self._output_columns_names]

        encoded_for_ai_evaluation_dataframe_outputs = self._nnengine._enc_dec.encode_for_ai(user_pre_encoded_evaluation_dataframe_outputs)

        # we evaluate the global LOSS for the validation pre_encoded_dataframe
        average_row_prediction_loss_on_df_evaluation = self._calculate_prediction_loss_between_df_inputs_solved_and_df_output_ref(
            dataframe_input_to_compare_pre_encoded = user_pre_encoded_evaluation_dataframe_inputs,
            output_dataframe_reference_to_compare_with_encoded_for_ai= encoded_for_ai_evaluation_dataframe_outputs,
        )
        if average_row_prediction_loss_on_df_evaluation == 0:
            logger.warning(
                    "While evaluating global loss we was unable to find a loss (predictions was 100% perfect on dataframe) - "
                    "this mean we cannot evaluate Column Importance - "
                    "reverting to _generate_configuration_default_minimum()"
                    )
            self._generate_configuration_default_minimum()


        # iteration of each input column
        #   we calculate Loss for pre_encoded_dataframe with minimal values
        #   we calculate Loss for pre_encoded_dataframe with maximal values
        #   the importance of the column is LOSS_Global - ( (LOSS_Min+LOSS_Max)/2 )
        #   result for each column is stored in loss_delta_by_column
        #   NOTE: the result cannot have negative value lower than 0

        sum_negative_loss_delta = 0
        loss_delta_by_column = dict()
        for one_input_column_name in self._input_columns_names:

            # to get the min of this column, we can use the stored value if the datayype is numeric otherwise there is no min values stored, so we calculate it  (DATETIME, LANGUAGE, ETC..)
            if self._columns_data_type[one_input_column_name] == DatasetColumnDataType.FLOAT:
                min_value = self._machine.db_machine.mdc_columns_values_min[ one_input_column_name ]
            else:
                min_value = min( user_pre_encoded_evaluation_dataframe_inputs[ one_input_column_name ].dropna(), default=None )
            # build pre_encoded_dataframe with only current columns filled all with with minimal values
            temp_minimal_dataframe_user_pre_encoded = user_pre_encoded_evaluation_dataframe_inputs.copy( )
            temp_minimal_dataframe_user_pre_encoded[one_input_column_name] = [
                min_value for _ in range(len(temp_minimal_dataframe_user_pre_encoded[one_input_column_name]))
            ]

            # to get the max of the column , we can use the stored value if the datayype is numeric otherwise there is no max values stored, so we calculate it (DATETIME, LANGUAGE, ETC..)
            if self._columns_data_type[one_input_column_name] == DatasetColumnDataType.FLOAT:
                max_value = self._machine.db_machine.mdc_columns_values_max[ one_input_column_name ]
            else:
                max_value = max( user_pre_encoded_evaluation_dataframe_inputs[ one_input_column_name ].dropna(), default=None )
            # build pre_encoded_dataframe with only current columns filled all with with maximal values
            temp_maximal_dataframe_user_pre_encoded = user_pre_encoded_evaluation_dataframe_inputs.copy( )
            temp_maximal_dataframe_user_pre_encoded[ one_input_column_name ] = [
                max_value for _ in range( len( temp_maximal_dataframe_user_pre_encoded[ one_input_column_name ] ) )
            ]

            # compute loss for pre_encoded_dataframe minimal value
            column_loss_for_minimal_values = self._calculate_prediction_loss_between_df_inputs_solved_and_df_output_ref(
                dataframe_input_to_compare_pre_encoded = temp_minimal_dataframe_user_pre_encoded,
                output_dataframe_reference_to_compare_with_encoded_for_ai= encoded_for_ai_evaluation_dataframe_outputs,
            )
            # compute loss for pre_encoded_dataframe maximal value
            column_loss_for_maximal_values = self._calculate_prediction_loss_between_df_inputs_solved_and_df_output_ref(
                dataframe_input_to_compare_pre_encoded = temp_maximal_dataframe_user_pre_encoded,
                output_dataframe_reference_to_compare_with_encoded_for_ai= encoded_for_ai_evaluation_dataframe_outputs
            )

            # compute the average difference between Loss_global , Loss_Min and Loss_Max
            loss_delta_this_column = (
                        (column_loss_for_maximal_values - average_row_prediction_loss_on_df_evaluation) + (column_loss_for_minimal_values-average_row_prediction_loss_on_df_evaluation)
                    ) / 2
            if loss_delta_this_column < 0:
                sum_negative_loss_delta += loss_delta_this_column
                if loss_delta_this_column < -0.1:
                     logger.warning( f"while disabling the column {one_input_column_name} the loss is better - the model is not performing well on this column" )
                loss_delta_this_column = 0

            loss_delta_by_column[one_input_column_name] = loss_delta_this_column

        # normalisation :
        # importance will be 0 when loss_delta_by_column is highest
        # the sum of all importance will be 1
        # = 1 - ( sum(X)-X ) /sum(X) )
        sum_loss_delta_by_column = sum(loss_delta_by_column.values())
        if sum_loss_delta_by_column == 0:
            logger.warning(        "Unable to compute _generate_configuration_best - "
                                            "it seem the NNmodel is not performing correctly at all - "
                                            "reverting to _generate_configuration_default_minimum()"
                                            )
            self._generate_configuration_default_minimum()
            return

        ici_by_column = {}
        for one_input_column_name in self._input_columns_names:
            ici_by_column[one_input_column_name] = loss_delta_by_column[one_input_column_name] /  sum_loss_delta_by_column
        self._column_importance_evaluation = ici_by_column

        self._configuration_reliability_percentage = ( 1 - (sum_negative_loss_delta / average_row_prediction_loss_on_df_evaluation) ) * 100

        self._fe_columns_inputs_importance_find_delay_sec = delay_total_started_at - default_timer()

        logger.info(f"Best ICI done with reliability: {self._configuration_reliability_percentage}% in {self._fe_columns_inputs_importance_find_delay_sec} secs, ICI: {self._column_importance_evaluation}" )

        # ==================================================
        # record in MachineEasyAutoML the delay and reliability
        from ML import MachineEasyAutoML
        MachineEasyAutoML_ici = MachineEasyAutoML( "__Results_Delay_Find_Best_ICI__" )
        dict_machine_overview = self._machine.get_machine_overview_information(
            with_base_info=True,
            with_fec_encdec_info=True,
            with_nn_model_info=True,
            with_training_infos=True,
            with_training_cycle_result=False,
            with_training_eval_result=False
        )
        MachineEasyAutoML_ici.learn_this_inputs_outputs(
            inputsOnly_or_Both_inputsOutputs = dict_machine_overview,
            outputs_optional= {
            "ICI reliability percentage": self._configuration_reliability_percentage,
            "ICI find delay sec": self._fe_columns_inputs_importance_find_delay_sec,
            "ICI found": self._column_importance_evaluation,
        })


    def _generate_configuration_default_minimum(self) -> NoReturn:
        """
        Sets default columns importance 1/(count of input columns)

        :return: NoReturn
        """
        average_importance = 1 / len(self._input_columns_names)
        self._column_importance_evaluation = {
            column_name: average_importance for column_name in self._input_columns_names
        }
        self._fe_columns_inputs_importance_find_delay_sec = 0

        if ENABLE_LOGGER_DEBUG_InputsColumnsImportance: logger.debug( f"_generate_configuration_default_minimum for {self._machine} done" )


    def save_configuration_in_machine(self) -> "InputsColumnsImportance":
        """
        Sets new columns importance percentage
        from self._column_importance_evaluation
        to self._machine.db_machine.fe_columns_inputs_importance_evaluation

        :return: InputsColumnsImportance computed
        """
        self._machine.db_machine.fe_columns_inputs_importance_evaluation = (self._column_importance_evaluation)
        self._machine.db_machine.fe_columns_inputs_importance_find_delay_sec = self._fe_columns_inputs_importance_find_delay_sec

        if ENABLE_LOGGER_DEBUG_InputsColumnsImportance: logger.debug( f"Updated column importance for machine_source {self._machine}" )

        return self


    def _is_configuration_default_minimum(self) -> bool:
        """
        Checks is each column has importance like 1/(count of input columns)

        :return: True if all column importance percentage is equal 1/(count of input columns)
        else False
        :rtype: bool
        """
        minimum_importance = 1 / len(self._input_columns_names)
        return all(
            column_importance == minimum_importance
            for column_importance in self._column_importance_evaluation.values()
        )


    def _is_configuration_empty(self) -> bool:
        """

        :return: True if sum of all column importance percentage is not exists or equal 0
        else False
        :rtype: bool
        """
        return (
            None
            if not self._column_importance_evaluation
            else sum(self._column_importance_evaluation.values())
        )


    def _is_configuration_generated_best(self) -> bool:
        """
        Checks is config calculated for such machine

        :return: True if (sum of all column importance equal 1 with delta=0.001
        and not _is_config_default_minimum()
        and not _is_config_default_minimum())
        else False
        :rtype: bool
        """
        return (
                not self._is_configuration_default_minimum()
                and not self._is_configuration_empty()
                and 0.999<sum(self._column_importance_evaluation.values)<1.001
        )


