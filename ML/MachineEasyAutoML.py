import pandas as pd
import numpy as np
import math
import datetime
import random
from typing import Optional,NoReturn,Union
import os
from ML import EasyAutoMLDBModels, __getlogger
from SharedConstants import *


logger = __getlogger()


# when buffering experience-datalines, the flush will be when the buffer reach this line count (of course , if the instance MachineEasyAutoML is deleted from memory the flush will be done immediately)
# Must be larger than 10 to allow the automatic detection of datatypes
MachineEasyAutoML_EXPERIENCES_BUFFER_FLUSH_WHEN_COUNT= 250


class MachineEasyAutoML:
    """
    MachineEasyAutoML is an easy way to use Machines, to create them and train them on the fly
    when prediction are not available yet , an alternative Experimenter may replace the prediction
    until there is enough dataLine to train the machine_source
    """


    def __repr__(self):
        return "MachineEasyAutoML:" + self._machine_name


    def __init__( self,
                        machine_name: str,
                        optional_experimenter: Optional = None,
                        record_experiments: bool = True,
                        access_user_id: int = None,
                        access_team_id: int = None,
                        decimal_separator : str = ".",
                        date_format : str = "MDY",
    ):
        """
        Creates a new MachineEasyAutoML object with given name

        :param machine_name: name of MachineEasyAutoML, if there is no machine_source created yet on deletion of this class machine_source will
            be created with this name, if machine_source exists machine_source with this name will be loaded
        :param optional_experimenter: when machine_source is not created or not trained yet, MachineEasyAutoML will use
            experimenter to predict instead. You can provide an 'Experimenter' instance to use experimenter. If this argument is None,
            MachineEasyAutoML will always return None in do_predict

        :param record_experiments: indicates whether to record predictions of experimenter
            if True predictions of experimenter will all be recorded and saved to the machine_source

        :param access_user_id: ID of machine_source owner

        """
        from ML import Machine

        if machine_name.startswith( "__") or machine_name.endswith( "__"):
            # only EasyAutoML Admin Team can create machine with __ so we know the user/team
            self._current_access_user_id =  EasyAutoMLDBModels( ).User.get_super_admin( ).id
            self._current_access_team_id = EasyAutoMLDBModels( ).Team.get_super_admin_team( ).id
        elif access_user_id is None:
            logger.warning("DEPRECATED : missing machine_owner_id when loading  MachineEasyAutoML ! ")
        else:
            self._current_access_user_id = access_user_id
            self._current_access_team_id = access_team_id

        self._machine_name=machine_name
        self._input_column_names=[]
        self._output_column_names=[]
        self._record_experiments=record_experiments
        self._machine=None
        self._dataset_user_experiences=pd.DataFrame()
        self._last_do_predict_inputs_df=pd.DataFrame( )
        self._experimenter=None
        self._nn_engine=None
        self._decimal_separator = decimal_separator
        self._date_format = date_format
        self._count_run_predict_by_experimenter = 0
        self._count_run_predict_by_solving = 0
        self._percentage_of_force_experimenter = 100

        from ML.Experimenter import Experimenter
        if isinstance(optional_experimenter,Experimenter):
            self._experimenter=optional_experimenter

        self._machine=None
        if Machine.is_this_machine_exist_and_authorized( machine_identifier_or_name=machine_name, machine_check_access_user_id=self._current_access_user_id ):
            self._machine = Machine(machine_identifier_or_name=machine_name, machine_access_check_with_user_id=self._current_access_user_id )
            # because the machine is created already , so we know already what are the columns experience_inputs and outputs
            self._input_column_names=self._machine.get_list_of_columns_name(ColumnDirectionType.INPUT)
            self._output_column_names=self._machine.get_list_of_columns_name(ColumnDirectionType.OUTPUT)
            self._percentage_of_force_experimenter = 100
            if self.ready_to_predict() and self._machine.db_machine.training_eval_loss_sample_evaluation:
                if self._machine.scale_loss_to_user_loss( float(self._machine.db_machine.training_eval_loss_sample_evaluation) ) < 0.1:
                    self._percentage_of_force_experimenter = 1
                elif self._machine.scale_loss_to_user_loss( float(self._machine.db_machine.training_eval_loss_sample_evaluation) ) < 0.2:
                    self._percentage_of_force_experimenter = 10
                elif self._machine.scale_loss_to_user_loss( float(self._machine.db_machine.training_eval_loss_sample_evaluation) ) < 0.25:
                    self._percentage_of_force_experimenter = 25
        else:
            # Machine does not exist yet
            # If we have an experimenter, we can create the machine later when predict() is called
            # If record_experiments is True, the machine will be created on first prediction
            if not self._record_experiments and not self._experimenter:
                # No way to create the machine - raise exception
                raise Exception(f"Machine '{machine_name}' does not exist or is not authorized for user {self._current_access_user_id}")
            # Otherwise, allow self._machine to remain None - it will be created later by predict()


    def ready_to_predict( self )->bool:
        """
        :return: True if this MachineEasyAutoML is ready for predicting with NNEngine
        """
        return self._machine and self._machine.is_nn_solving_ready( )


    def _get_nn_engine(self ):
        """
        Caching NNEngine to avoid reloading each time
        """
        if self._machine is None:
            logger.error("Machine is none !")

        from ML.NNEngine import NNEngine
        if self._nn_engine is None and self._machine is not None:
            self._nn_engine=NNEngine(self._machine , allow_re_run_configuration=False )

        return self._nn_engine


    def do_predict(self, data_inputs_to_predict: Union[dict,list,pd.DataFrame ] ) -> Optional[pd.DataFrame ]:
        """
        Gives the prediction based on experience_inputs
        If there is trained machine_source, gives NN prediction
        Else if there is experimenter, evaluates the result by the experimenter
        Else returns None

        :param data_inputs_to_predict: input data for prediction
            dict - one row
            list<dict> - multiple rows
            pd.Dataframe - one/multiple rows

        :return: the predictions (outputs)

        """
        if isinstance(data_inputs_to_predict,dict ):
            data_inputs_to_predict=pd.DataFrame([data_inputs_to_predict ] )
        elif isinstance(data_inputs_to_predict,list ):
            data_inputs_to_predict=pd.DataFrame(data_inputs_to_predict )
        self._input_column_names+=data_inputs_to_predict.columns.tolist( )
        self._input_column_names=list(set(self._input_column_names))

        decide_force_use_experimenter = False

        # POSSIBILITY 0 : Force experimenter if DEBUG_MachineEasyAutoML_FORCE_EXPERIMENTER
        # sometime even there is ready mode to predict it is necessary to keep experimenting because the machine have not learn enough to be very accurate
        if DEBUG_MachineEasyAutoML_FORCE_EXPERIMENTER and self._machine and self._experimenter:
            decide_force_use_experimenter = True

        # POSSIBILITY 1 : Force experimenter at some % rate depending on the machine loss on predictions
        # even if prediction are possible we do sometime experimenter at some % rate or always if DEBUG_MachineEasyAutoML_FORCE_EXPERIMENTER
        if not self._machine or not self._experimenter:
            pass
        elif DEBUG_MachineEasyAutoML_FORCE_EXPERIMENTER:
            decide_force_use_experimenter = True
        elif random.uniform(0, 100)  <= self._percentage_of_force_experimenter:
            decide_force_use_experimenter = True

        # POSSIBILITY 2 : Use Prediction if solving available
        if not decide_force_use_experimenter and self.ready_to_predict():
            # we check if this current machine is maybe doing configuration - it can be a machine EasyAutoML doing its own configuration recursively
            # we will never do recursively use the prediction, it can make recursive calls
            from ML.NNEngine import NNEngine
            if not NNEngine.machine_nn_engine_configuration_is_configurating( self._machine ):
                try:
                    outputs_predicted=self._get_nn_engine( ).do_solving_direct_dataframe_user(data_inputs_to_predict )
                    if ENABLE_LOGGER_DEBUG_MachineEasyAutoML: logger.debug( f"MachineEasyAutoML have evaluated with NNModel. {self} : { 'ok' if outputs_predicted is not None else 'None' }" )
                    self._count_run_predict_by_solving += 1
                    return outputs_predicted
                except Exception as e:
                    logger.warning(f"MachineEasyAutoML is unable to do a prediction with {self._machine} because error : {e} ")
                    pass

        # POSSIBILITY 3 : We use experimenter ( if available )
        if self._experimenter:
            outputs_predicted=self._run_experimenter_on_this( data_inputs_to_predict )
            self._count_run_predict_by_experimenter += 1
            if ENABLE_LOGGER_DEBUG_MachineEasyAutoML: logger.debug( f"MachineEasyAutoML have evaluated with experimenter. {self} : { 'ok' if outputs_predicted is not None else 'None' }" )
            return outputs_predicted

        if ENABLE_LOGGER_DEBUG_MachineEasyAutoML: logger.debug( f"MachineEasyAutoML was not able to to do any prediction/evaluation/experiment. {self}" )
        return None


    def learn_this_inputs_outputs(self,inputsOnly_or_Both_inputsOutputs: Union[dict,pd.DataFrame ],outputs_optional: Optional[Union[dict,pd.DataFrame ] ] = None, ) -> NoReturn:
        """
        Save the experience_inputs and outputs to dataset_user_experiences

        :param inputsOnly_or_Both_inputsOutputs: dict/dataframe of input column values
                            we store the dataframe in dataset_user_experiences

                            Important : If outputs_optional not None, then in inputsOnly_or_Both_inputsOutputs should be only input columns,
                            Important : If outputs_optional is None then the inputsOnly_or_Both_inputsOutputs must contains full dataframe

        :param outputs_optional: dict/dataframe of output column values, if None then inputsOnly_or_Both_inputsOutputs should contain all columns
        """

        if isinstance(inputsOnly_or_Both_inputsOutputs,dict):
            inputsOnly_or_Both_inputsOutputs=pd.DataFrame([inputsOnly_or_Both_inputsOutputs])
        if isinstance(outputs_optional,dict):
            outputs_optional=pd.DataFrame([outputs_optional])

        # we learn what are the experience_inputs and what are the outputs only if they are both given in argument
        if (inputsOnly_or_Both_inputsOutputs is not None and outputs_optional is not None):
            self._input_column_names+=(inputsOnly_or_Both_inputsOutputs.columns.tolist())
            self._input_column_names=list(set(self._input_column_names))

            self._output_column_names+=outputs_optional.columns.tolist()
            self._output_column_names=list(set(self._output_column_names))

        # we store the experience_inputs and outputs given in arguments
        user_inputs_outputs=pd.concat([inputsOnly_or_Both_inputsOutputs,outputs_optional],axis=1)
        self._dataset_user_experiences=pd.concat([self._dataset_user_experiences,user_inputs_outputs])
        self._dataset_user_experiences.reset_index(drop=True,inplace=True)

        # we flush buffer if there is enough lines in the buffer (but not if we are in rescaling mode)
        if len(self._dataset_user_experiences) >= MachineEasyAutoML_EXPERIENCES_BUFFER_FLUSH_WHEN_COUNT:
            self._flush_user_experiences_lines_buffer_to_Machine_DataLines( )


    def learn_this_part_inputs( self, inputs_only: Union[ dict, pd.DataFrame ] ) -> NoReturn:
        """
        we store the inputs , and we will wait a call to learn_this_part_outputs to store the full row (inputs+outputs)
        it is possible to do several calls in this order :
                nn_easyautoml.learn_this_part_inputs(pd.DataFrame([{ "temp": 6, "sky": "cloudy", "humidity": 0.6 } ] ) )
                nn_easyautoml.learn_this_part_inputs(pd.DataFrame([{ "temp": 7, "sky": "cloudy", "humidity": 0.7 } ] ) )
                nn_easyautoml.learn_this_part_outputs({ "raining": 0.77 } )
                nn_easyautoml.learn_this_part_outputs({ "raining": 0.66 } )

        :param inputs_only: the inputs to learn , we will temporary store them , and also learn what are the inputs columns names
        """
        if isinstance( inputs_only, dict ):
            for key in inputs_only.keys( ):
                self._input_column_names.append( key )
            inputs_only = pd.DataFrame( [ inputs_only ] )
        elif isinstance( inputs_only, pd.DataFrame ):
            self._input_column_names+=(inputs_only.columns.tolist())
            self._input_column_names=list(set(self._input_column_names))
        else:
            logger.error( f"inputs_only type incorrect : {type(inputs_only)}")

        self._last_do_predict_inputs_df=pd.concat([self._last_do_predict_inputs_df,inputs_only ] )
        self._last_do_predict_inputs_df.reset_index(drop=True,inplace=True )


    def learn_this_part_outputs(self,outputs_result: Union[dict,pd.DataFrame ] ) -> NoReturn:
        """
        We are receiving the outputs
        it must match the learn_this_part_inputs stored earlier
        it is possible to do several calls in this order :
                nn_easyautoml.learn_this_part_inputs(pd.DataFrame([{ "temp": 6, "sky": "cloudy", "humidity": 0.6 } ] ) )
                nn_easyautoml.learn_this_part_inputs(pd.DataFrame([{ "temp": 7, "sky": "cloudy", "humidity": 0.7 } ] ) )
                nn_easyautoml.learn_this_part_outputs({ "raining": 0.77 } )
                nn_easyautoml.learn_this_part_outputs({ "raining": 0.66 } )

        :param outputs_result: the dict or the dataframe containing the outputs which are suposed to match the previous experience_inputs stored
        """

        if self._last_do_predict_inputs_df.empty:
            raise ValueError("You are trying to learn_this_result, but there was no do_predict before")
        if isinstance(outputs_result,dict):
            outputs_result=pd.DataFrame( [outputs_result] )

        # learn that this columns are all outputs
        self._output_column_names += outputs_result.columns.tolist()
        self._output_column_names=list(set(self._output_column_names))

        if len(self._last_do_predict_inputs_df )==0 and len(outputs_result )>0:
            raise ValueError( f"There is no inputs defined before, use do_predict() to set the inputs" )
        elif len(self._last_do_predict_inputs_df )>=len(outputs_result )==1:
            pass
        else:
            raise ValueError( f"There is {len(self._last_do_predict_inputs_df )} inputs defined before and {len(outputs_result )} output, we cannot match inputs with outputs" )

        # adding the outputs to the experience_inputs stored earlier in _last_do_predict_inputs
        new_rows_to_add = pd.DataFrame()
        for i in range( len(outputs_result) , 0 , -1 ):
            c1 = self._last_do_predict_inputs_df.iloc[[ len(outputs_result ) - i - 1 ] ]
            c2 = outputs_result.iloc[[ i-1  ]]
            c1.reset_index(drop=True, inplace=True)
            c2.reset_index(drop=True, inplace=True)
            full_row_to_add = pd.concat( [  c1 , c2  ]   , axis='columns' , sort=False )
            full_row_to_add.reset_index(drop=True, inplace=True)
            self._dataset_user_experiences.reset_index(drop=True, inplace=True)
            self._dataset_user_experiences=pd.concat( [self._dataset_user_experiences, full_row_to_add ] )

        #self._dataset_user_experiences.reset_index(inplace=True,drop=True)

        # remove  _last_do_predict_inputs used
        self._last_do_predict_inputs_df=self._last_do_predict_inputs_df.iloc[ : -len(outputs_result ) ]
        #self._last_do_predict_inputs.reset_index( inplace=True, drop=True )

        # we flush buffer if there is enough lines in the buffer
        if len(self._dataset_user_experiences) >= MachineEasyAutoML_EXPERIENCES_BUFFER_FLUSH_WHEN_COUNT:
            self._flush_user_experiences_lines_buffer_to_Machine_DataLines( )


    def get_experience_data_saved( self,
            only_inputs: bool = False,
            only_outputs: bool = False
            ) -> pd.DataFrame:

        if not self._machine:
            return None

        df = self._machine.data_lines_read(
                only_column_direction_type= (ColumnDirectionType.INPUT if only_inputs else ColumnDirectionType.OUTPUT if only_outputs else None )
        )
        return df


    # todo please add function into the tutorial and the test functions
    def get_experiences_not_yet_saved(self ) -> pd.DataFrame:
        """
        this method is used by MachineEasyAutoMLAPI to export the buffer
        """
        return self._dataset_user_experiences


    def _set_experiences_buffer(self, new_experience:pd.DataFrame ):
        """
        this method is used by MachineEasyAutoMLAPI to export the buffer
        """
        if new_experience == None:
            self._dataset_user_experiences= None
        elif isinstance(new_experience , pd.DataFrame):
            self._dataset_user_experiences=new_experience
        else:
            logger.error(f"user_experiences should be of type pd.Dataframe or None but is of type {type(new_experience)}!")


    def get_list_input_columns_names(self ):
        """
        this method is used by MachineEasyAutoMLAPI to export the experience_inputs columns names
        """
        return self._input_column_names


    def set_list_input_columns_names(self,new_input_columns:list ):
        """
        this method is used by MachineEasyAutoMLAPI to set the experience_inputs columns names

        :param new_input_columns: the list of columns name for the experience_inputs
        """
        if isinstance(new_input_columns,list):
            self._input_column_names=new_input_columns
        else:
            logger.error(f"input_columns should be of type list but is of type {type(new_input_columns)}!")


    def get_list_output_columns_names(self ):
        """
        this method is used by MachineEasyAutoMLAPI to export the experience_inputs columns names
        """
        return self._output_column_names


    def set_list_output_columns_names(self, new_output_columns:list ):
        """
        this method is used by MachineEasyAutoMLAPI to set the outputs columns names

        :param new_output_columns: the list of columns name for the experience_inputs
        """
        if isinstance(new_output_columns,list):
            self._output_column_names=new_output_columns
        else:
            logger.error(f"output_columns should be of type list but is of type {type(new_output_columns)}!")


    def _run_experimenter_on_this(self, user_input_df: pd.DataFrame ) -> Optional[pd.DataFrame ]:
        """
        Gives experimenter prediction for all rows from user_input_df
        Store the predictions into self._dataset_user_experiences

        :param user_input_df: experience_inputs dataframe
        :return: outputs dataframe got by experimenter
        """
        #if ENABLE_LOGGER_DEBUG: logger.debug("MachineEasyAutoML predicting with experimenter")

        experimenter_experiment=self._experimenter.do(user_input_df)

        if experimenter_experiment.iloc[0] is None:
            return None

        if not self._record_experiments:
            return experimenter_experiment

        self._dataset_user_experiences=pd.concat([self._dataset_user_experiences,pd.concat([user_input_df,experimenter_experiment],axis=1),])

        self._dataset_user_experiences.reset_index(drop=True,inplace=True)

        self._output_column_names=list(
            set(self._output_column_names+experimenter_experiment.columns.tolist()
            if isinstance(experimenter_experiment,pd.DataFrame) else experimenter_experiment.index.tolist()))

        # we flush buffer if there is enough lines in the buffer
        if len(self._dataset_user_experiences) >= MachineEasyAutoML_EXPERIENCES_BUFFER_FLUSH_WHEN_COUNT:
            self._flush_user_experiences_lines_buffer_to_Machine_DataLines( )

        return experimenter_experiment


    def _flush_user_experiences_lines_buffer_to_Machine_DataLines(self ):
        """
        We store all experiences (experience_inputs+outputs) into a buffer, then when the buffer is full or when the MachineEasyAutoML instance is deleted, we flush the buffer in the machine DataLine
        If no machine exist yet, the machine will then be created now
        """

        if self._dataset_user_experiences is None or self._dataset_user_experiences.empty:
            return

        # it is possible that some columns are missing, but data will be none, it is ok
        #not_found_columns_input_output = set(self._input_column_names+self._output_column_names)-set(self._dataset_user_experiences.columns.tolist())
        #if not_found_columns_input_output:
        #    logger.error(f"The data stored are missing some column(s) : {not_found_columns_input_output}")

        # we have more columns that we know
        not_found_columns_dataset_user_experience=set(self._dataset_user_experiences.columns.tolist())-set(self._input_column_names+self._output_column_names)
        if not_found_columns_dataset_user_experience:
            logger.error(f"Some columns are unknown : {not_found_columns_dataset_user_experience}")

        # it is possible that some type of data sent in MachineEasyAutoML are object , so we will convert all objects to string
        for i in range( self._dataset_user_experiences.shape[0]): #iterate over rows
            for j in range( self._dataset_user_experiences.shape[1]): #iterate over columns
                value = self._dataset_user_experiences.iloc[i, j]
                if not pd.isnull( value ) and not isinstance( value, (int, float, np.generic, bool, str, datetime.datetime , datetime.date , datetime.time , list, dict ) ):
                    logger.warning( f"temporary warning check : we converted MachineEasyAutoML dataset value of type {type(value)} to string '{value}'  " )
                    self._dataset_user_experiences.iloc[i, j] = str( value )

        # control to check if the machine is created , but this is useful only if another process have created the machine after we did this self.init
        from ML.Machine import Machine
        if Machine.is_this_machine_exist_and_authorized( machine_identifier_or_name=self._machine_name, machine_check_access_user_id=self._current_access_user_id ):
            # if the machine exist already we just write the experience in the DB
            self._machine=Machine( self._machine_name, machine_access_check_with_user_id=self._current_access_user_id )
            from ML import MachineDataConfiguration
            machine_mdc = MachineDataConfiguration(self._machine)
            df_ok, problem_message = machine_mdc.verify_compatibility_additional_dataframe(
                        self._dataset_user_experiences,
                        self._machine,
                        decimal_separator = self._decimal_separator,
                        date_format = self._date_format,
                        )
            if not df_ok:
                logger.error( f"MachineEasyAutoML si unable to store the new data on the machine {self._machine} because : " + problem_message )

            self._machine.user_dataframe_format_then_save_in_db(
                        self._dataset_user_experiences,
                        columns_type=machine_mdc.columns_type_user_df ,
                        decimal_separator=self._decimal_separator ,
                        date_format=self._date_format ,
                        split_lines_in_learning_and_evaluation=True
                        )
        else:
            # if the machine is not existing
            # then we need first to check if columns experience_inputs and outputs have been identified
            if not self._input_column_names:
                logger.error("MachineEasyAutoML have not been able to identify ANY columns inputs. Please check you are using "
                                 "one or more of this method : learn_this(inputs and outputs), "
                                 "or learn_this_result() or do_predict")
                return

            if not self._output_column_names:
                logger.error("MachineEasyAutoML have not been able to identify ANY outputs in the dataframe. "
                                 "maybe you should use learn_this(inputs and outputs) (with 2 arguments instead of one) , "
                                 "also you may use learn_this_result() or do_predict ")
                return

            columns_description={input_column_name:"MachineEasyAutoML created this input column" for input_column_name in self._input_column_names}
            columns_description.update({output_column_name:"MachineEasyAutoML created this output column" for output_column_name in self._output_column_names})

            # we can create the machine MachineEasyAutoML
            if ENABLE_LOGGER_DEBUG_MachineEasyAutoML: logger.debug( f"Creating MachineEasyAutoML {self._machine_name} with {len(self._dataset_user_experiences )} lines" )
            # we need to replace the description filled with inputs and outputs by force_create_with_this_inputs and force_create_with_this_outputs
            self._machine=Machine(
                self._machine_name,
                user_dataset_unformatted = self._dataset_user_experiences,
                force_create_with_this_inputs = { col_name:True for col_name in self._input_column_names },
                force_create_with_this_outputs = { col_name:True for col_name in self._output_column_names },
                force_create_with_this_descriptions=columns_description,
                machine_create_user_id=self._current_access_user_id,
                machine_create_team_id=self._current_access_team_id,
                machine_description=f"MachineEasyAutoML created this machine from {len(self._dataset_user_experiences)} rows",
                decimal_separator = self._decimal_separator,
                date_format = self._date_format,
                )

            self._machine.save_machine_to_db()
            
            # Cache the MachineDataConfiguration for future flushes
            from ML import MachineDataConfiguration
            self._mdc = MachineDataConfiguration(self._machine)

        # buffer is flushed so we empty the buffer
        self._dataset_user_experiences=pd.DataFrame(columns=self._dataset_user_experiences.columns)


    # todo please add function into the tutorial and the test functions
    def save_data(self ):
        if self._dataset_user_experiences is None or self._dataset_user_experiences.empty:
            # not enough rows to create the machine
            return
        else:
            self._flush_user_experiences_lines_buffer_to_Machine_DataLines( )


    def delete( self ):
        self._dataset_user_experiences = None
        self._last_do_predict_inputs_df = None
        if self._machine:
            self._machine.delete( )


    def __del__(self):
        """
        when the instance is deleted from memory we will before flush the buffer
        """
        # Check if initialization was successful before attempting cleanup
        if not hasattr(self, '_machine_name'):
            # Initialization failed, nothing to clean up
            return
            
        try:
            self._flush_user_experiences_lines_buffer_to_Machine_DataLines( )
        except Exception as e:
            logger.error(f"Unable to flush buffer experience data lines in machine: {self._machine_name} {self._machine} because: {e} - self._input_column_names: {self._input_column_names} - self._output_column_names: {self._output_column_names} - self._dataset_user_experiences: {self._dataset_user_experiences}")

