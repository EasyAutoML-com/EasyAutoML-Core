import requests
import pandas as pd
from typing import Optional, NoReturn, Union

# Try to import from settings, use default if not available
try:
    from django.conf import settings
    MachineEasyAutoML_URL_API_SERVER = getattr(settings, 'MachineEasyAutoML_URL_API_SERVER', "https://easyautoml.com/api/")
except ImportError:
    # Default API server URL if settings is not available
    MachineEasyAutoML_URL_API_SERVER = "https://easyautoml.com/api/"

from ML import EasyAutoMLDBModels, __getlogger

logger = __getlogger()

class MachineEasyAutoMLAPI:
    """
    MachineEasyAutoML is an easy way to use Learning Machines, to create and use them
    MachineEasyAutoMLAPI will create one machine in EasyAutoML when the MachineEasyAutoMLAPi object is deleted or when the buffer is flushed with the method : MachineEasyAutoMLAPI.Flush_Experience_buffer()

    Because at the beginning there is not yet data to learn, or not enough data to create the machine, predictions may not be available until training is complete
    """

    # This is the default machine machine_level if it is not specified when instancing the MachineEasyAutoMLAPI
    # Note after the machine is created, the machine machine_level cannot not be changed
    CREATION_MACHINE_DEFAULT_LEVEL = 1

    def __init__(
        self,
        MachineEasyAutoMLAPI_Name: str,
        user_api_key: str,
        is_rescaling_numeric_output: bool = False,
        creation_machine_level=None,
    ):
        """
        Creates a new MachineEasyAutoMLAPI

        :param user_api_key: this is the user API secret key - it is available on easyautoml.com in the API page

        :param MachineEasyAutoMLAPI_Name: name of the MachineEasyAutoMLAPI, if there is no MachineEasyAutoML machine created in EasyAutoML then the machine will be created with this name when we call the method flush_experience_buffer or when the object MachineEasyAutoMLAPI is deleted from memory
                    be created with this name, if machine_source exists machine_source with this name will be loaded

        :param is_rescaling_numeric_output: If true will rescale numeric outputs when the buffer is flushed. The buffer is flushed when we call the method flush_experience_buffer or when the object MachineEasyAutoMLAPI is deleted from memory

        :param creation_machine_level: indicate what machine_level the remote machine should have when created, it will not update an existing machine
        """

        if MachineEasyAutoMLAPI_Name.startswith( "__" ) or MachineEasyAutoMLAPI_Name.endswith( "__" ):
            raise Exception( "double underscore at begining or end of the the name are reserved")
        self._machine_name = MachineEasyAutoMLAPI_Name
        self._user_api_key = user_api_key
        self._dataset_user_experiences = pd.DataFrame()
        self._input_column_names = []
        self._output_column_names = []
        self._is_rescaling_numeric_output = is_rescaling_numeric_output
        self._last_do_predict_inputs = pd.DataFrame()
        self._creation_machine_default_level = (
            creation_machine_level
            if creation_machine_level
            else self.CREATION_MACHINE_DEFAULT_LEVEL
        )
        # todo : we could load this values from API :
        #   self._input_column_names
        #   self._output_column_names
        if self._is_rescaling_numeric_output:
            # when we do rescaling, we will rescale only after deleting the object from memory or if calling flush_experience_buffer()
            self._experiences_buffer_flush_buffer_after_line_count = None
        else:
            # we will send remotely the data if the experience buffer reach this quantity of line or if we call flush_experience_buffer()
            self._experiences_buffer_flush_buffer_after_line_count = 1000

    def do_predict(
        self, experience_inputs: Union[dict, list, pd.DataFrame]
    ) -> Optional[pd.DataFrame]:
        """
        Gives the prediction based on inputs

        If there is trained machine already then it will gives prediction
        Otherwise it will return None

        :param experience_inputs: input data for prediction, it can be one dict of a single row, or a list of dict (multiple rows) or it can be a pandas pre_encoded_dataframe with one or more rows
        """
        if isinstance(experience_inputs, dict):
            experience_inputs = pd.DataFrame([experience_inputs])
        elif isinstance(experience_inputs, list):
            experience_inputs = pd.DataFrame(experience_inputs)

        self._input_column_names += experience_inputs.columns.tolist()
        self._input_column_names = list(set(self._input_column_names))
        self._last_do_predict_inputs = pd.concat(
            [self._last_do_predict_inputs, experience_inputs]
        )
        self._last_do_predict_inputs.reset_index(drop=True, inplace=True)

        result = requests.post(
            f"{MachineEasyAutoML_URL_API_SERVER}/do-predict",
            headers={"Authorization": self._user_api_key},
            json={
                "machine_name": self._machine_name,
                "data": experience_inputs.to_dict("records"),
            },
        )
        if result.status_code != 200:
            # 560 = apikey invalid
            # 562 = machine_source name no valid ( __ is not allowed anywhere \ / " ' $ * % & # < > -- ) MachineEasyAutoMLAPIService
            # 563 unable to create machine_source
            print(result.json())
            raise Exception(result.json()["error"])
        else:
            prediction = result.json()["prediction"]
            return pd.DataFrame(prediction)

    def learn_this(
        self,
        experience_inputs_or_inputsoutputs: Union[dict, pd.DataFrame],
        experience_outputs: Optional[Union[dict, pd.DataFrame]] = None,
    ) -> NoReturn:
        """
        Save the experience provided in a memory buffer
        It will train the machine when the buffer will be flushed

        :param experience_inputs_or_inputsoutputs: can be a dict or a pre_encoded_dataframe and contain the experience_inputs or both experience_inputs and outputs
        :param experience_outputs: it is optional and can be a dict or a pre_encoded_dataframe and contain the outputs

        There is 2 way to send the experience_inputs and outputs
                send a dict/pre_encoded_dataframe with both in the argument experience_inputs_or_inputsoutputs and None in experience_outputs
                send a dict/pre_encoded_dataframe for experience_inputs in experience_inputs_or_inputsoutputs and send a dict/pre_encoded_dataframe for outputs in experience_outputs (recomended)
        """
        if isinstance(experience_inputs_or_inputsoutputs, dict):
            experience_inputs_or_inputsoutputs = pd.DataFrame(
                [experience_inputs_or_inputsoutputs]
            )
        if isinstance(experience_outputs, dict):
            experience_outputs = pd.DataFrame([experience_outputs])
        if (
            experience_inputs_or_inputsoutputs is not None
            and experience_outputs is not None
        ):
            self._input_column_names += (
                experience_inputs_or_inputsoutputs.columns.tolist()
            )
            self._input_column_names = list(set(self._input_column_names))

            self._output_column_names += experience_outputs.columns.tolist()
            self._output_column_names = list(set(self._output_column_names))

        user_inputs_outputs = pd.concat(
            [experience_inputs_or_inputsoutputs, experience_outputs], axis=1
        )

        self._dataset_user_experiences = pd.concat(
            [self._dataset_user_experiences, user_inputs_outputs]
        )

        self._dataset_user_experiences.reset_index(drop=True, inplace=True)

        if self._experiences_buffer_flush_buffer_after_line_count:
            if (
                len(self._dataset_user_experiences.index)
                >= self._experiences_buffer_flush_buffer_after_line_count
            ):
                self._flush_experiences_buffer_remotely_to_machine()

    def learn_this_result(
        self, experience_outputs: Union[dict, pd.DataFrame]
    ) -> NoReturn:
        """
        Save the experience_outputs in the experiences buffer in memory and merge with experience_inputs already in memory buffer
        The experiences will be used to train the machine

        To use this method it is necessary before to call method do_predict which give the inputs
        It will merge the experience_inputs from do_predict which was called before (one or several times) and with experience_outputs provided in this method learn_this_result

        :param experience_outputs: dict or pre_encoded_dataframe of outputs to save
        """
        if self._last_do_predict_inputs.empty:
            logger.error(
                "You are trying to learn_this_result, but there was no do_predict before"
            )
        if isinstance(experience_outputs, dict):
            experience_outputs = pd.DataFrame([experience_outputs])

        self._output_column_names += experience_outputs.columns.tolist()
        self._output_column_names = list(set(self._output_column_names))

        if self._last_do_predict_inputs.shape[0] < experience_outputs.shape[0]:
            logger.error(
                "You are trying to learn_this_result more times than did do_predict"
            )

        self._last_do_predict_inputs[self._output_column_names] = experience_outputs[
            self._output_column_names
        ]

        self._dataset_user_experiences = pd.concat(
            [
                self._dataset_user_experiences,
                self._last_do_predict_inputs.iloc[: experience_outputs.shape[0]],
            ]
        )

        self._dataset_user_experiences.reset_index(inplace=True, drop=True)

        self._last_do_predict_inputs = self._last_do_predict_inputs.iloc[
            experience_outputs.shape[0] :
        ]
        self._last_do_predict_inputs.reset_index(inplace=True, drop=True)

        if self._experiences_buffer_flush_buffer_after_line_count:
            if (
                len(self._dataset_user_experiences.index)
                >= self._experiences_buffer_flush_buffer_after_line_count
            ):
                self._flush_experiences_buffer_remotely_to_machine()

    def _flush_experiences_buffer_remotely_to_machine(self):
        if self._dataset_user_experiences.empty:
            return

        result = requests.post(
            f"{MachineEasyAutoML_URL_API_SERVER}/save-lines",
            headers={"Authorization": self._user_api_key},
            json={
                "machine_name": self._machine_name,
                "user_experiences": self._dataset_user_experiences.to_dict(),
                "is_rescaling_numeric_output": self._is_rescaling_numeric_output,
                "input_columns": self._input_column_names,
                "output_columns": self._output_column_names,
                "machine_level": self._creation_machine_default_level,  # todo add machine_level in argument of django save-lines so API user can set machine level
            },
        )

        if result.status_code != 200:
            raise Exception(result.json()["error"])
            print(result.json())

        self._dataset_user_experiences = pd.DataFrame(
            columns=self._dataset_user_experiences.columns
        )


    def flush_experience_buffer(self):
        """
        This method will send all experience lines to the remote machine in easyautoml.com
        If the machine was not created yet, it will create it
        If enough lines are sent the machine will be trained after a few minutes
        """

        self._flush_experiences_buffer_remotely_to_machine()

    def __del__(self):
        """
        The MachineEasyAutoMLAPI instance is delete from memory, we will send remotely all data collected in the experience buffer
        """
        self._flush_experiences_buffer_remotely_to_machine()
