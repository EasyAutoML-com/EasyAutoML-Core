import os
from typing import Union, NoReturn

import pandas as pd
from django.db import models, connection, connections
from django.conf import settings
from django.utils.translation import gettext_lazy as _

# Import logger
def get_logger():
    try:
        from eamllogger.EasyAutoMLLogger import EasyAutoMLLogger
        return EasyAutoMLLogger()
        # Return logger instance directly from eamllogger
    except ImportError:
        # Fallback to models logger
        try:
            from models.logger import EasyAutoMLLogger
            return EasyAutoMLLogger()
        except ImportError:
            import logging
            return logging.getLogger(__name__)

logger = get_logger()

# Import constants
try:
    from SharedConstants import (
        POSSIBLE_MACHINES_LEVELS, 
        NNCONFIGURATION_LOSS_FUNCTION,
        NNCONFIGURATION_ALL_POSSIBLE_OPTIMIZER_FUNCTION,
        DatasetColumnDataType,
        ENABLE_LOGGER_DEBUG_Machine
    )
except ImportError:
    # Fallback values
    POSSIBLE_MACHINES_LEVELS = [1, 2, 3, 4, 5]
    NNCONFIGURATION_LOSS_FUNCTION = ['mse', 'mae', 'binary_crossentropy']
    NNCONFIGURATION_ALL_POSSIBLE_OPTIMIZER_FUNCTION = ['adam', 'sgd', 'rmsprop']
    
    # Simple DatasetColumnDataType fallback
    class DatasetColumnDataType:
        LABEL = "LABEL"
        FLOAT = "FLOAT"
        DATE = "DATE"
        TIME = "TIME"
        DATETIME = "DATETIME"
        JSON = "JSON"
        IGNORE = "IGNORE"
    
    ENABLE_LOGGER_DEBUG_Machine = False

# JSON Encoder
try:
    from models.JsonEncoderExtended import JsonEncoderExtended
except ImportError:
    from django.core.serializers.json import DjangoJSONEncoder
    JsonEncoderExtended = DjangoJSONEncoder

TYPE_MAPPING = {
    "LABEL": (models.TextField, {"null": True}),
    "LANGUAGE": (models.TextField, {"null": True}),
    "FLOAT": (models.DecimalField, {
        "null": True,
        "max_digits": getattr(settings, 'MODEL_DECIMAL_FIELD_DEFAULT_MAX_DIGITS', 20),
        "decimal_places": getattr(settings, 'MODEL_DECIMAL_FIELD_DEFAULT_DECIMAL_PLACES', 8),
    }),
    "DATE": (models.DateField, {"null": True}),
    "TIME": (models.TimeField, {"null": True}),
    "DATETIME": (models.DateTimeField, {"null": True}),
    "JSON": (models.TextField, {"null": True}),
    "IGNORE": (models.TextField, {"null": True}),
}


class Machine(models.Model):
    def __init__(self, *args, file_path=None, load_by_column_type=True, **kwargs):
        if isinstance(file_path, (str, bytes, os.PathLike, int)):
            self.file_path = file_path
        elif file_path is None:
            self.file_path = None
        else:
            logger.error(f"The path {file_path} is not valid")

        super(Machine, self).__init__(*args, **kwargs)
        if load_by_column_type:
            self.dfr_columns_type_user_df = {
                column_name: column_type if isinstance(column_type, DatasetColumnDataType) else DatasetColumnDataType[column_type]
                for column_name, column_type in self.dfr_columns_type_user_df.items()
            }
            self.mdc_columns_data_type = {
                column_name: column_type if isinstance(column_type, DatasetColumnDataType) else DatasetColumnDataType[column_type]
                for column_name, column_type in self.mdc_columns_data_type.items()
            }

    timestamp = models.DateTimeField(auto_now_add=True)
    machine_original = models.ForeignKey(
        "self",
        related_name="OriginalChild",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
    )

    machine_owner_user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, verbose_name=_("user"))

    machine_owner_team = models.ForeignKey("models.Team", on_delete=models.SET_NULL, null=True, blank=True, verbose_name=_("Team"))

    machine_name = models.CharField(_("name"), max_length=500, blank=False)
    machine_description = models.TextField(_("Description"), blank=True)

    machine_level = models.IntegerField(default=1, choices=zip(POSSIBLE_MACHINES_LEVELS, POSSIBLE_MACHINES_LEVELS))

    machine_data_source_is_public = models.BooleanField(default=False)
    machine_data_source_sample_only_public = models.BooleanField(default=False)
    machine_data_source_price_usd = models.DecimalField(
        null=True,
        blank=True,
        max_digits=settings.MODEL_DECIMAL_FIELD_DEFAULT_MAX_DIGITS,
        decimal_places=settings.MODEL_DECIMAL_FIELD_DEFAULT_DECIMAL_PLACES,
    )

    machine_api_solving_is_public = models.BooleanField(default=False)
    machine_api_solving_price_usd = models.DecimalField(
        null=True,
        blank=True,
        max_digits=settings.MODEL_DECIMAL_FIELD_DEFAULT_MAX_DIGITS,
        decimal_places=settings.MODEL_DECIMAL_FIELD_DEFAULT_DECIMAL_PLACES,
    )
    machine_machine_is_duplicatable = models.BooleanField(default=False)
    machine_machine_copy_cost_usd = models.DecimalField(
        null=True,
        blank=True,
        max_digits=settings.MODEL_DECIMAL_FIELD_DEFAULT_MAX_DIGITS,
        decimal_places=settings.MODEL_DECIMAL_FIELD_DEFAULT_DECIMAL_PLACES,
    )
    machine_machine_copy_update_cost_usd = models.DecimalField(
        null=True,
        blank=True,
        max_digits=settings.MODEL_DECIMAL_FIELD_DEFAULT_MAX_DIGITS,
        decimal_places=settings.MODEL_DECIMAL_FIELD_DEFAULT_DECIMAL_PLACES,
    )
    machine_billing_cost_solving_usd = models.DecimalField(
        null=True,
        blank=True,
        max_digits=settings.MODEL_DECIMAL_FIELD_DEFAULT_MAX_DIGITS,
        decimal_places=settings.MODEL_DECIMAL_FIELD_DEFAULT_DECIMAL_PLACES,
    )
    machine_billing_cost_training_usd = models.DecimalField(
        null=True,
        blank=True,
        max_digits=settings.MODEL_DECIMAL_FIELD_DEFAULT_MAX_DIGITS,
        decimal_places=settings.MODEL_DECIMAL_FIELD_DEFAULT_DECIMAL_PLACES,
    )

    machine_columns_errors = models.JSONField(default=dict, blank=True, encoder=JsonEncoderExtended)
    machine_columns_warnings = models.JSONField(default=dict, blank=True, encoder=JsonEncoderExtended)

    dfr_columns_description_user_df = models.JSONField(default=dict, blank=True, encoder=JsonEncoderExtended)
    dfr_columns_type_user_df = models.JSONField(default=dict, blank=True, encoder=JsonEncoderExtended)
    dfr_columns_python_user_df = models.JSONField(default=dict, blank=True, encoder=JsonEncoderExtended)

    mdc_columns_name_input_user_df = models.JSONField(null=True, blank=True, default=dict, encoder=JsonEncoderExtended)
    mdc_columns_name_output_user_df = models.JSONField(null=True, blank=True, default=dict, encoder=JsonEncoderExtended)

    mdc_columns_name_input = models.JSONField(default=dict, blank=True, encoder=JsonEncoderExtended)
    mdc_columns_name_output = models.JSONField(default=dict, blank=True, encoder=JsonEncoderExtended)
    mdc_columns_total_count = models.IntegerField(null=True, blank=True)
    mdc_columns_input_count = models.IntegerField(null=True, blank=True)
    mdc_columns_output_count = models.IntegerField(null=True, blank=True)
    mdc_columns_data_type = models.JSONField(default=dict, blank=True, encoder=JsonEncoderExtended)

    mdc_columns_count_of_datatypes = models.JSONField(default=dict, blank=True, encoder=JsonEncoderExtended)
    mdc_columns_input_count_of_datatypes = models.JSONField(default=dict, blank=True, encoder=JsonEncoderExtended)
    mdc_columns_output_count_of_datatypes = models.JSONField(default=dict, blank=True, encoder=JsonEncoderExtended)

    mdc_columns_json_structure = models.JSONField(default=dict, blank=True, encoder=JsonEncoderExtended)

    mdc_columns_unique_values_count = models.JSONField(default=dict, blank=True, encoder=JsonEncoderExtended)
    mdc_columns_missing_percentage = models.JSONField(default=dict, blank=True, encoder=JsonEncoderExtended)
    mdc_columns_most_frequent_values_count = models.JSONField(default=dict, blank=True, encoder=JsonEncoderExtended)

    mdc_columns_values_mean = models.JSONField(default=dict, blank=True, encoder=JsonEncoderExtended)
    mdc_columns_values_std_dev = models.JSONField(default=dict, blank=True, encoder=JsonEncoderExtended)
    mdc_columns_values_skewness = models.JSONField(default=dict, blank=True, encoder=JsonEncoderExtended)
    mdc_columns_values_kurtosis = models.JSONField(default=dict, blank=True, encoder=JsonEncoderExtended)
    mdc_columns_values_min = models.JSONField(default=dict, blank=True, encoder=JsonEncoderExtended)
    mdc_columns_values_max = models.JSONField(default=dict, blank=True, encoder=JsonEncoderExtended)

    mdc_columns_values_quantile02 = models.JSONField(blank=True, default=dict, encoder=JsonEncoderExtended)
    mdc_columns_values_quantile03 = models.JSONField(blank=True, default=dict, encoder=JsonEncoderExtended)
    mdc_columns_values_quantile07 = models.JSONField(blank=True, default=dict, encoder=JsonEncoderExtended)
    mdc_columns_values_quantile08 = models.JSONField(blank=True, default=dict, encoder=JsonEncoderExtended)
    mdc_columns_values_sem = models.JSONField(blank=True, default=dict, encoder=JsonEncoderExtended)
    mdc_columns_values_median = models.JSONField(blank=True, default=dict, encoder=JsonEncoderExtended)
    mdc_columns_values_mode = models.JSONField(blank=True, default=dict, encoder=JsonEncoderExtended)

    mdc_columns_values_str_percent_uppercase = models.JSONField(default=dict, blank=True, encoder=JsonEncoderExtended)
    mdc_columns_values_str_percent_lowercase = models.JSONField(default=dict, blank=True, encoder=JsonEncoderExtended)
    mdc_columns_values_str_percent_digit = models.JSONField(default=dict, blank=True, encoder=JsonEncoderExtended)
    mdc_columns_values_str_percent_punctuation = models.JSONField(default=dict, blank=True, encoder=JsonEncoderExtended)
    mdc_columns_values_str_percent_operators = models.JSONField(default=dict, blank=True, encoder=JsonEncoderExtended)
    mdc_columns_values_str_percent_underscore = models.JSONField(default=dict, blank=True, encoder=JsonEncoderExtended)
    mdc_columns_values_str_percent_space = models.JSONField(default=dict, blank=True, encoder=JsonEncoderExtended)

    mdc_columns_values_str_language_en = models.JSONField(default=dict, blank=True, encoder=JsonEncoderExtended)
    mdc_columns_values_str_language_fr = models.JSONField(default=dict, blank=True, encoder=JsonEncoderExtended)
    mdc_columns_values_str_language_de = models.JSONField(default=dict, blank=True, encoder=JsonEncoderExtended)
    mdc_columns_values_str_language_it = models.JSONField(default=dict, blank=True, encoder=JsonEncoderExtended)
    mdc_columns_values_str_language_es = models.JSONField(default=dict, blank=True, encoder=JsonEncoderExtended)
    mdc_columns_values_str_language_pt = models.JSONField(default=dict, blank=True, encoder=JsonEncoderExtended)
    mdc_columns_values_str_language_others = models.JSONField(default=dict, blank=True, encoder=JsonEncoderExtended)
    mdc_columns_values_str_language_none = models.JSONField(default=dict, blank=True, encoder=JsonEncoderExtended)

    machine_is_data_lines_training_count_limit_enabled = models.BooleanField(default=False)
    machine_data_lines_training_count_limit = models.IntegerField(null=True, blank=True)

    machine_is_re_run_mdc = models.BooleanField(default=False)
    machine_is_re_run_ici = models.BooleanField(default=False)
    machine_is_re_run_fe = models.BooleanField(default=False)
    machine_is_re_run_enc_dec = models.BooleanField(default=False)
    machine_is_re_run_nn_config = models.BooleanField(default=False)
    machine_is_re_run_model = models.BooleanField(default=False)

    fe_budget_total = models.DecimalField(
        null=True,
        blank=True,
        max_digits=settings.MODEL_DECIMAL_FIELD_DEFAULT_MAX_DIGITS,
        decimal_places=settings.MODEL_DECIMAL_FIELD_DEFAULT_DECIMAL_PLACES,
    )
    fe_columns_fet = models.JSONField(null=True, blank=True, default=dict, encoder=JsonEncoderExtended)
    fe_count_of_fet = models.JSONField(null=True, blank=True, default=dict, encoder=JsonEncoderExtended)
    fe_find_delay_sec = models.IntegerField(null=True, blank=True, default=None)

    fe_columns_inputs_importance_evaluation = models.JSONField(null=True, blank=True, default=dict, encoder=JsonEncoderExtended)
    fe_columns_inputs_importance_find_delay_sec = models.IntegerField(null=True, blank=True, default=None)

    enc_dec_configuration = models.OneToOneField(
        "models.EncDecConfiguration",
        related_name="machine_encdec_configuration",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
    )
    enc_dec_columns_info_input_encode_count = models.IntegerField(blank=True, null=True)
    enc_dec_columns_info_output_encode_count = models.IntegerField(blank=True, null=True)

    parameter_nn_loss = models.CharField(
        _("Loss"),
        max_length=100,
        null=True,
        blank=True,
        choices=zip(
            NNCONFIGURATION_LOSS_FUNCTION,
            NNCONFIGURATION_LOSS_FUNCTION,
        ),
        default="",
    )
    parameter_nn_loss_scaler = models.DecimalField(
        max_digits=settings.MODEL_DECIMAL_FIELD_DEFAULT_MAX_DIGITS,
        decimal_places=settings.MODEL_DECIMAL_FIELD_DEFAULT_DECIMAL_PLACES,
        null=True,
        blank=True,
    )
    parameter_nn_optimizer = models.CharField(
        _("Optimizer"),
        max_length=150,
        null=True,
        blank=True,
        choices=zip(
            NNCONFIGURATION_ALL_POSSIBLE_OPTIMIZER_FUNCTION,
            NNCONFIGURATION_ALL_POSSIBLE_OPTIMIZER_FUNCTION,
        ),
    )
    parameter_nn_shape = models.JSONField(null=True, blank=True, default=dict, encoder=JsonEncoderExtended)
    parameter_nn_find_delay_sec = models.IntegerField(blank=True, null=True)

    training_nn_model = models.OneToOneField(
        "models.NNModel",
        related_name="machine_nn_model",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
    )

    training_training_total_delay_sec = models.DecimalField(
        max_digits=settings.MODEL_DECIMAL_FIELD_DEFAULT_MAX_DIGITS,
        decimal_places=settings.MODEL_DECIMAL_FIELD_DEFAULT_DECIMAL_PLACES,
        null=True,
        blank=True,
    )
    training_training_cell_delay_sec = models.DecimalField(
        max_digits=settings.MODEL_DECIMAL_FIELD_DEFAULT_MAX_DIGITS,
        decimal_places=settings.MODEL_DECIMAL_FIELD_DEFAULT_DECIMAL_PLACES,
        null=True,
        blank=True,
    )
    training_date_time_machine_model = models.DateTimeField(null=True, blank=True)
    training_training_epoch_count = models.IntegerField(null=True, blank=True)
    training_training_batch_size = models.IntegerField(null=True, blank=True)
    training_total_training_line_count = models.IntegerField(null=True, blank=True)
    training_type_machine_hardware = models.TextField(null=True, blank=True)

    log_work_message = models.JSONField(null=True, blank=True, default=dict, encoder=JsonEncoderExtended)
    log_work_status = models.JSONField(null=True, blank=True, default=dict, encoder=JsonEncoderExtended)

    training_eval_loss_sample_training = models.DecimalField(
        max_digits=settings.MODEL_DECIMAL_FIELD_DEFAULT_MAX_DIGITS,
        decimal_places=settings.MODEL_DECIMAL_FIELD_DEFAULT_DECIMAL_PLACES,
        null=True,
        blank=True,
    )
    training_eval_loss_sample_evaluation = models.DecimalField(
        max_digits=settings.MODEL_DECIMAL_FIELD_DEFAULT_MAX_DIGITS,
        decimal_places=settings.MODEL_DECIMAL_FIELD_DEFAULT_DECIMAL_PLACES,
        null=True,
        blank=True,
    )
    training_eval_loss_sample_training_noise = models.DecimalField(
        max_digits=settings.MODEL_DECIMAL_FIELD_DEFAULT_MAX_DIGITS,
        decimal_places=settings.MODEL_DECIMAL_FIELD_DEFAULT_DECIMAL_PLACES,
        null=True,
        blank=True,
    )
    training_eval_loss_sample_evaluation_noise = models.DecimalField(
        max_digits=settings.MODEL_DECIMAL_FIELD_DEFAULT_MAX_DIGITS,
        decimal_places=settings.MODEL_DECIMAL_FIELD_DEFAULT_DECIMAL_PLACES,
        null=True,
        blank=True,
    )
    training_eval_accuracy_sample_training = models.DecimalField(
        max_digits=settings.MODEL_DECIMAL_FIELD_DEFAULT_MAX_DIGITS,
        decimal_places=settings.MODEL_DECIMAL_FIELD_DEFAULT_DECIMAL_PLACES,
        null=True,
        blank=True,
    )
    training_eval_accuracy_sample_evaluation = models.DecimalField(
        max_digits=settings.MODEL_DECIMAL_FIELD_DEFAULT_MAX_DIGITS,
        decimal_places=settings.MODEL_DECIMAL_FIELD_DEFAULT_DECIMAL_PLACES,
        null=True,
        blank=True,
    )
    training_eval_accuracy_sample_training_noise = models.DecimalField(
        max_digits=settings.MODEL_DECIMAL_FIELD_DEFAULT_MAX_DIGITS,
        decimal_places=settings.MODEL_DECIMAL_FIELD_DEFAULT_DECIMAL_PLACES,
        null=True,
        blank=True,
    )
    training_eval_accuracy_sample_evaluation_noise = models.DecimalField(
        max_digits=settings.MODEL_DECIMAL_FIELD_DEFAULT_MAX_DIGITS,
        decimal_places=settings.MODEL_DECIMAL_FIELD_DEFAULT_DECIMAL_PLACES,
        null=True,
        blank=True,
    )

    training_eval_outputs_cols_loss_sample_evaluation = models.JSONField(null=True, blank=True, default=dict, encoder=JsonEncoderExtended)
    training_eval_outputs_cols_accuracy_sample_evaluation = models.JSONField(null=True, blank=True, default=dict, encoder=JsonEncoderExtended)

    #used by page /machine_informations
    class Meta:
        app_label = 'models'
        db_table = "machine"
        verbose_name = _("machine")
        verbose_name_plural = _("Machines")
        indexes = [
            models.Index(fields=["-timestamp"]),
            models.Index(fields=["machine_owner_user"]),
            models.Index(fields=["machine_owner_team"]),
        ]

    def __str__(self):
        if self.id is not None and self.machine_name is not None:
            return f"<machine:{self.id}:{self.machine_name[0:30]}{'...' if len(self.machine_name)>25 else ''}>"
        elif self.id is not None:
            return f"<machine: id={self.id}, name=undefined>"
        else:
            return "<machine: Empty machine>"

    @property
    def training_nn_model_extfield(self) -> Union[None, bytes]:
        """
        Returns NN model of machine
        :return: nn model blob or None
        :rtype: Union[None, bytes]
        """
        return None if not self.training_nn_model else self.training_nn_model.nn_model

    @training_nn_model_extfield.setter
    def training_nn_model_extfield(self, nn_model_converted_in_bytes: bytes) -> NoReturn:
        """
        Sets a new nn model for machine, existing nn model will be overwritten
        :param nn_model_converted_in_bytes: new nn model
        :type nn_model_converted_in_bytes: bytes
        """
        if self.training_nn_model:
            self.training_nn_model.nn_model = nn_model_converted_in_bytes
        else:
            from models.nn_model import NNModel
            new_model_nnmodel = NNModel.objects.create(nn_model=nn_model_converted_in_bytes)
            self.training_nn_model = new_model_nnmodel

    @training_nn_model_extfield.deleter
    def training_nn_model_extfield(self) -> NoReturn:
        """
        Deletes nn model of machine
        """
        if self.training_nn_model:
            self.training_nn_model = None

    @property
    def enc_dec_configuration_extfield(self):
        """
        Returns encdec configuration of machine as dict
        :return: configuration dict or None
        :rtype: Union[None, dict]
        """
        if not self.enc_dec_configuration:
            return None
        else:
            return self.enc_dec_configuration.enc_dec_config

    @enc_dec_configuration_extfield.setter
    def enc_dec_configuration_extfield(self, configuration_to_set: dict) -> NoReturn:
        """
        Sets a new encdec configuration for machine, existing configuration will be overwritten
        :param configuration_to_set: new encdec configuration
        :type configuration_to_set: dict
        """
        if not self.enc_dec_configuration:
            from models.encdec_configuration import EncDecConfiguration
            self.enc_dec_configuration = EncDecConfiguration.objects.create(enc_dec_config=configuration_to_set)
        else:
            self.enc_dec_configuration.enc_dec_config = configuration_to_set

    @enc_dec_configuration_extfield.deleter
    def enc_dec_configuration_extfield(self):
        """
        Deletes encdec configuration of machine
        """
        if self.enc_dec_configuration:
            self.enc_dec_configuration = None

    def save(self, *args, **kwargs):
        if ENABLE_LOGGER_DEBUG_Machine:
            logger.debug(f"Saving Django Machine model {self} on disk")

        machine_name_is_reserved = self.machine_name.startswith("__") and self.machine_name.endswith("__")
        if machine_name_is_reserved and not self.machine_owner_user.is_superuser:
            raise ValueError("sorry but names with double underscore are reserved")

        # STEP 1/3
        # We save all the extfield attached to the Machine
        if self.enc_dec_configuration:
            self.enc_dec_configuration.save()
        if self.training_nn_model:
            self.training_nn_model.save()

        # STEP 2/3
        # we need to save this 2 fields as a dict of string so we convert them to string
        self.dfr_columns_type_user_df_backup = self.dfr_columns_type_user_df
        self.dfr_columns_type_user_df = {
            column_name: column_type.name if isinstance(column_type, DatasetColumnDataType) else column_type
            for column_name, column_type in self.dfr_columns_type_user_df.items()
        }
        self.mdc_columns_data_type_backup = self.mdc_columns_data_type
        if self.mdc_columns_data_type:
            self.mdc_columns_data_type = {
                column_name: column_type.name if isinstance(column_type, DatasetColumnDataType) else column_type
                for column_name, column_type in self.mdc_columns_data_type.items()
            }
        else:
            # yes in case of a bad dataset it can be None !
            self.mdc_columns_data_type = {}

        # STEP 3/3
        # saving the machine model django
        super(Machine, self).save(*args, **kwargs)

        # STEP 4/3
        # we need to revert the conversion we did for saving
        self.dfr_columns_type_user_df = self.dfr_columns_type_user_df_backup
        self.mdc_columns_data_type = self.mdc_columns_data_type_backup

    def get_machine_data_input_lines_list_columns(self, include_predefined=False):
        columns = [c for c, is_input in self.mdc_columns_name_input.items() if is_input]

        if include_predefined:
            columns.extend(self.get_machine_data_input_lines_list_predefined_columns().keys())

        return columns

    def get_machine_data_output_lines_list_columns(self, include_predefined=False):
        columns = [c for c, is_output in self.mdc_columns_name_output.items() if is_output]

        if include_predefined:
            columns.extend(self.get_machine_data_output_lines_list_predefined_columns().keys())

        return columns

    @staticmethod
    def get_machine_data_input_lines_list_predefined_columns():
        predefined_columns = {
            "Line_ID": models.BigIntegerField(primary_key=True),  # Changed from AutoField to BigIntegerField to avoid decimal conversion in SQLite
            "IsForLearning": models.BooleanField(default=False),
            "IsForSolving": models.BooleanField(default=False),
            "IsForEvaluation": models.BooleanField(default=False),
            "IsLearned": models.BooleanField(default=False),
            "IsSolved": models.BooleanField(default=False),
        }
        return predefined_columns

    def get_machine_data_output_lines_list_predefined_columns(self):
        input_model = self.get_machine_data_input_lines_model()
        predefined_columns = {
            "Line_ID": models.ForeignKey(
                to=input_model,
                db_constraint=False,
                on_delete=models.CASCADE,
                primary_key=True,
                db_column="Line_ID",
            ),
        }
        return predefined_columns

    def get_machine_data_input_lines_model(self):
        """
        Return class of the Machine_<ID>_DataInputLines
        """
        try:
            types = dict(self.dfr_columns_type_user_df)

            columns = [key for key, value in self.mdc_columns_name_input_user_df.items() if value]
            if len(columns) == 0:
                logger.error(f"there is no columns INPUTS in mdc_columns_name_input_user_df for machine {self}")

            predefined_columns = self.get_machine_data_input_lines_list_predefined_columns()

            # Import here to avoid circular import
            from models.dynamic_model import create_dynamic_model, DynamicModel

            # Merge predefined columns (Line_ID, IsForLearning, etc.) with user columns
            all_fields = {}
            all_fields.update(predefined_columns)  # Add predefined columns first
            if columns:
                all_fields.update(_get_model_fields_by_names_types(columns, types))  # Add user columns

            cls = create_dynamic_model(
                (models.Model, DynamicModel),
                f"Machine_{self.id}_DataInputLines",
                table=f"Machine_{self.id}_DataInputLines",
                fields=all_fields,
                app_label="models",
                module_name="",
                primary_key_column="Line_ID",
            )

            cls._meta.id = self.id

            cls._meta.indexes = [
                models.Index(fields=["Line_ID"], name=f"Machine_{self.id}_Line_ID"),
                models.Index(fields=["IsForLearning"], name=f"Machine_{self.id}_IsForLearning"),
                models.Index(fields=["IsForSolving"], name=f"Machine_{self.id}_IsForSolving"),
                models.Index(fields=["IsForEvaluation"], name=f"Machine_{self.id}_IsForEvaluation"),
                models.Index(fields=["IsLearned"], name=f"Machine_{self.id}_IsLearned"),
                models.Index(fields=["IsSolved"], name=f"Machine_{self.id}_IsSolved"),
            ]

            dil = cls
        except Exception:
            logger.error(f"Unable to get data_input_lines_model for machine:{self}")
            dil = None

        return dil

    def get_machine_data_output_lines_model(self):
        """Return class of the Machine_<ID>_DataOutputLines"""

        try:
            types = dict(self.dfr_columns_type_user_df)

            columns = [key for key, value in self.mdc_columns_name_output_user_df.items() if value]
            if len(columns) == 0:
                logger.error(f"there is no columns OUTPUT in mdc_columns_name_input_user_df for {self}")

            predefined_columns = self.get_machine_data_output_lines_list_predefined_columns()

            # Import here to avoid circular import
            from models.dynamic_model import create_dynamic_model, DynamicModel

            # Merge predefined columns (Line_ID) with user columns
            all_fields = {}
            all_fields.update(predefined_columns)  # Add predefined columns first (Line_ID)
            if columns:
                all_fields.update(_get_model_fields_by_names_types(columns, types))  # Add user columns

            cls = create_dynamic_model(
                (models.Model, DynamicModel),
                f"Machine_{self.id}_DataOutputLines",
                table=f"Machine_{self.id}_DataOutputLines",
                fields=all_fields,
                app_label="models",
                module_name="",
                primary_key_column="Line_ID",
            )

            cls._meta.id = self.id

            cls._meta.indexes = [
                models.Index(fields=["Line_ID"], name=f"Machine_{self.id}_LineOutput_ID"),
            ]

            dil = cls
        except Exception:
            logger.error(f"Unable to get data_output_lines_model for machine:{self}")
            dil = None

        return dil

    def get_pre_encoded_list_input_column(self):
        return [column_name for column_name, is_column in self.mdc_columns_name_input.items() if is_column]

    def get_pre_encoded_list_output_column(self):
        return [column_name for column_name, is_column in self.mdc_columns_name_output.items() if is_column]

    @property
    def get_user_df_list_names_input_column(self):
        """
        Function should return [`Columns`]
        [`Columns`] - it is names of columns for Machine_<ID>_DataInputLines

        return: [`Columns`]
        """
        return [column_name for column_name, is_column in self.mdc_columns_name_input_user_df.items() if is_column]

    @property
    def get_user_df_list_names_output_column(self):
        """
        Function should return [`Columns`]
        [`Columns`] - it is names of columns for Machine_<ID>_DataOutputLines

        return: [`Columns`]
        """
        return [column_name for column_name, is_column in self.mdc_columns_name_output_user_df.items() if is_column]

    @property
    def get_user_df_list_all_column(self):
        """
        Function should return [`Columns`]
        [`Columns`] - it is names of columns for Machine_<ID>_DataOutputLines

        return: [`Columns`]
        """
        input_columns = [column_name for column_name, is_column in self.mdc_columns_name_input_user_df.items() if is_column]
        output_columns = [column_name for column_name, is_column in self.mdc_columns_name_output_user_df.items() if is_column]
        all_columns = input_columns + output_columns
        return all_columns

    def update_field_directly_by_sql(self, field_name):
        """
        Updates a filed in machine table

        :param field_name: name os the filed to update
        """
        assert field_name in [f.name for f in Machine._meta.get_fields()], f"Undefined field name: {field_name}."
        field_value = getattr(self, field_name)
        if isinstance(field_value, dict):
            for k, v in field_value.items():
                field_value[k] = v if not isinstance(v, str) else v.replace('"', "").replace("'", "")
            field_value = str(field_value)

        if isinstance(field_value, str):
            field_value = field_value.replace("'", '"').replace("\\n", "")
            field_value = f"'{field_value}'"

        with connection.cursor() as cursor:
            cursor.execute(f"UPDATE machine SET {field_name} = {field_value} WHERE id={self.id}")
            cursor.fetchone()

    def get_machine_info_by_field_name_and_column_name(self, field_name, column_name):
        is_field_return_length = field_name.startswith("=>LEN;")
        if is_field_return_length:
            field_name = field_name.split(";")[1]
        field_value = getattr(self, field_name)
        return len(field_value.get(column_name, "")) if is_field_return_length else field_value.get(column_name, "")

    def read_data_lines_from_db(
        self,
        input_columns: list = None,
        output_columns: list = None,
        where_clause_dict: dict = None,
        sort_by: str = None,
        rows_count_limit: int = None,
        is_random: bool = False,
    ) -> pd.DataFrame:

        _where_clause_dict = {}
        if where_clause_dict:
            where_clause_dict_clean = {
                column_name: column_value
                for column_name, column_value in where_clause_dict.items()
                if column_name not in ("with_reserved_columns", "offset", "")
            }
            _where_clause_dict.update({column_name: "=1" for column_name, column_value in where_clause_dict_clean.items() if column_value is True})
            _where_clause_dict.update({column_name: "=0" for column_name, column_value in where_clause_dict_clean.items() if column_value is False})

        _limit_clause = f"LIMIT {rows_count_limit}" if rows_count_limit else ""

        if sort_by and is_random:
            logger.error("It is not possible to use simultaneously sort_by and is_random")
        elif sort_by:
            _sort_clause_direction = "DESC" if sort_by.startswith("-") else "ASC"
            _sort_clause = f"ORDER BY `{sort_by}` {_sort_clause_direction}"
        elif is_random:
            _sort_clause = "ORDER BY RAND()"
        else:
            _sort_clause = ""

        if isinstance(input_columns, list) and isinstance(output_columns, list) and len(input_columns) != 0 and len(output_columns) != 0:
            _where_clause = (
                "WHERE " + " AND ".join([f"{column_name}{column_clause}" for column_name, column_clause in _where_clause_dict.items()])
                if _where_clause_dict
                else ""
            )

            _sql_query = (
                f"SELECT {', '.join([f'`{column_name}`' for column_name in input_columns])}, {', '.join([f'`{column_name}`' for column_name in output_columns])} "
                f"FROM Machine_{self.id}_DataInputLines "
                f"LEFT JOIN Machine_{self.id}_DataOutputLines "
                f"ON Machine_{self.id}_DataInputLines.Line_ID = Machine_{self.id}_DataOutputLines.Line_ID "
                f"{_where_clause} "
                f"{_sort_clause} {_limit_clause};"
            )

        elif isinstance(input_columns, list) and len(input_columns) != 0 and (output_columns is None or output_columns == []):
            _where_clause = (
                "WHERE " + " AND ".join([f"{column_name}{column_clause}" for column_name, column_clause in _where_clause_dict.items()])
                if _where_clause_dict
                else ""
            )

            _sql_query = f"SELECT {', '.join([f'`{column_name}`' for column_name in input_columns])} FROM Machine_{self.id}_DataInputLines {_where_clause} {_sort_clause} {_limit_clause}"

        elif isinstance(output_columns, list) and len(output_columns) != 0 and (input_columns is None or input_columns == []):
            _where_clause = (
                "WHERE " + " AND ".join([f"{column_name}{column_clause}" for column_name, column_clause in _where_clause_dict.items()])
                if _where_clause_dict
                else ""
            )

            _sql_query = f"SELECT {', '.join([f'`{column_name}`' for column_name in output_columns])} FROM Machine_{self.id}_DataOutputLines {_where_clause} {_sort_clause} {_limit_clause}"

        elif isinstance(output_columns, list) and len(output_columns) != 0 and not input_columns and _where_clause_dict:
            _where_clause = (
                "WHERE " + " AND ".join([f"{column_name}{column_clause}" for column_name, column_clause in _where_clause_dict.items()])
                if _where_clause_dict
                else ""
            )

            _sql_query = (
                f"SELECT {', '.join([f'`{column_name}`' for column_name in output_columns])} "
                f"FROM Machine_{self.id}_DataInputLines "
                f"LEFT JOIN Machine_{self.id}_DataOutputLines "
                f"ON Machine_{self.id}_DataInputLines.Line_ID = Machine_{self.id}_DataOutputLines.Line_ID "
                f"{_where_clause} "
                f"{_sort_clause} {_limit_clause};"
            )

        else:
            return self.read_data_lines_from_db(
                input_columns=[
                    column_name for column_name, column_direction in self.mdc_columns_name_input_user_df.items() if column_direction
                ],
                output_columns=[
                    column_name for column_name, column_direction in self.mdc_columns_name_output_user_df.items() if column_direction
                ],
                where_clause_dict=where_clause_dict,
                sort_by=sort_by,
                rows_count_limit=rows_count_limit,
                is_random=is_random,
            )

        _df = pd.read_sql_query(_sql_query, connections["default"])

        return _df


def _get_model_fields_by_names_types(columns, types):
    fields = {}
    for column in columns:
        try:
            (field_callable, field_args) = TYPE_MAPPING[
                types[column].name if isinstance(types[column], DatasetColumnDataType) else types[column]
            ]
            fields[column] = field_callable(**field_args)
        except KeyError:
            logger.error(f"Unknown column type for column {column}, using TextField")
            fields[column] = models.TextField(null=True)

    return fields
