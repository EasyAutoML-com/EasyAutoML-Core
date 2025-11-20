from typing import Union, Optional, NoReturn
from timeit import default_timer

from ML import EasyAutoMLDBModels, __getlogger

from ML import Machine
from ML import MachineDataConfiguration
from ML.FeatureEngineeringTemplate import *
from ML import FeatureEngineeringTemplate

from SharedConstants import *

logger = __getlogger()



class FeatureEngineeringConfiguration:
    """
    Find the FET to enable for each dataframe columns
    Using a budget (how many columns to create at maximum)
    Using ICI (Inputs columns Importance)
    And also using MachineEasyAutoML to predict benefits of the FET
    """

    def __init__(self,
                machine: Machine,
                global_dataset_budget: Optional[int] = None,
                nn_engine_for_searching_best_config: Optional = None,
                force_configuration_simple_minimum: Optional[bool] = False,
                ):
        """
        Creates or loads a FeatureEngineeringConfiguration object

        :param machine_or_nnengine: a machine object if we want to load the configuration , or a nnengine if we want to search the best config, or none if we do force_configuration_simple_minimum
        :param global_dataset_budget:
            If None:    loads a FeatureEngineeringConfiguration object using machine object
            If int:     create a FeatureEngineeringConfiguration object with total budget value

        """

        self._activated_fet_list_per_column = dict()
        self._cost_per_columns = dict()
        self._fe_find_delay_sec = None
        self._machine = machine

        if not isinstance( machine , Machine ):
            logger.error( f"Machine is not a machine object == {Machine}")

        if force_configuration_simple_minimum:
            if nn_engine_for_searching_best_config:
                logger.error( f"nn_engine_for_searching_best_config cannot be defined if force_configuration_simple_minimum")
            self._init_create_configuration(force_configuration_simple_minimum=True)

        elif global_dataset_budget is None:
            if nn_engine_for_searching_best_config:
                logger.error( f"nn_engine_for_searching_best_config cannot be defined if force_configuration_simple_minimum")
            self._init_load_configuration( machine )

        else:
            # create configuration BEST with budget
            if not nn_engine_for_searching_best_config:
                logger.error( f"nn_engine_for_searching_best_config must be defined for searching best config if global_dataset_budget is defined")
            self._init_create_configuration(
                global_all_fec_budget = global_dataset_budget,
                nn_engine_to_use_in_trial = nn_engine_for_searching_best_config,
                force_configuration_simple_minimum = False,
            )


    def _init_load_configuration(self , machine_to_load_config_from:Machine ) -> NoReturn:
        """
        Load the FeatureEngineeringConfiguration object using an instance of the machine class
        """

        if ENABLE_LOGGER_DEBUG_FeatureEngineeringConfiguration: logger.debug( f"Loading FeatureEngineeringConfiguration for {machine_to_load_config_from} " )

        self._activated_fet_list_per_column = machine_to_load_config_from.db_machine.fe_columns_fet
        self._fe_find_delay_sec = machine_to_load_config_from.db_machine.fe_find_delay_sec


    def _init_create_configuration(
            self,
            global_all_fec_budget: Optional[int ] = 0,
            nn_engine_to_use_in_trial = None,
            force_configuration_simple_minimum: Optional[bool] = False,
            ) -> NoReturn:
        """
        Defines all FeatureEngineeringColumns with best mode or simple mode

        :global_all_fec_budget: the maximum budget value of the sum of all columns cost
        :param force_configuration_simple_minimum: init will create the configuration minimum_simple
        """

        if global_all_fec_budget and force_configuration_simple_minimum:
            logger.error("We cannot define a Budget for the FET when doing force_configuration_simple_minimum")

        if force_configuration_simple_minimum or DEBUG_DISABLE_find_machine_best_FE_configuration:
            if ENABLE_LOGGER_DEBUG_FeatureEngineeringConfiguration: logger.debug( f"Creation FEC Minimum for {self._machine} starting..." )
            # make minimum FEC for all columns

            for this_column_name, this_data_type in self._machine.db_machine.mdc_columns_data_type.items():
                if not isinstance( this_data_type , DatasetColumnDataType ):
                    logger.error(f"mdc_columns_data_type should have type DatasetColumnDataType, but is '{type( this_data_type )}' ")
                if this_data_type == DatasetColumnDataType.IGNORE:
                    # the IGNORE column must have no FET
                    pass
                else:
                    # create the minimum configuration for this FEC
                    fec_minimum = FeatureEngineeringColumn(
                            this_column_datas_infos=self.get_all_column_datas_infos(this_column_name),
                            force_configuration_simple_minimum=True,
                            )
                    self.store_this_fec_to_fet_list_configuration( fec_minimum, this_column_name )
            self._fe_find_delay_sec = 0
        else:
            # make the best configuration for all the columns
            if ENABLE_LOGGER_DEBUG_FeatureEngineeringConfiguration: logger.debug( f"Creation FEC Best for {self._machine} starting, budget: {global_all_fec_budget} " )
            delay_total_started_at = default_timer( )

            columns_importance_evaluation = self._machine.db_machine.fe_columns_inputs_importance_evaluation
            if not columns_importance_evaluation:
                logger.error( "computing best possible configuration for the column require columns_importance_evaluation and it is not defined")

            # compute the budget for each column
            global_all_fec_budget_inputs = global_all_fec_budget / self._machine.db_machine.mdc_columns_total_count* self._machine.db_machine.mdc_columns_input_count
            global_all_fec_budget_outputs = global_all_fec_budget / self._machine.db_machine.mdc_columns_total_count* self._machine.db_machine.mdc_columns_output_count
            budget_per_columns = {}
            for column_name in self._machine.db_machine.mdc_columns_data_type:
                if self._machine.db_machine.mdc_columns_name_input[ column_name ]:
                    budget_per_columns[column_name] = max( 1 , round(global_all_fec_budget_inputs * columns_importance_evaluation[column_name ]) )
                elif self._machine.db_machine.mdc_columns_name_output[ column_name ]:
                    budget_per_columns[column_name] = max( 1 , round(global_all_fec_budget_outputs / len(self._machine.db_machine.mdc_columns_name_output) ) )

            # create all the FEC for each column
            # will find the best configuration fitting the budget
            # - it do for every column FEC._set_configuration_best_having_column_budget
            for this_column_name, budget_this_column in budget_per_columns.items():
                fet_for_this_column = None
                if budget_this_column < 1:
                    # budget is too low -> no fet at all
                    if ENABLE_LOGGER_DEBUG_FeatureEngineeringConfiguration: logger.debug( f"Set FEC for column '{this_column_name}' disabled because budget: {budget_this_column} is < 1" )
                    fet_for_this_column = FeatureEngineeringColumn(
                            this_column_datas_infos=self.get_all_column_datas_infos(this_column_name),
                            force_configuration_empty=True,
                            )
                elif budget_this_column <= FEC_SIMPLE_MINIMUM_CONFIGURATION_WHEN_BUDGET_BELOW:
                    # budget is too low to search the best configuration -> config minimum
                    if ENABLE_LOGGER_DEBUG_FeatureEngineeringConfiguration: logger.debug( f"Set FEC for column '{this_column_name}' minimum because budget : {budget_this_column} is <= {FEC_SIMPLE_MINIMUM_CONFIGURATION_WHEN_BUDGET_BELOW}" )
                    fet_for_this_column = FeatureEngineeringColumn(
                            this_column_datas_infos=self.get_all_column_datas_infos(this_column_name),
                            force_configuration_simple_minimum=True,
                            )
                else:
                    # generate FEC with best configuration
                    if ENABLE_LOGGER_DEBUG_FeatureEngineeringConfiguration: logger.debug( f"Search best FEC for column '{this_column_name}' with budget : {budget_this_column}" )
                    fet_for_this_column = FeatureEngineeringColumn(
                            nn_engine_to_use_in_trial= nn_engine_to_use_in_trial,
                            this_column_datas_infos=self.get_all_column_datas_infos(this_column_name),
                            this_fec_budget_maximum=budget_this_column,
                        )
                    if ENABLE_LOGGER_DEBUG_FeatureEngineeringConfiguration: logger.info( f"Done found best FEC for column '{this_column_name}' in {int(default_timer( ) - delay_total_started_at)} seconds - Budget: {budget_this_column} - Cost: {fet_for_this_column._fec_cost} - FEC have {len( fet_for_this_column._list_class_enabled_fet )} FET(s).. " )

                # set the FEC for each column
                self.store_this_fec_to_fet_list_configuration( fet_for_this_column, this_column_name )

            # all columns FEC have been set
            # total delay for all columns FEC
            self._fe_find_delay_sec = default_timer() - delay_total_started_at

        # configuration FEC all columns done (minimum or best)


        # checking if the configuration is ok for the columns inputs
        # if not any( self._activated_fet_list_per_column[ col_name ] != [ ] for col_name,enabled in self._mdc.columns_name_input.items() if enabled ):
        #     logger.error("all inputs have ALL no FET !")
        all_fet_list_empty = True
        for col_name,input_enabled in self._machine.db_machine.mdc_columns_name_input.items():
            if input_enabled:
                if self._activated_fet_list_per_column[ col_name ] != [ ]:
                    all_fet_list_empty = False
                    break
        if all_fet_list_empty:
            logger.error("all inputs have ALL no FET !  - this is not allowed")

        # checking if the configuration is ok for the columns outputs
        # if not all( self._activated_fet_list_per_column[ col_name ] != [ ] for col_name,enabled in self._mdc.columns_name_output.items() if enabled ):
        #     logger.error("there is some outputs without any FET - this is not allowed")
        for col_name,output_enabled in self._machine.db_machine.mdc_columns_name_output.items():
            if output_enabled:
                if self._activated_fet_list_per_column[ col_name ] == [ ] or self._activated_fet_list_per_column[ col_name ] == [ FET_TO_ENABLE_WHEN_NONE_PRESENT ]:
                    logger.error( f"The ouput column {col_name} , {self._machine.db_machine.mdc_columns_data_type[col_name]} , have no FET - this is not allowed")

        if ENABLE_LOGGER_DEBUG_FeatureEngineeringConfiguration:
            if force_configuration_simple_minimum:
                logger.info( f"Done FEC with force_configuration_simple_minimum=True" )
            else:
                logger.info( f"Done FEC best with budget : {global_all_fec_budget} in {self._fe_find_delay_sec} seconds " )


    def save_configuration_in_machine(self) -> "FeatureEngineeringConfiguration":
        """
        Stores FeatureEngineeringConfiguration attributes inside the machine
        :return: return itself
        """

        if ENABLE_LOGGER_DEBUG_FeatureEngineeringConfiguration: logger.debug( f"Saving FeatureEngineeringConfiguration in {self._machine} " )

        _fe_count_of_fet = {}
        for fet_name in LIST_ALL_FET_NAMES:
            _fe_count_of_fet[fet_name] = 0
        for col_name, list_fet_enabled in self._activated_fet_list_per_column.items():
            for fet_enabled in list_fet_enabled:
                for fet_name in LIST_ALL_FET_NAMES:
                    if fet_enabled.startswith( fet_name ):
                        _fe_count_of_fet[ fet_name ] += 1
        self._machine.db_machine.fe_count_of_fet = _fe_count_of_fet
        self._machine.db_machine.fe_columns_fet = self._activated_fet_list_per_column
        self._machine.db_machine.fe_find_delay_sec = self._fe_find_delay_sec

        return self


    def get_all_column_datas_infos(self, column_name: str) -> Column_datas_infos:
        """
        give all possible information about a column so we can identify later what FET is more useful
        structure Column_datas_infos is defined in module FEC

        :param column_name: the name of the column to get get_all_datas_infos for
        :return: Column_datas_infos for the column
        """
        _all_datas_infos = Column_datas_infos(
            column_name,
            self._machine.db_machine.mdc_columns_name_input[column_name],
            self._machine.db_machine.mdc_columns_name_output[column_name],
            self._machine.db_machine.mdc_columns_data_type[column_name],

            self._machine.db_machine.dfr_columns_description_user_df.get(column_name),

            self._machine.db_machine.mdc_columns_unique_values_count.get(column_name),
            self._machine.db_machine.mdc_columns_missing_percentage.get(column_name),
            self._machine.db_machine.mdc_columns_most_frequent_values_count.get(column_name),

            self._machine.db_machine.mdc_columns_values_min.get(column_name),
            self._machine.db_machine.mdc_columns_values_max.get(column_name),
            self._machine.db_machine.mdc_columns_values_mean.get(column_name),
            self._machine.db_machine.mdc_columns_values_std_dev.get(column_name ),
            self._machine.db_machine.mdc_columns_values_skewness.get(column_name),
            self._machine.db_machine.mdc_columns_values_kurtosis.get(column_name),

            self._machine.db_machine.mdc_columns_values_quantile02.get(column_name),
            self._machine.db_machine.mdc_columns_values_quantile03.get(column_name),
            self._machine.db_machine.mdc_columns_values_quantile07.get(column_name),
            self._machine.db_machine.mdc_columns_values_quantile08.get(column_name),
            self._machine.db_machine.mdc_columns_values_sem.get(column_name),
            self._machine.db_machine.mdc_columns_values_median.get(column_name),
            self._machine.db_machine.mdc_columns_values_mode.get(column_name),

            self._machine.db_machine.mdc_columns_values_str_percent_uppercase.get(column_name),
            self._machine.db_machine.mdc_columns_values_str_percent_lowercase.get(column_name),
            self._machine.db_machine.mdc_columns_values_str_percent_digit.get(column_name),
            self._machine.db_machine.mdc_columns_values_str_percent_punctuation.get(column_name),
            self._machine.db_machine.mdc_columns_values_str_percent_operators.get(column_name),
            self._machine.db_machine.mdc_columns_values_str_percent_underscore.get(column_name),
            self._machine.db_machine.mdc_columns_values_str_percent_space.get(column_name),

            self._machine.db_machine.mdc_columns_values_str_language_en.get(column_name),
            self._machine.db_machine.mdc_columns_values_str_language_fr.get(column_name),
            self._machine.db_machine.mdc_columns_values_str_language_de.get(column_name),
            self._machine.db_machine.mdc_columns_values_str_language_it.get(column_name),
            self._machine.db_machine.mdc_columns_values_str_language_es.get(column_name),
            self._machine.db_machine.mdc_columns_values_str_language_pt.get(column_name),
            self._machine.db_machine.mdc_columns_values_str_language_others.get(column_name),
            self._machine.db_machine.mdc_columns_values_str_language_none.get(column_name),

            self._activated_fet_list_per_column.get(column_name, list()),
            )

        return _all_datas_infos


    def set_this_fec_in_columns_configuration(self, fec: "FeatureEngineeringColumn" ) -> NoReturn:
        """
        Adds a FeatureEngineeringColumn to FeatureEngineeringConfiguration

        :param fec: the FeatureEngineeringColumn object to be added
        """
        column_name = fec._fec_column_datas_infos.name

        if column_name not in self._mdc.columns_data_type:
            logger.error(f"FE can not add fec on '{column_name}' column because datatype is not defined !")

        self._activated_fet_list_per_column[column_name] = [fet_class.__name__
                    for fet_class in fec._list_class_enabled_fet]
        self._cost_per_columns[column_name] = fec._fec_cost


    def store_this_fec_to_fet_list_configuration(self, the_new_fec: "FeatureEngineeringColumn", column_name: str ) -> NoReturn:
        """
        Add or replace a FeatureEngineeringColumn to self.FeatureEngineeringConfiguration in the indicated column_name

        :param the_new_fec: the FeatureEngineeringColumn object to be added/replaced
        :param column_name: the column where to set this FEC
        """
        list_fet_names = [fet_class.__name__ for fet_class in the_new_fec._list_class_enabled_fet]
        if ENABLE_LOGGER_DEBUG_FeatureEngineeringConfiguration: logger.debug( f"Set column '{column_name}' FEC to ({list_fet_names})" )

        if column_name not in self._machine.db_machine.mdc_columns_data_type:
            logger.error(f"FE can not add/replace fec with to this not existing column name : '{column_name}' ")

        elif self._machine.db_machine.mdc_columns_data_type[column_name] == DatasetColumnDataType.IGNORE:
            logger.error(f"FE can not add/replace fec in the empty column '{column_name}' ")

        self._activated_fet_list_per_column[column_name] = list_fet_names
        self._cost_per_columns[column_name] = the_new_fec._fec_cost


#=========================================================================================================================
class FeatureEngineeringColumn:
    """
    Contains configuration about FE for one single user_column
    The configuration is only which FET are enabled for the column

    when we create a FEC (FeatureEngineeringColumn) it search the best combination of FET to enable for this column which fit in the defined budget
    The budget is an integer which represent how many column enabled FET can create at maximum
    The budget 0 to 1 mean it will be no FET
    The budget >1 mean there must be at least one FET LOSSLESS

    The best combination of FET is the combination bringing the more benefits and with a cost <= budget
    """

    def __init__(
            self,
            nn_engine_to_use_in_trial: Optional = None,
            this_column_datas_infos: Optional[Column_datas_infos] = None,
            this_fec_budget_maximum: Optional[float ] = None,
            force_configuration_simple_minimum: Optional[bool] = False,
            force_configuration_empty: Optional[bool] = False,
            dict_force_load_this_fet_names: Optional[dict] = None,
    ):
        """
        Create or load the FEC (FeatureEngineeringColumn)

        :param this_column_datas_infos: give all possible information about column data to help decide what FET to enable
        :param this_fec_budget_maximum: how many columns we can create total with all selected FET
        :param force_configuration_simple_minimum: We will select only the minimum lossless FET
        """

        def _round_budget(budget: int) -> int:
            if budget >= 640:
                return 1024
            elif budget >= 160:
                return 256
            elif budget >= 40:
                return 64
            else:
                return 16

        self._fec_column_datas_infos = this_column_datas_infos
        self._list_class_enabled_fet = []
        self._fec_cost = 0


        #if ENABLE_LOGGER_DEBUG: logger.debug(f"Creation FeatureEngineeringColumn for column '{self._fec_column_datas_infos.name}' ")

        if force_configuration_empty:
            # this FEC will be empty
            return

        # IGNORE column will have no fet at all
        if self._fec_column_datas_infos.datatype is DatasetColumnDataType.IGNORE:
            return

        # If this column is not an input and is not output we do not add any FET
        if not self._fec_column_datas_infos.is_input and not self._fec_column_datas_infos.is_output:
            return

        if dict_force_load_this_fet_names:
            if force_configuration_simple_minimum:
                logger.error( "sorry but when creating FEC we can use force_configuration_simple_minimum OR force_load_this_fet_names_list not both" )
            if not this_column_datas_infos:
                logger.error( "Creating configuration from fet_names_list_to_load require also argument this_column_datas_infos")
            if this_fec_budget_maximum:
                logger.error("when using dict_force_load_this_fet_names you cannot set : this_fec_budget_maximum")
            self._set_configuration_to_this(dict_force_load_this_fet_names=dict_force_load_this_fet_names)

        elif force_configuration_simple_minimum:
            if this_fec_budget_maximum:
                logger.error("when using _set_configuration_simple_minimum you cannot set : this_fec_budget_maximum")
            self._set_configuration_simple_minimum()

        # if the column do not have many values, it is faster to do MINIMUM_CONFIGURATION
        elif (
                isinstance(this_column_datas_infos, Column_datas_infos) and
                this_column_datas_infos.unique_value_count <= 5
        ):
            self._set_configuration_simple_minimum()

        elif (
                isinstance(this_column_datas_infos, Column_datas_infos)
                and ( nn_engine_to_use_in_trial )
                and isinstance(this_fec_budget_maximum, (int, float) )
        ):
            self._set_configuration_best_having_column_budget(
                _round_budget(int(this_fec_budget_maximum)),
                nn_engine_to_use_in_trial,
            )

        else:
            logger.error("FEC cannot be loaded or created using these combination of arguments")


    def _set_configuration_to_this(self, dict_force_load_this_fet_names: dict) -> NoReturn:
        """
        load the FEC (configuration of FE for the column)
        :param dict_force_load_this_fet_names: the list of FET for this column FEC , keys are columns names , and values are false or true
        """

        if ENABLE_LOGGER_DEBUG_FeatureEngineeringConfiguration: logger.debug( f"Set column '{self._fec_column_datas_infos.name}' FEC to: {[ fet for fet, enabled in dict_force_load_this_fet_names.items( ) if enabled ]} " )
        try:
            self._list_class_enabled_fet = [ ]
            for fet_name, fet_value in dict_force_load_this_fet_names.items():
                if fet_value:
                    fec_instance = getattr(FeatureEngineeringTemplate, fet_name)
                    self._list_class_enabled_fet.append( fec_instance )
        except Exception as e:
            logger.error(f"unable to set the list of FET instance from fet list : {dict_force_load_this_fet_names}")

        self._set_cost_of_enabled_fets()

        if not self._check_if_current_configuration_is_valid():
            logger.error(f"this combination of FET not valid for column '{self._fec_column_datas_infos.name}' ")



    def _set_configuration_simple_minimum(self) -> NoReturn:
        """
        Return the list of fet for the columns : FEC_CONFIGURATION_SIMPLE_MINIMUM_LIST_FET
        and also FEC_MISSING_VALUES_MUST_ENABLE_THIS_FET  if there are some missing values in the column
        """

        if ENABLE_LOGGER_DEBUG_FeatureEngineeringConfiguration: logger.debug( f"Set column '{self._fec_column_datas_infos.name}' FEC to minimum" )

        # add in the FEC all fets in FEC_CONFIGURATION_SIMPLE_MINIMUM_LIST_FET
        list_class_fet_enabled_for_this_fec = list()
        fet_activated_for_this_column = FEC_CONFIGURATION_SIMPLE_MINIMUM_LIST_FET.copy()
        if self._fec_column_datas_infos.missing_percentage > 0:
            fet_activated_for_this_column.append( FET_TO_ENABLE_WHEN_NONE_PRESENT )
        for one_base_fet_name in fet_activated_for_this_column:

            # If the column has a datetime type we use 2 FET : for the date and for time data types
            if self._fec_column_datas_infos.datatype is DatasetColumnDataType.DATETIME:
                class_fet_to_add = getattr(FeatureEngineeringTemplate, f"{one_base_fet_name}Date")
                if class_fet_to_add.cls_is_possible_to_enable_this_fet_with_this_infos(self._fec_column_datas_infos ):
                    list_class_fet_enabled_for_this_fec.append(class_fet_to_add)
                class_fet_to_add = getattr(FeatureEngineeringTemplate, f"{one_base_fet_name}Time")
                if class_fet_to_add.cls_is_possible_to_enable_this_fet_with_this_infos(self._fec_column_datas_infos ):
                    list_class_fet_enabled_for_this_fec.append(class_fet_to_add)
            else:
                # we add the FET to this FEC
                class_fet_to_add = getattr(    FeatureEngineeringTemplate,
                                                            f"{one_base_fet_name}{self._fec_column_datas_infos.datatype.name.capitalize()}")
                if class_fet_to_add.cls_is_possible_to_enable_this_fet_with_this_infos(self._fec_column_datas_infos ):
                    if (        class_fet_to_add.fet_is_encoder and self._fec_column_datas_infos.is_input or
                                class_fet_to_add.fet_is_decoder and self._fec_column_datas_infos.is_output ):
                        list_class_fet_enabled_for_this_fec.append(class_fet_to_add)
                    else:
                        # we skip the FET because it have not the enocoder/decoder matching the input/output
                        pass

        self._list_class_enabled_fet = list_class_fet_enabled_for_this_fec
        self._set_cost_of_enabled_fets()

        # some FET cannot be set if there is not enough rows for example (LANGUAGE require 5 rows min)
        if list_class_fet_enabled_for_this_fec == [ ]:
            logger.warning( f"FET Minimum have defined no FET at all for this column {self._fec_column_datas_infos}" )
        # check the validity of the FEC MINIMUM
        elif not self._check_if_current_configuration_is_valid( ):
            logger.error( f"for the column {self._fec_column_datas_infos.name} invalid configuration {list_class_fet_enabled_for_this_fec}" )


    def _set_configuration_best_having_column_budget(self,
            budget_for_this_fec: int,
            nn_engine_to_use_in_trial,
    ) -> NoReturn:
        """
        Search the combination of the FeatureEngineeringTemplates
         - having a cost less than the argument : budget_for_this_fec
         - having maximum performance (lower loss)

        the combination of FET is a list stored in : self._list_class_enabled_fet
        """
        if ENABLE_LOGGER_DEBUG_FeatureEngineeringConfiguration: logger.debug( f"Searching FEC best configuration for column ({self._fec_column_datas_infos.name} " )


        # ======> for a output column => we will choose the most expensive FET within budget_for_this_fec
        if self._fec_column_datas_infos.is_output:

            if self._fec_column_datas_infos.datatype is DatasetColumnDataType.DATETIME:
                # output column is a Date+Time , we will search FET Date + Fet Time
                cost_found = 0
                most_expensive_fet_found_1 = None
                most_expensive_fet_found_2 = None
                for one_base_fet_name in LIST_ALL_FET_NAMES_FEC_SELECTABLE:
                        class_fet_to_check_1 = getattr(FeatureEngineeringTemplate, f"{one_base_fet_name}Date")
                        class_fet_to_check_2 = getattr(FeatureEngineeringTemplate, f"{one_base_fet_name}Time")
                        if ((class_fet_to_check_1.cls_is_possible_to_enable_this_fet_with_this_infos(self._fec_column_datas_infos ) and
                             cost_found < class_fet_to_check_1.cls_get_activation_cost( self._fec_column_datas_infos ) <= budget_for_this_fec ) and
                            (class_fet_to_check_2.cls_is_possible_to_enable_this_fet_with_this_infos(self._fec_column_datas_infos ) and
                             cost_found < class_fet_to_check_2.cls_get_activation_cost( self._fec_column_datas_infos ) <= budget_for_this_fec )):
                            most_expensive_fet_found_1 = class_fet_to_check_1
                            most_expensive_fet_found_2 = class_fet_to_check_2
                            cost_found = (most_expensive_fet_found_1.cls_get_activation_cost( self._fec_column_datas_infos ) +
                                                most_expensive_fet_found_2.cls_get_activation_cost( self._fec_column_datas_infos ) )
                if not most_expensive_fet_found_1 or not most_expensive_fet_found_2:
                    # apparently not enough budget for any FET so we set the minimum configuration
                    self._set_configuration_simple_minimum( )
                    return
                else:
                    self._list_class_enabled_fet = [ getattr(FeatureEngineeringTemplate, most_expensive_fet_found_1 ) ,
                                                                    getattr(FeatureEngineeringTemplate, most_expensive_fet_found_2 ) ]
                    self._set_cost_of_enabled_fets()
                    return

            else:
                # output column is not a Date+Time , we will search ONE best FEC
                cost_found = 0
                most_expensive_fet_found = None
                for one_base_fet_name in LIST_ALL_FET_NAMES_FEC_SELECTABLE:
                    class_fet_to_check = getattr(
                                FeatureEngineeringTemplate,
                                f"{one_base_fet_name}" + self._fec_column_datas_infos.datatype.name.capitalize()
                    )
                    if (class_fet_to_check.cls_is_possible_to_enable_this_fet_with_this_infos(self._fec_column_datas_infos ) and
                            cost_found < class_fet_to_check.cls_get_activation_cost( self._fec_column_datas_infos ) <= budget_for_this_fec ):
                        most_expensive_fet_found = class_fet_to_check
                        cost_found = class_fet_to_check.cls_get_activation_cost( self._fec_column_datas_infos )

                if not most_expensive_fet_found:
                    # apparently not enough budget for any FET so we set the minimum configuration
                    self._set_configuration_simple_minimum( )
                    return
                else:
                    # store the fet found in self
                    self._list_class_enabled_fet = [ most_expensive_fet_found ] #self._list_class_enabled_fet = [ getattr(FeatureEngineeringTemplate, most_expensive_fet_found ) ]
                    self._set_cost_of_enabled_fets()
                    return


        # ======>  for a input columns we will choose a combination of FET activatable for this column within budget_for_this_fec
        list_class_fet_activatable = []
        for one_base_fet_name in LIST_ALL_FET_NAMES_FEC_SELECTABLE:
            if self._fec_column_datas_infos.datatype is DatasetColumnDataType.DATETIME:
                class_fet_to_add_1 = getattr(FeatureEngineeringTemplate, f"{one_base_fet_name}Date")
                class_fet_to_add_2 = getattr(FeatureEngineeringTemplate, f"{one_base_fet_name}Time")
                if (
                        class_fet_to_add_1.cls_is_possible_to_enable_this_fet_with_this_infos(self._fec_column_datas_infos ) and
                        class_fet_to_add_2.cls_is_possible_to_enable_this_fet_with_this_infos(self._fec_column_datas_infos ) and
                        class_fet_to_add_1.cls_get_activation_cost( self._fec_column_datas_infos ) +
                        class_fet_to_add_2.cls_get_activation_cost( self._fec_column_datas_infos ) <= budget_for_this_fec):
                    list_class_fet_activatable.append(class_fet_to_add_1)
                    list_class_fet_activatable.append(class_fet_to_add_2)
            else:
                class_fet_to_add = getattr(FeatureEngineeringTemplate,
                                           f"{one_base_fet_name}" + self._fec_column_datas_infos.datatype.name.capitalize())
                if (class_fet_to_add.cls_is_possible_to_enable_this_fet_with_this_infos(self._fec_column_datas_infos ) and
                    class_fet_to_add.cls_get_activation_cost( self._fec_column_datas_infos ) <= budget_for_this_fec):
                    list_class_fet_activatable.append(class_fet_to_add)

        list_names_fet_activatable = [
            fet_class.__name__
            for fet_class in list_class_fet_activatable
        ]

        if list_names_fet_activatable == [ ]:
            # apparently not enough budget for any FET so we set the minimum configuration
            self._set_configuration_simple_minimum( )
            return

        if len(list_names_fet_activatable) == 1:
            # no choice here but to use the only available FET
            # store the solution found in self
            self._list_class_enabled_fet = [ getattr(FeatureEngineeringTemplate, list_names_fet_activatable[0]) ]
            self._set_cost_of_enabled_fets()
            return


        # for the SolutionFinder make the list of information-inputs
        dict_solution_score_constants = { "budget_for_this_fec": budget_for_this_fec, }
        dict_solution_score_constants.update( self.get_column_data_overview_information( ) )
        dict_solution_score_constants.update( nn_engine_to_use_in_trial._machine.get_machine_overview_information(
                                                                                                                                    with_base_info=True,
                                                                                                                                    with_fec_encdec_info=False,
                                                                                                                                    with_nn_model_info=False,
                                                                                                                                    with_training_infos=False,
                                                                                                                                    with_training_cycle_result=False ) )
        #  for the SolutionFinder make the list of OUTPUTS
        dict_solution_score_score_evaluation = {
            "Result_loss_scaled": "---(70%)",
            "Result_epoch_done_percent" : "+++(20%)",
            "Result_fec_cost_percent_budget": "---(10%)",
        }
        #  for the SolutionFinder make the list of FET inside the INPUTS VARIABLES but with a prefix to distinguish them from others inputs for later to extract them
        dict_solution_score_FET_possible_values = {
                        (EXPERIMENTER_FET_PREFIX_FET_NAME+fet_name): [False, True]
                        for fet_name in list_names_fet_activatable
                        }

        from ML import SolutionScore, ExperimenterColumnFETSelector, SolutionFinder
        solution_finder = SolutionFinder(f"ColumnFETSelector--Budget={budget_for_this_fec}--FETs=({'+'.join( sorted( list_names_fet_activatable ))})")
        solution_fet_found = solution_finder.find_solution(
            dict_solution_score_constants,
            dict_solution_score_FET_possible_values,
            SolutionScore(dict_solution_score_score_evaluation),
            ExperimenterColumnFETSelector(
                                                                 nn_engine_to_use_in_trial= nn_engine_to_use_in_trial,
                                                                 column_datas_infos=self._fec_column_datas_infos,
                                                                 fec_budget_max=budget_for_this_fec,
                                                                ),
            )

        if not solution_fet_found or not any(solution_fet_found.values()):
            logger.warning(f"SolutionFinder have returned all FET disabled for the column ({self._fec_column_datas_infos.name}) with budget {budget_for_this_fec} - Reverting to default minimum FEC for this column - solution_fet_found:{solution_fet_found}")
            self._set_configuration_simple_minimum( )
            return

        # we have added before a prefix to the inputs-variables, so we need to remove it now
        solution_fet_found_renamed = {key[len(EXPERIMENTER_FET_PREFIX_FET_NAME):] if key.startswith(EXPERIMENTER_FET_PREFIX_FET_NAME) else key: value for key, value in solution_fet_found.items()}
        # store the solution found in self
        self._list_class_enabled_fet = [
                        getattr(FeatureEngineeringTemplate, fet_name)
                            for fet_name, fet_status_enabled in solution_fet_found_renamed.items()
                                if fet_name in list_names_fet_activatable and fet_status_enabled]

        # We add the FET specific to handle missing data
        if self._fec_column_datas_infos.missing_percentage > 0:
            if self._fec_column_datas_infos.datatype is DatasetColumnDataType.DATETIME:
                self._list_class_enabled_fet.append( getattr(FeatureEngineeringTemplate, FET_TO_ENABLE_WHEN_NONE_PRESENT + "Date" ) )
                self._list_class_enabled_fet.append( getattr(FeatureEngineeringTemplate, FET_TO_ENABLE_WHEN_NONE_PRESENT + "Time" ) )
            else:
                self._list_class_enabled_fet.append( getattr(FeatureEngineeringTemplate, FET_TO_ENABLE_WHEN_NONE_PRESENT + self._fec_column_datas_infos.datatype.name.capitalize( ) ) )

        # compute the total cost for all the FET enabled
        self._set_cost_of_enabled_fets()

        # ==================================================
        # record in MachineEasyAutoML the delay of finding the best FET
        from ML import MachineEasyAutoML
        MachineEasyAutoML_fec = MachineEasyAutoML( "__Results_Find_Best_FEC__" )
        dict_column_overview = self.get_column_data_overview_information( )
        dict_column_overview.update( {"budget_for_this_fec": budget_for_this_fec } )
        # better than solution_with_all_fet_found will always contains all possible FET because solution_fet_found_renamed contains only some
        solution_with_all_fet_found = { fetname:0 for fetname in LIST_ALL_FET_NAMES_FEC_SELECTABLE }
        solution_with_all_fet_found.update( solution_fet_found_renamed )
        MachineEasyAutoML_fec.learn_this_inputs_outputs(
            inputsOnly_or_Both_inputsOutputs = dict_column_overview ,
            outputs_optional = {
                "result_delay sec":  solution_finder.result_delay_sec,
                "result_solution_fets_found": solution_fet_found_renamed,
                "result_evaluate_count_better_score" : solution_finder.result_evaluate_count_better_score,
                "result_best_solution_final_score" : solution_finder.result_best_solution_final_score,
                "result_evaluate_count_run" : solution_finder.result_evaluate_count_run,
                "result_shorter_cycles_enabled" : solution_finder.result_shorter_cycles_enabled
            })

        if not self._check_if_current_configuration_is_valid( ):
            logger.error( f"it is impossible but _set_configuration_best_having_column_budget generated an invalid configuration {self._list_class_enabled_fet}")



    def _set_cost_of_enabled_fets(self) -> NoReturn:

        if not self._check_if_current_configuration_is_valid():
            return

        self._fec_cost = sum(
            fet_class.cls_get_activation_cost(self._fec_column_datas_infos)
                for fet_class in self._list_class_enabled_fet)


    def _check_if_current_configuration_is_valid(self) -> bool:
        """
        check if the current list of fet enabled for this FEC are all possible to enable for this FEC (mostly by checking the datatype)
        """

        # a column can have no FET enabled at all
        # ??? output columns here ?
        if self._list_class_enabled_fet == []:
            return True

        return all(
            fet_class.cls_is_possible_to_enable_this_fet_with_this_infos(self._fec_column_datas_infos )
            for fet_class in self._list_class_enabled_fet
        )


    def get_column_data_overview_information(self) -> dict:
        """
        give all possible information about a column - this will be used to determine the FET to enable
        IMPORTANT : we must not provide any information about the FET enabled because it will disturb the NN FEC selector  - only fixed data (context) from MDC should be here
        IMPORTANT : the values of the keys must be single values : string/bool/float , cannot be list/dict

        :return: Returns a flat dict (dict values must be float/bool/string never lists or dicts)
        """

        _column_data_overview_information = dict(self._fec_column_datas_infos._asdict())

        del _column_data_overview_information[ "fet_list" ]    # no list and no infos about previous fet selection

        # _column_data_overview_information.update(
        #     {
        #         "is_column_dataset_fill_missing_value": self._fec_column_datas_infos.is_dataframe_fill_missing_values_enabled and self._fec_column_datas_infos.missing_percentage > 0,
        #         "is_column_dataset_skip_missing_value": self._fec_column_datas_infos.is_dataframe_skip_values_enabled and self._fec_column_datas_infos.missing_percentage > 0,
        #         "is_column_dataset_predict_missing_value": self._fec_column_datas_infos.is_dataframe_predict_values_enabled and self._fec_column_datas_infos.missing_percentage > 0,
        #     }
        # )

        return _column_data_overview_information

