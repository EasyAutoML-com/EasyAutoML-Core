from typing import List, Optional, NoReturn, Union
import copy
from functools import lru_cache
import numpy as np
import pandas as pd
import random
import pickle

from ML import MachineDataConfiguration, FeatureEngineeringConfiguration, Machine, FeatureEngineeringTemplate, EasyAutoMLDBModels
from SharedConstants import *

from ML import __getlogger

logger = __getlogger()


class EncDec:
	"""
	EncDec can Encode and decode the dataset for the NeuralNetwork
	First it need to create the configuration from the dataset
	Then using the configuration it can encode/decode ( to do this it also use the FET.methods )

	EncDec work after the module FeatureEngineeringConfiguration have generated his configuration
	The FeatureEngineeringConfiguration define which FeatureEngineeringTemplate will be enabled for every PreEncoded Dataset columns (experience_inputs and outputs)
	Then EncDec will create the EncDec.configuration

	EncDec can perform 3 operations :
		Encode a PreEncoded Dataset
		Decode a PreEncoded Dataset
		Create the configuration information to encode and decode a PreEncoded Dataset

	EncDec.configuration contains all scaling, multiplexing information to encode and decode.
	To create the global EncDec.configuration EncDec use FeatureEngineeringTemplate methods of each enabled FET
	also to encode and  decode EncDec use FeatureEngineeringTemplate methods of each enabled FET

	self._enc_dec_configuration =
			{
				pre_encoded_column_name:
					{
					"is_input":
					"is_output":
					#"is_with_FETIsNone": ((if this column have missing values))
					"column_datatype_enum":
					"column_datatype_name":
					"fet_list":   [
										{	"fet_column_name",
											"fet_class" : ,
											"fet_class_name" :
											"list_encoded_columns_name":
											"fet_serialized_config" :
										}
									]
					}
			}
	"""


	def __init__( self, machine: Machine, dataframe_pre_encoded: Optional[ pd.DataFrame ] = None ):
		"""
		Create or load a EncDec object

		Creating a EncDec is the operation to load or generate the configuration
		With EncDec configuration it is possible to encdec.encode and encdec.decode
		The configuration must be saved it into the machine_source with the method set_machine_properties()

		:param machine: Machine object (object need need to have information about MDC user data)
		:param dataframe_pre_encoded: if present then the EncDec configuration will be created by using MDC and the dataframe_pre_encoded
														if not present the configuration will be loaded from machine
		"""
		self._machine = machine
		self._enc_dec_configuration = None

		self._columns_input_encoded_count = None
		self._columns_output_encoded_count = None

		if isinstance( machine, Machine ) and dataframe_pre_encoded is None:
			self._init_load_configuration( )

		elif isinstance( machine, Machine ) and isinstance( dataframe_pre_encoded, pd.DataFrame ):
			self._init_create_configuration( dataframe_pre_encoded )

		else:
			error_msg = f"Combination of EncDec parameters given in the INIT are not possible - _enc_dec_configuration cannot be loaded or created using this input parameters"
			logger.error( error_msg )
			raise ValueError( error_msg )


	def _init_load_configuration( self ) -> NoReturn:
		"""
		load EncDec configuration from the Machine
		"""
		if ENABLE_LOGGER_DEBUG_EncDec: logger.debug( f"Loading EncDec for {self._machine} starting" )

		self._columns_input_encoded_count = self._machine.db_machine.enc_dec_columns_info_input_encode_count
		self._columns_output_encoded_count = self._machine.db_machine.enc_dec_columns_info_output_encode_count

		self._enc_dec_configuration = self._load_config_from_machine_and_deserialize_it()

		#if ENABLE_LOGGER_DEBUG: logger.debug(f"{machine} EncDec loaded")


	def _init_create_configuration( self, dataframe_pre_encoded: pd.DataFrame ) -> NoReturn:
		"""
		create the EncDec configuration in self._enc_dec_configuration using MDC information and dataframe_pre_encoded

		:param dataframe_pre_encoded: EncDec need to check the values in the pre_encoded_dataframe to create the configuration (for example we need for labels columns the list of all possible labels)
		:return: None
		"""

		if ENABLE_LOGGER_DEBUG_EncDec:
			logger.debug(f"Creating EncDec configuration for {self._machine} starting" )

		mdc_ = MachineDataConfiguration( self._machine )
		fe_ = FeatureEngineeringConfiguration( self._machine )

		columns_name_input = mdc_.columns_name_input
		columns_name_output = mdc_.columns_name_output
		columns_datatype = mdc_.columns_data_type

		if not any( columns_name_input.values( ) ):
			logger.error( "for creation EncDec _enc_dec_configuration we need at least one input column" )
		if not any( columns_name_output.values( ) ):
			logger.error( "for creation EncDec _enc_dec_configuration we need at least one output column" )

		# FET need several values to work correctly
		if len(dataframe_pre_encoded) < 5:
			logger.error(f"There is less than 5 rows , only {len(dataframe_pre_encoded)} rows to create encdec config ")

		# if no FET at all , FE was probably never defined
		if not fe_._activated_fet_list_per_column:
			logger.error( f"No FET at all defined, is FE have been initialized ?")

		fet_instance_per_columns = dict( )
		for column_name, fet_names_list in fe_._activated_fet_list_per_column.items():
			if column_name not in dataframe_pre_encoded:
				logger.error( f"Dataframe is missing the column : {column_name} (check if it is df_pre_encoded , not user_df)" )

			# preparing the data for doing the configuration
			column_data_nparray = None
			column_data_nparray_date = None
			column_data_nparray_time = None
			# If the column is DATETIME , we need to SPLIT DATETIME for  the FET because there is no FET-DATETIME
			if columns_datatype[ column_name ] == DatasetColumnDataType.DATETIME:
				column_data_nparray_date = dataframe_pre_encoded[ column_name ].dt.date.to_numpy( )
				column_data_nparray_time = dataframe_pre_encoded[ column_name ].dt.time.to_numpy( )
			elif columns_datatype[ column_name ] == DatasetColumnDataType.DATE:
					column_data_nparray_date = dataframe_pre_encoded[ column_name ].to_numpy()
			elif columns_datatype[ column_name ] == DatasetColumnDataType.TIME:
					column_data_nparray_time = dataframe_pre_encoded[ column_name ].to_numpy()
			else:
				column_data_nparray = dataframe_pre_encoded[ column_name ].to_numpy()

			column_info = fe_.get_all_column_datas_infos( column_name )

			# create the configuration for each FET of the column
			fet_instances_list = []
			for fet_name in fet_names_list:
				# run the FET INIT for all fet of the column
				fet_instance = None
				if IS_RUNNING_IN_DEBUG_MODE:
					if fet_name.upper( ).endswith( "DATE" ):
						# for FET-DATE we use prepared data DATE
						fet_instance = getattr( FeatureEngineeringTemplate, fet_name )( column_data_nparray_date, column_info )
					elif fet_name.upper( ).endswith( "TIME" ):
						# for FET-TIME we use prepared data TIME
						fet_instance = getattr( FeatureEngineeringTemplate, fet_name )( column_data_nparray_time, column_info )
					else:
						fet_instance = getattr( FeatureEngineeringTemplate, fet_name )( column_data_nparray, column_info )
				else:
					try:
						if fet_name.upper( ).endswith( "DATE" ):
							# for FET-DATE we use prepared data DATE
							fet_instance = getattr( FeatureEngineeringTemplate, fet_name )( column_data_nparray_date, column_info )
						elif fet_name.upper( ).endswith( "TIME" ):
							# for FET-TIME we use prepared data TIME
							fet_instance = getattr( FeatureEngineeringTemplate, fet_name )( column_data_nparray_time, column_info )
						else:
							fet_instance = getattr( FeatureEngineeringTemplate, fet_name )( column_data_nparray, column_info )
					except Exception as e:
						# error while doing encdec configuration, we will skip this FET but continue to do configuration
						# this error mean for example that the data have changed since MDC and the informations in data_columns_infos are not accurate and a wrong FET have been enabled
						self._store_column_error_in_machine(
							column_name,
							f"Error while starting training because data in column '{column_name}' - The Machine '{self._machine}' will restart the training later from the beginning to adapt to the new data" ,
							f"Error while doing configuration of fet '{fet_name}' in column '{column_name}' of machine '{self._machine}' - skipping FET. Error is {e} " )
						fet_instance = None

				# fet created without fatal  error - add to the list of fet, check if we have to log warning
				if fet_instance:
					fet_instances_list.append( fet_instance )
					if fet_instance.warning_message:
						self._store_column_warning_in_machine(
							column_name,
							f"Warning while starting training because the data are out of range in column '{column_name}' - The Machine '{self._machine}' will restart the training later to adapt to the new data" ,
							f"Warning while doing configuration of fet '{fet_name}' in column '{column_name}' of machine '{self._machine}'. FET will still used. Warning is : {fet_instance.warning_message} " )

			fet_instance_per_columns[ column_name ] = fet_instances_list

		del mdc_, fe_

		# store ENCDEC configurations for all columns
		columns_enc_dec_config_info = {}
		for pre_encoded_column_name in dataframe_pre_encoded.columns:
			if pre_encoded_column_name in fet_instance_per_columns:
				fet_list = [
					{
						"fet_column_name": pre_encoded_column_name,
						"fet_class": one_column_fet,
						"fet_class_name": type(one_column_fet).__name__,
						"list_encoded_columns_name": one_column_fet.get_list_encoded_columns_name(pre_encoded_column_name ),
						"fet_serialized_config": one_column_fet.serialize_fet_configuration(),
					} for one_column_fet in fet_instance_per_columns[pre_encoded_column_name]
				]
			else:
				fet_list = []
				if pre_encoded_column_name in columns_datatype and columns_datatype[ pre_encoded_column_name ] == DatasetColumnDataType.IGNORE:
					# this input column have no fet at all
					pass
				elif pre_encoded_column_name in columns_name_output and columns_name_output[ pre_encoded_column_name ]:
					# this output column have no fet at all
					logger.error( f"The column '{pre_encoded_column_name}' have no fet and is an output")

			enc_dec_config_info = {
				pre_encoded_column_name: {
					"is_input": columns_name_input[pre_encoded_column_name],
					"is_output": columns_name_output[pre_encoded_column_name],
					"column_datatype_enum": columns_datatype[pre_encoded_column_name],
					"column_datatype_name": columns_datatype[ pre_encoded_column_name].name,
					"fet_list": fet_list
				}
			}
			columns_enc_dec_config_info.update( enc_dec_config_info )

		self._enc_dec_configuration = columns_enc_dec_config_info

		self._columns_input_encoded_count, self._columns_output_encoded_count = self._get_input_and_output_encoded_for_ai_columns_count( columns_enc_dec_config_info )


	def save_configuration_in_machine( self ) -> "EncDec":
		"""
		Stores EncDec self.properties inside the machine_source

		:return: EncDec computed
		"""

		db_machine = self._machine.db_machine
		db_machine.enc_dec_columns_info_input_encode_count = (self._columns_input_encoded_count)
		db_machine.enc_dec_columns_info_output_encode_count = (self._columns_output_encoded_count)

		self._serialize_configuration_and_save_it_in_machine( self._enc_dec_configuration, self._machine )

		if ENABLE_LOGGER_DEBUG_EncDec: logger.debug( f"EncDec attributes was saved inside {self._machine} machine_source model" )

		return self


	def _store_column_error_in_machine(
			self,
			column_name: str,
			error_message_user: str,
			error_message_internal: str,
	) -> NoReturn:
		"""
		add an error in the machine._columns_errors
		we save it trough SQL because the machine may not be saved later

		:param column_name: the column where to put the error message
		:param error_message_user: the error message to put in the column for the user (no technical details)
		:param error_message_internal: the error message detailled to log for EasyAutoML.com
		"""
		if ENABLE_LOGGER_DEBUG_EncDec:
			logger.warning( f"Error for machine '{self._machine.id}' in column '{column_name}' : '{error_message_internal}'")

		error_message_to_store = error_message_internal if DEBUG_WRITE_ERROR_CLEARLY_INSIDE_MACHINES else error_message_user
		# got some error while saving \\ in the json field
		error_message_to_store = error_message_to_store.replace( "\\" , "/")

		self._machine.store_error( column_name , error_message_to_store )
		# this instance of machine will maybe not be saved, so we update the warning directly inside the database
		self._machine.db_machine.update_field_directly_by_sql( "machine_columns_errors" )

		# in case of error all the training configurations will be rebuilt
		self._machine.db_machine.machine_is_re_run_mdc = True
		self._machine.db_machine.update_field_directly_by_sql( "machine_is_re_run_mdc" )


	def _store_column_warning_in_machine(self,
			column_name: str,
			warning_message_user: str,
			warning_message_internal: str):
		"""
		add a warning in the machine._columns_warning
		we save it trough SQL because the machine may not be saved later

		:param column_name: the column where to put the warning message
		:param warning_message_user: the warning message to put in the column
		"""
		if ENABLE_LOGGER_DEBUG_EncDec:
			logger.warning(f"Warning for machine '{self._machine.id}' in column '{column_name}' : '{warning_message_internal}'")

		warning_messages_to_store = warning_message_internal if DEBUG_WRITE_ERROR_CLEARLY_INSIDE_MACHINES else warning_message_user
		# got some error while saving \\ in the json field
		warning_messages_to_store = warning_messages_to_store.replace( "\\" , "/")

		self._machine.store_warning( column_name ,  warning_messages_to_store )
		# this instance of machine will maybe not be saved, so we update the warning directly inside the database
		self._machine.db_machine.update_field_directly_by_sql( "machine_columns_warnings" )
		
		if "[is_re_run_mdc]" in warning_message_internal:
			self._machine.db_machine.machine_is_re_run_mdc = True
			self._machine.db_machine.update_field_directly_by_sql( "machine_is_re_run_mdc" )
		elif "[is_re_run_ici]" in warning_message_internal:
			self._machine.db_machine.machine_is_re_run_ici = True
			self._machine.db_machine.update_field_directly_by_sql( "machine_is_re_run_ici" )
		elif "[is_re_run_fe]" in warning_message_internal:
			self._machine.db_machine.machine_is_re_run_fe = True
			self._machine.db_machine.update_field_directly_by_sql( "machine_is_re_run_fe" )
		elif "[is_re_run_enc_dec]" in warning_message_internal:
			self._machine.db_machine.machine_is_re_run_enc_dec = True
			self._machine.db_machine.update_field_directly_by_sql( "machine_is_re_run_enc_dec" )
		elif "[is_re_run_nn_config]" in warning_message_internal:
			self._machine.db_machine.machine_is_re_run_nn_config = True
			self._machine.db_machine.update_field_directly_by_sql( "machine_is_re_run_nn_config" )
		elif "[is_re_run_model]" in warning_message_internal:
			self._machine.db_machine.machine_is_re_run_model = True
			self._machine.db_machine.update_field_directly_by_sql( "machine_is_re_run_model" )



	def encode_for_ai( self, pre_encoded_dataframe: pd.DataFrame ) -> pd.DataFrame:
		"""
		After Pre_encoding we need to encode_for_ai to convert all data to columns with numeric 0-1 for the neural network model
		usually we encode the inputs, the outputs or both

		:param pre_encoded_dataframe: pre_encoded_dataframe to encode_for_ai
		:return: the pre_encoded_dataframe encoded_for_ai
		"""


		def encode_for_ai_encode_one_column_one_fet( machine_id, fet_instance, column_name, values_to_encode_nparray, list_encoded_columns  ) -> pd.DataFrame:
			"""
			execute the encoding and return a pandas dataframe with the right columns names title (but not yet the correct index)
			"""

			# no row to encode so we return empty dataframe
			if len(values_to_encode_nparray)==0:
				return pd.DataFrame( columns=list_encoded_columns )

			if IS_RUNNING_IN_DEBUG_MODE:
				# if debugger is active we will disable the warning system and trigger debugger
				np_encoded_data = fet_encoder_caching_system_do_encode_with_cache( machine_id, fet_instance, column_name, values_to_encode_nparray )
			else:
				try:
					np_encoded_data = fet_encoder_caching_system_do_encode_with_cache( machine_id, fet_instance, column_name, values_to_encode_nparray )
				except Exception as e:
					msg = f"Python Error in FET ENCODE {fet_instance} for the column {column_name}, error is {e} - [is_re_run_enc_dec] "
					logger.warning( msg )
					fet_instance.add_warning( msg )
					return None

			if np_encoded_data is None or len(np_encoded_data)==0:
				logger.error( "We have no encoding data but this should never happend as we always have data to encode" )

			pd_encoded_data = pd.DataFrame( np_encoded_data, columns=list_encoded_columns )
			# control of the result columns count
			np_encoded_columns_count = 1 if len( np_encoded_data.shape ) == 1 else np_encoded_data.shape[ 1 ]
			if np_encoded_columns_count != len( list_encoded_columns ):
				logger.error(     f"EncDec encoded the column {column_name} to {np_encoded_columns_count} columns, "
										f"but encdec configuration is {len( list_encoded_columns )} columns : {list_encoded_columns}" )
			return pd_encoded_data


		if not isinstance( pre_encoded_dataframe, pd.DataFrame ):
			logger.error( "input pre_encoded_dataframe for EncDec encoding must be of type pd.DataFrame" )

		if not self._enc_dec_configuration:
			raise RuntimeError( "you need to create an EncDec configuration before encoding" )

		if ENABLE_LOGGER_DEBUG_EncDec: logger.debug( f"Encoding for AI dataframe (pre_encoded) of {pre_encoded_dataframe.shape[ 0 ]} rows X {pre_encoded_dataframe.shape[ 1 ]} cols" )


		all_columns_results_df = None
		for column_name in pre_encoded_dataframe.columns:
			if ( column_name not in self._enc_dec_configuration or
				self._enc_dec_configuration[ column_name ][ "fet_list" ] is None or
				self._enc_dec_configuration[ column_name ][ "fet_list" ] == [ ]
			):
				continue # nothing to encode

			values_to_encode_nparray = None
			values_to_encode_nparray_date = None
			values_to_encode_nparray_time = None
			# If the column is DATETIME , we need to SPLIT DATETIME for  the FET because there is no FET-DATETIME
			if self._enc_dec_configuration[ column_name ]['column_datatype_enum'] == DatasetColumnDataType.DATETIME:
				values_to_encode_nparray_date = pre_encoded_dataframe[ column_name ].dt.date.to_numpy( )
				values_to_encode_nparray_time = pre_encoded_dataframe[ column_name ].dt.time.to_numpy( )
			elif self._enc_dec_configuration[ column_name ][ 'column_datatype_enum' ] == DatasetColumnDataType.DATE:
					values_to_encode_nparray_date = pre_encoded_dataframe[ column_name ].to_numpy()
			elif self._enc_dec_configuration[ column_name ][ 'column_datatype_enum' ] == DatasetColumnDataType.TIME:
					values_to_encode_nparray_time = pre_encoded_dataframe[ column_name ].to_numpy()
			else:
				values_to_encode_nparray = pre_encoded_dataframe[ column_name ].to_numpy()

			for encdec_config_fet_list in self._enc_dec_configuration[ column_name ][ "fet_list" ]:
				fet_instance = encdec_config_fet_list[ "fet_class" ]
				fet_instance.clear_warning_message()
				list_encoded_columns = encdec_config_fet_list[ "list_encoded_columns_name" ]

				if str( fet_instance ).upper().endswith( "DATE" ) or str( fet_instance ).upper().endswith( "DATE()"):
					# for FET-DATE we use prepared data DATE
					pd_encoded_data = encode_for_ai_encode_one_column_one_fet( self._machine.db_machine.id , fet_instance, column_name, values_to_encode_nparray_date, list_encoded_columns  )
				elif str( fet_instance ).upper().endswith( "TIME" ) or str( fet_instance ).upper().endswith( "TIME()"):
					# for FET-TIME we use prepared data TIME
					pd_encoded_data = encode_for_ai_encode_one_column_one_fet( self._machine.db_machine.id , fet_instance, column_name, values_to_encode_nparray_time, list_encoded_columns  )
				else:
					pd_encoded_data = encode_for_ai_encode_one_column_one_fet( self._machine.db_machine.id , fet_instance, column_name, values_to_encode_nparray, list_encoded_columns  )

				if fet_instance.warning_message:
					# problem occurred in FET.encoder but we can continue
					self._store_column_warning_in_machine(
						column_name,
						f"Problem in column '{column_name}' - We will restart the training to train the machine to the new datas" ,
						f"Unable to encode column '{column_name}' because {fet_instance.warning_message} " )

				# concatenate all results into one single pandas dataframe
				if all_columns_results_df is not None:
					all_columns_results_df = pd.concat( [ all_columns_results_df , pd_encoded_data ] , axis=1)
				else:
					all_columns_results_df = pd_encoded_data

		# all columns , all FET encoders , done
		if all_columns_results_df is None:
			logger.error( "The encoded dataframe is empty - probably because there is no FET in any columns" )

		# set same line numbers as data to encode
		all_columns_results_df.index = pre_encoded_dataframe.index

		# check the column count is correct - it can be inputs,outputs or both (it is not a perfect verification but ok)
		# when doing FEC trial , EncDec is not saved in machine (to make it faster) so we cannot check and compare with machine data : fec, encdec
		if (all_columns_results_df.shape[ 1 ] != (self._columns_input_encoded_count) and
				all_columns_results_df.shape[ 1 ] != (self._columns_output_encoded_count) and
				all_columns_results_df.shape[ 1 ] != (self._columns_input_encoded_count + self._columns_output_encoded_count) ):
			missing_cols = [col for col in all_columns_results_df.columns if not any(col.startswith(f"{key}-") for key in self._enc_dec_configuration.keys() if len( self._enc_dec_configuration[ key ][ 'fet_list'] )>0)]
			missing_keys = [key for key in self._enc_dec_configuration.keys() if len( self._enc_dec_configuration[ key ][ 'fet_list'] )>0 and not any(col.startswith(f"{key}-") for col in all_columns_results_df.columns )]
			logger.error( f"the dataframe encoded_for_ai have {all_columns_results_df.shape[ 1 ]} cols, but it do not match encdec._columns_input_encoded_count={self._columns_input_encoded_count} and/or encdec._columns_output_encoded_count={self._columns_output_encoded_count} , missing_cols={missing_cols} , missing_keys={missing_keys} "  )

		#if ENABLE_LOGGER_DEBUG: logger.debug(f"encode_for_ai pre_encoded_dataframe done")
		return all_columns_results_df


	def decode_from_ai( self, data_encoded_from_ai: pd.DataFrame ) -> pd.DataFrame:
		"""
		we need to convert all data of columns with numeric 0-1 from the neural network model to pre_encoded_dataframe decoded_from_ai which will be post_decoded to be like user pre_encoded_dataframe
		Important : it do process only outputs columns, inputs columns are discarded
		:param data_encoded_from_ai: pre_encoded_dataframe returned by neural network, all values are from 0-1
		:return: the pre_encoded_dataframe decoded_from_ai
		"""
		if not isinstance( data_encoded_from_ai, pd.DataFrame ):
			logger.error( "input pre_encoded_dataframe for EncDec decoding must be of type pd.DataFrame" )

		if ENABLE_LOGGER_DEBUG_EncDec: logger.debug( f"Decoding dataframe (encoded_from_ai) of {data_encoded_from_ai.shape[ 0 ]} rows X {data_encoded_from_ai.shape[ 1 ]} cols" )

		#check the column count is correct
		#check the column count is correct - it can be inputs,outputs or both
		# if (        data_encoded_from_ai.shape[ 1 ] != self._machine.db_machine.enc_dec_columns_info_input_encode_count and
		# 			data_encoded_from_ai.shape[ 1 ] != self._machine.db_machine.enc_dec_columns_info_output_encode_count and
		# 			data_encoded_from_ai.shape[ 1 ] != self._machine.db_machine.enc_dec_columns_info_input_encode_count + self._machine.db_machine.enc_dec_columns_info_output_encode_count
		# ):
		# 	logger.error( f"the dataframe to decode_from_ai have {data_encoded_from_ai.shape[ 1 ]} cols, but we was expecting {self._machine.db_machine.enc_dec_columns_info_input_encode_count}" )
		if (        data_encoded_from_ai.shape[ 1 ] != self._columns_input_encoded_count and
					data_encoded_from_ai.shape[ 1 ] != self._columns_output_encoded_count and
					data_encoded_from_ai.shape[ 1 ] != self._columns_input_encoded_count + self._columns_output_encoded_count
		):
			logger.error( f"the dataframe to decode_from_ai have {data_encoded_from_ai.shape[ 1 ]} cols, but we was expecting Inputs:{self._columns_input_encoded_count} or Outputs:{self._columns_output_encoded_count} or Both" )

		information_for_decoding_output_columns = { output_column_name: column_encoding_info for output_column_name, column_encoding_info in self._enc_dec_configuration.items( ) if
			column_encoding_info[ "is_output" ] }

		for info_column_name, decoding_info in information_for_decoding_output_columns.items():
			decoding_info[ "fet_list" ] = [ encdec_config_fet_list for encdec_config_fet_list in decoding_info[ "fet_list" ] if
				isinstance( encdec_config_fet_list[ "fet_class" ], FeatureEngineeringTemplate.FeatureEngineeringTemplate ) and encdec_config_fet_list[ "fet_class" ].fet_is_decoder ]
			decoding_info[ "output_encoded_columns" ] = [ encoded_column_name for encdec_config_fet_list in decoding_info[ "fet_list" ] for encoded_column_name in
				encdec_config_fet_list[ "list_encoded_columns_name" ] ]

		pandas_decoded_merged = pd.DataFrame()
		for (column_name, decoding_info) in information_for_decoding_output_columns.items( ):
			encoded_columns_name = decoding_info[ "output_encoded_columns" ]

			# in case there is several fet results to merge for the column, we will decode all fet and store in a dict dict_with_decoded_arrays_all_fet where the key is the fet_class_name
			dict_by_fet_name_nparrays_decoded_values = dict()
			for encdec_config_fet_list in decoding_info[ "fet_list" ]:
				fet_class_name = encdec_config_fet_list[ "fet_class_name" ]
				column_names_to_decode = [ column_to_decode for column_to_decode in encoded_columns_name if fet_class_name in column_to_decode ]
				fet_instance = encdec_config_fet_list[ "fet_class" ]
				fet_instance.clear_warning_message()
				try:
					dict_by_fet_name_nparrays_decoded_values[ fet_class_name ] = fet_instance.decode( data_encoded_from_ai[column_names_to_decode ].values ).ravel( )
				except Exception as e:
					msg = f"Python Error in FET decode {fet_instance} for the column {column_name}, error is {e} [is_re_run_enc_dec] "
					logger.warning( msg )
					fet_instance.add_warning( msg )

				if fet_instance.warning_message:
					# problem occurred in FET.decoder but we can continue
					self._store_column_warning_in_machine(
						column_name,
						f"Problem in column '{column_name}' - We will restart the training to train the machine to the new data" ,
						f"Unable to decode column '{column_names_to_decode}' with '{fet_instance}' because {fet_instance.warning_message} " )

			# ALL FET have decoded , and results are in dict_with_decoded_arrays_all_fet
			all_fet_names = list( dict_by_fet_name_nparrays_decoded_values.keys( ) )
			all_fet_names_count = len( all_fet_names )
			if all_fet_names_count == 0:
				logger.error( f"There is no FET to decode for the column : {column_name}")
			elif all_fet_names_count == 1:
				# no need to combine, only one FET => one single fet for this column we store the result
				pandas_decoded_merged[ column_name ] = list( dict_by_fet_name_nparrays_decoded_values.values() )[0]
			else:
				# more than one fet , so we need to merge all the fet_results by averaging result values

				how_many_fet_float = sum ( [ 1 if name.upper().endswith( "FLOAT" ) else 0 for name in all_fet_names ] )
				how_many_fet_date = sum ( [ 1 if name.upper().endswith( "DATE" ) else 0 for name in all_fet_names ] )
				how_many_fet_time = sum ( [ 1 if name.upper().endswith( "TIME" ) else 0 for name in all_fet_names ] )
				how_many_fet_label = sum ( [ 1 if name.upper().endswith( "LABEL" ) else 0 for name in all_fet_names ] )

				# checking if FET types decoded are correct
				if how_many_fet_float + how_many_fet_date + how_many_fet_time + how_many_fet_label != all_fet_names_count:
					logger.error( f"Detected in column '{column_name}'  some unknown FET DATATYPE in {all_fet_names} " )
				elif (
							(how_many_fet_float > 0 and how_many_fet_date>0) or
							(how_many_fet_float > 0 and how_many_fet_time > 0) or
							(how_many_fet_float > 0 and how_many_fet_label > 0)
				):
					logger.error( f"Detected in column '{column_name}'  some mixed FET DATATYPE in {all_fet_names} ")
				elif (
							(how_many_fet_label > 0 and how_many_fet_date>0) or
							(how_many_fet_label > 0 and how_many_fet_time > 0) or
							(how_many_fet_label > 0 and how_many_fet_float > 0)
				):
					logger.error( f"Detected in column '{column_name}'  some mixed FET DATATYPE in {all_fet_names} ")

				if how_many_fet_float == all_fet_names_count:
					# we average_similar_values for all results in dict_with_decoded_arrays_all_fet
					data_np_array = np.stack( [ v for v in dict_by_fet_name_nparrays_decoded_values.values( ) ] ).T.astype( float )
					result_average_similar_values = FeatureEngineeringTemplate.decoder_merger_average_2d_nparray_similar_values_per_rows( data_np_array )
				elif how_many_fet_label == all_fet_names_count:
					# we check all items , if they are similar we keep else we None
					data_np_array = np.stack( [ v for v in dict_by_fet_name_nparrays_decoded_values.values( ) ] ).T
					result_average_similar_values = FeatureEngineeringTemplate.decoder_merger_merge_2d_array_labels_similar_per_rows( data_np_array )
				elif how_many_fet_time == all_fet_names_count and how_many_fet_date == 0:
					# we will merge all TIME
					data_np_array = np.stack( [ v for v in dict_by_fet_name_nparrays_decoded_values.values( ) ] ).T
					result_average_similar_values = FeatureEngineeringTemplate.decoder_float_to_time(
										FeatureEngineeringTemplate.decoder_merger_average_2d_nparray_similar_values_per_rows(
													FeatureEngineeringTemplate.encoder_time_to_float(
																data_np_array ) ) )
				elif how_many_fet_date == all_fet_names_count and how_many_fet_time == 0:
					# we will merge all DATE
					data_np_array = np.stack( [ v for v in dict_by_fet_name_nparrays_decoded_values.values( ) ] ).T
					result_average_similar_values = FeatureEngineeringTemplate.decoder_float_to_date(
										FeatureEngineeringTemplate.decoder_merger_average_2d_nparray_similar_values_per_rows(
													FeatureEngineeringTemplate.encoder_date_to_float(
																data_np_array ) ) )
				elif how_many_fet_date + how_many_fet_time == all_fet_names_count:
					# we will find FET DATE / TIME -> merge all FET DATE  -> merge all FET TIME -> combine both in datetime
					list_fet_name_date =  [ name for name in all_fet_names if name.upper().endswith( "DATE" ) ]
					data_np_array = np.stack( [ v for k, v in dict_by_fet_name_nparrays_decoded_values.items( ) if k in list_fet_name_date ] ).T
					result_average_similar_values_date = FeatureEngineeringTemplate.decoder_float_to_date(
										FeatureEngineeringTemplate.decoder_merger_average_2d_nparray_similar_values_per_rows(
													FeatureEngineeringTemplate.encoder_date_to_float(
																data_np_array ) ) )
					list_fet_name_time =  [ name for name in all_fet_names if name.upper().endswith( "TIME" ) ]
					data_np_array = np.stack( [ v for k, v in dict_by_fet_name_nparrays_decoded_values.items( ) if k in list_fet_name_time ] ).T
					result_average_similar_values_time = FeatureEngineeringTemplate.decoder_float_to_time(
										FeatureEngineeringTemplate.decoder_merger_average_2d_nparray_similar_values_per_rows(
													FeatureEngineeringTemplate.encoder_time_to_float(
																data_np_array ) ) )
					result_average_similar_values = FeatureEngineeringTemplate.decoder_merger_combine_1D_nparray_date_time(
								result_average_similar_values_date ,
								result_average_similar_values_time )
				else:
					# merging several FET but not a combinaison valid
					logger.error( f"merging several FET in column '{column_name}' but combinaison of TYPE are not supported : {all_fet_names}" )

				# storing the merged FETs into the result
				pandas_decoded_merged[ column_name ] = result_average_similar_values

		if pandas_decoded_merged.empty:
			logger.error( "The output have no columns to decode ! - there is no outputs columns at all or all outputs columns have no FET-decoders enabled at all" )

		#if ENABLE_LOGGER_DEBUG: logger.debug(f"decode_from_ai done")
		return pandas_decoded_merged


	def _load_config_from_machine_and_deserialize_it( self ) -> dict:
		"""
		load configuration from the machine and convert some data type

		:return: dict Enc_Dec_Configuration
		"""
		loaded_encdec_configuration = copy.deepcopy( self._machine.db_machine.enc_dec_configuration_extfield )

		if not loaded_encdec_configuration:
			logger.error( "Unable to load EncDec configuration because database do not have any - create EncDec config first !")

		for info_column_name, column_enc_dec_info in loaded_encdec_configuration.items():
			# object column_datatype_enum was discarded when saving, now we instance it
			column_enc_dec_info[ "column_datatype_enum" ] = DatasetColumnDataType[ column_enc_dec_info[ "column_datatype_name" ] ]
			for a_fet in column_enc_dec_info[ "fet_list" ]:
				# object fet_class was discarded when saving, now we instance it
				try:
					a_fet[ "fet_class" ] = getattr( FeatureEngineeringTemplate, a_fet[ "fet_class_name" ] )( a_fet[ "fet_serialized_config" ] )
				except Exception as e:
					logger.error( f"Unable to instance FET Class {a_fet[ 'fet_class_name' ]} with config {a_fet[ 'fet_serialized_config' ]}")

		return loaded_encdec_configuration


	def _serialize_configuration_and_save_it_in_machine( self, enc_dec_configuration_to_save, machine: Machine ) -> NoReturn:
		"""
		update the enc_dec_configuration by adding some parameters
		then save it into the table

		:param enc_dec_configuration_to_save: the config to save
		:param machine: the machine where to save it
		"""

		enc_dec_configuration_to_save_serialized = copy.deepcopy( enc_dec_configuration_to_save )

		for column_enc_dec_info_key in enc_dec_configuration_to_save_serialized:
			if "column_datatype_enum" in enc_dec_configuration_to_save_serialized[column_enc_dec_info_key]:
				# column_datatype_enum is discarded then only column_datatype_name will be saved
				enc_dec_configuration_to_save_serialized[column_enc_dec_info_key][ "column_datatype_enum" ] = None
			if "fet_list" in enc_dec_configuration_to_save_serialized[column_enc_dec_info_key]:
				for fet_idx in range(len(enc_dec_configuration_to_save_serialized[column_enc_dec_info_key]["fet_list"])):
					# we discard all fet_class objects because it cannot be saved in json, but fet_class_name and fet_serialized_config are saved
					enc_dec_configuration_to_save_serialized[column_enc_dec_info_key]["fet_list"][fet_idx][ "fet_class" ] = None

		machine.db_machine.enc_dec_configuration_extfield = enc_dec_configuration_to_save_serialized


	def get_encoded_for_ai_columns_names_by_the_pre_encoded_column_name( self, pre_encoded_column_name: str ) -> List[ str ]:
		"""
		Returns the encoded columns name created from the pre_encoded column name

		:params user_column_name: the user column (from user dataframe)
		:return: all columns PRE-ENCODED (JSON) and AI-ENCODED (FET)
		"""
		if pre_encoded_column_name not in self._enc_dec_configuration:
			logger.error( f"The pre_encoded_column_name '{pre_encoded_column_name} cannot be found in the Encode_For_AI configuration , there is no FET at all for this column ? Or it is a user_dataframe json column ? ")

		return [ encoded_column for encdec_config_fet_list in self._enc_dec_configuration[ pre_encoded_column_name ][ "fet_list" ] \
							if encdec_config_fet_list for encoded_column in
									encdec_config_fet_list[ "list_encoded_columns_name" ]
						]


	def _get_input_and_output_encoded_for_ai_columns_count( self, columns_enc_dec_config_info ):
		"""
		count how many inputs and outputs columns are in the dataframe_encoded_for_ai using encDec_configuration
		:param columns_enc_dec_config_info: the EncDec.configuration
		"""
		input_encode_column_count, output_encode_column_count = 0, 0
		for column_encoding_info in columns_enc_dec_config_info.values():
			if column_encoding_info["fet_list"] is None or column_encoding_info["fet_list"] == [ ]:
				count_of_columns_after_encoding_for_ai = 0
			else:
				count_of_columns_after_encoding_for_ai = sum( len( encdec_config_fet_list[ "list_encoded_columns_name" ] ) for encdec_config_fet_list in column_encoding_info[ "fet_list" ] )

			if column_encoding_info[ "is_output" ]:
				output_encode_column_count += count_of_columns_after_encoding_for_ai

			elif column_encoding_info["is_input"]:
				input_encode_column_count += count_of_columns_after_encoding_for_ai

			else:
				pass

		return input_encode_column_count, output_encode_column_count


	def nested_data_structure_convert_isnull_to_____________________( self , nested_data_structure , value_to_replace_by ):
		# If the input is a dictionary, apply the function recursively to all its values
		if isinstance( nested_data_structure, dict ):
			return { k: self.nested_data_structure_convert_isnull_to( v, value_to_replace_by ) for k, v in nested_data_structure.items( ) }
		# If the input is a list, apply the function recursively to all its elements
		elif isinstance( nested_data_structure, list ):
			return [ self.nested_data_structure_convert_isnull_to( v, value_to_replace_by ) for v in nested_data_structure ]
		# If the input is a numpy array, replace all NaN and None values with the specified value
		elif isinstance( nested_data_structure, np.ndarray ):
			# Use pandas.isnull to detect NaN and None values inside the numpy array
			nested_data_structure[ pd.isnull( nested_data_structure ) ] = value_to_replace_by
			return nested_data_structure
		# If the input is None, return the specified value
		elif pd.isnull( nested_data_structure ):
			return value_to_replace_by
		# Otherwise, return the input unchanged
		else:
			return nested_data_structure



def fet_encoder_caching_system_do_encode_with_cache( machine_id , fet_instance, column_name, values_to_encode_nparray: np.array ) -> np.array:
	"""
	encoding the values with the FET in argument with caching
	"""
	if  (values_to_encode_nparray.size < 250 or
		DEBUG_DISABLE_FET_ENCODER_CACHE
		):
		# we do save memory and we do not cache small array, because calling fet_encoder_caching_system_do_real_caching is heavy as we need to pickle everything
		encoded_without_cache = fet_instance.encode( values_to_encode_nparray )
		return encoded_without_cache
	else:
		import hashlib
		encoded_with_cache = fet_encoder_caching_system_do_real_caching( machine_id , column_name, pickle.dumps( fet_instance ), pickle.dumps( values_to_encode_nparray ) )
		#if ENABLE_LOGGER_DEBUG_EncDec:logger.debug( f"LRU info : {fet_encoder_caching_system_do_real_caching.cache_info( )} " )
		# sometime we check if the cache is working well
		if True or True or random.randrange( 1, 1000 ) == 1:
			encoded_without_cache = fet_instance.encode( values_to_encode_nparray )
			if not np.array_equiv( encoded_with_cache, encoded_without_cache ):
				logger.error( f"FET encoding caching error for FET {fet_instance} for column '{column_name}' \n encoded_without_cache : {values_to_encode_nparray} \n encoded_without_cache: {encoded_without_cache}" )
		return encoded_with_cache


@lru_cache( maxsize=999, typed=True )
def fet_encoder_caching_system_do_real_caching( machine_id , column_name, fet_pickled, values_to_encode_pickled ):
	"""
	function with pickled argument, so they are hashable , and are working with LRU
	"""
	fet_instance = pickle.loads(fet_pickled)
	values_to_encode_nparray = pickle.loads( values_to_encode_pickled )
	return fet_instance.encode( values_to_encode_nparray )

