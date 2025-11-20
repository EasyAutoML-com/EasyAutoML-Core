import pandas as pd
import numpy as np

from typing import Union, NoReturn

from ML import EasyAutoMLDBModels, __getlogger


logger = __getlogger()


class SolutionScore:
    """
    SolutionScore use a dataframe containing criteria and scores,
    then convert it into a formula python ,
    which can be evaluated quickly using data of another dataset

    -CRITERIA-FILTER-
    LIST : the VALUE must be equal to one of the LIST.item  ( Work with Label and numeric )

    <MAX_VALUE : if VALUE is numeric the VALUE must be less than MAX_VALUE,
                            if VALUE is label the VALUE must be lower than MAX_VALUE in lexicographical ordering

    >MIN_VALUE : if VALUE is numeric: the VALUE must be greater than MIN_VALUE,
                            if VALUE is label: the VALUE must be upper than MIN_VALUE in lexicographical ordering

    -SCORE-EVALUATION-
    scores: --- : score is maximum when the value is lowest ( Work with numeric only )
    scores: +++ : score is maximum when the value is highest ( Work with numeric only )
    scores: ~TARGET : score is maximum when the value is closest to the TARGET ( Work with numeric only )
    """

    def __init__(self, raw_formula: Union[dict, pd.DataFrame]) -> NoReturn:
        """
        Constructor method

        :param raw_formula: Formula which should be converted to Python expression and evaluated on pd.DataFrame
        :type raw_formula: Union[dict, pd.DataFrame]
        """
        self._column_types = dict()
        self._formula_converted_to_dict = self.__convert_data_into_dict(raw_formula)
        self._python_formula = self.__convert_expression_to_python_expression(self._formula_converted_to_dict)

    @staticmethod
    def __convert_list_expression(operand: str, expression: Union[list, tuple]) -> str:
        """
        Converts expression which contains list of possible values

        :param operand: column title
        :type operand: str
        :param expression: set of possible values
        :type expression: Union[list, tuple]

        :return: part of Python ternary expression
        :rtype: str
        """
        return f"{operand} in {expression}"

    @staticmethod
    def __convert_numeric_expression(operand: str, expression: Union[int, float]) -> str:
        """
        Converts expression which contains numbers

        :param operand: column title
        :type operand: str
        :param expression: number
        :type expression: str

        :return: part of Python ternary expression
        :rtype: str
        """
        return f"{operand} == {expression}"

    @staticmethod
    def __convert_bool_expression(operand: str, expression: bool) -> str:
        """
        Converts expression which contains bool value

        :param operand: column title
        :type operand: str
        :param expression: number
        :type expression: bool

        :return: part of Python ternary expression
        :rtype: str
        """
        return f"{operand} is {expression}"

    def __convert_expression_to_python_expression(self, expression: dict) -> str:
        """
        Converts dict of expressions to Python ternary expression

        :param expression: contains as key column title and as value expression like "+++" or ">10"
        :type expression: dict

        :return: Python ternary expression ready to evaluation
        :rtype: str
        """
        python_conditions = []
        python_expressions = []
        for key, value in expression.items():
            if type(value) in (list, tuple):
                python_conditions.append(self.__convert_list_expression(key, value))
                self._column_types[key] = [type(item) for item in value]

            elif type(value) is str:
                if value.startswith("+++"):
                    if "%" in value:
                        _importance_percentage = float(value[4:-2])
                        if 0 < _importance_percentage < 100:
                            python_expressions.append(f"+{key}*({_importance_percentage}/100)")
                        else:
                            logger.error(f"Percentage in '{value}' is not correct!")
                    else:
                        python_expressions.append(f"+{key}")
                    self._column_types[key] = float

                elif value.startswith("---"):
                    if "%" in value:
                        _importance_percentage = float(value[4:-2])
                        if 0 < _importance_percentage < 100:
                            python_expressions.append(f"-{key}*({_importance_percentage}/100)")
                        else:
                            logger.error(f"Percentage in '{value}' is not correct!")
                    else:
                        python_expressions.append(f"-{key}")
                    self._column_types[key] = float

                elif value.startswith("~"):
                    if value[1:].replace(".", "", 1).isdigit():
                        python_expressions.append(f"+abs({key} - {value[1:]})")
                        self._column_types[key] = float

                    else:
                        logger.error(f"Unsupported type in expression {value}, {value[1:]} should be a number.")

                elif value.startswith("<="):
                    python_conditions.append(f"{key} <= {value[2:]}")
                    self._column_types[key] = float if value[2:].replace(".", "", 1).isdigit() else str

                elif value.startswith(">="):
                    python_conditions.append(f"{key} >= {value[2:]}")
                    self._column_types[key] = float if value[2:].replace(".", "", 1).isdigit() else str

                elif value.startswith("<"):
                    python_conditions.append(f"{key} < {value[1:]}")
                    self._column_types[key] = float if value[1:].replace(".", "", 1).isdigit() else str

                elif value.startswith(">"):
                    python_conditions.append(f"{key} > {value[1:]}")
                    self._column_types[key] = float if value[1:].replace(".", "", 1).isdigit() else str

                else:
                    python_conditions.append(f'{key} == "{value}"')
                    self._column_types[key] = str

            elif type(value) in (int, float):
                python_conditions.append(self.__convert_numeric_expression(key, value))
                self._column_types[key] = type(value)

            elif type(value) is bool:
                python_conditions.append(self.__convert_bool_expression(key, value))
                self._column_types[key] = bool

            else:
                logger.error(f"Unsupported type of {key}, type should be str, int, float, bool, list or tuple not {type(value)}")
        if python_conditions:
            return f"{' '.join(python_expressions)} if {' and '.join(python_conditions)} else None"
        return " ".join(python_expressions)

    @staticmethod
    def __convert_data_into_dict(data_to_convert: Union[dict, pd.DataFrame]) -> dict:
        """
        If input is dict -> returns it, if - pd.DataFrame
        converting to dict

        :param data_to_convert: initialisation data
        :type data_to_convert: Union[dict, pd.DataFrame]

        :raises TypeError: if data_to_convert in not dict or pd.DataFrame

        :return: converted data to dict
        :rtype: dict
        """
        if isinstance(data_to_convert, dict):
            return data_to_convert
        elif isinstance(data_to_convert, pd.DataFrame):
            df_to_dict = data_to_convert.where(pd.notnull(data_to_convert), None).to_dict()
            df_to_dict = {column_name: list(expression.values()) for column_name, expression in df_to_dict.items()}
            df_to_dict = {
                column_name: [item for item in expression if item not in (None, np.nan)] for column_name, expression in df_to_dict.items()
            }
            df_to_dict = {
                column_name: expression if len(expression) > 1 else expression[0] for column_name, expression in df_to_dict.items()
            }
            return df_to_dict
        else:
            logger.error(f"Formula should be dict or pd.DataFrame, not {type(data_to_convert)}")

    @staticmethod
    def __validate_user_dataframe(user_dataframe: pd.DataFrame) -> NoReturn:
        """
        Checking type of dataframe

        :raises TypeError: if dataframe is not instance of pd.DataFrame
        """
        if not isinstance(user_dataframe, pd.DataFrame):
            logger.error(f"User dataframe should be pd.DataFrame, not {type(user_dataframe)}")

    def __validate_columns_in_both_datasets(self, user_dataframe: pd.DataFrame) -> NoReturn:
        """
        Validating columns in users dataframe

        :param user_dataframe: dataframe for evaluating
        :type user_dataframe: pd.DataFrame

        :raises KeyError: if formula contains column name which is not in dataframe
        """
        columns_in_dict_evaluate_dataframe = list(self._column_types.keys())
        columns_in_user_dataframe = list(user_dataframe)
        for column_title in columns_in_dict_evaluate_dataframe:
            if column_title not in columns_in_user_dataframe:
                raise ValueError(f"User dataframe ({user_dataframe}) doesn`t contain required column {column_title}")


    def __convert_columns_to_right_type(self, user_dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Compliance check columns types in user dataframe and machine dataframe

        :param user_dataframe: dataframe for evaluating
        :type user_dataframe: pd.DataFrame

        :return: NoReturn
        """
        LABEL_TYPES = ("O", "string", "object")
        FLOAT_TYPES = ("int64", "float64")

        user_dataframe_types = user_dataframe.dtypes.to_dict()
        for column_title, column_type in self._column_types.items():
            if column_type in (int, float) and user_dataframe_types[column_title] not in FLOAT_TYPES:
                user_dataframe = user_dataframe.astype({column_title: float})
            elif column_type == str and user_dataframe_types[column_title] not in LABEL_TYPES:
                user_dataframe = user_dataframe.astype({column_title: str})
            elif column_type == bool and user_dataframe_types[column_title] not in (
                "bool",
                "boolean",
            ):
                user_dataframe = user_dataframe.astype({column_title: bool})
            elif type(column_type) == list:
                if type(column_type[0]) is str:
                    user_dataframe = user_dataframe.astype({column_title: str})
                elif type(column_type[0]) in (int, float):
                    user_dataframe = user_dataframe.astype({column_title: float})
                elif type(column_type[0]) is bool:
                    user_dataframe = user_dataframe.astype({column_title: bool})

        return user_dataframe

    def eval(self, user_dataframe: pd.DataFrame) -> list:
        """
        Evaluating formulas to each row of machine dataframe

        :param user_dataframe: dataframe for evaluating
        :type user_dataframe: pd.DataFrame

        :return: dict(formula: [list of scores])
        """
        self.__validate_user_dataframe(user_dataframe)
        self.__validate_columns_in_both_datasets(user_dataframe)
        user_dataframe = self.__convert_columns_to_right_type(user_dataframe)
        total_score = []
        for index, row in user_dataframe.iterrows():
            total_score.append(eval(self._python_formula, row.to_dict()))
        return total_score

    def scored_columns_list(self) -> dict:
        """
        Calculating list of columns in user dataframe

        :return: list of columns, which has "+++", "---" or "~" in formula
        :rtype: list
        """
        scored_columns = {}
        for column_title, expression in self._formula_converted_to_dict.items():
            if str(expression).startswith(("+++", "---", "~")):
                scored_columns[column_title] = expression
        return scored_columns

    def how_many_columns_having_criteria(self) -> int:
        """
        Calculating count of columns with any criteria

        :return: count of columns with criteria
        :rtype: int
        """
        return len(self._column_types) - len(self.scored_columns_list().values())

    def how_many_columns_having_criteria_numeric(self) -> int:
        """
        Calculating count of columns with numeric criteria

        :return: count of columns with numeic criteria
        :rtype: int
        """
        count_of_columns_with_numeric_criteria = 0
        for column_type in self._column_types.values():
            if column_type in (int, float):
                count_of_columns_with_numeric_criteria += 1
            elif column_type in (list, tuple) and column_type[0] in (int, float):
                count_of_columns_with_numeric_criteria += 1
        return count_of_columns_with_numeric_criteria

    def how_many_columns_having_criteria_label(self) -> int:
        """
        Calculating count of columns with label criteria

        :return: count of columns with label criteria
        :rtype: int
        """
        count_of_columns_with_label_criteria = 0
        for column_type in self._column_types.values():
            if column_type is str:
                count_of_columns_with_label_criteria += 1
            elif type(column_type) in (list, tuple) and column_type[0] is str:
                count_of_columns_with_label_criteria += 1
        return count_of_columns_with_label_criteria

    def how_many_columns_having_criteria_list(self) -> int:
        """
        Calculating columns with criteria list

        :return: count of columns with possible values
        :rtype: int
        """

        return self._python_formula.count(" in ")

    def __get_columns_with_criteria_values(self) -> dict:
        """
        Calculating columns which has possible value or values, like:
            "Country": "Ukraine,USA,Germany"
            "Manufacture": "Audi"

        :return: dict of columns with possible values
        :rtype: dict
        """
        columns_with_criteria_possible_values = {}
        for column_title, expression in self._formula_converted_to_dict.items():
            if type(expression) not in (list, tuple) and not str(expression).startswith(("+++", "---", "~", ">", "<")):
                columns_with_criteria_possible_values[column_title] = expression
            elif type(expression) in (list, tuple):
                columns_with_criteria_possible_values[column_title] = expression

        return columns_with_criteria_possible_values

    def get_columns_with_criteria_compare_values(self) -> dict:
        """
        Calculating dict of columns with boundary criteria

        :return: dict of columns with boundary criteria
        :rtype: dict
        """
        columns_with_criteria_compare_values = dict()
        for column_title, expression in self._formula_converted_to_dict.items():
            if type(expression) not in (tuple, list) and expression.startswith(
                (
                    "<",
                    ">",
                    "<=",
                    ">=",
                )
            ):
                columns_with_criteria_compare_values[column_title] = expression

        return columns_with_criteria_compare_values

    def get_columns_with_criteria_possible_values(self) -> dict:
        """
        Calculating dict of columns with possible values

        :return: dict of columns with possible values
        :rtype: dict
        """
        columns_with_criteria_possible_values = dict()
        for column_title, expression in self._formula_converted_to_dict.items():
            if type(expression) not in (tuple, list):
                if not expression.startswith(("<", ">", "<=", ">=", "+++", "---", "~")):
                    columns_with_criteria_possible_values[column_title] = expression
            else:
                columns_with_criteria_possible_values[column_title] = expression

        return columns_with_criteria_possible_values

    def get_columns_with_criteria_boundary_max(self) -> dict:
        """
        Calculating columns with boundary max condition
        example: "<10", "<-1", etc.

        :return: columns with criteria boundary max
        :rtype: dict
        """
        columns_with_criteria_criteria_max = dict()

        for column_name, expression in self._formula_converted_to_dict.items():
            if type(expression) not in (list, tuple) and str(expression).startswith("<") and str(expression)[1] != "=":
                columns_with_criteria_criteria_max[column_name] = expression

        return columns_with_criteria_criteria_max

    def get_columns_with_criteria_boundary_min(self) -> dict:
        """
        Calculating columns with boundary min condition
        example: ">10", ">-1", etc.

        :return: columns with criteria boundary min
        :rtype: dict
        """
        columns_with_criteria_criteria_min = dict()

        for column_name, expression in self._formula_converted_to_dict.items():
            if type(expression) not in (list, tuple) and str(expression).startswith(">") and str(expression)[1] != "=":
                columns_with_criteria_criteria_min[column_name] = expression

        return columns_with_criteria_criteria_min

    def get_columns_with_score(self) -> list:
        """
        Calculates list of column names with "+++", "---" or "~" in formula

        :return: list of column names
        :rtype: list
        """
        columns_titles_with_score = list(self.scored_columns_list().items())
        return list(set(columns_titles_with_score))

    def total_count_of_possible_combinations_of_criteria_values(self) -> int:
        """
        Calculates count of premutations based on possible values

        :return: count of premutations
        :rtype: int
        """
        result = 1
        for value in self.__get_columns_with_criteria_values().values():
            if type(value) in (list, tuple):
                result *= len(value)
        return result

    def get_columns_with_varying(self):
        return {
            key: val for key, val in self._formula_converted_to_dict.items() if isinstance(val, tuple) or isinstance(val, list)
        }

    def get_columns_with_constants(self):
        res = {}
        for key, val in self._formula_converted_to_dict.items():
            if not isinstance(val, list) and not isinstance(val, tuple):
                if isinstance(val, float) or isinstance(val, int):
                    res[key] = val
                elif isinstance(val, str) and val[0] not in ("+", "~", ">", "<"):
                    if len(val) > 2 and val[1] != "-":
                        res[key] = val
                    elif len(val) == 1:
                        res[key] = val
        return res
