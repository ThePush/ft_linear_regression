import pandas as pd
import numpy as np
import os
import sys


def count(x: list) -> int:
    '''
    Computes the number of elements in a given list or array x. The method
    returns the number of elements as an integer, otherwise None if x is an empty
    list or array.

    Args:
        x (list): list or array of elements

    Returns:
        int: number of elements in x
    '''
    if len(x) == 0:
        return None
    return len(x)


def mean(x: list) -> float:
    '''
    Computes the mean of a given non-empty list or array x, using a for-loop.
    The method returns the mean as a float, otherwise None if x is an empty list or
    array.

    Args:
        x (list): list or array of elements

    Returns:
        float: mean of x
    '''
    if len(x) == 0:
        return None
    return sum(x) / len(x)


def median(x: list) -> float:
    '''
    Computes the median of a given non-empty list or array x. The method
    returns the median as a float, otherwise None if x is an empty list or array.

    Args:
        x (list): list or array of elements

    Returns:
        float: median of x
    '''
    if len(x) == 0:
        return None
    x = sorted(x)
    if len(x) % 2 == 0:
        return (float(x[int(len(x) / 2)]) + float(x[int(len(x) / 2) - 1])) / 2
    else:
        return float(x[int(len(x) / 2)])


def quartile(x: list) -> tuple:
    '''
    Computes the 1st and 3rd quartiles of a given non-empty array x.
    The method returns the quartile as a float, otherwise None if x is an empty list or
    array.

    Args:
        x (list): list or array of elements

    Returns:
        tuple: 1st and 3rd quartiles of x
    '''
    x = sorted(x)
    return ([median(x[1:len(x) // 2]), median(x[len(x) // 2:])])


def var(x: list) -> float:
    '''
    Computes the variance of a given non-empty list or array x, using a for-
    loop. The method returns the variance as a float, otherwise None if x is
    an empty list or array.

    Args:
        x (list): list or array of elements

    Returns:
        float: variance of x
    '''
    if len(x) == 0:
        return None
    _mean = mean(x)
    return sum([(i - _mean) ** 2 for i in x]) / len(x)


def std_var(x: list) -> float:
    '''
    Computes the standard deviation of a given non-empty list or array x,
    using a for-loop. The method returns the standard deviation as a float, otherwise
    None if x is an empty list or array.

    Args:
        x (list): list or array of elements

    Returns:
        float: standard deviation of x
    '''
    if len(x) == 0:
        return None
    return var(x) ** 0.5


def is_numeric(x: str) -> bool:
    '''
    Checks if a given string x is numeric. The method returns True if x is
    numeric, otherwise False.

    Args:
        x (str): string to check

    Returns:
        bool: True if x is numeric, False otherwise
    '''
    try:
        float(x)
        return True
    except ValueError:
        return False


def max(x: list) -> float:
    '''
    Computes the maximum of a given non-empty list or array x, using a for-
    loop. The method returns the maximum as a float, otherwise None if x is an empty
    list or array.

    Args:
        x (list): list or array of elements

    Returns:
        float: maximum of x
    '''
    if len(x) == 0:
        return None
    _max = x[0]
    for i in x:
        if i > _max:
            _max = i
    return _max


def min(x: list) -> float:
    '''
    Computes the minimum of a given non-empty list or array x, using a for-
    loop. The method returns the minimum as a float, otherwise None if x is an empty
    list or array.

    Args:
        x (list): list or array of elements

    Returns:
        float: minimum of x
    '''
    if len(x) == 0:
        return None
    _min = x[0]
    for i in x:
        if i < _min:
            _min = i
    return _min


def drop_missing_rows(data: pd.DataFrame) -> pd.DataFrame:
    '''
    Removes rows with missing values from a DataFrame by keeping only
    the rows where the mask is False (no missing values).

    Args:
        data (pd.DataFrame): DataFrame to clean

    Returns:
        pd.DataFrame: DataFrame with missing rows removed
    '''
    clean_data = data.copy()
    clean_data = clean_data[~clean_data.isnull().any(axis=1)]
    return clean_data


def drop_non_numeric_columns(data: pd.DataFrame) -> pd.DataFrame:
    '''
    Removes columns that do not contain numerical values from a DataFrame.

    Args:
        data (pd.DataFrame): DataFrame to clean


    Returns:
        pd.DataFrame: DataFrame with non-numeric columns removed
    '''
    clean_data = data.copy()
    numeric_columns = []
    for col in clean_data.columns:
        try:
            clean_data[col] = pd.to_numeric(clean_data[col])
            numeric_columns.append(col)
        except ValueError:
            pass
    return clean_data[numeric_columns]


def drop_column_by_name(data: pd.DataFrame, column_name: str) -> pd.DataFrame:
    '''
    Removes a column from a DataFrame by its name.

    Args:
        data (pd.DataFrame): DataFrame to clean
        column_name (str): name of the column to remove

    Returns:
        pd.DataFrame: DataFrame with column removed
    '''
    clean_data = data.copy()
    columns = [col for col in clean_data.columns if col != column_name]
    clean_data = clean_data[columns]
    return clean_data


def clean_dataset(data, column_name):
    '''
    Cleans a dataset by removing missing rows, non-numeric columns and a given column.

    Args:
        data (pd.DataFrame): DataFrame to clean
        column_name (str): name of the column to remove

    Returns:
        pd.DataFrame: DataFrame with missing rows, non-numeric columns and given column removed
    '''
    data = drop_column_by_name(data, column_name)
    data = drop_missing_rows(data)
    data = drop_non_numeric_columns(data)
    return data


def check_file(filename: str) -> bool:
    '''
    Checks if a file exists, is a CSV file and is not empty.

    Args:
        filename (str): name of the file to check

    Returns:
        bool: True if file exists, is a CSV file and is not empty, False otherwise
    '''
    assert os.path.exists(filename), sys.exit(
        f'File does not exist: {filename}')
    assert filename.endswith('.csv'), sys.exit(
        f'File is not a CSV file: {filename}')
    assert os.path.getsize(filename) > 0, sys.exit(
        f'File is empty: {filename}')
    return True


def normalize_array(X: np.array) -> np.array:
    '''
    Normalize np.array data to be between 0 and 1.

    Args:
        X (np.array): array to normalize

    Returns:
        np.array: normalized array
    '''
    return (X - X.min()) / (X.max() - X.min())


def normalize_df(data: pd.DataFrame) -> pd.DataFrame:
    '''
    Normalizes the values of a DataFrame between 0 and 1.

    Args:
        data (pd.DataFrame): DataFrame to normalize

    Returns:
        pd.DataFrame: normalized DataFrame
    '''
    normalized_data = data.copy()
    for col in normalized_data.columns:
        normalized_data[col] = (normalized_data[col] - normalized_data[col].min()) / (
            normalized_data[col].max() - normalized_data[col].min())
    return normalized_data


def normalize_column(column: pd.Series) -> pd.Series:
    '''
    Normalizes the values of a pd.Series column between 0 and 1.

    Args:
        column (pd.Series): column to normalize

    Returns:
        pd.Series: normalized column
    '''
    normalized_column = column.copy()
    normalized_column = (normalized_column - normalized_column.min()) / (
        normalized_column.max() - normalized_column.min())
    return normalized_column


def predict(theta0: float, theta1: float, x_i: float) -> float:
    '''
    Linear function of type y = mx + b.

    Args:
        theta0 (float): b, the y-intercept
        theta1 (float): m, the slope
        x_i (float): x, the input

    Returns:
        float: predicted value
    '''
    return theta1 * x_i + theta0


def error(theta0: float, theta1: float, x_i: float, y_i: float) -> float:
    '''
    Difference between predicted and actual value.

    Args:
        theta0 (float): b, the y-intercept
        theta1 (float): m, the slope
        x_i (float): x, the input
        y_i (float): y, the actual value

    Returns:
        float: error
    '''
    return predict(theta0, theta1, x_i) - y_i


def sum_of_squared_errors(theta0: float, theta1: float, X: list, Y: list) -> float:
    '''
    Sum of squared errors between predicted and actual values.

    Args:
        theta0 (float): b, the y-intercept
        theta1 (float): m, the slope
        X (list): list of x, the inputs
        Y (list): list of y, the actual values

    Returns:
        float: sum of squared errors
    '''
    return sum(error(theta0, theta1, x_i, y_i) ** 2
               for x_i, y_i in zip(X, Y))


def mean_squared_error(theta0: float, theta1: float, X: list, Y: list) -> float:
    '''
    Mean squared error between predicted and actual values.

    Args:
        theta0 (float): b, the y-intercept
        theta1 (float): m, the slope
        X (list): list of x, the inputs
        Y (list): list of y, the actual values

    Returns:
        float: mean squared error
    '''
    return sum_of_squared_errors(theta0, theta1, X, Y) / len(X)


def cost(theta0: float, theta1: float, X: list, Y: list) -> float:
    '''
    Average squared error between predicted and actual values.

    Args:
        theta0 (float): b, the y-intercept
        theta1 (float): m, the slope
        X (list): list of x, the inputs
        Y (list): list of y, the actual values

    Returns:
        float: average squared error
    '''
    return sum_of_squared_errors(theta0, theta1, X, Y) / (2 * len(X))


def correlation(x: pd.DataFrame, y: pd.DataFrame) -> float:
    '''
    Calculate the Pearson (linear) correlation between two variables.

    Args:
        x (pd.DataFrame): first variable
        y (pd.DataFrame): second variable

    Returns:
        float: r correlation
    '''
    n = len(x)
    numerator = n * sum(x * y) - sum(x) * sum(y)
    denominator = ((n * sum(x**2) - sum(x)**2)**.5) * \
        ((n * sum(y**2) - sum(y)**2)**.5)
    return numerator / denominator


def r_squared(X: list, Y: list) -> float:
    '''
    R-squared coefficient of determination.

    Args:
        X (list): first variable
        Y (list): second variable

    Returns:
        float: r-squared
    '''
    return correlation(pd.Series(X), pd.Series(Y))**2


def std_err_of_estimate(theta0: float, theta1: float, X: list, Y: list) -> float:
    '''
    Standard error of estimate.

    Args:
        theta0 (float): b, the y-intercept
        theta1 (float): m, the slope
        X (list): list of x, the inputs
        Y (list): list of y, the actual values

    Returns:
        float: standard error of estimate
    '''
    return (sum_of_squared_errors(theta0, theta1, X, Y) / (len(X) - 2))**.5


def check_datafile(filename: str) -> None:
    '''
    Check if filename is a exists and is a none empty .csv file.

    Args:
        filename (str): name of file to check

    Raises:
        FileNotFoundError: if file does not exist
        TypeError: if file is not a .csv file
        ValueError: if file is empty
    '''
    if not os.path.exists(filename):
        raise FileNotFoundError('File ' + filename + ' does not exist')
    if not filename.endswith('.csv'):
        raise TypeError('File ' + filename + ' is not a CSV file')
    if os.stat(filename).st_size == 0:
        raise ValueError('File ' + filename + ' is empty')


def check_csv(self, filepath: str):
    '''
    Check if csv is two columns of numbers with no null values.

    Args:
        filepath (str): path to csv file

    Raises:
        Exception: if csv is not two columns of numbers with no null values
    '''
    line_num = 0
    with open(filepath, 'r') as file:
        for line in file:
            line_num += 1
            values = line.strip().split(',')
            if len(values) != 2:
                raise Exception(
                    f'Dataset must have two columns at line {line_num}')
            if any([value == '' for value in values]):
                raise Exception(
                    f'Dataset must not have null values at line {line_num}')
            try:
                values = [float(value) for value in values]
            except ValueError:
                raise Exception(
                    f'Dataset must contain only numbers at line {line_num}')
