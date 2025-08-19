# Title: COMP - 4254 - Assignment 2
# Author: Gunbold Boldsaikhan
# BCIT ID: A01363684
# Date: 2024/12/01
# Description:

import numpy as np
import translations
import pandas as pd
from matplotlib import pyplot as plt


# Plot normalized distributions for all columns except target
def plot_distributions(data, target_name):
    X = data.copy()
    y = X.pop(target_name)
    X_norm = (X - X.mean(numeric_only=True)) / (X.std(numeric_only=True))
    X_norm = X_norm.join(y)
    X_norm.hist(figsize=(20, 20))
    plt.show()


def translate_values(df):
    df['Make'] = df['Make'].replace(translations.TRANSLATION_DICT_MAKE)
    df['Model'] = df['Model'].replace(translations.TRANSLATION_DICT_MODEL)
    df['Transmission'] = df['Transmission'].replace(translations.TRANSLATION_DICT_TRANSMISSION)
    df['Steering'] = df['Steering'].replace(translations.TRANSLATION_DICT_STEERING)
    df['BodyType'] = df['BodyType'].replace(translations.TRANSLATION_DICT_BODY_TYPE)
    df['ColorExterior'] = df['ColorExterior'].replace(translations.TRANSLATION_DICT_COLOR_EXTERIOR)
    df['Fuel'] = df['Fuel'].replace(translations.TRANSLATION_DICT_FUEL)
    df['ColorInterior'] = df['ColorInterior'].replace(translations.TRANSLATION_DICT_COLOR_INTERIOR)
    df['Leasing'] = df['Leasing'].replace(translations.TRANSLATION_DICT_LEASING)
    df['Drivetrain'] = df['Drivetrain'].replace(translations.TRANSLATION_DICT_DRIVETRAIN)
    df['Condition'] = df['Condition'].replace(translations.TRANSLATION_DICT_CONDITION)
    #df['Province'] = df['Province'].replace(translations.TRANSLATION_DICT_PROVINCE)
    #df['District'] = df['District'].replace(translations.TRANSLATION_DICT_DISTRICT)
    return df


def get_data_types_and_convert_numeric(df):
    columns = {}  # Initiate empty dictionary
    for column in df.columns:
        # Try to convert the column to numeric, coerce errors to NaN
        converted_col = pd.to_numeric(df[column], errors='coerce')
        # Calculate the percentage of non-NaN values
        non_nan_percentage = converted_col.notna().mean()
        # If more than 50% of the values are numeric, consider the column as numeric
        if non_nan_percentage > 0.5:
            columns[column] = 'numeric'  # Save column datatype to dictionary
            df[column] = converted_col  # Replace the column with the new numeric column
        # Else, consider the column as containing string values.
        else:
            columns[column] = 'string'  # Save column datatype to dictionary
    return columns, df


def remove_outliers(column_name, df):
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)

    # Calculate the interquartile range (IQR)
    IQR = Q3 - Q1

    # Define left and right whiskers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter the DataFrame to include only non-outliers
    filtered_df = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]
    return filtered_df


def impute_all_null_values(columns, df):
    for column_name, column_type in columns.items():
        # Impute null values if the column type is numeric and if the number of non-nulls
        # does not equal number of data points, meaning only columns with nulls.
        if column_type == 'numeric' and df[column_name].notnull().sum() != len(df):
            print(column_name + ' has NULLS!')
            # Create two new column names based on original column name.
            indicator_col_name = 'm_' + column_name  # Tracks whether imputed.
            imputed_col_name = 'imp_' + column_name  # Stores original & imputed data.
            imputed_value = df[column_name].mean()  # Get mean.
            # Populate new columns with data.
            imputed_column = []
            indicator_column = []
            for i in range(len(df)):
                is_imputed = False
                # imp_OriginalName column stores imputed & original data.
                if np.isnan(df.loc[i][column_name]):
                    is_imputed = True
                    imputed_column.append(imputed_value)
                else:
                    imputed_column.append(df.loc[i][column_name])
                # m_OriginalName column tracks if is imputed (1) or not (0).
                if is_imputed:
                    indicator_column.append(1)
                else:
                    indicator_column.append(0)

            # Append new columns to dataframe but always keep original column.
            df[indicator_col_name] = indicator_column
            df[imputed_col_name] = imputed_column
            del df[column_name]  # Drop old column.
    return df


def get_all_dummy_variables(columns, df):
    string_columns = [column_name for column_name, column_type in columns.items() if column_type == 'string']
    df = pd.get_dummies(df, columns=string_columns, dtype=int)
    return df
