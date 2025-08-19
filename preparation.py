# Title: COMP - 4254 - Assignment 2
# Author: Gunbold Boldsaikhan
# BCIT ID: A01363684
# Date: 2024/12/01
# Description:

import pandas as pd
import datetime
import prep_functions

PATH = "datasets/"  # Set path for data
CSV_RAW_DATA = "Car_Data_Unegui_Backup_20241118.csv"  # Set dataset file name
FILENAME = "car_data"  # Set dataset file name

# Since dataset is large it might be faster to specify datatypes, rather than pandas guessing
# These are only initial datatypes, it will be processed and transformed
dtypes_dict = {
    'AdID': 'int64',
    'AdLink': 'object',
    'Make': 'object',
    'Model': 'object',
    'Price': 'int64',
    'EngineCC': 'object',
    'Transmission': 'object',
    'Steering': 'object',
    'BodyType': 'object',
    'ColorExterior': 'object',
    'ManufactureYear': 'int64',
    'ImportYear': 'int64',
    'Fuel': 'object',
    'ColorInterior': 'object',
    'Leasing': 'object',
    'Location': 'object',
    'Drivetrain': 'object',
    'Km': 'object',
    'Condition': 'object',
    'Doors': 'int64',
    'Description': 'object'
}

# Read CSV into pandas dataframe
dataset = pd.read_csv(
    PATH + CSV_RAW_DATA,
    dtype=dtypes_dict,
    parse_dates=['AdDate'],
    encoding="UTF-8",
    sep='=')
pd.set_option('display.max_columns', None)  # Show all columns of dataframe
pd.set_option('display.width', 1000)  # Increase number of columns that display on one line

# Data Cleaning and Preparation

# # The location column consists of strings separated by the long dash symbol
# # Split the 'Location' column by '—' (long dash) and store split values into two new columns
# dataset[['Province', 'District']] = dataset['Location'].str.split('—', n=1, expand=True)
# # Split 'Province' and 'District' columns by ',' and only keep the first part
# # Only the first part is city ir province name
# dataset['Province'] = dataset['Province'].str.split(',', expand=True)[0]
# # Only the first part is district name, rest is street name, building name etc
# dataset['District'] = dataset['District'].str.split(',', expand=True)[0]
# dataset['Province'] = dataset['Province'].str.strip()  # Take out leading and trailing spaces
# dataset['District'] = dataset['District'].str.strip()

# Create IsCity flag feature to flag if ad is in the city or not
# Most ads are from the city, and this reduces number of features when we are creating dummy variables
dataset['IsCity'] = dataset['Location'].apply(lambda x: 1 if 'УБ' in x else 0)
dataset.drop('Location', axis=1, inplace=True)  # Drop the old Location column

# Translate all the string in the dataset into English
dataset = prep_functions.translate_values(dataset)

# Remove string ' км' from values of Km column
dataset['Km'] = dataset['Km'].str.replace(' км', '')
# Km has numbers too big for int, will filter those below
dataset['Km'] = dataset['Km'].astype(float)

# Some final cleaning to filter out unwanted and inaccurate data
# Filter the dataframe to exclude rows where price is more than 2 billion
# Price more than 2 billion are mostly from fake/joke ads
# Keep rows with prices below 2 billion
dataset = dataset[dataset.Price <= 2_000_000_000]
# Filter the dataframe to exclude rows where km is more than 500,000 kms
# Also from fake/joke ads
# Keep rows with km less than 500,000
dataset = dataset[dataset.Km <= 500_000]

# Exclude cars that are not new but has low mileage
dataset = dataset[~((datetime.datetime.now().year - dataset['ManufactureYear'] > 2) & (dataset['Km'] < 10_000))]
# Exclude cars that are new but very cheap
dataset = dataset[~((datetime.datetime.now().year - dataset['ManufactureYear'] < 2) & (dataset['Price'] < 15_000_000))]
# Exclude cars that are very cheap but has low mileage
dataset = dataset[~((dataset['Km'] < 20_000) & (dataset['Price'] < 15_000_000))]

dataset['Price'] = dataset['Price'].astype(int)
dataset['Km'] = dataset['Km'].astype(int)

dataset.to_csv(PATH + FILENAME + '_base.csv', index=False)

# For electric vehicles the value for EngineCC is the string 'Цахилгаан' (means electric in Mongolian)
# Create a new flag column IsElectric where the value is 1 or 0 depending on EngineCC value
dataset['IsElectric'] = (dataset['EngineCC'] == 'Цахилгаан').astype(int)
# Replace all 'Цахилгаан' values with 0
dataset['EngineCC'] = dataset['EngineCC'].replace('Цахилгаан', 0)
dataset['IsManual'] = (dataset['Transmission'] == 'Manual').astype(int)
dataset['IsLeftSteering'] = (dataset['Steering'] == 'Left').astype(int)
dataset.drop(columns=['Transmission', 'Steering'], inplace=True)  # Drop the old Transmission and Steering columns

# Derive some basic information such as length of characters and words.
dataset['DescriptionCharacters'] = dataset['Description'].str.len()
dataset['DescriptionCharacters'] = dataset['DescriptionCharacters'].fillna(0).astype(int)
dataset['DescriptionWords'] = dataset['Description'].fillna('').str.split().str.len()
dataset.drop('Description', axis=1, inplace=True)  # Drop the Description column

# Derive new age features from the ManufactureYear and ImportYear columns
# dataset['ManufactureAge'] = datetime.datetime.now().year - dataset['ManufactureYear']
# dataset['ImportAge'] = datetime.datetime.now().year - dataset['ImportYear']

# Derive new features from AdDate column
dataset['AdMonth'] = dataset['AdDate'].dt.month  # Month number (1-12)
dataset['AdWeek'] = dataset['AdDate'].dt.isocalendar().week  # Week number (1-52/53)
dataset['AdWeekDay'] = dataset['AdDate'].dt.weekday  # Day of week (0 = Monday, 6 = Sunday)
dataset['AdMonthDay'] = dataset['AdDate'].dt.day  # Day of month (1-31)

# Convert numeric columns to numeric
columnTypes, dataset = prep_functions.get_data_types_and_convert_numeric(dataset)

dataset.to_csv(PATH + FILENAME + '_prepped.csv', index=False)

# Drop columns that have no use anymore
columns_to_drop = ['AdID', 'AdLink', 'AdDate', 'Model']
for column in columns_to_drop:
    columnTypes.pop(column, None)
dataset = dataset.drop(columns=columns_to_drop)

# Plot distributions for all numeric features
prep_functions.plot_distributions(dataset, 'Price')

# Impute any and all null values, if there is any
dataset = prep_functions.impute_all_null_values(columnTypes, dataset)
# Create dummy variables for all string/categorical columns
dataset = prep_functions.get_all_dummy_variables(columnTypes, dataset)

# Export final data
dataset.to_csv(PATH + FILENAME + '_ready.csv', index=False)
print('Data Shape')
print(f'Number of columns:  {dataset.shape[1]}\n'
      f'Number of rows:     {dataset.shape[0]}')
features_names = dataset.columns.tolist()
features_df = pd.DataFrame(features_names, columns=['Feature'])
features_df.to_csv(PATH + FILENAME + '_features.csv', index=False)
