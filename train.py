# Title: COMP - 4254 - Assignment 2
# Author: Gunbold Boldsaikhan
# BCIT ID: A01363684
# Date: 2024/12/01
# Description:

import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

PATH = "datasets/"  # Set path for data
CSV_CLEAN_DATA = "car_data_ready.csv"  # Set clean dataset file name
dataset = pd.read_csv(PATH + CSV_CLEAN_DATA, encoding="ISO-8859-1", sep=',')  # Read CSV into pandas dataframe
pd.set_option('display.max_columns', None)  # Show all columns.
pd.set_option('display.width', 1000)  # Increase number of columns that display on one line.

selected_features = [
    'EngineCC',
    'ManufactureYear',
    'ImportYear',
    'Km',
    'IsManual',
    'IsLeftSteering',
    'DescriptionCharacters',
    'DescriptionWords',
    'AdWeek',
    'BodyType_SUV',
    'Drivetrain_AWD/4WD',
    'Make_Lexus',
    'Make_Mercedes-Benz',
    'Make_Toyota'
]

X_data = dataset[selected_features]
y_data = dataset['Price']

# Create a random sample of the data (20% of the data)
sample_size = 0.001
X = X_data.sample(frac=sample_size, random_state=42)
y = y_data.loc[X.index]

k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
results = []

# Base models
base_models = [
    RandomForestRegressor(random_state=42),
    GradientBoostingRegressor(random_state=42),
    SVR()
]

# Stacked model
stacked_model = LinearRegression()

for train_index, test_index in k_fold.split(X):
    X_scaler = MinMaxScaler()
    X_train_scaled = X_scaler.fit_transform(X.iloc[train_index])
    X_test_scaled = X_scaler.transform(X.iloc[test_index])

    # No scaling for y
    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    # Train the base model
    base_model_preds_train = []
    base_model_preds_test = []
    for model in base_models:
        model.fit(X_train_scaled, y_train)
        base_model_preds_train.append(model.predict(X_train_scaled))
        base_model_preds_test.append(model.predict(X_test_scaled))

    # Convert base model predictions into a new feature set for the stacked model
    base_model_preds_train = np.column_stack(base_model_preds_train)
    base_model_preds_test = np.column_stack(base_model_preds_test)

    # Train the stacked model on the base model predictions
    stacked_model.fit(base_model_preds_train, y_train)

    # Make predictions
    y_pred = stacked_model.predict(base_model_preds_test)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Store the results
    results.append({
        "K-fold": str(len(results)),
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R²": r2
    })

    # Print metrics
    print("\n***K-fold: " + str(len(results)))
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Root Squared Error (RMSE): {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")

    # Scatter plot of Actual vs. Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label="Ideal Fit")
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title(f"Actual vs. Predicted Prices (Fold {len(results)})")
    plt.legend()
    plt.savefig("K-fold-" + str(len(results)))

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Calculate overall metrics
mae_mean = results_df['MAE'].mean()
mae_std = results_df['MAE'].std()
mse_mean = results_df['MSE'].mean()
mse_std = results_df['MSE'].std()
rmse_mean = results_df['RMSE'].mean()
rmse_std = results_df['RMSE'].std()
r2_mean = results_df['R²'].mean()
r2_std = results_df['R²'].std()

print("\n*** Overall Metrics Across All Folds ***")
print(f"Mean Absolute Error (MAE): {mae_mean:.4f} ± {mae_std:.4f}")
print(f"Mean Squared Error (MSE): {mse_mean:.4f} ± {mse_std:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_mean:.4f} ± {rmse_std:.4f}")
print(f"R² Score: {r2_mean:.4f} ± {r2_std:.4f}")

mean_price = dataset['Price'].mean()
relative_error = (rmse_mean / mean_price) * 100
print(f"Mean Price: {mean_price:.2f}")
print(f"Relative Error: {relative_error:.2f}%")

# Save results to a CSV file
results_df.to_csv("results_kfold.csv", index=False)

# Finalize model on full dataset
X_final_scaled = X_scaler.fit_transform(X)

# Train the base models on full dateset
base_model_preds = []
for model in base_models:
    model.fit(X_final_scaled, y)
    base_model_preds.append(model.predict(X_final_scaled))

base_model_preds = np.column_stack(base_model_preds)

# Train stacked model on the base model predictions
stacked_model.fit(base_model_preds, y)

# Save scalers and model
with open("scaler_X.pkl", "wb") as scalerXFile:
    pickle.dump(X_scaler, scalerXFile)

with open("base_models.pkl", "wb") as base_models_file:
    pickle.dump(base_models, base_models_file)

with open("stacked_model.pkl", "wb") as stacked_model_file:
    pickle.dump(stacked_model, stacked_model_file)
