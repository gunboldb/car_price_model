# Title: COMP - 4254 - Assignment 2
# Author: Gunbold Boldsaikhan
# BCIT ID: A01363684
# Date: 2024/12/01
# Description:
import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, SelectKBest, chi2, RFECV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

PATH = "datasets/"  # Set path for data
CSV_CLEAN_DATA = "car_data_ready.csv"  # Set clean dataset file name
dataset = pd.read_csv(PATH + CSV_CLEAN_DATA, encoding="ISO-8859-1", sep=',')  # Read CSV into pandas dataframe
pd.set_option('display.max_columns', None)  # Show all columns.
pd.set_option('display.width', 1000)  # Increase number of columns that display on one line.

X = dataset.drop('Price', axis=1)
y = dataset['Price']

all_features = list(X.keys())

# Create a random sample of the data (10% of the data)
sample_size = 0.001
X_sample = X.sample(frac=sample_size, random_state=42)
y_sample = y.loc[X_sample.index]

# Implement scaling for X and y.
X_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

# Fit the scaler and perform the scaling transform.
X_sample_scaled = X_scaler.fit_transform(X_sample)
y_sample_scaled = y_scaler.fit_transform(y_sample.values.reshape(-1, 1))


# X_scaled = X_scaler.fit_transform(X)

# Use Random Forest for its feature importance to search for potential features
def random_forest(features, target):
    model = RandomForestRegressor(random_state=42)
    model.fit(features, target)
    feature_list = list(X.columns)
    importances = list(model.feature_importances_)
    df_random_forest = pd.DataFrame()
    for i in range(0, len(feature_list)):
        df_random_forest = df_random_forest._append(
            {
                "Feature": feature_list[i],
                "Importance": importances[i]
            },
            ignore_index=True)

    df_random_forest = df_random_forest.sort_values(
        by=["Importance"],
        ascending=False)
    df_random_forest.to_csv(PATH + 'features_random_forest.csv', index=False)

    potential_features = df_random_forest[df_random_forest["Importance"] > 0.001]["Feature"].tolist()
    return potential_features

    # # Create a bar plot for feature importance
    # plt.figure(figsize=(10, 10))
    # plt.barh(feature_list_sorted, importances_sorted, color='skyblue')
    # plt.xlabel('Gini Importance')
    # plt.title('Feature Importance - Gini Importance')
    # plt.gca().invert_yaxis()  # Invert y-axis for better visualization
    # plt.show()

    # potential_features = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    # return potential_features


# Backward Feature Elimination or Recursive Feature Elimination (RFE)
def recursive_feature_elimination(features, target, n_features):
    model = RandomForestRegressor()
    rfe = RFE(model, n_features_to_select=n_features)
    rfe = rfe.fit(features, target)
    potential_features = []
    df_cfe = pd.DataFrame()
    for i in range(0, len(features[0])):
        if (rfe.support_[i]):
            potential_features.append(X.columns[i])
            df_cfe = df_cfe._append({'feature': all_features[i]},
                                    ignore_index=True)
    df_cfe.to_csv(PATH + 'features_cfe_' + str(n_features) + '.csv', index=False)
    return potential_features
    # feature_ranking = pd.DataFrame({'Feature': X.columns, 'Ranking': rfe.ranking_})
    # return feature_ranking.sort_values(by='Ranking').head(n_features)


def select_k_best_features(features, target):
    test = SelectKBest(score_func=chi2, k='all')
    chi_scores = test.fit(features, target)
    np.set_printoptions(precision=3)
    chi_square_potential_features = []
    df_chi_square = pd.DataFrame()
    for i in range(0, len(chi_scores.scores_)):
        # Check if chi score is greater than 3.8
        if (chi_scores.scores_[i] > 3.8):
            chi_square_potential_features.append(all_features[i])
            df_chi_square = df_chi_square._append({'chi square': chi_scores.scores_[i],
                                                   'feature': all_features[i]},
                                                  ignore_index=True)
    df_chi_square.to_csv(PATH + 'features_chi_square.csv', index=False)
    return chi_square_potential_features


# RFE with Cross-Validation
def rfe_cross_validation(features, target):
    model = RandomForestRegressor()
    rfecv = RFECV(model, step=1, cv=5, scoring='neg_mean_absolute_error')  # Robust against outliers compared to MSE
    # rfecv = RFECV(model, step=1, cv=5, scoring='neg_mean_squared_error ')  # Penalizes larger errors more heavily than MAE
    rfecv = rfecv.fit(features, target)
    potential_features = []
    df_rfe_cv = pd.DataFrame()
    for i in range(0, len(features[0])):
        if (rfecv.support_[i]):
            potential_features.append(X.columns[i])
            df_rfe_cv = df_rfe_cv._append({'feature': all_features[i]},
                                          ignore_index=True)
    df_rfe_cv.to_csv(PATH + 'features_rfe_cv.csv', index=False)
    return potential_features


def evaluate_feature_models(feature_sets):
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale y_train and y_test
    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
    y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1))

    # Initialize a results dictionary
    results = []

    print(f"Total number of features: {X_train.columns.values}")

    # Loop through each feature set
    for name, selected_features in feature_sets.items():
        print(f"\nEvaluating features selected by {name}:")
        print(f"Selected Features: {selected_features}")
        print(f"Number of Features: {len(selected_features)}")

        # Select the columns corresponding to the selected features
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]

        # Train the model
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train_selected, y_train_scaled.ravel())

        # Make predictions
        y_pred_scaled = model.predict(X_test_selected)

        # Inverse transform the predictions and the test target
        y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
        y_test_original = y_scaler.inverse_transform(y_test_scaled)

        # Calculate evaluation metrics
        mae = mean_absolute_error(y_test_original, y_pred)
        mse = mean_squared_error(y_test_original, y_pred)
        rmse = math.sqrt(mse)
        r2 = r2_score(y_test_original, y_pred)

        # Store the results
        results.append({
            "Algorithm": name,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R²": r2
        })

        # Print metrics
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Mean Root Squared Error (RMSE): {rmse:.4f}")
        print(f"R² Score: {r2:.4f}")

    # Convert results to DataFrame for better readability
    results_df = pd.DataFrame(results)
    return results_df


feature_sets = {
    "RFE 5": recursive_feature_elimination(X_sample_scaled, y_sample_scaled.ravel(), n_features=5),
    "RFE 10": recursive_feature_elimination(X_sample_scaled, y_sample, n_features=10),
    "RFE 15": recursive_feature_elimination(X_sample_scaled, y_sample, n_features=15),
    "Chi-Squared": select_k_best_features(X_sample_scaled, y_sample),
    "RFECV": rfe_cross_validation(X_sample_scaled, y_sample),
    "Random Forest": random_forest(X_sample_scaled, y_sample)
}

results_df = evaluate_feature_models(feature_sets)
results_df.to_csv(PATH + 'features_results.csv', index=False)
print("\nEvaluation Summary:")
print(results_df)
