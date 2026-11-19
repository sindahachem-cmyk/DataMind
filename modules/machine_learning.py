
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# Train a Linear Regression model
def train_linear_regression(df, target_column, feature_columns):

    # Keep only the selected columns
    data = df[feature_columns + [target_column]].copy()

    # Keep numerical data only
    data = data.apply(pd.to_numeric, errors="coerce")

    # Remove rows containing missing values
    data = data.dropna()

    if len(data) < 5:
        raise ValueError(
            "At least 5 complete rows are required."
        )

    # Input variables
    X = data[feature_columns]

    # Value we want to predict
    y = data[target_column]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Create the model
    model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Evaluate model performance
    metrics = {
        "MAE": mean_absolute_error(y_test, predictions),
        "MSE": mean_squared_error(y_test, predictions),
        "R2 Score": r2_score(y_test, predictions)
    }

    return model, metrics, predictions, y_test

# Train a Random Forest Regression model
def train_random_forest(df, target_column, feature_columns):

    data = df[feature_columns + [target_column]].copy()
    data = data.apply(pd.to_numeric, errors="coerce")
    data = data.dropna()

    if len(data) < 5:
        raise ValueError("At least 5 complete rows are required.")

    X = data[feature_columns]
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, predictions),
        "MSE": mean_squared_error(y_test, predictions),
        "R2 Score": r2_score(y_test, predictions)
    }

    return model, metrics, predictions, y_test
def train_decision_tree(df, target_column, feature_columns):

    # Select the required columns
    data = df[feature_columns + [target_column]].copy()

    # Convert values to numbers
    data = data.apply(pd.to_numeric, errors="coerce")

    # Remove missing values
    data = data.dropna()

    if len(data) < 5:
        raise ValueError("At least 5 complete rows are required.")

    # Define inputs and target
    X = data[feature_columns]
    y = data[target_column]

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create the Decision Tree model
    model = DecisionTreeRegressor(
        max_depth=4,
        random_state=42
    )

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate performance
    metrics = {
        "MAE": mean_absolute_error(y_test, predictions),
        "MSE": mean_squared_error(y_test, predictions),
        "R2 Score": r2_score(y_test, predictions)
    }

    return model, metrics, predictions, y_test