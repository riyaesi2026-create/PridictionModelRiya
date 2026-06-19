"""
House Price Prediction (Regression)
-------------------------------------
Pipeline: load CSV -> clean (missing values + outlier removal) ->
preprocess/transform -> train/test split -> train Random Forest Regressor
-> evaluate -> validate against a fresh house listing never seen by the
model -> final score (R^2).

Usage:
    python house_price_prediction.py
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

CSV_PATH = "../datasets/house_prices.csv"
RANDOM_STATE = 2


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {path} -> shape {df.shape}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    before = len(df)
    df = df.drop_duplicates()
    if len(df) != before:
        print(f"Dropped {before - len(df)} duplicate rows")

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing):
        print(f"Filling missing values with column medians:\n{missing}")
        df = df.fillna(df.median(numeric_only=True))

    # Remove extreme price outliers using the IQR rule (catches the bad
    # data-entry rows where price was multiplied by a typo factor).
    q1, q3 = df["price"].quantile([0.25, 0.75])
    iqr = q3 - q1
    upper = q3 + 3 * iqr
    n_outliers = (df["price"] > upper).sum()
    if n_outliers:
        print(f"Removing {n_outliers} extreme price outlier(s) above {upper:,.0f}")
        df = df[df["price"] <= upper]

    return df


def preprocess(df: pd.DataFrame):
    X = df.drop(columns="price")
    y = df["price"]

    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)

    return X_scaled, y, scaler, X.columns.tolist()


def train_model(X_train, y_train) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=200, max_depth=10, random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_train, y_train, X_test, y_test):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    mae = mean_absolute_error(y_test, test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, test_pred))

    print(f"\nTraining R^2 : {train_r2:.4f}")
    print(f"Test R^2     : {test_r2:.4f}")
    print(f"Test MAE     : {mae:,.0f}")
    print(f"Test RMSE    : {rmse:,.0f}")

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
    print(f"5-fold CV R^2: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    return train_r2, test_r2


def predict_unseen_sample(model, scaler, feature_names):
    """A house listing that was never part of the CSV."""
    new_house = {
        "sqft_living": 2150, "bedrooms": 3, "bathrooms": 2.0, "floors": 2.0,
        "house_age": 8, "dist_to_city_km": 5.4, "quality_score": 7,
    }
    row = pd.DataFrame([new_house])[feature_names]
    row_scaled = scaler.transform(row)
    pred = model.predict(row_scaled)[0]

    print(f"\nUnseen sample input: {new_house}")
    print(f"Predicted price: ${pred:,.0f}")
    return pred


def main():
    df = load_data(CSV_PATH)
    df = clean_data(df)
    X, y, scaler, feature_names = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"Train/test shapes: {X_train.shape} / {X_test.shape}")

    model = train_model(X_train, y_train)
    train_r2, test_r2 = evaluate(model, X_train, y_train, X_test, y_test)
    predict_unseen_sample(model, scaler, feature_names)

    print(f"\nFinal prediction score (test R^2): {test_r2:.4f}")
    return test_r2


if __name__ == "__main__":
    main()
