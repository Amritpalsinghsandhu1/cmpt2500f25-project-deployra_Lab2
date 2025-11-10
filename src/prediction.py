# src/prediction.py
from preprocess import load_and_clean_data, prepare_used_data, split_data
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle

def train_and_evaluate(X_train, X_test, y_train, y_test):
    results = {}

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    results["LinearRegression"] = {
        "R2": r2_score(y_test, y_pred_lr),
        "MAE": mean_absolute_error(y_test, y_pred_lr),
        "RMSE": mean_squared_error(y_test, y_pred_lr, squared=False)
    }

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    results["RandomForest"] = {
        "R2": r2_score(y_test, y_pred_rf),
        "MAE": mean_absolute_error(y_test, y_pred_rf),
        "RMSE": mean_squared_error(y_test, y_pred_rf, squared=False)
    }

    # XGBoost
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    results["XGBoost"] = {
        "R2": r2_score(y_test, y_pred_xgb),
        "MAE": mean_absolute_error(y_test, y_pred_xgb),
        "RMSE": mean_squared_error(y_test, y_pred_xgb, squared=False)
    }

    return results, rf

if __name__ == "__main__":
    # Load and preprocess
    df = load_and_clean_data("data/raw/CBB_Listings.csv")
    X, y, encoders = prepare_used_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train & evaluate
    results, best_model = train_and_evaluate(X_train, X_test, y_train, y_test)

    # Print metrics
    for model, metrics in results.items():
        print(f"\n{model}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    # Save best model (Random Forest) + encoders
    with open("models/rf_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    with open("models/label_encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)
