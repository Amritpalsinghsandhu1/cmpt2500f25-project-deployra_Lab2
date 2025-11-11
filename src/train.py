# src/train.py
import argparse, yaml, pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/train_config.yaml")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Load the cleaned dataset
    df = pd.read_csv("data/processed/df_clean.csv")
    target_col = cfg.get("target", "target")  # default fallback

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model training
    model = RandomForestClassifier(**cfg["params"]["random_forest"])
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Model trained with accuracy: {acc:.4f}")

    # Save model
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/random_forest_latest.pkl")
    print("💾 Model saved to models/random_forest_latest.pkl")

if __name__ == "__main__":
    main()
