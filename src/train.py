cat > src/train.py << 'PY'
import argparse, yaml, pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/train_config.yaml")
    args = parser.parse_args()

    print("🔧 Using config:", args.config)
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    df_path = "data/processed/df_clean.csv"
    print("📦 Loading data:", df_path)
    df = pd.read_csv(df_path)

    target_col = cfg.get("target", "target")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Columns: {list(df.columns)}")

    X = df.drop(columns=[target_col]); y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    params = cfg["params"]["random_forest"]
    print("🤖 Training RandomForest with params:", params)
    model = RandomForestClassifier(**params).fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"✅ Accuracy: {acc:.4f}")

    Path("models").mkdir(parents=True, exist_ok=True)
    out = Path("models/random_forest_latest.pkl").resolve()
    joblib.dump(model, out)
    print("💾 Model saved to:", out)

if __name__ == "__main__":
    main()
PY
