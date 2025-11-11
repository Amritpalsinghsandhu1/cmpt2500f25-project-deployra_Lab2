from pathlib import Path
import joblib

# make sure models/ exists and tell us the full path
Path("models").mkdir(parents=True, exist_ok=True)
out_path = Path("models") / "random_forest_latest.pkl"
joblib.dump(model, out_path)
print(f"💾 Model save attempt to: {out_path.resolve()}")

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

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # load cleaned data produced by preprocess step
    df = pd.read_csv("data/processed/df_clean.csv")
    target_col = cfg.get("target", "target")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not in df_clean.csv columns: {list(df.columns)}")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    params = cfg["params"]["random_forest"]
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"✅ Accuracy: {acc:.4f}")

    # ensure models dir exists and save with full path print
    Path("models").mkdir(parents=True, exist_ok=True)
    out_path = Path("models") / "random_forest_latest.pkl"
    joblib.dump(model, out_path)
    print(f"💾 Model saved to: {out_path.resolve()}")

if __name__ == "__main__":
    main()
PY
