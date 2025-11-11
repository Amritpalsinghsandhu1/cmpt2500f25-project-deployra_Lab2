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
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_path = "data/processed/df_clean.csv"
    print("📦 Loading:", data_path)
    df = pd.read_csv(data_path)

    # pick the correct target
    target = cfg.get("target")
    if target is None:
        # common CBB columns – change if needed
        for c in ["listing_type","status","is_sold","Sold","label","target"]:
            if c in df.columns: target = c; break
    if target is None or target not in df.columns:
        raise ValueError(f"Target column not found. Set 'target' in config/train_config.yaml. Columns: {list(df.columns)}")

    print("🎯 Target:", target)
    X, y = df.drop(columns=[target]), df[target]

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    params = cfg.setdefault("params", {}).setdefault("random_forest",
        {"n_estimators": 100, "max_depth": 10, "min_samples_split": 2, "min_samples_leaf": 1})
    print("🤖 RandomForest params:", params)

    model = RandomForestClassifier(**params).fit(Xtr, ytr)
    acc = accuracy_score(yte, model.predict(Xte))
    print(f"✅ Accuracy: {acc:.4f}")

    Path("models").mkdir(parents=True, exist_ok=True)
    out = Path("models/random_forest_latest.pkl").resolve()
    joblib.dump(model, out)
    print("💾 Model saved to:", out)

if __name__ == "__main__":
    main()
PY
