cat > src/train.py << 'PY'
import argparse, yaml, pandas as pd, joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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

    target = cfg.get("target", "listing_type")
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found. Columns: {list(df.columns)}")

    print("🎯 Target:", target)
    y = df[target]
    X = df.drop(columns=[target])

    # Split numeric vs categorical features automatically
    cat_cols = X.select_dtypes(include=["object","category","string"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number","bool"]).columns.tolist()
    print(f"🧱 Features -> cat: {len(cat_cols)} | num: {len(num_cols)}")

    # Preprocess: OneHot for categoricals (ignore unseen), passthrough numerics
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop"
    )

    rf_params = cfg.setdefault("params", {}).setdefault("random_forest", {
        "n_estimators": 200, "max_depth": 12, "min_samples_split": 2, "min_samples_leaf": 1
    })
    print("🤖 RF params:", rf_params)

    model = Pipeline([
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(**rf_params))
    ])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if y.nunique()<50 else None)
    model.fit(Xtr, ytr)
    acc = accuracy_score(yte, model.predict(Xte))
    print(f"✅ Accuracy: {acc:.4f}")

    Path("models").mkdir(parents=True, exist_ok=True)
    out = Path("models/cbb_rf_pipeline.pkl").resolve()
    joblib.dump(model, out)
    print("💾 Pipeline saved to:", out)

if __name__ == "__main__":
    main()
PY
