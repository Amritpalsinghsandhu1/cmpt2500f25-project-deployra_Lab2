cat > src/train.py << 'PY'
import argparse, yaml, numpy as np, joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

def load_yaml(p):
    with open(p, "r") as f: return yaml.safe_load(f)

def get_model(kind, params):
    if kind == "random_forest": return RandomForestClassifier(**params)
    if kind == "logistic_regression": return LogisticRegression(**params)
    if kind == "gradient_boosting": return GradientBoostingClassifier(**params)
    if kind == "adaboost": return AdaBoostClassifier(**params)
    raise ValueError(f"Unknown model type: {kind}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/train_config.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    model_type = cfg["model"]["type"]
    params = cfg["params"][model_type]

    data = np.load("data/processed/preprocessed_data.npy", allow_pickle=True).item()
    X_train, y_train = data["X_train"], data["y_train"]

    model = get_model(model_type, params)
    model.fit(X_train, y_train)

    Path("models").mkdir(exist_ok=True)
    out = f"models/{model_type}_latest.pkl"
    joblib.dump(model, out)
    print(f"✅ Model saved: {out}")

if __name__ == "__main__":
    main()
PY
