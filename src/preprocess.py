import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_clean_data(path: str) -> pd.DataFrame:
    """Load dataset and apply all cleaning steps."""
    df = pd.read_csv(path)

    # Keep only sold listings
    df = df[df["listing_dropoff_date"].notnull()]

    # Drop unnecessary columns
    drop_cols = [
        "listing_id", "listing_heading", "listing_url", "listing_first_date",
        "days_on_market", "dealer_id", "dealer_name", "dealer_street",
        "dealer_city", "dealer_province", "dealer_postal_code", "dealer_url",
        "dealer_email", "dealer_phone", "dealer_type", "vehicle_id", "vin",
        "uvc", "series", "style", "has_leather", "has_navigation",
        "price_analysis", "wheelbase_from_vin", "number_price_changes",
        "price_history_delimited", "distance_to_dealer", "location_score",
        "listing_dropoff_date", "exterior_color", "interior_color"
    ]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Price filters
    df = df[df["price"] >= 1000]
    price_99th = df["price"].quantile(0.99)
    df = df[df["price"] <= price_99th]

    # Fix unrealistic model years
    df = df[(df["model_year"] >= 1900) & (df["model_year"] <= 2025)]

    # Fill missing color categories
    for col in ["exterior_color_category", "interior_color_category"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Add new/used flag
    df["is_new"] = df["mileage"].apply(lambda x: 1 if x < 100 else 0)

    return df


def prepare_used_data(df: pd.DataFrame):
    """Prepare used vehicle dataset for modeling."""
    df_used = df[df["is_new"] == 0].copy()

    # Add vehicle age
    df_used["age"] = 2025 - df_used["model_year"]

    # Select features and target
    X = df_used[[
        "make", "model", "model_year", "mileage",
        "fuel_type_from_vin", "transmission_from_vin",
        "stock_type", "age"
    ]].copy()
    y = df_used["price"]

    # Encode categorical variables
    label_encoders = {}
    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    return X, y, label_encoders


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


if __name__ == "__main__":
    # Run preprocessing
    df = load_and_clean_data("data/raw/CBB_Listings.csv")

    # Save cleaned dataset
    df.to_csv("data/processed/df_clean.csv", index=False)

    # Preview output
    print(df.head())
    print(f"\n Cleaned dataset saved to data/processed/df_clean.csv")
    print(f"Final shape: {df.shape}")
