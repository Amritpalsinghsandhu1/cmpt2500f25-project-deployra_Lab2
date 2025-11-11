def create_preprocessing_pipeline(numerical_features,
categorical_features):
 """
 Create sklearn preprocessing pipeline.

 Args:
 numerical_features: List of numerical feature names
 categorical_features: List of categorical feature names

 Returns:
 ColumnTransformer pipeline
 """
 from sklearn.compose import ColumnTransformer
 from sklearn.preprocessing import StandardScaler, OneHotEncoder

 preprocessor = ColumnTransformer(
 transformers=[
 ('num', StandardScaler(), numerical_features),
 ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'),
 categorical_features)
 ],
 remainder='passthrough' # Keep other columns as-is
 )

 return preprocessor
    def main():
 # ... load data ...

 # Define feature groups
 numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges',
'SeniorCitizen']
 categorical_features = ['gender', 'Partner', 'Dependents',
'PhoneService',
 'MultipleLines', 'InternetService',
'OnlineSecurity',
 'OnlineBackup', 'DeviceProtection',
'TechSupport',
 'StreamingTV', 'StreamingMovies', 'Contract',
 'PaperlessBilling', 'PaymentMethod']

 # Create pipeline
 pipeline = create_preprocessing_pipeline(numerical_features,
categorical_features)

 # Fit and transform
 X_train_transformed = pipeline.fit_transform(X_train)
 X_test_transformed = pipeline.transform(X_test)

 # Save pipeline
 joblib.dump(pipeline, 'data/processed/preprocessing_pipeline.pkl')
# Save data
 data = {
 'X_train': X_train_transformed,
 'X_test': X_test_transformed,
 'y_train': y_train,
 'y_test': y_test
 }
 np.save('data/processed/preprocessed_data.npy', data)
