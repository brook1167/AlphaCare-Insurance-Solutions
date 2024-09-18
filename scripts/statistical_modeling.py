import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import shap


def prepare_features(df, numeric_features=None, categorical_features=None):
    # Default to all numeric and categorical features if none are provided
    if numeric_features is None:
        numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    if categorical_features is None:
        categorical_features = df.select_dtypes(include=['object', 'bool']).columns
    
    # Remove target variables from features
    features = list(set(numeric_features) | set(categorical_features) - set(['TotalPremium', 'TotalClaims']))
    
    return features


def label_encode_columns(df, categorical_features):
    # Apply LabelEncoder to each categorical column separately
    for col in categorical_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df

def preprocess_data(df, features):
    # Separate features and target
    X = df[features]
    y_premium = df['TotalPremium']
    y_claims = df['TotalClaims']
    
    # Split the data
    X_train, X_test, y_premium_train, y_premium_test, y_claims_train, y_claims_test = train_test_split(
        X, y_premium, y_claims, test_size=0.2, random_state=42)
    
    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'bool']).columns
    
    # Apply LabelEncoder to categorical features
    X_train = label_encode_columns(X_train, categorical_features)
    X_test = label_encode_columns(X_test, categorical_features)
    
    # Create preprocessing steps (now only for numeric features since categorical are label encoded)
    numeric_transformer = SimpleImputer(strategy='mean')
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ],
        remainder='passthrough'  # To keep the label-encoded categorical columns
    )
    
    return X_train, X_test, y_premium_train, y_premium_test, y_claims_train, y_claims_test, preprocessor

def build_models():
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
    }
    return models

def train_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor, models):
    results = {}
    for name, model in models.items():
        # Create a pipeline with preprocessor and model
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        # Fit the pipeline
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {'MSE': mse, 'R2': r2, 'Model': pipeline}
    
    return results

def analyze_feature_importance(model, X):
    # For tree-based models, we can use feature_importances_
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance = pd.DataFrame({'feature': X.columns, 'importance': importances})
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        return feature_importance
    else:
        return None

def interpret_model_with_shap(model, X):
    # Create a SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X)
    
    return shap_values, X


def summarize_model_performance():
    # Summary
    print("\nSummary:")
    print("Based on the R2 scores, the best model for predicting both TotalPremium and TotalClaims.")

    # Identify the best model based on R2 score
    top_premium_model = max(premium_results, key=lambda model: premium_results[model]['R2'])
    top_claims_model = max(claims_results, key=lambda model: claims_results[model]['R2'])
    print(f"Best model for TotalPremium: {top_premium_model}")
    print(f"Best model for TotalClaims: {top_claims_model}")

    print("\nKey features for predicting TotalPremium include:")
    print(premium_importance['feature'][:5].tolist())

    print("\nKey features for predicting TotalClaims include:")
    print(claims_importance['feature'][:5].tolist())

