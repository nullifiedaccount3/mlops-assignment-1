"""
Utility functions for machine learning workflow.
This module contains generic functions for data loading, preprocessing,
model training, and evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


def load_data():
    """
    Load the Boston Housing dataset from the original source.
    
    Returns:
        pd.DataFrame: DataFrame containing features and target variable (MEDV)
    """
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
    
    # Split into data and target
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    
    # Feature names based on the original dataset
    feature_names = [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
        'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
    ]
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=feature_names)
    df['MEDV'] = target  # MEDV is the target variable
    
    return df


def preprocess_data(df, target_column='MEDV', test_size=0.2, random_state=42, scale=True):
    """
    Preprocess the dataset by splitting into train/test sets and optionally scaling.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        target_column (str): Name of the target column
        test_size (float): Proportion of dataset to include in test split
        random_state (int): Random state for reproducibility
        scale (bool): Whether to apply standard scaling to features
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
               scaler is None if scale=False
    """
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    scaler = None
    if scale:
        # Apply standard scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler


def train_model(model, X_train, y_train):
    """
    Train a machine learning model.
    
    Args:
        model: Scikit-learn model instance
        X_train: Training features
        y_train: Training target
    
    Returns:
        Trained model
    """
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained scikit-learn model
        X_test: Test features
        y_test: Test target
    
    Returns:
        dict: Dictionary containing predictions and MSE score
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate MSE
    mse = mean_squared_error(y_test, y_pred)
    
    return {
        'predictions': y_pred,
        'mse': mse,
        'actual': y_test
    }


def display_results(model_name, results):
    """
    Display model evaluation results.
    
    Args:
        model_name (str): Name of the model
        results (dict): Results dictionary from evaluate_model
    """
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    print(f"Mean Squared Error (MSE): {results['mse']:.4f}")
    print(f"Root Mean Squared Error (RMSE): {np.sqrt(results['mse']):.4f}")
    print(f"{'='*60}\n")
