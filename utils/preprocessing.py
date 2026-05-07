# feature engineering functions
# utils/preprocessing.py
import pandas as pd
import numpy as np

def scale_features(df, scaler):
    """
    Apply StandardScaler to Amount and Time.
    Must use the SAME scaler fitted in Phase 2 — never refit.
    """
    df = df.copy()
    df[['scaled_amount', 'scaled_time']] = scaler.transform(
        df[['Amount', 'Time']]
    )
    df = df.drop(['Amount', 'Time'], axis=1)
    return df

def add_engineered_features(df):
    """
    Add the four features created in Phase 3.
    Must be applied identically to any data the model will score.
    """
    df = df.copy()
    df['is_micro_txn'] = (df['scaled_amount'] < -0.35).astype(int)
    df['V14_x_V17']    = df['V14'] * df['V17']
    df['V14_x_V12']    = df['V14'] * df['V12']
    df['V17_x_V10']    = df['V17'] * df['V10']
    df['fraud_signal_magnitude'] = (
        df[['V14','V17','V12','V10','V11','V4']].abs().sum(axis=1)
    )
    return df

def prepare_for_prediction(df, scaler):
    """
    Full preprocessing pipeline for any new data.
    Call this on X_test or any new transaction DataFrame.
    """
    df = scale_features(df, scaler)
    df = add_engineered_features(df)
    return df

def predict_with_threshold(model, X, threshold=0.30):
    """
    Get predictions using a custom threshold instead of default 0.5.
    Returns both the probability and the binary prediction.
    """
    probabilities = model.predict_proba(X)[:, 1]
    predictions   = (probabilities >= threshold).astype(int)
    return probabilities, predictions