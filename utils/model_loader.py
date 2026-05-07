# load models + scaler (cached)
# utils/model_loader.py
import streamlit as st
import joblib
import pandas as pd

@st.cache_resource          # Load once, cache forever in this session
def load_models():
    """Load both trained models from disk."""
    rf  = joblib.load('models/random_forest_best.pkl')
    lr  = joblib.load('models/logistic_regression.pkl')
    return rf, lr

@st.cache_resource
def load_scaler():
    """Load the StandardScaler fitted in Phase 2."""
    return joblib.load('preprocessed/scaler.pkl')

@st.cache_data              # Cache the dataset — 284k rows, load once
def load_dataset():
    """Load the raw creditcard.csv for EDA pages."""
    df = pd.read_csv('creditcard.csv')
    df = df.drop_duplicates()
    return df