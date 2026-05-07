# dashboard overview
# pages/1_home.py
import streamlit as st
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from utils.model_loader    import load_models, load_scaler, load_dataset
from utils.preprocessing   import prepare_for_prediction, predict_with_threshold
from utils.charts          import (plot_confusion_matrix, plot_roc_curve,
                                   plot_class_distribution)

st.title('📊 Dashboard')
st.markdown('Full evaluation of both models on the held-out test set.')

# Load everything
rf_model, lr_model = load_models()
scaler              = load_scaler()

# Load test data
# In a real project load from phase3_output/X_test_final.csv
try:
    X_test = pd.read_csv('phase3_output/X_test_final.csv')
    y_test = pd.read_csv('phase3_output/y_test_final.csv').squeeze()
except FileNotFoundError:
    st.error(
        'Test data not found. Run the pipeline phases first '
        'to generate phase3_output/X_test_final.csv'
    )
    st.stop()

# Get threshold from global sidebar
threshold = st.session_state.get('threshold', 0.30)
model_name = st.session_state.get('selected_model', 'Random Forest')
model = rf_model if model_name == 'Random Forest' else lr_model

# Predictions
rf_probs, rf_preds = predict_with_threshold(rf_model, X_test, threshold)
lr_probs, lr_preds = predict_with_threshold(lr_model, X_test, threshold)
probs, preds       = (rf_probs, rf_preds) if model_name == 'Random Forest' \
                     else (lr_probs, lr_preds)

# ─── METRIC CARDS ────────────────────────────────────────────────────────
from sklearn.metrics import recall_score, precision_score

st.subheader(f'Active model: {model_name}  |  Threshold: {threshold}')
m1, m2, m3, m4 = st.columns(4)

recall    = recall_score(y_test, preds)
precision = precision_score(y_test, preds, zero_division=0)
f1        = f1_score(y_test, preds)
auc       = roc_auc_score(y_test, probs)

m1.metric('Recall',    f'{recall:.3f}',    help='Fraction of fraud caught')
m2.metric('Precision', f'{precision:.3f}', help='Fraction of alerts that are real fraud')
m3.metric('F1 Score',  f'{f1:.3f}',        help='Harmonic mean of precision and recall')
m4.metric('ROC-AUC',   f'{auc:.4f}',       help='Overall model discrimination ability')

st.markdown('---')

# ─── CHARTS ROW ──────────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.plotly_chart(
        plot_confusion_matrix(y_test, preds),
        use_container_width=True
    )

with col_right:
    st.plotly_chart(
        plot_roc_curve(y_test, lr_probs, rf_probs),
        use_container_width=True
    )

# ─── CLASS DISTRIBUTION ──────────────────────────────────────────────────
st.markdown('---')
st.subheader('Dataset class distribution')
st.plotly_chart(
    plot_class_distribution(y_test),
    use_container_width=True
)