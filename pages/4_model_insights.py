# feature importance, ROC, PR curves
# pages/4_model_insights.py
import streamlit as st
import pandas as pd
import numpy as np
from utils.model_loader import load_models, load_scaler, load_dataset
from utils.charts       import plot_feature_importance, plot_roc_curve
from utils.preprocessing import prepare_for_prediction, predict_with_threshold

st.title('🧠 Model Insights')
st.markdown('Feature importances, ROC/PR curves, and threshold analysis.')

rf_model, lr_model = load_models()
scaler             = load_scaler()

# ─── FEATURE IMPORTANCE ──────────────────────────────────────────────────
st.subheader('Feature importances — Random Forest')
st.caption(
    'Red = strongest fraud signals from EDA (V14, V17). '
    'Purple = engineered features. Blue = other V features.'
)

try:
    X_test = pd.read_csv('phase3_output/X_test_final.csv')
    feature_names = X_test.columns.tolist()
    st.plotly_chart(
        plot_feature_importance(rf_model, feature_names),
        use_container_width=True
    )
except FileNotFoundError:
    st.warning('Run the pipeline to generate test data first.')

# ─── THRESHOLD ANALYSIS ──────────────────────────────────────────────────
st.markdown('---')
st.subheader('Threshold sweep — precision vs recall trade-off')

try:
    y_test  = pd.read_csv('phase3_output/y_test_final.csv').squeeze()
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    lr_probs = lr_model.predict_proba(X_test)[:, 1]

    thresholds = np.arange(0.05, 0.95, 0.05)
    rows = []
    for t in thresholds:
        from sklearn.metrics import recall_score, precision_score, f1_score
        preds = (rf_probs >= t).astype(int)
        rows.append({
            'Threshold':  round(float(t), 2),
            'Recall':     round(recall_score(y_test, preds),                  3),
            'Precision':  round(precision_score(y_test, preds, zero_division=0), 3),
            'F1':         round(f1_score(y_test, preds, zero_division=0),     3),
            'FN (missed)': int((y_test == 1) & (preds == 0)).sum()
        })

    thresh_df = pd.DataFrame(rows)

    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=thresh_df['Threshold'], y=thresh_df['Recall'],
                             name='Recall', line=dict(color='#E24B4A', width=2)))
    fig.add_trace(go.Scatter(x=thresh_df['Threshold'], y=thresh_df['Precision'],
                             name='Precision', line=dict(color='#378ADD', width=2)))
    fig.add_trace(go.Scatter(x=thresh_df['Threshold'], y=thresh_df['F1'],
                             name='F1', line=dict(color='#1D9E75', width=2)))

    current = st.session_state.get('threshold', 0.30)
    fig.add_vline(x=current, line_dash='dash', line_color='gray',
                  annotation_text=f'Current: {current}')
    fig.update_layout(
        title='Recall / Precision / F1 at every threshold (Random Forest)',
        xaxis_title='Threshold',
        yaxis_title='Score',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(thresh_df, use_container_width=True)

    # ROC curve
    st.markdown('---')
    st.subheader('ROC Curve — both models')
    st.plotly_chart(
        plot_roc_curve(y_test, lr_probs, rf_probs),
        use_container_width=True
    )

except FileNotFoundError:
    st.warning('Run the pipeline phases first to generate evaluation data.')