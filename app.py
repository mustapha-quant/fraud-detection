# main entry point (run this)
# app.py
import streamlit as st

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────
# Must be the very first Streamlit command in the file
st.set_page_config(
    page_title  = 'Fraud Detection ML',
    page_icon   = '🛡️',
    layout      = 'wide',           # Use full screen width
    initial_sidebar_state = 'expanded'
)

# ─── GLOBAL SIDEBAR CONTROLS ─────────────────────────────────────────────
# These controls appear on every page
with st.sidebar:
    st.title('🛡️ Fraud Detector')
    st.markdown('---')

    # Model selector — stored in session state so all pages see it
    st.session_state['selected_model'] = st.selectbox(
        'Active model',
        options=['Random Forest', 'Logistic Regression'],
        index=0
    )

    # Threshold slider
    st.session_state['threshold'] = st.slider(
        'Decision threshold',
        min_value=0.05,
        max_value=0.95,
        value=0.30,
        step=0.05,
        help='Transactions with fraud probability above this value are flagged. '
             'Lower = catch more fraud. Higher = fewer false alarms.'
    )

    st.markdown('---')
    st.caption('Built with scikit-learn + Streamlit')
    st.caption('Dataset: Kaggle Credit Card Fraud')

# ─── LANDING PAGE ────────────────────────────────────────────────────────
st.title('Credit Card Fraud Detection')
st.markdown(
    'A complete ML pipeline for real-time fraud scoring. '
    'Use the sidebar to navigate between pages.'
)

col1, col2, col3, col4 = st.columns(4)
col1.metric('Total Transactions', '284,807')
col2.metric('Fraud Rate',         '0.17%')
col3.metric('Model ROC-AUC',      '0.9823')
col4.metric('Recall',             '95.4%')

st.info(
    'Navigate using the sidebar. Start with **Home** for a full dashboard, '
    'or jump to **Single Prediction** to score a transaction right now.'
)