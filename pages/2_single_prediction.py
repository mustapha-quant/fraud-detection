# predict one transaction
# pages/2_single_prediction.py
import streamlit as st
import pandas as pd
import numpy as np
from utils.model_loader  import load_models, load_scaler
from utils.preprocessing import prepare_for_prediction, predict_with_threshold
from utils.charts        import fraud_gauge

st.title('🔍 Single Transaction Prediction')
st.markdown(
    'Adjust the feature sliders to describe a transaction, '
    'then click **Predict** to score it.'
)

rf_model, lr_model = load_models()
scaler             = load_scaler()
threshold          = st.session_state.get('threshold', 0.30)
model_name         = st.session_state.get('selected_model', 'Random Forest')
model              = rf_model if model_name == 'Random Forest' else lr_model

# ─── INPUT FORM ──────────────────────────────────────────────────────────
st.subheader('Transaction details')
st.caption(
    'V1–V28 are PCA-transformed features. '
    'Adjust Amount and Time freely — the rest simulate feature values.'
)

col1, col2 = st.columns(2)

with col1:
    amount = st.number_input('Amount (€)', min_value=0.0,
                              max_value=30000.0, value=50.0, step=1.0)
    time   = st.number_input('Time (seconds since first txn)',
                              min_value=0.0, max_value=200000.0, value=50000.0)
    v1     = st.slider('V1',  -30.0, 30.0, -1.36, 0.01)
    v2     = st.slider('V2',  -30.0, 30.0, -0.07, 0.01)
    v3     = st.slider('V3',  -30.0, 30.0,  2.54, 0.01)
    v4     = st.slider('V4',  -20.0, 20.0,  1.38, 0.01)
    v5     = st.slider('V5',  -20.0, 20.0, -0.34, 0.01)
    v6     = st.slider('V6',  -20.0, 20.0,  0.46, 0.01)
    v7     = st.slider('V7',  -30.0, 30.0,  0.24, 0.01)
    v8     = st.slider('V8',  -20.0, 20.0,  0.10, 0.01)
    v9     = st.slider('V9',  -20.0, 20.0,  0.36, 0.01)
    v10    = st.slider('V10', -30.0, 30.0,  0.09, 0.01)
    v11    = st.slider('V11', -20.0, 20.0, -0.55, 0.01)
    v12    = st.slider('V12', -20.0, 20.0,  0.73, 0.01)
    v13    = st.slider('V13', -10.0, 10.0, -1.39, 0.01)
    v14    = st.slider('V14', -20.0, 20.0, -0.05, 0.01)

with col2:
    v15    = st.slider('V15', -10.0, 10.0,  0.21, 0.01)
    v16    = st.slider('V16', -20.0, 20.0, -1.16, 0.01)
    v17    = st.slider('V17', -30.0, 30.0,  0.06, 0.01)
    v18    = st.slider('V18', -10.0, 10.0, -0.18, 0.01)
    v19    = st.slider('V19', -10.0, 10.0,  0.17, 0.01)
    v20    = st.slider('V20', -30.0, 30.0,  0.13, 0.01)
    v21    = st.slider('V21', -30.0, 30.0, -0.15, 0.01)
    v22    = st.slider('V22', -10.0, 10.0, -0.07, 0.01)
    v23    = st.slider('V23', -10.0, 10.0,  0.03, 0.01)
    v24    = st.slider('V24', -5.0,  5.0,   0.08, 0.01)
    v25    = st.slider('V25', -5.0,  5.0,   0.13, 0.01)
    v26    = st.slider('V26', -5.0,  5.0,  -0.19, 0.01)
    v27    = st.slider('V27', -30.0, 30.0,  0.03, 0.01)
    v28    = st.slider('V28', -10.0, 10.0,  0.01, 0.01)

# ─── PREDICT BUTTON ──────────────────────────────────────────────────────
if st.button('Predict transaction', type='primary', use_container_width=True):
    # Build DataFrame matching the raw CSV format
    txn = pd.DataFrame([{
        'Time': time, 'Amount': amount,
        'V1':v1,'V2':v2,'V3':v3,'V4':v4,'V5':v5,
        'V6':v6,'V7':v7,'V8':v8,'V9':v9,'V10':v10,
        'V11':v11,'V12':v12,'V13':v13,'V14':v14,'V15':v15,
        'V16':v16,'V17':v17,'V18':v18,'V19':v19,'V20':v20,
        'V21':v21,'V22':v22,'V23':v23,'V24':v24,'V25':v25,
        'V26':v26,'V27':v27,'V28':v28
    }])

    # Apply identical preprocessing to Phase 2 and 3
    txn_processed = prepare_for_prediction(txn, scaler)

    # Predict
    probs, preds = predict_with_threshold(model, txn_processed, threshold)
    probability  = float(probs[0])
    is_fraud     = bool(preds[0])

    st.markdown('---')
    st.subheader('Result')

    # Gauge chart
    st.plotly_chart(fraud_gauge(probability), use_container_width=True)

    # Verdict banner
    if is_fraud:
        st.error(
            f'🚨 **FRAUD DETECTED**  '
            f'Fraud probability: **{probability*100:.1f}%**  '
            f'(threshold: {threshold*100:.0f}%)'
        )
    else:
        st.success(
            f'✅ **NORMAL TRANSACTION**  '
            f'Fraud probability: **{probability*100:.1f}%**  '
            f'(threshold: {threshold*100:.0f}%)'
        )

    # Feature breakdown
    with st.expander('Show engineered feature values'):
        st.dataframe(txn_processed.T.rename(columns={0: 'Value'}))
