# upload a csv, get all predictions
# pages/3_batch_upload.py
import streamlit as st
import pandas as pd
from utils.model_loader  import load_models, load_scaler
from utils.preprocessing import prepare_for_prediction, predict_with_threshold

st.title('📁 Batch Prediction')
st.markdown(
    'Upload a CSV with the same columns as `creditcard.csv` '
    '(without the `Class` column). The model will score every row.'
)

rf_model, lr_model = load_models()
scaler     = load_scaler()
threshold  = st.session_state.get('threshold', 0.30)
model_name = st.session_state.get('selected_model', 'Random Forest')
model      = rf_model if model_name == 'Random Forest' else lr_model

uploaded = st.file_uploader('Upload CSV file', type='csv')

if uploaded is not None:
    df = pd.read_csv(uploaded)

    # Drop Class column if it exists (evaluation set scenario)
    if 'Class' in df.columns:
        y_true = df['Class'].copy()
        df     = df.drop('Class', axis=1)
        has_labels = True
    else:
        has_labels = False

    st.write(f'Loaded **{len(df):,} transactions**')
    st.dataframe(df.head(3))

    if st.button('Run batch prediction', type='primary'):
        with st.spinner('Preprocessing and scoring all transactions...'):
            X_processed  = prepare_for_prediction(df, scaler)
            probs, preds = predict_with_threshold(model, X_processed, threshold)

        # Build results DataFrame
        results = df.copy()
        results['fraud_probability'] = probs.round(4)
        results['prediction']        = preds
        results['verdict']           = results['prediction'].map(
            {0: 'NORMAL', 1: 'FRAUD'}
        )

        # Summary metrics
        n_fraud   = preds.sum()
        n_normal  = len(preds) - n_fraud
        fraud_pct = n_fraud / len(preds) * 100

        c1, c2, c3 = st.columns(3)
        c1.metric('Total transactions', f'{len(preds):,}')
        c2.metric('Flagged as fraud',   f'{n_fraud:,}')
        c3.metric('Fraud rate detected', f'{fraud_pct:.3f}%')

        # Show flagged transactions
        st.subheader('Flagged transactions (FRAUD)')
        fraud_rows = results[results['prediction'] == 1].sort_values(
            'fraud_probability', ascending=False
        )
        st.dataframe(fraud_rows[['fraud_probability', 'verdict', 'Amount', 'Time']]
                     if 'Amount' in fraud_rows.columns
                     else fraud_rows[['fraud_probability', 'verdict']])

        # If labels available, show evaluation metrics
        if has_labels:
            from sklearn.metrics import classification_report
            st.subheader('Evaluation (labels found in uploaded file)')
            report = classification_report(
                y_true, preds,
                target_names=['Normal', 'Fraud'],
                output_dict=True
            )
            st.dataframe(pd.DataFrame(report).T.round(3))

        # Download results
        csv_out = results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label     = 'Download predictions as CSV',
            data      = csv_out,
            file_name = 'fraud_predictions.csv',
            mime      = 'text/csv'
        )