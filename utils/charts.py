# reusable Plotly chart functions
# utils/charts.py
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

FRAUD_COLOR  = '#E24B4A'
NORMAL_COLOR = '#378ADD'
ACCENT_COLOR = '#1D9E75'

def plot_class_distribution(y):
    counts = y.value_counts()
    fig = go.Figure(go.Bar(
        x=['Normal', 'Fraud'],
        y=[counts.get(0, 0), counts.get(1, 0)],
        marker_color=[NORMAL_COLOR, FRAUD_COLOR],
        text=[f'{counts.get(0,0):,}', f'{counts.get(1,0):,}'],
        textposition='outside'
    ))
    fig.update_layout(
        title='Class Distribution',
        yaxis_title='Count',
        showlegend=False,
        height=350
    )
    return fig

def plot_confusion_matrix(y_true, y_pred):
    cm    = confusion_matrix(y_true, y_pred)
    total = cm.sum()
    labels = [
        [f"TN: {cm[0,0]:,}<br>({cm[0,0]/total*100:.2f}%)",
         f"FP: {cm[0,1]:,}<br>({cm[0,1]/total*100:.2f}%)"],
        [f"FN: {cm[1,0]:,}<br>({cm[1,0]/total*100:.2f}%)",
         f"TP: {cm[1,1]:,}<br>({cm[1,1]/total*100:.2f}%)"]
    ]
    fig = go.Figure(go.Heatmap(
        z=cm,
        x=['Predicted Normal', 'Predicted Fraud'],
        y=['Actual Normal',    'Actual Fraud'],
        text=labels,
        texttemplate="%{text}",
        colorscale='Blues',
        showscale=False
    ))
    fig.update_layout(title='Confusion Matrix', height=350)
    return fig

def plot_roc_curve(y_true, lr_probs, rf_probs):
    fig = go.Figure()
    for probs, name, color in [
        (lr_probs, 'Logistic Regression', '#D85A30'),
        (rf_probs, 'Random Forest',       '#185FA5')
    ]:
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc         = roc_auc_score(y_true, probs)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode='lines',
            name=f'{name} (AUC={auc:.4f})',
            line=dict(color=color, width=2.5)
        ))
    fig.add_trace(go.Scatter(
        x=[0,1], y=[0,1], mode='lines',
        name='Random baseline',
        line=dict(color='gray', dash='dash', width=1)
    ))
    fig.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate (Recall)',
        height=400
    )
    return fig

def plot_feature_importance(model, feature_names, top_n=15):
    imp = pd.Series(model.feature_importances_, index=feature_names)
    imp = imp.nlargest(top_n).sort_values()
    fig = go.Figure(go.Bar(
        x=imp.values,
        y=imp.index,
        orientation='h',
        marker_color=[
            FRAUD_COLOR if 'V14' in f or 'V17' in f
            else '#7F77DD' if any(e in f for e in ['_x_', 'magnitude', 'micro'])
            else NORMAL_COLOR
            for f in imp.index
        ]
    ))
    fig.update_layout(
        title=f'Top {top_n} Feature Importances',
        xaxis_title='Importance Score',
        height=500
    )
    return fig

def fraud_gauge(probability):
    """A speedometer-style gauge showing fraud probability."""
    fig = go.Figure(go.Indicator(
        mode='gauge+number+delta',
        value=probability * 100,
        number={'suffix': '%', 'font': {'size': 36}},
        delta={'reference': 30, 'suffix': '%'},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar':  {'color': FRAUD_COLOR if probability >= 0.30 else ACCENT_COLOR},
            'steps': [
                {'range': [0,  30],  'color': '#E1F5EE'},
                {'range': [30, 60],  'color': '#FAEEDA'},
                {'range': [60, 100], 'color': '#FCEBEB'}
            ],
            'threshold': {
                'line':  {'color': 'black', 'width': 2},
                'thickness': 0.75,
                'value': 30
            }
        },
        title={'text': 'Fraud Probability', 'font': {'size': 18}}
    ))
    fig.update_layout(height=280)
    return fig