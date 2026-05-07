# explore the original dataset
# pages/5_eda_explorer.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from utils.model_loader import load_dataset

st.title('🔬 EDA Explorer')
st.markdown('Explore the raw dataset interactively.')

df = load_dataset()

# ─── TOP STATS ───────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric('Total rows',     f'{len(df):,}')
c2.metric('Fraud cases',    f'{df["Class"].sum():,}')
c3.metric('Fraud rate',     f'{df["Class"].mean()*100:.4f}%')
c4.metric('Median amount',  f'€{df["Amount"].median():.2f}')

st.markdown('---')

# ─── FEATURE SELECTOR ────────────────────────────────────────────────────
st.subheader('Feature distribution by class')
feature = st.selectbox(
    'Select feature to explore',
    options=[c for c in df.columns if c != 'Class']
)

fig = px.histogram(
    df, x=feature, color='Class',
    color_discrete_map={0: '#378ADD', 1: '#E24B4A'},
    labels={'Class': 'Transaction type'},
    nbins=60, barmode='overlay', opacity=0.7,
    title=f'Distribution of {feature} by class'
)
fig.update_layout(height=380)
st.plotly_chart(fig, use_container_width=True)

# ─── CORRELATION WITH FRAUD ───────────────────────────────────────────────
st.markdown('---')
st.subheader('Feature correlation with fraud (Class column)')
corr = df.corr()['Class'].drop('Class').sort_values()
colors = ['#E24B4A' if v < 0 else '#378ADD' for v in corr.values]
fig2 = go.Figure(go.Bar(
    x=corr.values, y=corr.index,
    orientation='h',
    marker_color=colors
))
fig2.update_layout(
    title='Pearson correlation with Class (fraud=1)',
    xaxis_title='Correlation coefficient',
    height=600
)
st.plotly_chart(fig2, use_container_width=True)

# ─── AMOUNT STATS TABLE ───────────────────────────────────────────────────
st.markdown('---')
st.subheader('Transaction amount: normal vs fraud')
st.dataframe(
    df.groupby('Class')['Amount']
      .describe()
      .rename(index={0: 'Normal', 1: 'Fraud'})
      .round(2),
    use_container_width=True
)