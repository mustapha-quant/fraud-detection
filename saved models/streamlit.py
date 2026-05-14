"""
╔══════════════════════════════════════════════════════════════╗
║     💊 Pharmaceutical Inventory Intelligence Dashboard       ║
║     Streamlit App — All ML Deliverables                      ║
╚══════════════════════════════════════════════════════════════╝

Run with:
    pip install streamlit pandas numpy scikit-learn statsmodels openpyxl plotly
    streamlit run pharma_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_absolute_error, mean_squared_error,
    r2_score, silhouette_score, confusion_matrix, roc_curve
)
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PharmaIQ Dashboard",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS — Dark pharmaceutical theme
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp { background: #0d1117; color: #e6edf3; }

section[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #30363d;
}
section[data-testid="stSidebar"] * { color: #e6edf3 !important; }

.dash-header {
    background: linear-gradient(135deg, #1a2332 0%, #0d1117 60%, #1a1f2e 100%);
    border: 1px solid #30363d;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
}
.dash-badge {
    display: inline-block;
    background: rgba(0,210,110,0.15);
    border: 1px solid rgba(0,210,110,0.3);
    color: #00d26a;
    padding: .2rem .8rem;
    border-radius: 20px;
    font-size: .72rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: .7rem;
}
.dash-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    color: #ffffff;
    margin: 0;
    letter-spacing: -.5px;
}
.dash-subtitle { color: #8b949e; font-size: .95rem; margin-top: .3rem; font-weight: 300; }

.kpi-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 1rem; margin-bottom: 1.5rem; }
.kpi-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    position: relative;
    overflow: hidden;
}
.kpi-card::after {
    content:''; position:absolute; bottom:0; left:0; right:0;
    height:3px; border-radius:0 0 12px 12px;
}
.kpi-card.red::after   { background:#f85149; }
.kpi-card.amber::after { background:#d29922; }
.kpi-card.green::after { background:#3fb950; }
.kpi-card.blue::after  { background:#58a6ff; }

.kpi-label { font-size:.72rem; color:#8b949e; text-transform:uppercase; letter-spacing:1px; font-weight:500; }
.kpi-value { font-size:1.9rem; font-weight:700; color:#fff; line-height:1.2; margin:.3rem 0 0; }
.kpi-sub   { font-size:.75rem; color:#8b949e; margin-top:.2rem; }

.section-title {
    font-family:'DM Serif Display',serif;
    font-size:1.3rem; color:#fff;
    border-left:3px solid #00d26a;
    padding-left:.8rem;
    margin: 1.5rem 0 .8rem;
}
.alert-box { border-radius:10px; padding:.9rem 1.2rem; margin-bottom:.8rem; border-left:4px solid; font-size:.88rem; }
.alert-red   { background:rgba(248,81,73,.1);  border-color:#f85149; }
.alert-amber { background:rgba(210,153,34,.1); border-color:#d29922; }
.alert-green { background:rgba(63,185,80,.1);  border-color:#3fb950; }
.alert-blue  { background:rgba(88,166,255,.1); border-color:#58a6ff; }

.metric-row { display:flex; justify-content:space-between; align-items:center; padding:.5rem 0; border-bottom:1px solid #21262d; }
.metric-name  { color:#8b949e; font-size:.83rem; }
.metric-value { color:#fff; font-weight:600; font-size:.92rem; }
.metric-good  { color:#3fb950; }
.metric-warn  { color:#d29922; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# PLOTLY DARK THEME
# ─────────────────────────────────────────────────────────────
PLOTLY_THEME = dict(
    paper_bgcolor='#161b22',
    plot_bgcolor='#161b22',
    font=dict(color='#e6edf3', family='DM Sans'),
    xaxis=dict(gridcolor='#21262d', linecolor='#30363d'),
    yaxis=dict(gridcolor='#21262d', linecolor='#30363d'),
    margin=dict(l=40, r=20, t=40, b=40),
    colorway=['#00d26a','#58a6ff','#f85149','#d29922','#bc8cff','#fb8500']
)
def apply_theme(fig):
    fig.update_layout(**PLOTLY_THEME)
    return fig

# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data(file):
    xls    = pd.ExcelFile(file)
    sheets = xls.sheet_names
    dfs    = [pd.read_excel(xls, sheet_name=s) for s in sheets[:4]]
    while len(dfs) < 4:
        dfs.append(pd.DataFrame())
    ops_df, hr_df, purchase_df, expense_df = dfs

    for d in [ops_df, hr_df, purchase_df, expense_df]:
        if not d.empty:
            num = d.select_dtypes(include='number').columns
            d[num] = d[num].fillna(d[num].median())
            for c in d.select_dtypes(include='object').columns:
                if d[c].mode().shape[0]:
                    d[c] = d[c].fillna(d[c].mode()[0])

    ops_df['Date'] = pd.to_datetime(ops_df['Date'], errors='coerce')
    if not purchase_df.empty and 'Date' in purchase_df.columns:
        purchase_df['Date'] = pd.to_datetime(purchase_df['Date'], errors='coerce')

    return ops_df, hr_df, purchase_df, expense_df

@st.cache_data
def engineer(ops_df, purchase_df):
    df = ops_df.copy()
    df['Month']      = df['Date'].dt.month
    df['Quarter']    = df['Date'].dt.quarter
    df['DayOfWeek']  = df['Date'].dt.dayofweek
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    df['Year']       = df['Date'].dt.year

    df['Remaining_Stock']     = df['Stock'] - df['Units_Sold']
    df['Stock_Buffer']        = df['Stock'] - df['Reorder_Level']
    df['Stock_Cover_Days']    = df['Stock'] / df['Units_Sold'].replace(0, 0.1)
    df['Days_Until_Stockout'] = df['Stock_Cover_Days'] - df['Delay_Days']
    df['Revenue']             = df['Units_Sold'] * df['Unit_Price']
    df['Stockout_Gap']        = df['Reorder_Level'] - df['Remaining_Stock']
    df['Expiry_Risk_Score']   = df['Units_Sold'] / df['Days_To_Expiry'].replace(0, 0.1)
    df['Will_Expire_Before_Sold'] = (df['Days_To_Expiry'] < df['Stock_Cover_Days']).astype(int)

    if not purchase_df.empty and 'Supplier' in purchase_df.columns and 'Delivery_Days' in purchase_df.columns:
        sup = purchase_df.groupby('Supplier')['Delivery_Days'].mean().reset_index()
        sup.columns = ['Supplier', 'Avg_Delivery_Days']
        df = df.merge(sup, on='Supplier', how='left')
    else:
        df['Avg_Delivery_Days'] = df['Delay_Days']

    df = df.sort_values(['Medicine', 'Date'])
    df['Rolling_Avg_Sales_7d']  = df.groupby('Medicine')['Units_Sold'].transform(lambda x: x.rolling(7,  min_periods=1).mean())
    df['Rolling_Avg_Sales_30d'] = df.groupby('Medicine')['Units_Sold'].transform(lambda x: x.rolling(30, min_periods=1).mean())
    df['Sales_Trend']           = df['Rolling_Avg_Sales_7d'] - df['Rolling_Avg_Sales_30d']

    df['Stockout_Risk'] = ((df['Remaining_Stock'] <= df['Reorder_Level']) | (df['Days_Until_Stockout'] <= 0)).astype(int)
    df['Expiry_Risk']   = ((df['Days_To_Expiry'] <= 30) | (df['Will_Expire_Before_Sold'] == 1)).astype(int)
    return df

@st.cache_resource
def train_all(df):
    CAT  = ['Medicine','Category','Region','Facility','Supplier']
    NUM  = ['Stock','Units_Sold','Reorder_Level','Unit_Price','Delay_Days',
            'Days_To_Expiry','Month','Quarter','DayOfWeek',
            'Remaining_Stock','Stock_Buffer','Stock_Cover_Days','Revenue',
            'Avg_Delivery_Days','Rolling_Avg_Sales_7d','Rolling_Avg_Sales_30d',
            'Sales_Trend','Expiry_Risk_Score','Stockout_Gap']
    FEAT = CAT + NUM

    df_enc = df.copy()
    le = LabelEncoder()
    for c in CAT:
        df_enc[c] = le.fit_transform(df_enc[c].astype(str))

    X  = df_enc[FEAT].fillna(df_enc[FEAT].median())
    yc = df_enc['Stockout_Risk']
    yr = df_enc['Days_Until_Stockout'].clip(lower=0)
    ye = df_enc['Expiry_Risk']

    # Stockout classifier
    Xtc, Xtec, ytc, ytec = train_test_split(X, yc, test_size=.2, random_state=42, stratify=yc)
    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight='balanced', n_jobs=-1)
    clf.fit(Xtc, ytc)
    ypc = clf.predict(Xtec);  ypc_p = clf.predict_proba(Xtec)[:,1]

    # Stockout regressor
    Xtr, Xter, ytr, yter = train_test_split(X, yr, test_size=.2, random_state=42)
    reg = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    reg.fit(Xtr, ytr);  ypr = reg.predict(Xter)

    # Expiry classifier
    Xte, Xtee, yte, ytee = train_test_split(X, ye, test_size=.2, random_state=42, stratify=ye)
    exp_clf = GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=42)
    exp_clf.fit(Xte, yte);  ype = exp_clf.predict(Xtee);  ype_p = exp_clf.predict_proba(Xtee)[:,1]

    # Clustering
    dp = df.groupby('Medicine').agg(
        Avg_Stock=('Stock','mean'), Avg_Units_Sold=('Units_Sold','mean'),
        Avg_Days_Expiry=('Days_To_Expiry','mean'), Avg_Unit_Price=('Unit_Price','mean'),
        Total_Revenue=('Revenue','sum'), Stockout_Rate=('Stockout_Risk','mean'),
        Expiry_Risk_Rate=('Expiry_Risk','mean'), Avg_Stock_Cover=('Stock_Cover_Days','mean')
    ).reset_index()
    CF  = ['Avg_Stock','Avg_Units_Sold','Avg_Days_Expiry','Avg_Unit_Price',
           'Total_Revenue','Stockout_Rate','Expiry_Risk_Rate','Avg_Stock_Cover']
    Xcl = dp[CF].fillna(0)
    sc  = StandardScaler();  Xs = sc.fit_transform(Xcl)

    best_k, best_s = 3, -1
    for k in range(2, min(8, len(dp))):
        lb = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(Xs)
        s  = silhouette_score(Xs, lb)
        if s > best_s: best_s, best_k = s, k

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    dp['Cluster'] = km.fit_predict(Xs)
    pca_coords    = PCA(n_components=2, random_state=42).fit_transform(Xs)
    dp['PCA_1'], dp['PCA_2'] = pca_coords[:,0], pca_coords[:,1]

    # Full-dataset predictions
    df['Stockout_Prob']      = clf.predict_proba(X)[:,1]
    df['Stockout_Risk_Pred'] = clf.predict(X)
    df['Days_To_Stockout']   = reg.predict(X).clip(min=0)
    df['Expiry_Prob']        = exp_clf.predict_proba(X)[:,1]
    df['Expiry_Risk_Pred']   = exp_clf.predict(X)

    def badge(p):
        return '🔴 HIGH' if p>=.7 else '🟡 MEDIUM' if p>=.4 else '🟢 LOW'
    df['Stockout_Level'] = df['Stockout_Prob'].apply(badge)
    df['Expiry_Level']   = df['Expiry_Prob'].apply(badge)

    fpr, tpr, _ = roc_curve(ytec, ypc_p)
    metrics = {
        'clf': dict(accuracy=accuracy_score(ytec,ypc), precision=precision_score(ytec,ypc,zero_division=0),
                    recall=recall_score(ytec,ypc,zero_division=0), f1=f1_score(ytec,ypc,zero_division=0),
                    roc_auc=roc_auc_score(ytec,ypc_p), cm=confusion_matrix(ytec,ypc), fpr=fpr, tpr=tpr),
        'reg': dict(mae=mean_absolute_error(yter,ypr), rmse=float(np.sqrt(mean_squared_error(yter,ypr))),
                    r2=r2_score(yter,ypr), y_te=yter.values, y_pred=ypr),
        'exp': dict(accuracy=accuracy_score(ytee,ype), precision=precision_score(ytee,ype,zero_division=0),
                    recall=recall_score(ytee,ype,zero_division=0), f1=f1_score(ytee,ype,zero_division=0),
                    roc_auc=roc_auc_score(ytee,ype_p)),
        'cluster': dict(silhouette=silhouette_score(Xs, dp['Cluster']), k=best_k),
    }
    fi_clf = pd.Series(clf.feature_importances_, index=FEAT).nlargest(12)
    fi_exp = pd.Series(exp_clf.feature_importances_, index=FEAT).nlargest(10)
    return df, dp, metrics, fi_clf, fi_exp

@st.cache_data
def demand_forecast(ops_df, n=6):
    daily   = ops_df.groupby(['Date','Medicine'])['Units_Sold'].sum().reset_index()
    top     = ops_df.groupby('Medicine')['Units_Sold'].sum().nlargest(n).index.tolist()
    results = {}
    for m in top:
        s = daily[daily['Medicine']==m].set_index('Date')['Units_Sold'].resample('D').sum().fillna(0)
        if len(s) < 7: continue
        try:
            fit   = ExponentialSmoothing(s, trend='add',
                        seasonal='add' if len(s)>=14 else None,
                        seasonal_periods=7).fit(optimized=True)
            fcast = fit.forecast(30).clip(lower=0)
            results[m] = {'history': s.tail(60), 'forecast': fcast}
        except:
            pass
    return results

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 💊 PharmaIQ")
    st.markdown("---")
    uploaded = st.file_uploader("Upload Excel Workbook", type=['xlsx','xls'])
    if uploaded:
        st.success("✅ File loaded")
    st.markdown("---")
    page = st.radio("Navigation", [
        "🏠 Overview", "📦 Stockout Model", "📈 Demand Forecast",
        "⏰ Expiry Risk", "🔵 Segmentation", "📊 Model Metrics",
        "🎯 Recommendations", "📤 Export"
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### Filters")

# ─────────────────────────────────────────────────────────────
# UPLOAD GATE
# ─────────────────────────────────────────────────────────────
if not uploaded:
    st.markdown("""
    <div class="dash-header">
        <div class="dash-badge">PHARMACEUTICAL ML SYSTEM</div>
        <h1 class="dash-title">💊 PharmaIQ Intelligence Dashboard</h1>
        <p class="dash-subtitle">Upload your pharmaceutical Excel workbook (Operations, HR, Purchasing, Expenses) to launch.</p>
    </div>""", unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    for col, icon, title, desc in [
        (c1,"📦","Stockout Prediction","Classify & forecast when drugs will run out"),
        (c2,"⏰","Expiry Risk","Flag products expiring before being sold"),
        (c3,"📈","Demand Forecast","30-day exponential smoothing per medicine"),
        (c4,"🔵","Drug Segments","KMeans clustering into strategic groups"),
    ]:
        with col:
            st.markdown(f"""<div class="kpi-card blue" style="text-align:center;padding:1.5rem">
                <div style="font-size:2rem">{icon}</div>
                <div style="font-weight:600;margin-top:.5rem;color:#fff">{title}</div>
                <div style="font-size:.78rem;color:#8b949e;margin-top:.3rem">{desc}</div>
            </div>""", unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────────
# LOAD & TRAIN
# ─────────────────────────────────────────────────────────────
with st.spinner("🔬 Training models — this takes ~30 seconds on first load..."):
    ops_df, hr_df, purchase_df, expense_df = load_data(uploaded)
    df        = engineer(ops_df, purchase_df)
    df, dp, metrics, fi_clf, fi_exp = train_all(df)
    fc_results = demand_forecast(ops_df)

# Sidebar filters
with st.sidebar:
    sel_region = st.selectbox("Region",   ['All'] + sorted(df['Region'].unique().tolist()))
    sel_cat    = st.selectbox("Category", ['All'] + sorted(df['Category'].unique().tolist()))
    risk_thr   = st.slider("Risk Threshold", 0.0, 1.0, 0.5, 0.05)

fdf = df.copy()
if sel_region != 'All': fdf = fdf[fdf['Region']==sel_region]
if sel_cat    != 'All': fdf = fdf[fdf['Category']==sel_cat]

# ═════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("""<div class="dash-header">
        <div class="dash-badge">LIVE INTELLIGENCE</div>
        <h1 class="dash-title">Inventory Overview</h1>
        <p class="dash-subtitle">Real-time pharmaceutical inventory intelligence across all facilities</p>
    </div>""", unsafe_allow_html=True)

    n_crit  = int((fdf['Stockout_Prob']>=.7).sum())
    n_warn  = int(((fdf['Stockout_Prob']>=.4)&(fdf['Stockout_Prob']<.7)).sum())
    n_exp   = int((fdf['Days_To_Expiry']<=30).sum())
    tot_rev = fdf['Revenue'].sum()

    st.markdown(f"""<div class="kpi-grid">
        <div class="kpi-card red">
            <div class="kpi-label">🔴 Critical Stockout Risk</div>
            <div class="kpi-value">{n_crit:,}</div>
            <div class="kpi-sub">Records ≥ 70% probability</div>
        </div>
        <div class="kpi-card amber">
            <div class="kpi-label">🟡 Warning Stockout</div>
            <div class="kpi-value">{n_warn:,}</div>
            <div class="kpi-sub">40–70% probability</div>
        </div>
        <div class="kpi-card amber">
            <div class="kpi-label">🟠 Expiry Alerts</div>
            <div class="kpi-value">{n_exp:,}</div>
            <div class="kpi-sub">Expiring within 30 days</div>
        </div>
        <div class="kpi-card green">
            <div class="kpi-label">💰 Total Revenue (GHS)</div>
            <div class="kpi-value">{tot_rev/1e6:.1f}M</div>
            <div class="kpi-sub">Across all facilities</div>
        </div>
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">Risk Level Distribution</div>', unsafe_allow_html=True)
        n_safe = int((fdf['Stockout_Prob']<.4).sum())
        fig = px.pie(values=[n_safe, n_warn, n_crit],
                     names=['🟢 LOW','🟡 MEDIUM','🔴 HIGH'],
                     color_discrete_sequence=['#3fb950','#d29922','#f85149'], hole=.5)
        apply_theme(fig); st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Monthly Sales Trend</div>', unsafe_allow_html=True)
        monthly = fdf.groupby(fdf['Date'].dt.to_period('M').astype(str))['Units_Sold'].sum().reset_index()
        monthly.columns = ['Month','Units_Sold']
        fig = px.line(monthly, x='Month', y='Units_Sold', markers=True,
                      color_discrete_sequence=['#00d26a'])
        fig.update_traces(fill='tozeroy', fillcolor='rgba(0,210,106,.08)')
        apply_theme(fig); fig.update_xaxes(tickangle=40)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="section-title">Average Stock by Region</div>', unsafe_allow_html=True)
        rs = fdf.groupby('Region')['Stock'].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(rs, x='Region', y='Stock', color='Stock', color_continuous_scale='Blues')
        apply_theme(fig); fig.update_xaxes(tickangle=30)
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.markdown('<div class="section-title">Top 10 Revenue Medicines</div>', unsafe_allow_html=True)
        tr = fdf.groupby('Medicine')['Revenue'].sum().nlargest(10).reset_index()
        fig = px.bar(tr, x='Revenue', y='Medicine', orientation='h',
                     color='Revenue', color_continuous_scale='Greens')
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        apply_theme(fig); st.plotly_chart(fig, use_container_width=True)

# ═════════════════════════════════════════════════════════════
# PAGE: STOCKOUT MODEL
# ═════════════════════════════════════════════════════════════
elif page == "📦 Stockout Model":
    st.markdown("""<div class="dash-header">
        <div class="dash-badge">ML MODEL</div>
        <h1 class="dash-title">Stockout Prediction</h1>
        <p class="dash-subtitle">Random Forest — Classification (will it stockout?) + Regression (when?)</p>
    </div>""", unsafe_allow_html=True)

    m = metrics['clf']
    cols = st.columns(5)
    for col, label, val in zip(cols,
        ['Accuracy','Precision','Recall','F1','ROC-AUC'],
        [m['accuracy'],m['precision'],m['recall'],m['f1'],m['roc_auc']]
    ):
        color = '#3fb950' if val>=.8 else '#d29922' if val>=.6 else '#f85149'
        with col:
            st.markdown(f"""<div class="kpi-card blue" style="text-align:center">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value" style="color:{color};font-size:1.5rem">{val:.3f}</div>
            </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)
        fig = px.imshow(m['cm'], text_auto=True, color_continuous_scale='Blues',
                        x=['Predicted Safe','Predicted At Risk'],
                        y=['Actual Safe','Actual At Risk'])
        apply_theme(fig); st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">ROC Curve</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=m['fpr'], y=m['tpr'], mode='lines',
                                  name=f"AUC={m['roc_auc']:.3f}", line=dict(color='#00d26a',width=2)))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random',
                                  line=dict(color='#8b949e',dash='dash')))
        fig.update_layout(xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
        apply_theme(fig); st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="section-title">Feature Importances</div>', unsafe_allow_html=True)
        fi_df = fi_clf.reset_index(); fi_df.columns=['Feature','Importance']
        fig = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                     color='Importance', color_continuous_scale='Blues')
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        apply_theme(fig); st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.markdown('<div class="section-title">Top At-Risk Medicines</div>', unsafe_allow_html=True)
        ar = fdf[fdf['Stockout_Prob']>=risk_thr].groupby('Medicine')['Stockout_Prob'].mean().nlargest(15).reset_index()
        fig = px.bar(ar, x='Stockout_Prob', y='Medicine', orientation='h',
                     color='Stockout_Prob', color_continuous_scale='Reds')
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, xaxis_range=[0,1])
        apply_theme(fig); st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">At-Risk Records</div>', unsafe_allow_html=True)
    tbl = fdf[fdf['Stockout_Prob']>=risk_thr][
        ['Medicine','Region','Facility','Stock','Units_Sold','Reorder_Level',
         'Stockout_Prob','Days_To_Stockout','Stockout_Level']
    ].sort_values('Stockout_Prob', ascending=False).head(50)
    tbl['Stockout_Prob'] = tbl['Stockout_Prob'].round(3)
    tbl['Days_To_Stockout'] = tbl['Days_To_Stockout'].round(1)
    st.dataframe(tbl, use_container_width=True, height=350)

# ═════════════════════════════════════════════════════════════
# PAGE: DEMAND FORECAST
# ═════════════════════════════════════════════════════════════
elif page == "📈 Demand Forecast":
    st.markdown("""<div class="dash-header">
        <div class="dash-badge">TIME SERIES</div>
        <h1 class="dash-title">30-Day Demand Forecast</h1>
        <p class="dash-subtitle">Exponential Smoothing — per top medicine</p>
    </div>""", unsafe_allow_html=True)

    if not fc_results:
        st.warning("Not enough data to generate forecasts (need ≥7 days per medicine).")
    else:
        sel = st.selectbox("Select Medicine", list(fc_results.keys()))
        res = fc_results[sel]
        hist, fcast = res['history'], res['forecast']

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name='Historical',
                                  line=dict(color='#58a6ff', width=2)))
        fig.add_trace(go.Scatter(x=fcast.index, y=fcast.values, name='30-Day Forecast',
                                  line=dict(color='#00d26a', width=2, dash='dash')))
        fig.add_trace(go.Scatter(
            x=list(fcast.index)+list(fcast.index[::-1]),
            y=list(fcast.values*1.15)+list(fcast.values[::-1]*0.85),
            fill='toself', fillcolor='rgba(0,210,106,.1)',
            line=dict(color='rgba(0,0,0,0)'), name='±15% CI'
        ))
        fig.add_vline(x=str(hist.index[-1]), line_dash='dot', line_color='#8b949e')
        fig.update_layout(title=f'Demand Forecast — {sel}', xaxis_title='Date', yaxis_title='Units')
        apply_theme(fig); st.plotly_chart(fig, use_container_width=True)

        cc = st.columns(4)
        for col, label, val in zip(cc,
            ['Avg Daily','Total 30-Day','Peak Day','Min Day'],
            [f"{fcast.mean():.1f}", f"{fcast.sum():.0f}", f"{fcast.max():.1f}", f"{fcast.min():.1f}"]
        ):
            with col:
                st.markdown(f"""<div class="kpi-card blue" style="text-align:center">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value" style="font-size:1.4rem">{val}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-title">All Medicines — 30-Day Total Demand</div>', unsafe_allow_html=True)
        rows = [{'Medicine':m,'Total 30D':round(r['forecast'].sum(),0)} for m,r in fc_results.items()]
        sdf  = pd.DataFrame(rows).sort_values('Total 30D', ascending=False)
        fig2 = px.bar(sdf, x='Medicine', y='Total 30D',
                      color='Total 30D', color_continuous_scale='Blues')
        apply_theme(fig2); fig2.update_xaxes(tickangle=30)
        st.plotly_chart(fig2, use_container_width=True)

# ═════════════════════════════════════════════════════════════
# PAGE: EXPIRY RISK
# ═════════════════════════════════════════════════════════════
elif page == "⏰ Expiry Risk":
    st.markdown("""<div class="dash-header">
        <div class="dash-badge">EXPIRY INTELLIGENCE</div>
        <h1 class="dash-title">Expiry Risk Model</h1>
        <p class="dash-subtitle">Gradient Boosting — predict drugs that will expire before being sold</p>
    </div>""", unsafe_allow_html=True)

    m = metrics['exp']
    cols = st.columns(5)
    for col, label, val in zip(cols,
        ['Accuracy','Precision','Recall','F1','ROC-AUC'],
        [m['accuracy'],m['precision'],m['recall'],m['f1'],m['roc_auc']]
    ):
        color = '#3fb950' if val>=.8 else '#d29922' if val>=.6 else '#f85149'
        with col:
            st.markdown(f"""<div class="kpi-card amber" style="text-align:center">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value" style="color:{color};font-size:1.5rem">{val:.3f}</div>
            </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">Expiry Risk by Category</div>', unsafe_allow_html=True)
        ec = fdf.groupby('Category')['Expiry_Prob'].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(ec, x='Category', y='Expiry_Prob', color='Expiry_Prob', color_continuous_scale='Oranges')
        apply_theme(fig); fig.update_xaxes(tickangle=30)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Days to Expiry vs Stock Cover</div>', unsafe_allow_html=True)
        samp = fdf.sample(min(2000,len(fdf)), random_state=42)
        fig  = px.scatter(samp, x='Days_To_Expiry', y='Stock_Cover_Days',
                          color='Expiry_Level',
                          color_discrete_map={'🔴 HIGH':'#f85149','🟡 MEDIUM':'#d29922','🟢 LOW':'#3fb950'},
                          opacity=.5)
        fig.add_shape(type='line', x0=0, y0=0, x1=365, y1=365,
                      line=dict(color='#8b949e', dash='dot'))
        apply_theme(fig); st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Feature Importances — Expiry Model</div>', unsafe_allow_html=True)
    fi_e = fi_exp.reset_index(); fi_e.columns=['Feature','Importance']
    fig  = px.bar(fi_e, x='Importance', y='Feature', orientation='h',
                  color='Importance', color_continuous_scale='Oranges')
    fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=330)
    apply_theme(fig); st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Expiry Alert Table</div>', unsafe_allow_html=True)
    et = fdf[fdf['Expiry_Prob']>=.5][
        ['Medicine','Category','Region','Facility','Stock','Units_Sold',
         'Days_To_Expiry','Expiry_Prob','Expiry_Level']
    ].sort_values('Days_To_Expiry').head(50)
    et['Expiry_Prob'] = et['Expiry_Prob'].round(3)
    st.dataframe(et, use_container_width=True, height=350)

# ═════════════════════════════════════════════════════════════
# PAGE: SEGMENTATION
# ═════════════════════════════════════════════════════════════
elif page == "🔵 Segmentation":
    st.markdown("""<div class="dash-header">
        <div class="dash-badge">CLUSTERING</div>
        <h1 class="dash-title">Drug Segmentation</h1>
        <p class="dash-subtitle">KMeans — group medicines into strategic inventory segments</p>
    </div>""", unsafe_allow_html=True)

    k   = metrics['cluster']['k']
    sil = metrics['cluster']['silhouette']
    cc  = st.columns(3)
    with cc[0]:
        st.markdown(f"""<div class="kpi-card blue" style="text-align:center">
            <div class="kpi-label">Optimal K</div><div class="kpi-value">{k}</div></div>""", unsafe_allow_html=True)
    with cc[1]:
        st.markdown(f"""<div class="kpi-card green" style="text-align:center">
            <div class="kpi-label">Silhouette Score</div><div class="kpi-value">{sil:.3f}</div></div>""", unsafe_allow_html=True)
    with cc[2]:
        st.markdown(f"""<div class="kpi-card amber" style="text-align:center">
            <div class="kpi-label">Drugs Segmented</div><div class="kpi-value">{len(dp)}</div></div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">Cluster Map (PCA 2D)</div>', unsafe_allow_html=True)
        fig = px.scatter(dp, x='PCA_1', y='PCA_2', color='Cluster',
                         hover_data=['Medicine','Avg_Units_Sold','Stockout_Rate'],
                         color_continuous_scale='Viridis', size='Avg_Units_Sold', size_max=20)
        apply_theme(fig); st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Stockout Risk by Cluster</div>', unsafe_allow_html=True)
        ca = dp.groupby('Cluster')[['Avg_Units_Sold','Stockout_Rate','Expiry_Risk_Rate','Total_Revenue']].mean().reset_index()
        fig = px.bar(ca, x='Cluster', y='Stockout_Rate', color='Cluster',
                     color_continuous_scale='Reds', title='Avg Stockout Rate per Cluster')
        apply_theme(fig); st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="section-title">Avg Sales by Cluster</div>', unsafe_allow_html=True)
        fig = px.bar(ca, x='Cluster', y='Avg_Units_Sold', color='Cluster',
                     color_continuous_scale='Blues')
        apply_theme(fig); st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.markdown('<div class="section-title">Revenue Share by Cluster</div>', unsafe_allow_html=True)
        fig = px.pie(ca, values='Total_Revenue', names='Cluster',
                     color_discrete_sequence=['#00d26a','#58a6ff','#f85149','#d29922','#bc8cff'])
        apply_theme(fig); st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Drug Cluster Assignments</div>', unsafe_allow_html=True)
    disp = dp[['Medicine','Cluster','Avg_Stock','Avg_Units_Sold','Avg_Days_Expiry',
               'Stockout_Rate','Expiry_Risk_Rate','Total_Revenue']].copy()
    for c in disp.columns[2:]: disp[c] = disp[c].round(2)
    st.dataframe(disp.sort_values('Cluster'), use_container_width=True, height=350)

# ═════════════════════════════════════════════════════════════
# PAGE: MODEL METRICS
# ═════════════════════════════════════════════════════════════
elif page == "📊 Model Metrics":
    st.markdown("""<div class="dash-header">
        <div class="dash-badge">EVALUATION</div>
        <h1 class="dash-title">Model Evaluation Metrics</h1>
        <p class="dash-subtitle">Performance breakdown — all trained models</p>
    </div>""", unsafe_allow_html=True)

    comp = pd.DataFrame({
        'Metric':['Accuracy','Precision','Recall','F1','ROC-AUC'],
        'Stockout Classifier':[metrics['clf']['accuracy'],metrics['clf']['precision'],
                               metrics['clf']['recall'],metrics['clf']['f1'],metrics['clf']['roc_auc']],
        'Expiry Risk Model':  [metrics['exp']['accuracy'],metrics['exp']['precision'],
                               metrics['exp']['recall'],metrics['exp']['f1'],metrics['exp']['roc_auc']],
    })
    fig = go.Figure()
    for col, color in [('Stockout Classifier','#00d26a'),('Expiry Risk Model','#58a6ff')]:
        fig.add_trace(go.Bar(name=col, x=comp['Metric'], y=comp[col],
                             marker_color=color, text=comp[col].round(3), textposition='outside'))
    fig.add_hline(y=.8, line_dash='dash', line_color='#8b949e', annotation_text='0.80 benchmark')
    fig.update_layout(barmode='group', yaxis_range=[0,1.1], title='Classification Models Comparison')
    apply_theme(fig); st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    r = metrics['reg']
    with c1:
        st.markdown('<div class="section-title">Regression — Days to Stockout</div>', unsafe_allow_html=True)
        st.markdown(f"""<div class="kpi-card blue">
            <div class="metric-row"><span class="metric-name">MAE</span><span class="metric-value">{r['mae']:.2f} days</span></div>
            <div class="metric-row"><span class="metric-name">RMSE</span><span class="metric-value">{r['rmse']:.2f} days</span></div>
            <div class="metric-row"><span class="metric-name">R²</span>
              <span class="metric-value {'metric-good' if r['r2']>=.8 else 'metric-warn'}">{r['r2']:.4f}</span></div>
        </div>""", unsafe_allow_html=True)
        idx = np.random.choice(len(r['y_te']), min(500,len(r['y_te'])), replace=False)
        fig2 = px.scatter(x=r['y_te'][idx], y=r['y_pred'][idx],
                          labels={'x':'Actual Days','y':'Predicted Days'}, opacity=.4,
                          color_discrete_sequence=['#58a6ff'])
        fig2.add_shape(type='line', x0=float(r['y_te'].min()), y0=float(r['y_te'].min()),
                       x1=float(r['y_te'].max()), y1=float(r['y_te'].max()),
                       line=dict(color='#f85149', dash='dash'))
        fig2.update_layout(title=f"Actual vs Predicted (R²={r['r2']:.3f})")
        apply_theme(fig2); st.plotly_chart(fig2, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Clustering Quality</div>', unsafe_allow_html=True)
        sil = metrics['cluster']['silhouette']
        st.markdown(f"""<div class="kpi-card green">
            <div class="metric-row"><span class="metric-name">Algorithm</span><span class="metric-value">KMeans</span></div>
            <div class="metric-row"><span class="metric-name">Optimal K</span><span class="metric-value">{metrics['cluster']['k']}</span></div>
            <div class="metric-row"><span class="metric-name">Silhouette</span>
              <span class="metric-value {'metric-good' if sil>=.5 else 'metric-warn'}">{sil:.4f}</span></div>
        </div>""", unsafe_allow_html=True)
        res_vals = r['y_te'] - r['y_pred']
        fig3 = px.histogram(x=res_vals, nbins=30, color_discrete_sequence=['#d29922'],
                            labels={'x':'Residual (days)'}, title='Regression Residuals')
        fig3.add_vline(x=0, line_dash='dash', line_color='#f85149')
        apply_theme(fig3); st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="section-title">Complete Metrics Summary</div>', unsafe_allow_html=True)
    all_m = pd.DataFrame([
        {'Model':'Stockout Classifier','Metric':'Accuracy', 'Value':f"{metrics['clf']['accuracy']:.4f}", 'Status':'✅' if metrics['clf']['accuracy']>=.8 else '⚠️'},
        {'Model':'Stockout Classifier','Metric':'ROC-AUC',  'Value':f"{metrics['clf']['roc_auc']:.4f}",  'Status':'✅' if metrics['clf']['roc_auc']>=.8  else '⚠️'},
        {'Model':'Stockout Classifier','Metric':'F1 Score',  'Value':f"{metrics['clf']['f1']:.4f}",       'Status':'✅' if metrics['clf']['f1']>=.7       else '⚠️'},
        {'Model':'Days-to-Stockout Reg','Metric':'R²',        'Value':f"{metrics['reg']['r2']:.4f}",       'Status':'✅' if metrics['reg']['r2']>=.7       else '⚠️'},
        {'Model':'Days-to-Stockout Reg','Metric':'MAE (days)','Value':f"{metrics['reg']['mae']:.2f}",      'Status':'—'},
        {'Model':'Expiry Risk Model',  'Metric':'Accuracy', 'Value':f"{metrics['exp']['accuracy']:.4f}", 'Status':'✅' if metrics['exp']['accuracy']>=.8 else '⚠️'},
        {'Model':'Expiry Risk Model',  'Metric':'ROC-AUC',  'Value':f"{metrics['exp']['roc_auc']:.4f}",  'Status':'✅' if metrics['exp']['roc_auc']>=.8  else '⚠️'},
        {'Model':'Clustering',         'Metric':'Silhouette','Value':f"{metrics['cluster']['silhouette']:.4f}", 'Status':'✅' if metrics['cluster']['silhouette']>=.4 else '⚠️'},
        {'Model':'Clustering',         'Metric':'K Clusters','Value':str(metrics['cluster']['k']),         'Status':'—'},
    ])
    st.dataframe(all_m, use_container_width=True, hide_index=True)

# ═════════════════════════════════════════════════════════════
# PAGE: RECOMMENDATIONS
# ═════════════════════════════════════════════════════════════
elif page == "🎯 Recommendations":
    st.markdown("""<div class="dash-header">
        <div class="dash-badge">ACTION INTELLIGENCE</div>
        <h1 class="dash-title">Optimization Recommendations</h1>
        <p class="dash-subtitle">AI-generated action items from all model outputs</p>
    </div>""", unsafe_allow_html=True)

    crit  = fdf[(fdf['Stockout_Risk_Pred']==1)&(fdf['Days_To_Stockout']<=7)]
    warn  = fdf[(fdf['Stockout_Risk_Pred']==1)&(fdf['Days_To_Stockout']>7)&(fdf['Days_To_Stockout']<=14)]
    exp_a = fdf[(fdf['Expiry_Risk_Pred']==1)&(fdf['Days_To_Expiry']<=30)]
    slow  = fdf[(fdf['Expiry_Risk_Pred']==1)&(fdf['Rolling_Avg_Sales_30d']<fdf['Rolling_Avg_Sales_30d'].quantile(.25))]

    cc = st.columns(4)
    for col, label, val, color in [
        (cc[0],"🔴 Critical Reorders", len(crit), "red"),
        (cc[1],"🟡 Warning Reorders",  len(warn), "amber"),
        (cc[2],"🟠 Expiry Alerts",     len(exp_a),"amber"),
        (cc[3],"📉 Slow Movers",       len(slow), "blue"),
    ]:
        with col:
            st.markdown(f"""<div class="kpi-card {color}" style="text-align:center">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{val:,}</div>
                <div class="kpi-sub">records</div>
            </div>""", unsafe_allow_html=True)

    t1,t2,t3,t4,t5 = st.tabs(["🔴 Critical","🟡 Warning","🟠 Expiry","📦 Suppliers","📈 Demand Trends"])

    with t1:
        st.markdown("### Reorder IMMEDIATELY — Stockout within 7 days")
        if len(crit)==0:
            st.markdown('<div class="alert-box alert-green">✅ No critical alerts. All products safe.</div>', unsafe_allow_html=True)
        else:
            tbl = crit.groupby(['Medicine','Region']).agg(
                Avg_Stock=('Stock','mean'), Days_Left=('Days_To_Stockout','mean'), Risk=('Stockout_Prob','mean')
            ).round(2).sort_values('Days_Left').reset_index().head(30)
            st.dataframe(tbl, use_container_width=True)
            fig = px.bar(tbl.head(15), x='Medicine', y='Days_Left',
                         color='Risk', color_continuous_scale='Reds', title='Days Remaining Before Stockout')
            apply_theme(fig); fig.update_xaxes(tickangle=30)
            st.plotly_chart(fig, use_container_width=True)

    with t2:
        st.markdown("### Plan Reorder — Stockout in 8–14 days")
        if len(warn)==0:
            st.markdown('<div class="alert-box alert-green">✅ No warning-level alerts.</div>', unsafe_allow_html=True)
        else:
            tbl2 = warn.groupby(['Medicine','Region']).agg(
                Days_Left=('Days_To_Stockout','mean'), Risk=('Stockout_Prob','mean')
            ).round(2).reset_index().head(30)
            st.dataframe(tbl2, use_container_width=True)

    with t3:
        st.markdown("### Prioritize Dispensing / Return to Supplier")
        if len(exp_a)==0:
            st.markdown('<div class="alert-box alert-green">✅ No imminent expiry alerts.</div>', unsafe_allow_html=True)
        else:
            et = exp_a.groupby(['Medicine','Region']).agg(
                Days_Expiry=('Days_To_Expiry','mean'), Stock=('Stock','mean'), Sales=('Units_Sold','mean')
            ).round(2).sort_values('Days_Expiry').reset_index().head(30)
            st.dataframe(et, use_container_width=True)
            fig = px.scatter(et, x='Days_Expiry', y='Stock', size='Sales',
                             color='Days_Expiry', hover_data=['Medicine','Region'],
                             color_continuous_scale='Oranges_r', title='Expiry Risk — Stock vs Days Remaining')
            apply_theme(fig); st.plotly_chart(fig, use_container_width=True)

    with t4:
        st.markdown("### Supplier Delay Performance")
        sp = fdf.groupby('Supplier')['Delay_Days'].agg(Avg_Delay='mean', Max_Delay='max', Orders='count').round(2).sort_values('Avg_Delay', ascending=False).reset_index()
        st.dataframe(sp, use_container_width=True)
        fig = px.bar(sp.head(15), x='Supplier', y='Avg_Delay',
                     color='Avg_Delay', color_continuous_scale='Reds', title='Average Supplier Delay (days)')
        apply_theme(fig); fig.update_xaxes(tickangle=30)
        st.plotly_chart(fig, use_container_width=True)

    with t5:
        ca, cb = st.columns(2)
        with ca:
            st.markdown("### 📈 Rising Demand")
            rising = fdf[fdf['Sales_Trend']>0].groupby('Medicine')['Sales_Trend'].mean().nlargest(10).reset_index()
            fig = px.bar(rising, x='Sales_Trend', y='Medicine', orientation='h',
                         color='Sales_Trend', color_continuous_scale='Greens', title='Fastest Rising')
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            apply_theme(fig); st.plotly_chart(fig, use_container_width=True)
        with cb:
            st.markdown("### 📉 Falling Demand")
            falling = fdf[fdf['Sales_Trend']<0].groupby('Medicine')['Sales_Trend'].mean().nsmallest(10).reset_index()
            fig = px.bar(falling, x='Sales_Trend', y='Medicine', orientation='h',
                         color='Sales_Trend', color_continuous_scale='Reds', title='Fastest Falling')
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            apply_theme(fig); st.plotly_chart(fig, use_container_width=True)

# ═════════════════════════════════════════════════════════════
# PAGE: EXPORT
# ═════════════════════════════════════════════════════════════
elif page == "📤 Export":
    st.markdown("""<div class="dash-header">
        <div class="dash-badge">DATA EXPORT</div>
        <h1 class="dash-title">Export All Results</h1>
        <p class="dash-subtitle">Download model predictions and reports as Excel</p>
    </div>""", unsafe_allow_html=True)

    import io
    buf = io.BytesIO()
    ecols = ['Medicine','Category','Region','Facility','Stock','Units_Sold',
             'Reorder_Level','Days_To_Expiry','Supplier','Delay_Days',
             'Stockout_Risk_Pred','Stockout_Prob','Days_To_Stockout','Stockout_Level',
             'Expiry_Risk_Pred','Expiry_Prob','Expiry_Level']

    exp_tbl = fdf[(fdf['Expiry_Risk_Pred']==1)&(fdf['Days_To_Expiry']<=30)].groupby(['Medicine','Region']).agg(
        Avg_Days_Expiry=('Days_To_Expiry','mean'), Avg_Stock=('Stock','mean')
    ).round(2).reset_index()
    sup_df = fdf.groupby('Supplier')['Delay_Days'].agg(Avg_Delay='mean', Orders='count').round(2).reset_index()

    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        fdf[ecols].round(3).to_excel(writer, sheet_name='All_Predictions', index=False)
        fdf[fdf['Stockout_Risk_Pred']==1][['Medicine','Region','Days_To_Stockout','Stockout_Prob']].sort_values('Days_To_Stockout').to_excel(writer, sheet_name='Stockout_Alerts', index=False)
        exp_tbl.to_excel(writer, sheet_name='Expiry_Alerts', index=False)
        dp[['Medicine','Cluster','Avg_Units_Sold','Stockout_Rate','Expiry_Risk_Rate','Total_Revenue']].to_excel(writer, sheet_name='Drug_Segments', index=False)
        sup_df.to_excel(writer, sheet_name='Supplier_Performance', index=False)
    buf.seek(0)

    st.markdown('<div class="alert-box alert-blue">📁 Export contains 5 sheets: All Predictions, Stockout Alerts, Expiry Alerts, Drug Segments, Supplier Performance</div>', unsafe_allow_html=True)
    st.download_button(
        label="⬇️  Download pharma_ml_results.xlsx",
        data=buf,
        file_name="pharma_ml_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.markdown('<div class="section-title">Preview — First 100 Records</div>', unsafe_allow_html=True)
    st.dataframe(fdf[ecols].round(3).head(100), use_container_width=True, height=400)