"""
💊 Pharmaceutical Inventory Intelligence Dashboard
Streamlit app — extension of the ML project notebook.
Run with: streamlit run pharma_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── Sklearn ──────────────────────────────────────────────────
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score,
    silhouette_score, precision_recall_curve
)

# ── Time Series ──────────────────────────────────────────────
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Pharma Inventory Intelligence",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = ["#2E86AB", "#E84855", "#F4A261", "#2A9D8F", "#8338EC", "#FB8500"]

# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Loading pharmacy data…")
def load_data():
    sheet_id = "158Ig9EaUlfhUdMU9U5PeQpENWLYOCgNB6Uwqbwaan3s"
    sheets   = ["HR", "Operations", "Expenses", "Purchasing"]
    data     = {}
    for sheet in sheets:
        url = (
            f"https://docs.google.com/spreadsheets/d/{sheet_id}"
            f"/gviz/tq?tqx=out:csv&sheet={sheet}"
        )
        data[sheet.lower()] = pd.read_csv(url)

    ops = data["operations"].copy()
    if "Unnamed: 13" in ops.columns:
        ops.drop(columns=["Unnamed: 13"], inplace=True)
    ops["Date"] = pd.to_datetime(ops["Date"], format="%d/%m/%Y")
    return data["hr"], ops, data["expenses"], data["purchasing"]


@st.cache_data(show_spinner="Engineering features & training models…")
def build_features_and_models(_ops, _purchasing):
    df = _ops.copy()

    # ── Date features
    df["Month"]      = df["Date"].dt.month
    df["Quarter"]    = df["Date"].dt.quarter
    df["DayOfWeek"]  = df["Date"].dt.dayofweek
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Year"]       = df["Date"].dt.year

    # ── Inventory features
    df["Remaining_Stock"]     = df["Stock"] - df["Units_Sold"]
    df["Stock_Buffer"]        = df["Stock"] - df["Reorder_Level"]
    df["Stock_Cover_Days"]    = df["Stock"] / df["Units_Sold"].replace(0, 0.1)
    df["Days_Until_Stockout"] = df["Stock_Cover_Days"] - df["Delay_Days"]
    df["Revenue"]             = df["Units_Sold"] * df["Unit_Price"]
    df["Stockout_Gap"]        = df["Reorder_Level"] - df["Remaining_Stock"]
    df["Expiry_Risk_Score"]   = df["Units_Sold"] / df["Days_To_Expiry"].replace(0, 0.1)
    df["Will_Expire_Before_Sold"] = (df["Days_To_Expiry"] < df["Stock_Cover_Days"]).astype(int)

    # ── Supplier merge
    supplier_stats = _purchasing.groupby("Supplier").agg(
        Avg_Delivery_Days   =("Delivery_Days",           "mean"),
        Total_Purchased     =("Quantity_Purchased",       "sum"),
        Avg_Purchase_Price  =("Purchase_Price_Per_Unit",  "mean"),
    ).reset_index()
    df = df.merge(supplier_stats, on="Supplier", how="left")

    # ── Rolling / lag features
    df = df.sort_values(["Medicine", "Date"])
    df["Rolling_Avg_Sales_7d"]  = df.groupby("Medicine")["Units_Sold"].transform(
        lambda x: x.rolling(7,  min_periods=1).mean()
    )
    df["Rolling_Avg_Sales_30d"] = df.groupby("Medicine")["Units_Sold"].transform(
        lambda x: x.rolling(30, min_periods=1).mean()
    )
    df["Sales_Trend"] = df["Rolling_Avg_Sales_7d"] - df["Rolling_Avg_Sales_30d"]

    # ── Targets
    df["Stockout_Risk"] = (
        (df["Remaining_Stock"] <= df["Reorder_Level"]) |
        (df["Days_Until_Stockout"] <= 0)
    ).astype(int)
    df["Days_Until_Stockout_Target"] = df["Days_Until_Stockout"].clip(lower=0)

    df["Expiry_Risk"] = (
        (df["Days_To_Expiry"] <= 30) |
        (df["Will_Expire_Before_Sold"] == 1)
    ).astype(int)
    df["Expiry_Severity"] = np.clip(
        (1 - df["Days_To_Expiry"] / 365) * 100 * df["Will_Expire_Before_Sold"], 0, 100
    )

    # ── Encoding for models
    cat_cols = ["Medicine", "Category", "Region", "Facility", "Supplier"]
    num_cols = [
        "Stock", "Units_Sold", "Reorder_Level", "Unit_Price", "Delay_Days",
        "Days_To_Expiry", "Month", "Quarter", "DayOfWeek", "WeekOfYear",
        "Remaining_Stock", "Stock_Buffer", "Stock_Cover_Days", "Revenue",
        "Avg_Delivery_Days", "Rolling_Avg_Sales_7d", "Rolling_Avg_Sales_30d",
        "Sales_Trend", "Expiry_Risk_Score", "Stockout_Gap",
    ]
    FEATURE_COLS = cat_cols + num_cols

    df_enc = df.copy()
    le = LabelEncoder()
    for col in cat_cols:
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))

    X = df_enc[FEATURE_COLS].fillna(df_enc[FEATURE_COLS].median())
    y_class = df_enc["Stockout_Risk"]
    y_reg   = df_enc["Days_Until_Stockout_Target"]

    # ── Stockout classifier
    X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(
        X, y_class, test_size=0.2, random_state=42, stratify=y_class
    )
    clf = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_split=5,
        random_state=42, class_weight="balanced", n_jobs=-1
    )
    clf.fit(X_tr_c, y_tr_c)
    y_pred_c     = clf.predict(X_te_c)
    y_pred_proba = clf.predict_proba(X_te_c)[:, 1]

    # ── Stockout regressor
    X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )
    reg = RandomForestRegressor(
        n_estimators=300, max_depth=12, min_samples_split=5,
        random_state=42, n_jobs=-1
    )
    reg.fit(X_tr_r, y_tr_r)
    y_pred_r = reg.predict(X_te_r)

    # ── Expiry classifier
    expiry_features = [
        "Medicine", "Category", "Region", "Supplier",
        "Stock", "Units_Sold", "Days_To_Expiry", "Unit_Price",
        "Stock_Cover_Days", "Expiry_Risk_Score", "Rolling_Avg_Sales_7d",
        "Rolling_Avg_Sales_30d", "Sales_Trend", "Month", "Quarter",
    ]
    df_exp = df.copy()
    le2 = LabelEncoder()
    for col in ["Medicine", "Category", "Region", "Supplier"]:
        df_exp[col] = le2.fit_transform(df_exp[col].astype(str))

    X_exp = df_exp[expiry_features].fillna(df_exp[expiry_features].median())
    y_exp = df_exp["Expiry_Risk"]

    X_tr_e, X_te_e, y_tr_e, y_te_e = train_test_split(
        X_exp, y_exp, test_size=0.2, random_state=42, stratify=y_exp
    )
    exp_clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
    )
    exp_clf.fit(X_tr_e, y_tr_e)
    y_pred_e      = exp_clf.predict(X_te_e)
    y_pred_e_prob = exp_clf.predict_proba(X_te_e)[:, 1]

    # ── Clustering
    drug_profile = df.groupby("Medicine").agg(
        Avg_Stock        =("Stock",          "mean"),
        Avg_Units_Sold   =("Units_Sold",     "mean"),
        Avg_Days_Expiry  =("Days_To_Expiry", "mean"),
        Avg_Unit_Price   =("Unit_Price",     "mean"),
        Avg_Delay        =("Delay_Days",     "mean"),
        Total_Revenue    =("Revenue",        "sum"),
        Stockout_Rate    =("Stockout_Risk",  "mean"),
        Expiry_Risk_Rate =("Expiry_Risk",    "mean"),
        Avg_Stock_Cover  =("Stock_Cover_Days","mean"),
        Avg_Sales_Trend  =("Sales_Trend",    "mean"),
    ).reset_index()

    cluster_features = [
        "Avg_Stock", "Avg_Units_Sold", "Avg_Days_Expiry",
        "Avg_Unit_Price", "Avg_Delay", "Total_Revenue",
        "Stockout_Rate", "Expiry_Risk_Rate", "Avg_Stock_Cover",
    ]
    X_cl = drug_profile[cluster_features].fillna(0)
    scaler = StandardScaler()
    X_cl_scaled = scaler.fit_transform(X_cl)

    sil_scores = []
    K_range = range(2, min(len(X_cl_scaled), 10))
    for k in K_range:
        km  = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = km.fit_predict(X_cl_scaled)
        sil_scores.append(silhouette_score(X_cl_scaled, lbl))
    best_k = list(K_range)[int(np.argmax(sil_scores))]

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    drug_profile["Cluster"] = kmeans.fit_predict(X_cl_scaled)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_cl_scaled)
    drug_profile["PCA_1"] = X_pca[:, 0]
    drug_profile["PCA_2"] = X_pca[:, 1]

    cluster_summary = drug_profile.groupby("Cluster")[cluster_features].mean().round(2)

    def label_cluster(row):
        if row["Stockout_Rate"] > 0.5 and row["Avg_Units_Sold"] > cluster_summary["Avg_Units_Sold"].mean():
            return "🔴 High Demand / High Stockout Risk"
        elif row["Expiry_Risk_Rate"] > 0.4:
            return "🟠 High Expiry Risk / Slow Movers"
        elif row["Avg_Units_Sold"] > cluster_summary["Avg_Units_Sold"].mean():
            return "🟢 Fast Movers / Low Risk"
        else:
            return "🔵 Stable / Low Volume"

    cluster_summary["Label"] = cluster_summary.apply(label_cluster, axis=1)
    drug_profile["Cluster_Label"] = drug_profile["Cluster"].map(cluster_summary["Label"])

    # ── Attach predictions to master df
    df["Stockout_Risk_Pred"]    = clf.predict(X)
    df["Stockout_Probability"]  = clf.predict_proba(X)[:, 1]
    df["Days_To_Stockout_Pred"] = reg.predict(X).clip(min=0)
    df["Expiry_Risk_Pred"]      = exp_clf.predict(X_exp)
    df["Expiry_Probability"]    = exp_clf.predict_proba(X_exp)[:, 1]

    def risk_badge(prob):
        if prob >= 0.7:   return "🔴 HIGH"
        elif prob >= 0.4: return "🟡 MEDIUM"
        else:             return "🟢 LOW"

    df["Stockout_Risk_Level"] = df["Stockout_Probability"].apply(risk_badge)
    df["Expiry_Risk_Level"]   = df["Expiry_Probability"].apply(risk_badge)
    df = df.merge(drug_profile[["Medicine", "Cluster", "Cluster_Label"]], on="Medicine", how="left")

    # ── Demand forecast
    daily_sales = _ops.groupby(["Date", "Medicine"])["Units_Sold"].sum().reset_index()
    daily_sales = daily_sales.sort_values(["Medicine", "Date"])
    top_meds = _ops.groupby("Medicine")["Units_Sold"].sum().nlargest(6).index.tolist()

    FORECAST_DAYS  = 30
    forecast_results = {}
    for med in top_meds:
        series = (
            daily_sales[daily_sales["Medicine"] == med]
            .set_index("Date")["Units_Sold"]
            .resample("D").sum().fillna(0)
        )
        try:
            model = ExponentialSmoothing(
                series, trend="add",
                seasonal="add" if len(series) >= 14 else None,
                seasonal_periods=7,
            )
            fit   = model.fit(optimized=True)
            fcast = fit.forecast(FORECAST_DAYS).clip(lower=0)
            forecast_results[med] = {"history": series, "forecast": fcast}
        except Exception:
            pass

    forecast_summary = []
    for med, res in forecast_results.items():
        fcast = res["forecast"]
        forecast_summary.append({
            "Medicine":              med,
            "Avg_Daily_Forecast":    round(float(fcast.mean()), 1),
            "Total_30Day_Forecast":  round(float(fcast.sum()),  0),
            "Peak_Day_Forecast":     round(float(fcast.max()),  1),
            "Min_Day_Forecast":      round(float(fcast.min()),  1),
        })
    forecast_df = pd.DataFrame(forecast_summary)

    metrics = {
        "clf": {
            "accuracy":  accuracy_score(y_te_c, y_pred_c),
            "precision": precision_score(y_te_c, y_pred_c),
            "recall":    recall_score(y_te_c, y_pred_c),
            "f1":        f1_score(y_te_c, y_pred_c),
            "roc_auc":   roc_auc_score(y_te_c, y_pred_proba),
            "y_te":      y_te_c,
            "y_pred":    y_pred_c,
            "y_proba":   y_pred_proba,
            "feature_importance": pd.Series(clf.feature_importances_, index=FEATURE_COLS),
        },
        "reg": {
            "mae":   mean_absolute_error(y_te_r, y_pred_r),
            "rmse":  float(np.sqrt(mean_squared_error(y_te_r, y_pred_r))),
            "r2":    r2_score(y_te_r, y_pred_r),
            "y_te":  y_te_r,
            "y_pred":y_pred_r,
        },
        "exp": {
            "accuracy":  accuracy_score(y_te_e, y_pred_e),
            "precision": precision_score(y_te_e, y_pred_e),
            "recall":    recall_score(y_te_e, y_pred_e),
            "f1":        f1_score(y_te_e, y_pred_e),
            "roc_auc":   roc_auc_score(y_te_e, y_pred_e_prob),
            "y_te":      y_te_e,
            "y_pred":    y_pred_e,
            "y_proba":   y_pred_e_prob,
            "feature_importance": pd.Series(exp_clf.feature_importances_, index=expiry_features),
        },
        "cluster": {
            "silhouette": silhouette_score(X_cl_scaled, drug_profile["Cluster"]),
            "K":          best_k,
            "n_drugs":    len(drug_profile),
        },
    }

    return df, drug_profile, cluster_features, forecast_results, forecast_df, metrics


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
st.sidebar.title("💊 Pharma Intelligence")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Overview",
        "📦 Stockout Risk",
        "⏳ Expiry Risk",
        "📈 Demand Forecast",
        "🔬 Drug Segmentation",
        "📊 Model Evaluation",
        "🚨 Alerts & Actions",
    ],
)

# ── Load data
hr, operations, expenses, purchasing = load_data()
df, drug_profile, cluster_features, forecast_results, forecast_df, metrics = (
    build_features_and_models(operations, purchasing)
)

# ── Sidebar filters (used across pages)
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Filters")
regions    = ["All"] + sorted(df["Region"].dropna().unique().tolist())
categories = ["All"] + sorted(df["Category"].dropna().unique().tolist())
sel_region   = st.sidebar.selectbox("Region",   regions)
sel_category = st.sidebar.selectbox("Category", categories)

dff = df.copy()
if sel_region   != "All": dff = dff[dff["Region"]   == sel_region]
if sel_category != "All": dff = dff[dff["Category"] == sel_category]


# ═══════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("🏥 Pharmaceutical Inventory Intelligence Dashboard")
    st.markdown("A real-time ML-powered operations view built on the pharmacy dataset.")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Records",      f"{len(dff):,}")
    c2.metric("Unique Medicines",   dff["Medicine"].nunique())
    c3.metric("Regions",            dff["Region"].nunique())
    c4.metric("High Stockout Risk", f"{(dff['Stockout_Probability'] >= 0.7).sum():,}")
    c5.metric("Expiry Alerts",      f"{(dff['Expiry_Risk_Pred'] == 1).sum():,}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Monthly Revenue Trend")
        monthly_rev = dff.groupby(dff["Date"].dt.to_period("M"))["Revenue"].sum()
        monthly_rev.index = monthly_rev.index.astype(str)
        st.line_chart(monthly_rev)

    with col2:
        st.subheader("Units Sold by Category")
        cat_sales = dff.groupby("Category")["Units_Sold"].sum().sort_values(ascending=False)
        st.bar_chart(cat_sales)

    st.subheader("Risk Distribution")
    col3, col4 = st.columns(2)
    with col3:
        risk_dist = dff["Stockout_Risk_Level"].value_counts()
        fig, ax   = plt.subplots(figsize=(4, 3))
        ax.pie(risk_dist.values, labels=risk_dist.index,
               colors=["tomato", "gold", "mediumseagreen"],
               autopct="%1.1f%%", startangle=90)
        ax.set_title("Stockout Risk Levels")
        st.pyplot(fig)
        plt.close()
    with col4:
        exp_dist = dff["Expiry_Risk_Level"].value_counts()
        fig, ax  = plt.subplots(figsize=(4, 3))
        ax.pie(exp_dist.values, labels=exp_dist.index,
               colors=["tomato", "gold", "mediumseagreen"],
               autopct="%1.1f%%", startangle=90)
        ax.set_title("Expiry Risk Levels")
        st.pyplot(fig)
        plt.close()


# ═══════════════════════════════════════════════════════════════
# PAGE: STOCKOUT RISK
# ═══════════════════════════════════════════════════════════════
elif page == "📦 Stockout Risk":
    st.title("📦 Stockout Risk Analysis")

    c1, c2, c3 = st.columns(3)
    c1.metric("Stockout Rate",      f"{dff['Stockout_Risk'].mean()*100:.1f}%")
    c2.metric("High-Risk Records",  f"{(dff['Stockout_Probability']>=0.7).sum():,}")
    c3.metric("Avg Days to Stockout (At-Risk)",
              f"{dff.loc[dff['Stockout_Risk_Pred']==1,'Days_To_Stockout_Pred'].mean():.1f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 10 Highest Stockout-Risk Medicines")
        top10 = dff.groupby("Medicine")["Stockout_Probability"].mean().nlargest(10)
        colors = ["tomato" if v >= 0.7 else "gold" if v >= 0.4 else "mediumseagreen"
                  for v in top10.values]
        fig, ax = plt.subplots(figsize=(6, 4))
        top10.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
        ax.set_xlabel("Avg Stockout Probability")
        ax.invert_yaxis()
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader("Stockout Risk by Region")
        reg_risk = dff.groupby("Region")["Stockout_Probability"].mean().sort_values(ascending=False)
        fig, ax  = plt.subplots(figsize=(6, 4))
        reg_risk.plot(kind="bar", ax=ax, color=PALETTE[0], edgecolor="white")
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylabel("Avg Risk Probability")
        st.pyplot(fig); plt.close()

    st.subheader("Days Until Stockout — At-Risk Products")
    at_risk_days = dff[dff["Stockout_Risk_Pred"] == 1]["Days_To_Stockout_Pred"]
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.hist(at_risk_days, bins=25, color="coral", edgecolor="white")
    ax.axvline(7,  color="red",    linestyle="--", lw=1.5, label="7 days")
    ax.axvline(14, color="orange", linestyle="--", lw=1.5, label="14 days")
    ax.set_xlabel("Predicted Days Until Stockout")
    ax.legend()
    st.pyplot(fig); plt.close()

    st.subheader("Stockout Dual-Risk Map")
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(dff["Stockout_Probability"], dff["Expiry_Probability"],
                    c=dff["Stockout_Risk_Pred"], cmap="RdYlGn_r", alpha=0.3, s=10)
    ax.axhline(0.5, color="gray", lw=0.8, linestyle="--")
    ax.axvline(0.5, color="gray", lw=0.8, linestyle="--")
    ax.set_xlabel("Stockout Probability"); ax.set_ylabel("Expiry Probability")
    ax.set_title("Quadrant Risk Map")
    plt.colorbar(sc, ax=ax, label="Stockout Predicted")
    st.pyplot(fig); plt.close()

    st.subheader("Raw Predictions Table")
    cols_show = ["Medicine", "Category", "Region", "Facility",
                 "Stock", "Units_Sold", "Reorder_Level",
                 "Stockout_Probability", "Days_To_Stockout_Pred", "Stockout_Risk_Level"]
    st.dataframe(
        dff[cols_show].sort_values("Stockout_Probability", ascending=False)
        .head(200).reset_index(drop=True)
    )


# ═══════════════════════════════════════════════════════════════
# PAGE: EXPIRY RISK
# ═══════════════════════════════════════════════════════════════
elif page == "⏳ Expiry Risk":
    st.title("⏳ Expiry Risk Analysis")

    c1, c2, c3 = st.columns(3)
    c1.metric("Expiry Risk Rate",     f"{dff['Expiry_Risk'].mean()*100:.1f}%")
    c2.metric("High-Risk Products",   f"{(dff['Expiry_Probability']>=0.7).sum():,}")
    c3.metric("Avg Days To Expiry",   f"{dff['Days_To_Expiry'].mean():.0f} days")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Expiry Risk Rate by Drug Category")
        exp_cat = dff.groupby("Category")["Expiry_Probability"].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6, 4))
        exp_cat.plot(kind="bar", ax=ax, color=PALETTE[1], edgecolor="white")
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylabel("Avg Expiry Probability")
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader("Expiry Risk Rate by Region")
        exp_reg = dff.groupby("Region")["Expiry_Probability"].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6, 4))
        exp_reg.plot(kind="bar", ax=ax, color=PALETTE[4], edgecolor="white")
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylabel("Avg Expiry Probability")
        st.pyplot(fig); plt.close()

    st.subheader("Expiry vs Stock Cover Days (Red = At Risk)")
    sample = dff.sample(min(2000, len(dff)), random_state=42)
    fig, ax = plt.subplots(figsize=(9, 5))
    sc = ax.scatter(sample["Days_To_Expiry"], sample["Stock_Cover_Days"],
                    c=sample["Expiry_Risk"], cmap="RdYlGn_r", alpha=0.4, s=12)
    ax.axline((0, 0), slope=1, color="black", linestyle="--", lw=1, label="Break-even")
    ax.set_xlabel("Days To Expiry"); ax.set_ylabel("Stock Cover Days")
    plt.colorbar(sc, ax=ax, label="Expiry Risk")
    ax.legend(); st.pyplot(fig); plt.close()

    st.subheader("Top Expiry-Risk Drugs (Soonest First)")
    exp_table = (
        dff[dff["Expiry_Risk_Pred"] == 1]
        [["Medicine", "Category", "Region", "Facility",
          "Stock", "Units_Sold", "Days_To_Expiry", "Expiry_Probability"]]
        .sort_values("Days_To_Expiry")
        .head(30)
        .reset_index(drop=True)
    )
    st.dataframe(exp_table)


# ═══════════════════════════════════════════════════════════════
# PAGE: DEMAND FORECAST
# ═══════════════════════════════════════════════════════════════
elif page == "📈 Demand Forecast":
    st.title("📈 30-Day Demand Forecast")
    st.info("Exponential Smoothing (Holt-Winters) for top 6 medicines by historical volume.")

    st.subheader("Forecast Summary Table")
    st.dataframe(forecast_df, use_container_width=True)

    st.subheader("Forecast Charts")
    meds = list(forecast_results.keys())
    sel_med = st.selectbox("Select Medicine", meds)

    if sel_med:
        res   = forecast_results[sel_med]
        hist  = res["history"].tail(60)
        fcast = res["forecast"]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(hist.index,  hist.values,  color=PALETTE[0], lw=2, label="Historical")
        ax.plot(fcast.index, fcast.values, color=PALETTE[1], lw=2, linestyle="--", label="Forecast")
        ax.fill_between(fcast.index,
                        fcast.values * 0.85, fcast.values * 1.15,
                        alpha=0.2, color=PALETTE[1], label="±15% CI")
        ax.axvline(x=hist.index[-1], color="gray", linestyle=":", lw=1)
        ax.set_title(f"{sel_med} — 30-Day Forecast")
        ax.set_xlabel("Date"); ax.set_ylabel("Units Sold")
        ax.legend(); ax.tick_params(axis="x", rotation=30)
        st.pyplot(fig); plt.close()

    st.subheader("All 6 Medicines — Forecast Grid")
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()
    for i, (med, res) in enumerate(forecast_results.items()):
        ax    = axes[i]
        hist  = res["history"].tail(60)
        fcast = res["forecast"]
        ax.plot(hist.index,  hist.values,  color=PALETTE[0], lw=2, label="Historical")
        ax.plot(fcast.index, fcast.values, color=PALETTE[1], lw=2, linestyle="--", label="Forecast")
        ax.fill_between(fcast.index, fcast.values*0.85, fcast.values*1.15,
                        alpha=0.2, color=PALETTE[1])
        ax.axvline(x=hist.index[-1], color="gray", linestyle=":", lw=1)
        ax.set_title(med, fontweight="bold")
        ax.set_xlabel("Date"); ax.set_ylabel("Units Sold")
        ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    st.pyplot(fig); plt.close()


# ═══════════════════════════════════════════════════════════════
# PAGE: DRUG SEGMENTATION
# ═══════════════════════════════════════════════════════════════
elif page == "🔬 Drug Segmentation":
    st.title("🔬 Drug Segmentation (K-Means Clustering)")

    K = metrics["cluster"]["K"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Optimal Clusters",  K)
    c2.metric("Silhouette Score",  f"{metrics['cluster']['silhouette']:.4f}")
    c3.metric("Drugs Segmented",   metrics["cluster"]["n_drugs"])

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("PCA Cluster Map (2D)")
        fig, ax = plt.subplots(figsize=(6, 5))
        for cl in range(K):
            mask = drug_profile["Cluster"] == cl
            ax.scatter(
                drug_profile.loc[mask, "PCA_1"],
                drug_profile.loc[mask, "PCA_2"],
                label=f"Cluster {cl}", s=60, alpha=0.7,
                color=PALETTE[cl % len(PALETTE)]
            )
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.legend(); st.pyplot(fig); plt.close()

    with col2:
        st.subheader("Segment Distribution")
        seg_dist = drug_profile["Cluster_Label"].value_counts()
        fig, ax  = plt.subplots(figsize=(6, 4))
        seg_dist.plot(kind="barh", ax=ax, color=PALETTE[:K], edgecolor="white")
        ax.set_xlabel("Number of Drugs")
        st.pyplot(fig); plt.close()

    st.subheader("Cluster Profiles")
    cluster_summary = drug_profile.groupby("Cluster")[cluster_features].mean().round(2)
    st.dataframe(cluster_summary, use_container_width=True)

    st.subheader("Drug-Level Segment Table")
    st.dataframe(
        drug_profile[["Medicine", "Cluster", "Cluster_Label",
                      "Avg_Units_Sold", "Total_Revenue",
                      "Stockout_Rate", "Expiry_Risk_Rate"]]
        .sort_values("Stockout_Rate", ascending=False)
        .reset_index(drop=True),
        use_container_width=True,
    )


# ═══════════════════════════════════════════════════════════════
# PAGE: MODEL EVALUATION
# ═══════════════════════════════════════════════════════════════
elif page == "📊 Model Evaluation":
    st.title("📊 Model Evaluation")

    st.subheader("Classification Models Summary")
    summ = pd.DataFrame({
        "Model":     ["Stockout Classifier", "Expiry Risk Model"],
        "Accuracy":  [metrics["clf"]["accuracy"],  metrics["exp"]["accuracy"]],
        "Precision": [metrics["clf"]["precision"], metrics["exp"]["precision"]],
        "Recall":    [metrics["clf"]["recall"],    metrics["exp"]["recall"]],
        "F1":        [metrics["clf"]["f1"],        metrics["exp"]["f1"]],
        "ROC-AUC":   [metrics["clf"]["roc_auc"],   metrics["exp"]["roc_auc"]],
    }).set_index("Model").round(4)
    st.dataframe(summ, use_container_width=True)

    tab1, tab2, tab3 = st.tabs(["Stockout Model", "Expiry Model", "Regression"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Confusion Matrix**")
            cm  = confusion_matrix(metrics["clf"]["y_te"], metrics["clf"]["y_pred"])
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=["Safe","At Risk"], yticklabels=["Safe","At Risk"])
            ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
            st.pyplot(fig); plt.close()

        with col2:
            st.markdown("**ROC Curve**")
            fpr, tpr, _ = roc_curve(metrics["clf"]["y_te"], metrics["clf"]["y_proba"])
            auc = metrics["clf"]["roc_auc"]
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.plot(fpr, tpr, color=PALETTE[0], lw=2, label=f"AUC={auc:.3f}")
            ax.plot([0,1],[0,1], "k--", lw=1)
            ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
            ax.legend(); st.pyplot(fig); plt.close()

        st.markdown("**Top Feature Importances**")
        fi = metrics["clf"]["feature_importance"].nlargest(12)
        fig, ax = plt.subplots(figsize=(8, 4))
        fi.plot(kind="barh", ax=ax, color=PALETTE[0]); ax.invert_yaxis()
        st.pyplot(fig); plt.close()

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            cm2 = confusion_matrix(metrics["exp"]["y_te"], metrics["exp"]["y_pred"])
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm2, annot=True, fmt="d", cmap="Oranges", ax=ax,
                        xticklabels=["Safe","At Risk"], yticklabels=["Safe","At Risk"])
            ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
            st.pyplot(fig); plt.close()

        with col2:
            fpr2, tpr2, _ = roc_curve(metrics["exp"]["y_te"], metrics["exp"]["y_proba"])
            auc2 = metrics["exp"]["roc_auc"]
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.plot(fpr2, tpr2, color=PALETTE[1], lw=2, label=f"AUC={auc2:.3f}")
            ax.plot([0,1],[0,1], "k--", lw=1)
            ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend()
            st.pyplot(fig); plt.close()

        fi2 = metrics["exp"]["feature_importance"].nlargest(10)
        fig, ax = plt.subplots(figsize=(8, 4))
        fi2.plot(kind="barh", ax=ax, color=PALETTE[2]); ax.invert_yaxis()
        st.pyplot(fig); plt.close()

    with tab3:
        st.metric("MAE",  f"{metrics['reg']['mae']:.2f} days")
        st.metric("RMSE", f"{metrics['reg']['rmse']:.2f} days")
        st.metric("R²",   f"{metrics['reg']['r2']:.4f}")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].scatter(metrics["reg"]["y_te"], metrics["reg"]["y_pred"],
                        alpha=0.3, color=PALETTE[1], s=15)
        m = max(metrics["reg"]["y_te"].max(), metrics["reg"]["y_pred"].max())
        axes[0].plot([0,m],[0,m], "r--", lw=2)
        axes[0].set_xlabel("Actual"); axes[0].set_ylabel("Predicted")
        axes[0].set_title(f"Actual vs Predicted (R²={metrics['reg']['r2']:.3f})")

        residuals = metrics["reg"]["y_te"] - metrics["reg"]["y_pred"]
        axes[1].hist(residuals, bins=30, color=PALETTE[2], edgecolor="white")
        axes[1].axvline(0, color="black", linestyle="--")
        axes[1].set_title("Regression Residuals")
        plt.tight_layout()
        st.pyplot(fig); plt.close()


# ═══════════════════════════════════════════════════════════════
# PAGE: ALERTS & ACTIONS
# ═══════════════════════════════════════════════════════════════
elif page == "🚨 Alerts & Actions":
    st.title("🚨 Alerts & Recommended Actions")

    # Critical Stockout (≤7 days)
    critical_so = (
        dff[(dff["Stockout_Risk_Pred"] == 1) & (dff["Days_To_Stockout_Pred"] <= 7)]
        .groupby(["Medicine", "Region"])
        .agg(Avg_Stock      =("Stock",                 "mean"),
             Avg_Days_Left  =("Days_To_Stockout_Pred", "mean"),
             Avg_Risk       =("Stockout_Probability",  "mean"))
        .round(2)
        .sort_values("Avg_Days_Left")
        .reset_index()
    )

    # Warning Stockout (8–14 days)
    warning_so = (
        dff[(dff["Stockout_Risk_Pred"] == 1) &
            (dff["Days_To_Stockout_Pred"] > 7) &
            (dff["Days_To_Stockout_Pred"] <= 14)]
        .groupby(["Medicine", "Region"])
        .agg(Avg_Days_Left=("Days_To_Stockout_Pred", "mean"))
        .round(2)
        .reset_index()
    )

    # Expiry Alerts (≤30 days)
    exp_alerts = (
        dff[(dff["Expiry_Risk_Pred"] == 1) & (dff["Days_To_Expiry"] <= 30)]
        .groupby(["Medicine", "Region"])
        .agg(Avg_Days_Expiry =("Days_To_Expiry", "mean"),
             Avg_Stock       =("Stock",          "mean"),
             Avg_Units_Sold  =("Units_Sold",     "mean"))
        .round(2)
        .sort_values("Avg_Days_Expiry")
        .reset_index()
    )

    # Supplier performance
    sup_perf = (
        dff.groupby("Supplier")["Delay_Days"]
        .agg(["mean","max","count"])
        .rename(columns={"mean":"Avg_Delay","max":"Max_Delay","count":"Records"})
        .sort_values("Avg_Delay", ascending=False)
        .round(2)
    )

    st.error(f"🔴 CRITICAL — Reorder Immediately ({len(critical_so)} drug-region pairs)")
    if not critical_so.empty:
        st.dataframe(critical_so, use_container_width=True)

    st.warning(f"🟡 WARNING — Plan Reorder in 8–14 days ({len(warning_so)} drug-region pairs)")
    if not warning_so.empty:
        st.dataframe(warning_so, use_container_width=True)

    st.warning(f"🟠 EXPIRY ALERT — Prioritize dispensing or return ({len(exp_alerts)} drug-region pairs)")
    if not exp_alerts.empty:
        st.dataframe(exp_alerts, use_container_width=True)

    st.subheader("📦 Supplier Performance")
    st.dataframe(sup_perf, use_container_width=True)

    st.subheader("📈 Demand Trend Insights")
    rising  = dff[dff["Sales_Trend"] > 0].groupby("Medicine")["Sales_Trend"].mean().nlargest(5)
    falling = dff[dff["Sales_Trend"] < 0].groupby("Medicine")["Sales_Trend"].mean().nsmallest(5)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**📈 Rising Demand — Increase Stock**")
        for med, trend in rising.items():
            st.success(f"{med}: +{trend:.2f} units/day")
    with col2:
        st.markdown("**📉 Falling Demand — Reduce Orders**")
        for med, trend in falling.items():
            st.error(f"{med}: {trend:.2f} units/day")

    # ── Download
    st.markdown("---")
    st.subheader("⬇️ Export Predictions")
    export_cols = [
        "Medicine", "Category", "Region", "Facility",
        "Stock", "Units_Sold", "Reorder_Level", "Days_To_Expiry",
        "Stockout_Risk_Pred", "Stockout_Probability", "Days_To_Stockout_Pred", "Stockout_Risk_Level",
        "Expiry_Risk_Pred",   "Expiry_Probability",   "Expiry_Risk_Level",
        "Cluster", "Cluster_Label",
    ]
    csv = dff[export_cols].round(3).to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Full Predictions as CSV",
        data=csv,
        file_name="pharma_predictions.csv",
        mime="text/csv",
    )