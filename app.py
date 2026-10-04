import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
import joblib
import shap


from services.data_service import (
    load_data,
    calculate_kpis,
    forecast_business,
    evaluate_forecast
)

from services.ai_service import (
    generate_financial_summary,
    generate_rag_answer,
    is_gemini_configured
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def pd_is_valid(value):
    return pd.notna(value)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="UdyamPulse AI",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("📊 UdyamPulse AI")

st.subheader(
    "Predictive Financial Analytics & "
    "Compliance Intelligence Platform for MSMEs"
)

st.warning(
    "Demo system: All financial data shown is "
    "synthetic and intended for academic use only."
)


# ============================================================
# LOAD DATA
# ============================================================

df = load_data()

# Calculate all financial KPIs
df = calculate_kpis(df)


# ============================================================
# LOAD FINANCIAL RISK MODEL
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "financial_risk_model.pkl"
)

# ============================================================
# LOAD ANOMALY DETECTION MODEL
# ============================================================

ANOMALY_MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "anomaly_detection_model.pkl"
)

try:
    risk_model = joblib.load(MODEL_PATH)
    anomaly_model = joblib.load(ANOMALY_MODEL_PATH)
except Exception as error:
    st.error(
        "A required Phase 2 model could not be loaded. "
        f"Check the model files and installed dependencies ({error.__class__.__name__})."
    )
    st.stop()


# ============================================================
# SIDEBAR - BUSINESS SELECTION
# ============================================================

st.sidebar.header("🏢 Business Selection")

business_list = sorted(
    df["business_id"].unique()
)

selected_business = st.sidebar.selectbox(
    "Select Business",
    business_list
)


# ============================================================
# FILTER SELECTED BUSINESS
# ============================================================

business_df = df[
    df["business_id"] == selected_business
].copy()

business_df = business_df.sort_values(
    "month"
)


# ============================================================
# LATEST MONTH DATA
# ============================================================

latest_data = business_df.iloc[-1]


# ============================================================
# BUSINESS INFORMATION
# ============================================================

st.sidebar.write(
    f"**Sector:** {latest_data['sector']}"
)

st.sidebar.write(
    f"**State:** {latest_data['state']}"
)

st.sidebar.write(
    f"**Employees:** {latest_data['employee_count']}"
)

st.sidebar.write(
    f"**Latest Month:** "
    f"{latest_data['month'].strftime('%B %Y')}"
)


# ============================================================
# BUSINESS OVERVIEW
# ============================================================

st.header("📌 Business Overview")


# ------------------------------------------------------------
# First KPI row
# ------------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Revenue",
    f"₹{latest_data['revenue']:,.0f}"
)

col2.metric(
    "Expenses",
    f"₹{latest_data['expenses']:,.0f}"
)

col3.metric(
    "Profit",
    f"₹{latest_data['profit']:,.0f}"
)

col4.metric(
    "Cash Balance",
    f"₹{latest_data['cash_balance']:,.0f}"
)


# ------------------------------------------------------------
# Second KPI row
# ------------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Receivables",
    f"₹{latest_data['accounts_receivable']:,.0f}"
)

col2.metric(
    "Payables",
    f"₹{latest_data['accounts_payable']:,.0f}"
)

col3.metric(
    "Loan Outstanding",
    f"₹{latest_data['loan_outstanding']:,.0f}"
)

col4.metric(
    "Working Capital",
    f"₹{latest_data['working_capital']:,.0f}"
)


# ============================================================
# REVENUE VS EXPENSES
# ============================================================

st.header("📈 Revenue vs Expenses")

trend_df = business_df[
    [
        "month",
        "revenue",
        "expenses"
    ]
].copy()

trend_df = trend_df.melt(
    id_vars="month",
    value_vars=[
        "revenue",
        "expenses"
    ],
    var_name="Metric",
    value_name="Amount"
)

fig = px.line(
    trend_df,
    x="month",
    y="Amount",
    color="Metric",
    markers=True,
    title="Monthly Revenue and Expenses"
)

fig.update_layout(
    xaxis_title="Month",
    yaxis_title="Amount (₹)"
)

st.plotly_chart(
    fig,
    width="stretch"
)


# ============================================================
# FINANCIAL KPIs
# ============================================================

st.header("📊 Financial KPIs")


# ------------------------------------------------------------
# Profitability and growth
# ------------------------------------------------------------

col1, col2, col3 = st.columns(3)

col1.metric(
    "Profit Margin",
    f"{latest_data['profit_margin']:.2f}%"
)


# Handle first-period NaN values safely

revenue_growth = latest_data["revenue_growth"]

if pd_is_valid(revenue_growth):
    revenue_growth_text = f"{revenue_growth:.2f}%"
else:
    revenue_growth_text = "N/A"


expense_growth = latest_data["expense_growth"]

if pd_is_valid(expense_growth):
    expense_growth_text = f"{expense_growth:.2f}%"
else:
    expense_growth_text = "N/A"


col2.metric(
    "Revenue Growth",
    revenue_growth_text
)

col3.metric(
    "Expense Growth",
    expense_growth_text
)


# ------------------------------------------------------------
# Liquidity and debt KPIs
# ------------------------------------------------------------

col1, col2, col3 = st.columns(3)

col1.metric(
    "Cash Ratio",
    f"{latest_data['cash_ratio']:.2f}"
)

col2.metric(
    "Receivable Ratio",
    f"{latest_data['receivable_ratio'] * 100:.2f}%"
)

col3.metric(
    "Debt Service Ratio",
    f"{latest_data['debt_service_ratio'] * 100:.2f}%"
)


# ------------------------------------------------------------
# Additional KPI
# ------------------------------------------------------------

col1, col2, col3 = st.columns(3)

col1.metric(
    "Payable Ratio",
    f"{latest_data['payable_ratio'] * 100:.2f}%"
)

col2.metric(
    "Inventory Ratio",
    f"{latest_data['inventory_ratio'] * 100:.2f}%"
)

col3.metric(
    "Working Capital",
    f"₹{latest_data['working_capital']:,.0f}"
)


# ============================================================
# FINANCIAL RISK PREDICTION
# ============================================================

st.header("⚠️ Financial Risk Prediction")

st.info(
    "This ML model provides a financial-risk indicator based "
    "on the selected business's current financial condition. "
    "It does not guarantee business failure."
)


# ------------------------------------------------------------
# Current business data
# ------------------------------------------------------------

current_data = latest_data


# ------------------------------------------------------------
# Calculate current features
# ------------------------------------------------------------

profit = (
    current_data["revenue"]
    - current_data["expenses"]
)

profit_margin = (
    profit
    / current_data["revenue"]
    * 100
)

working_capital = (
    current_data["accounts_receivable"]
    + current_data["inventory_value"]
    - current_data["accounts_payable"]
)

cash_ratio = (
    current_data["cash_balance"]
    / current_data["expenses"]
)

receivable_ratio = (
    current_data["accounts_receivable"]
    / current_data["revenue"]
)

payable_ratio = (
    current_data["accounts_payable"]
    / current_data["expenses"]
)

debt_service_ratio = (
    current_data["loan_emi"]
    / current_data["revenue"]
)

inventory_ratio = (
    current_data["inventory_value"]
    / current_data["revenue"]
)


# ------------------------------------------------------------
# Growth features
# ------------------------------------------------------------

revenue_growth = (
    business_df["revenue"]
    .pct_change()
    .iloc[-1]
    * 100
)

expense_growth = (
    business_df["expenses"]
    .pct_change()
    .iloc[-1]
    * 100
)

cash_growth = (
    business_df["cash_balance"]
    .pct_change()
    .iloc[-1]
    * 100
)

receivable_growth = (
    business_df["accounts_receivable"]
    .pct_change()
    .iloc[-1]
    * 100
)


# ------------------------------------------------------------
# Handle missing growth values
# ------------------------------------------------------------

revenue_growth = (
    0.0
    if pd.isna(revenue_growth)
    else revenue_growth
)

expense_growth = (
    0.0
    if pd.isna(expense_growth)
    else expense_growth
)

cash_growth = (
    0.0
    if pd.isna(cash_growth)
    else cash_growth
)

receivable_growth = (
    0.0
    if pd.isna(receivable_growth)
    else receivable_growth
)


# ------------------------------------------------------------
# Create model input
# ------------------------------------------------------------

risk_input = pd.DataFrame([{

    "employee_count":
        current_data["employee_count"],

    "revenue":
        current_data["revenue"],

    "expenses":
        current_data["expenses"],

    "cash_balance":
        current_data["cash_balance"],

    "accounts_receivable":
        current_data["accounts_receivable"],

    "accounts_payable":
        current_data["accounts_payable"],

    "inventory_value":
        current_data["inventory_value"],

    "loan_emi":
        current_data["loan_emi"],

    "loan_outstanding":
        current_data["loan_outstanding"],

    "profit":
        profit,

    "profit_margin":
        profit_margin,

    "working_capital":
        working_capital,

    "cash_ratio":
        cash_ratio,

    "receivable_ratio":
        receivable_ratio,

    "payable_ratio":
        payable_ratio,

    "debt_service_ratio":
        debt_service_ratio,

    "inventory_ratio":
        inventory_ratio,

    "revenue_growth":
        revenue_growth,

    "expense_growth":
        expense_growth,

    "cash_growth":
        cash_growth,

    "receivable_growth":
        receivable_growth

}])


# ------------------------------------------------------------
# Ensure exact feature order
# ------------------------------------------------------------

risk_features = [

    "employee_count",

    "revenue",

    "expenses",

    "cash_balance",

    "accounts_receivable",

    "accounts_payable",

    "inventory_value",

    "loan_emi",

    "loan_outstanding",

    "profit",

    "profit_margin",

    "working_capital",

    "cash_ratio",

    "receivable_ratio",

    "payable_ratio",

    "debt_service_ratio",

    "inventory_ratio",

    "revenue_growth",

    "expense_growth",

    "cash_growth",

    "receivable_growth"

]

risk_input = risk_input[
    risk_features
]


# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

risk_prediction = risk_model.predict(
    risk_input
)[0]

risk_probabilities = risk_model.predict_proba(
    risk_input
)[0]


risk_labels = {
    0: "LOW",
    1: "MEDIUM",
    2: "HIGH"
}


risk_label = risk_labels[
    risk_prediction
]
risk_probability_by_label = {
    risk_labels.get(int(class_id), str(class_id)): float(probability)
    for class_id, probability in zip(risk_model.classes_, risk_probabilities)
}


# ------------------------------------------------------------
# Display risk level
# ------------------------------------------------------------

if risk_label == "LOW":

    st.success(
        "🟢 Financial Risk: LOW"
    )

elif risk_label == "MEDIUM":

    st.warning(
        "🟡 Financial Risk: MEDIUM"
    )

else:

    st.error(
        "🔴 Financial Risk: HIGH"
    )


# ------------------------------------------------------------
# Risk probabilities
# ------------------------------------------------------------

col1, col2, col3 = st.columns(3)

col1.metric(
    "LOW Probability",
    f"{risk_probability_by_label.get('LOW', 0.0) * 100:.1f}%"
)

col2.metric(
    "MEDIUM Probability",
    f"{risk_probability_by_label.get('MEDIUM', 0.0) * 100:.1f}%"
)

col3.metric(
    "HIGH Probability",
    f"{risk_probability_by_label.get('HIGH', 0.0) * 100:.1f}%"
)

st.caption(
    "The prediction is an AI/ML-based financial-risk "
    "indicator and should not be interpreted as a guarantee "
    "of future business failure."
)


# ============================================================
# ANOMALY DETECTION
# ============================================================

st.header("🚨 Financial Anomaly Detection")

st.info(
    "Anomaly detection identifies unusual financial patterns "
    "in the selected business. An anomaly does not necessarily "
    "mean the business is financially unhealthy."
)


# ------------------------------------------------------------
# Same features used by anomaly detection model
# ------------------------------------------------------------

anomaly_features = [

    "employee_count",

    "revenue",

    "expenses",

    "cash_balance",

    "accounts_receivable",

    "accounts_payable",

    "inventory_value",

    "loan_emi",

    "loan_outstanding",

    "profit",

    "profit_margin",

    "working_capital",

    "cash_ratio",

    "receivable_ratio",

    "payable_ratio",

    "debt_service_ratio",

    "inventory_ratio",

    "revenue_growth",

    "expense_growth",

    "cash_growth",

    "receivable_growth"

]


# ------------------------------------------------------------
# Create anomaly input
# ------------------------------------------------------------

anomaly_input = risk_input[
    anomaly_features
].copy()


# ------------------------------------------------------------
# Predict anomaly
# ------------------------------------------------------------

anomaly_prediction = anomaly_model.predict(
    anomaly_input
)[0]

anomaly_score = anomaly_model.decision_function(
    anomaly_input
)[0]


# ------------------------------------------------------------
# Display anomaly result
# ------------------------------------------------------------

if anomaly_prediction == -1:

    st.error(
        "🚨 Financial Pattern: ANOMALY"
    )

    st.metric(
        "Anomaly Score",
        f"{anomaly_score:.4f}"
    )

    st.caption(
        "This business shows an unusual financial pattern "
        "compared with the patterns learned by the anomaly "
        "detection model."
    )

else:

    st.success(
        "✅ Financial Pattern: NORMAL"
    )

    st.metric(
        "Anomaly Score",
        f"{anomaly_score:.4f}"
    )

    st.caption(
        "No significant unusual financial pattern was "
        "detected for the latest available month."
    )


# ============================================================
# SHAP EXPLANATION
# ============================================================

st.header("🔍 AI Risk Explanation")

st.info(
    "SHAP explains which financial features contributed most "
    "to the model's risk prediction."
)


try:

    # --------------------------------------------------------
    # Create SHAP explainer
    # --------------------------------------------------------

    explainer = shap.TreeExplainer(
        risk_model
    )

    shap_values = explainer.shap_values(
        risk_input
    )


    # --------------------------------------------------------
    # Handle SHAP output format
    # --------------------------------------------------------

    if isinstance(shap_values, list):

        predicted_class = int(
            risk_prediction
        )

        class_shap_values = (
            shap_values[predicted_class][0]
        )

    else:

        shap_array = shap_values

        if len(shap_array.shape) == 3:

            predicted_class = int(
                risk_prediction
            )

            class_shap_values = shap_array[
                0,
                :,
                predicted_class
            ]

        else:

            class_shap_values = shap_array[0]


    # --------------------------------------------------------
    # Create explanation dataframe
    # --------------------------------------------------------

    shap_explanation = pd.DataFrame({

        "Feature":
            risk_input.columns,

        "SHAP Value":
            class_shap_values

    })


    shap_explanation[
        "Absolute Impact"
    ] = (
        shap_explanation[
            "SHAP Value"
        ].abs()
    )


    shap_explanation = (
        shap_explanation
        .sort_values(
            "Absolute Impact",
            ascending=False
        )
        .head(10)
    )


    # --------------------------------------------------------
    # Display top features
    # --------------------------------------------------------

    st.subheader(
        "Top Factors Influencing Risk Prediction"
    )


    st.dataframe(
        shap_explanation[
            [
                "Feature",
                "SHAP Value"
            ]
        ].reset_index(drop=True),
        width="stretch"
    )


    # --------------------------------------------------------
    # SHAP bar chart
    # --------------------------------------------------------

    shap_chart = px.bar(

        shap_explanation.sort_values(
            "Absolute Impact"
        ),

        x="SHAP Value",

        y="Feature",

        orientation="h",

        title="Top 10 Risk Prediction Factors"

    )


    shap_chart.update_layout(
        xaxis_title="SHAP Value",
        yaxis_title="Financial Feature"
    )


    st.plotly_chart(
        shap_chart,
        width="stretch"
    )


except Exception as e:

    st.warning(
        f"SHAP explanation could not be generated: {e}"
    )


# ============================================================
# REVENUE AND CASH FORECASTING
# ============================================================

st.header("🔮 Revenue and cash forecast")

try:
    forecast_df = forecast_business(df, selected_business, months_ahead=6)
    forecast_evaluation = evaluate_forecast(df, selected_business)
    latest_month = business_df["month"].max()

    revenue_change = (
        (forecast_df["revenue"].iloc[-1] / latest_data["revenue"] - 1) * 100
        if latest_data["revenue"] else float("nan")
    )
    cash_change = (
        (forecast_df["cash_balance"].iloc[-1] / latest_data["cash_balance"] - 1) * 100
        if latest_data["cash_balance"] else float("nan")
    )

    revenue_tab, cash_tab = st.tabs(["Revenue forecast", "Cash forecast"])
    for tab, metric, title in (
        (revenue_tab, "revenue", "Monthly revenue: history and six-month forecast"),
        (cash_tab, "cash_balance", "Monthly cash balance: history and six-month forecast"),
    ):
        with tab:
            historical = business_df[["month", metric]].dropna()
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=historical["month"],
                y=historical[metric],
                mode="lines+markers",
                name="Historical",
            ))
            forecast_line = pd.concat([
                historical.tail(1).rename(columns={metric: "value"}),
                forecast_df[["month", metric]].rename(columns={metric: "value"}),
            ], ignore_index=True)
            fig.add_trace(go.Scatter(
                x=forecast_line["month"],
                y=forecast_line["value"],
                mode="lines+markers",
                line={"dash": "dash"},
                name="Forecast",
            ))
            fig.add_vline(x=latest_month, line_dash="dot", annotation_text="Forecast starts")
            fig.update_layout(title=title, xaxis_title="Month", yaxis_title="Amount (₹)")
            st.plotly_chart(fig, width="stretch")

    forecast_col1, forecast_col2 = st.columns(2)
    forecast_col1.metric(
        "Revenue in 6 months",
        f"₹{forecast_df['revenue'].iloc[-1]:,.0f}",
        f"{revenue_change:+.1f}% vs latest month",
    )
    forecast_col2.metric(
        "Cash balance in 6 months",
        f"₹{forecast_df['cash_balance'].iloc[-1]:,.0f}",
        f"{cash_change:+.1f}% vs latest month",
    )
    st.caption(
        "Forecasts use a damped linear trend over up to 24 past monthly observations. "
        "They are synthetic-data estimates, not financial advice."
    )
    st.subheader("Six-month forecast table")
    st.dataframe(forecast_df, hide_index=True, width="stretch")
    st.download_button(
        "Download forecast CSV",
        forecast_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{selected_business}_six_month_forecast.csv",
        mime="text/csv",
    )

    st.subheader("Chronological holdout evaluation")
    st.caption("MAE and RMSE are calculated on the latest held-out months, not used to fit the forecast.")
    st.dataframe(forecast_evaluation, hide_index=True, width="stretch")
except (KeyError, ValueError) as error:
    forecast_df = pd.DataFrame()
    forecast_evaluation = pd.DataFrame()
    st.error(f"Forecast could not be generated: {error}")


# ============================================================
# CASH BALANCE TREND
# ============================================================

st.header("💰 Cash Balance Trend")

cash_fig = px.line(
    business_df,
    x="month",
    y="cash_balance",
    markers=True,
    title="Monthly Cash Balance"
)

cash_fig.update_layout(
    xaxis_title="Month",
    yaxis_title="Cash Balance (₹)"
)

st.plotly_chart(
    cash_fig,
    width="stretch"
)


# ============================================================
# PROFIT TREND
# ============================================================

st.header("💵 Profit Trend")

profit_fig = px.line(
    business_df,
    x="month",
    y="profit",
    markers=True,
    title="Monthly Profit"
)

profit_fig.update_layout(
    xaxis_title="Month",
    yaxis_title="Profit (₹)"
)

st.plotly_chart(
    profit_fig,
    width="stretch"
)


# ============================================================
# AI INSIGHTS
# ============================================================

st.header("🤖 AI Financial Explanation")

ai_tab_summary, ai_tab_rag = st.tabs([
    "Financial explanation",
    "RAG compliance assistant"
])


with ai_tab_summary:

    summary_text = generate_financial_summary(
        business_df,
        business_history=business_df,
        analysis_context={
            "risk_label": risk_label,
            "risk_probabilities": risk_probability_by_label,
            "anomaly_result": "anomaly" if anomaly_prediction == -1 else "normal",
            "revenue_forecast_change_pct": (
                float(revenue_change) if not forecast_df.empty else None
            ),
            "cash_forecast_change_pct": (
                float(cash_change) if not forecast_df.empty else None
            ),
            "forecast": (
                {
                    "revenue_next_six_months": forecast_df["revenue"].tolist(),
                    "cash_next_six_months": forecast_df["cash_balance"].tolist(),
                }
                if not forecast_df.empty else {}
            ),
        },
    )

    st.write(
        summary_text
    )
    if not is_gemini_configured():
        st.caption("Gemini API key not configured; showing the local financial explanation.")


with ai_tab_rag:

    default_question = (
        "How should an MSME monitor cash flow and "
        "working capital to avoid financial stress?"
    )

    question = st.text_area(
        "Ask about financial health, working capital, or compliance",
        value=default_question,
        height=120
    )


    if st.button("Generate Answer"):

        response = generate_rag_answer(
            question,
            doc_dir=os.path.join(os.path.dirname(__file__), "documents")
        )

        st.markdown(
            "**Answer**"
        )

        st.write(
            response["answer"]
        )

        if not response.get("used_gemini"):
            st.caption("Showing locally retrieved source passages; no generated answer was available.")

        if response["sources"]:

            st.markdown(
                "**Sources**"
            )

            for source in response["sources"]:

                st.write(
                    f"- {source}"
                )


# ============================================================
# FINANCIAL DATA
# ============================================================

with st.expander(
    "🔍 View Financial Data"
):

    st.dataframe(
        business_df,
        width="stretch"
    )
    st.download_button(
        "Download selected business data",
        business_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{selected_business}_financial_data.csv",
        mime="text/csv",
    )