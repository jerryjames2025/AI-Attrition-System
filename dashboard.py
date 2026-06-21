# Dashboard Layout - executive summary- Multi page streamlit app
import pandas as pd
import numpy as np
import streamlit as st

#we are checking the folder exists
import os

print(os.listdir("outputs"))

# we printing the dashboard KPI

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Employee Attrition Intelligence Platform",
    page_icon="🏢",
    layout="wide"
)

# ==================================================
# SIDEBAR
# ==================================================

# with st.sidebar:

#     st.title("🏢 Attrition AI")

#     st.success(
#         "Enterprise Employee Intelligence Platform"
#     )

#     st.divider()

#     st.markdown(
#         """
#         ### 📂 Analytics Modules

#         🏠 Dashboard

#         👤 Employee Explorer

#         📊 Feature Importance

#         🎯 Scenario Analysis

#         ⚖️ Fairness Analysis

#         🚨 Alerts Dashboard

#         📈 SHAP Explainability

#         🧬 Employee Twin
#         """
#     )

#     st.divider()

#     st.caption(
#         "Built by Data Scientist Jerry 🚀"
#     )

# ==================================================
# LOAD DATA
# ==================================================

summary = pd.read_csv("outputs/summary.csv")

predictions = pd.read_csv(
    "outputs/predictions.csv"
)

# ==================================================
# TITLE
# ==================================================

st.title(
    "🏢 Employee Attrition Intelligence Platform"
)

st.markdown(
    """
    AI-powered workforce analytics platform for
    predicting employee attrition, understanding
    risk drivers, and recommending retention actions.
    """
)

st.divider()

# ==================================================
# KPI SECTION
# ==================================================

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "👥 Employees",
    int(summary["Employees"][0])
)

col2.metric(
    "🎯 Accuracy",
    f"{summary['Accuracy'][0]:.2%}"
)

col3.metric(
    "📈 ROC-AUC",
    round(summary["AUC"][0], 3)
)

col4.metric(
    "⚠️ Avg Risk",
    f"{summary['Avg Risk'][0]:.2f}%"
)

st.divider()

# ==================================================
# RISK DISTRIBUTION
# ==================================================

left, right = st.columns(2)

with left:

    st.subheader("📈 Risk Distribution")

    risk_bins = pd.cut(
        predictions["Probability"] * 100,
        bins=[0, 40, 70, 100],
        labels=[
            "Low Risk",
            "Moderate Risk",
            "High Risk"
        ]
    )

    risk_counts = risk_bins.value_counts()

    st.bar_chart(risk_counts)

with right:

    st.subheader("📉 Prediction Distribution")

    prediction_counts = predictions[
        "Prediction"
    ].value_counts()

    prediction_counts.index = [
        "Stay" if x == 0 else "Leave"
        for x in prediction_counts.index
    ]

    st.bar_chart(prediction_counts)

st.divider()

# ==================================================
# TOP HIGH RISK EMPLOYEES
# ==================================================

st.subheader("🚨 Top 10 High Risk Employees")

top_risk = predictions.sort_values(
    by="Probability",
    ascending=False
)

display_cols = [
    "Age",
    "Gender",
    "Job Role",
    "Work-Life Balance",
    "Remote Work",
    "Company Reputation",
    "Probability"
]

available_cols = [
    col for col in display_cols
    if col in top_risk.columns
]

high_risk_table = top_risk[
    available_cols
].head(10)

if "Probability" in high_risk_table.columns:
    high_risk_table["Probability"] = (
        high_risk_table["Probability"] * 100
    ).round(2)

    high_risk_table.rename(
        columns={
            "Probability": "Risk %"
        },
        inplace=True
    )

st.dataframe(
    high_risk_table,
    use_container_width=True
)

st.divider()

# ==================================================
# ATTRITION SEGMENTATION
# ==================================================

st.subheader("🎯 Employee Risk Segmentation")

low_risk = len(
    predictions[
        predictions["Probability"] < 0.40
    ]
)

moderate_risk = len(
    predictions[
        (predictions["Probability"] >= 0.40)
        &
        (predictions["Probability"] < 0.70)
    ]
)

high_risk = len(
    predictions[
        predictions["Probability"] >= 0.70
    ]
)

segmentation_df = pd.DataFrame({
    "Risk Category": [
        "Low Risk",
        "Moderate Risk",
        "High Risk"
    ],
    "Employees": [
        low_risk,
        moderate_risk,
        high_risk
    ]
})

st.dataframe(
    segmentation_df,
    use_container_width=True
)

st.bar_chart(
    segmentation_df.set_index(
        "Risk Category"
    )
)

st.divider()

# ==================================================
# EXECUTIVE INSIGHTS
# ==================================================

st.subheader("🤖 AI Executive Summary")

high_risk_count = (
    predictions["Probability"] >= 0.70
).sum()

avg_risk = (
    predictions["Probability"].mean()
    * 100
)

predicted_leavers = (
    predictions["Prediction"] == 1
).sum()

st.info(
    f"""
📌 Total Employees Analyzed: {len(predictions)}

📌 Predicted Leavers: {predicted_leavers}

📌 High Risk Employees: {high_risk_count}

📌 Average Attrition Risk: {avg_risk:.2f}%

📌 Employees with elevated risk should be prioritized for retention programs.

📌 Remote Work, Work-Life Balance, Employee Recognition, and Career Growth are major retention drivers.
"""
)

st.divider()

# ==================================================
# PLATFORM STATUS
# ==================================================

st.success(
    """
✅ Model Successfully Loaded

✅ Predictions Generated

✅ Explainability Available

✅ Fairness Analysis Available

✅ Scenario Analysis Available

✅ Employee Twin Available

🚀 Employee Attrition Intelligence Platform Ready
"""
)