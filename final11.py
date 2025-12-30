import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Aviator Prediction Dashboard",
    layout="wide"
)

# -----------------------------
# CSS STYLING
# -----------------------------
st.markdown("""
<style>
body {
    background-color: #f5f7fa;
}
.main-title {
    font-size: 34px;
    font-weight: bold;
    color: #1f4e79;
    text-align: center;
}
.metric-box {
    background-color: white;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0px 0px 10px rgba(0,0,0,0.08);
    text-align: center;
}
.yes {
    color: green;
    font-weight: bold;
}
.no {
    color: red;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Aviator Target > 3 Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown("---")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("aviator_processed_data.csv")

df = load_data()

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
df["Momentum"] = df["Target"].diff()
df["Volatility"] = df["Target"].rolling(5).std()

df.fillna(0, inplace=True)

# Composite score
df["Score"] = (
    0.4 * np.abs(df["Vstatus"]) +
    0.4 * np.abs(df["Vstatus_LV"]) +
    0.2 * np.abs(df["Momentum"])
)

# -----------------------------
# PROBABILITY MODEL
# -----------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

df["Prob_Next_Target_GT_3"] = sigmoid(3 - df["Score"])

# -----------------------------
# SIDEBAR CONTROLS
# -----------------------------
st.sidebar.header("Model Controls")

threshold = st.sidebar.slider(
    "Probability Threshold",
    min_value=0.40,
    max_value=0.80,
    value=0.55,
    step=0.01
)

if st.sidebar.button("Clear Dashboard"):
    st.experimental_rerun()

# -----------------------------
# INDICATOR
# -----------------------------
df["Indicator"] = np.where(
    df["Prob_Next_Target_GT_3"] >= threshold,
    "YES",
    "NO"
)

# -----------------------------
# CURRENT SIGNAL
# -----------------------------
latest = df.iloc[-1]

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.metric("Probability Next Target > 3", f"{latest['Prob_Next_Target_GT_3']:.2f}")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.metric("Threshold", threshold)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    signal_class = "yes" if latest["Indicator"] == "YES" else "no"
    st.markdown(
        f"<span class='{signal_class}'>INDICATOR: {latest['Indicator']}</span>",
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# LAST 6 ROWS
# -----------------------------
st.markdown("### 📋 Last 6 Records")
st.dataframe(
    df.tail(6)[[
        "Target",
        "Vstatus",
        "Vstatus_LV",
        "Prob_Next_Target_GT_3",
        "Indicator"
    ]]
)

# -----------------------------
# VISUALIZATION
# -----------------------------
st.markdown("### 📈 Probability Trend")

fig, ax = plt.subplots()
ax.plot(df["Prob_Next_Target_GT_3"])
ax.axhline(threshold)
ax.set_ylabel("Probability")
ax.set_xlabel("Time Index")
st.pyplot(fig)

# -----------------------------
# DOWNLOAD BUTTON
# -----------------------------
st.markdown("### ⬇ Download Full Dataset")

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download CSV",
    data=csv,
    file_name="aviator_full_dataset.csv",
    mime="text/csv"
)
