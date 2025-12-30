import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Aviator Indicator Dashboard", layout="wide")
CSV_FILE = "aviator_processed_data.csv"

# =========================
# CSS STYLING
# =========================
st.markdown("""
<style>
body {
    background-color: #f6f8fa;
}
.metric-box {
    padding: 15px;
    border-radius: 10px;
    background-color: white;
    box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
    text-align: center;
    margin-bottom: 10px;
}
.good { color: green; font-weight: bold; }
.warn { color: orange; font-weight: bold; }
.bad { color: red; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# =========================
# FUNCTIONS
# =========================
def calculate_features(df):
    df["Variation_Tv"] = df["Target"].diff().fillna(0)
    df["Vstatus"] = df["Variation_Tv"].cumsum()
    df["lag_Target"] = df["Target"].shift(1)
    df["lag_Variation"] = df["Variation_Tv"].shift(1)
    df["Vstatus_LV"] = df["lag_Variation"].fillna(0).cumsum()
    df["Std_Tv"] = df["Variation_Tv"].expanding().std(ddof=1).fillna(0)
    df["Momentum"] = (df["Variation_Tv"] > 0).astype(int)
    df["Low_Volatility"] = (df["Std_Tv"] < df["Std_Tv"].rolling(3, min_periods=1).mean()).astype(int)
    df["Stable_Status"] = ((df["Vstatus"].abs() < 0.7) & (df["Vstatus_LV"].abs() < 0.9)).astype(int)

    # Composite Score & Probability
    df["Indicator_Score"] = df["Momentum"] + df["Low_Volatility"] + df["Stable_Status"]
    df["Prob_Next_Target_GT_3"] = 1 / (1 + np.exp(-(df["Indicator_Score"])))  # Sigmoid

    # Prediction based on threshold (default 0.55)
    df["Expect_Target_gt_3"] = np.where(df["Prob_Next_Target_GT_3"] >= 0.55, "YES", "NO")
    return df

def load_data():
    if os.path.exists(CSV_FILE):
        return pd.read_csv(CSV_FILE)
    return pd.DataFrame(columns=["Target"])

# =========================
# APP TITLE
# =========================
st.title("🚀 Aviator Early Indicator Dashboard")
st.write("Predictive indicators for **Next Target > 3** (statistical, not guaranteed)")

# =========================
# SIDEBAR CONTROLS
# =========================
st.sidebar.header("Model Controls")
threshold = st.sidebar.slider(
    "Probability Threshold",
    min_value=0.40,
    max_value=0.80,
    value=0.55,
    step=0.01
)
if st.sidebar.button("Clear Dashboard"):
    if os.path.exists(CSV_FILE):
        os.remove(CSV_FILE)
    st.experimental_rerun()

# =========================
# INPUT SECTION
# =========================
st.subheader("Enter Target Values (One per Line)")
user_input = st.text_area("")

if st.button("Process Data"):
    if user_input.strip():
        values = [float(x) for x in user_input.split("\n")]
        df = load_data()
        new_df = pd.DataFrame({"Target": values})
        df = pd.concat([df, new_df], ignore_index=True)
        df = calculate_features(df)
        # Apply threshold from sidebar
        df["Expect_Target_gt_3"] = np.where(df["Prob_Next_Target_GT_3"] >= threshold, "YES", "NO")
        df.to_csv(CSV_FILE, index=False)
        st.success("Data processed successfully!")

# =========================
# LOAD & DISPLAY DATA
# =========================
df = load_data()
if not df.empty:
    last = df.iloc[-1]

    st.subheader("📊 Current Indicator Status")
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(f"<div class='metric-box'>Momentum<br><span class='good'>{last['Momentum']}</span></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='metric-box'>Low Volatility<br><span class='good'>{last['Low_Volatility']}</span></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='metric-box'>Stable Status<br><span class='good'>{last['Stable_Status']}</span></div>", unsafe_allow_html=True)

    status_class = "good" if last["Expect_Target_gt_3"] == "YES" else "bad"
    col4.markdown(
        f"<div class='metric-box'>Next Target &gt; 3<br><span class='{status_class}'>{last['Expect_Target_gt_3']}</span></div>",
        unsafe_allow_html=True
    )

    # =========================
    # LAST 6 ROWS
    # =========================
    st.subheader("📋 Last 6 Records")
    st.dataframe(df.tail(6))

    # =========================
    # PROBABILITY TREND
    # =========================
    st.subheader("📈 Probability Trend")
    fig, ax = plt.subplots()
    ax.plot(df["Prob_Next_Target_GT_3"], marker='o', linestyle='-')
    ax.axhline(threshold, color='red', linestyle='--', label='Threshold')
    ax.set_ylabel("Probability")
    ax.set_xlabel("Index")
    ax.legend()
    st.pyplot(fig)

    # =========================
    # DOWNLOAD BUTTON
    # =========================
    st.download_button(
        label="⬇️ Download Full Dataset",
        data=df.to_csv(index=False),
        file_name=CSV_FILE,
        mime="text/csv"
    )
