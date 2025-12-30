import streamlit as st
import pandas as pd
import numpy as np
import os

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
    df["Std_Tv"] = df["Variation_Tv"].expanding().std(ddof=1)
    df["StdDev_Variation"] = df["Variation_Tv"].expanding().std(ddof=1)
    df["Ave_mean"] = df["Target"].expanding().mean()

    # Indicators
    df["Momentum"] = (df["Variation_Tv"] > 0).astype(int)
    df["Low_Volatility"] = (df["Std_Tv"] < df["Std_Tv"].rolling(3).mean()).astype(int)
    df["Stable_Status"] = (
        (df["Vstatus"].abs() < 0.7) &
        (df["Vstatus_LV"].abs() < 0.9)
    ).astype(int)

    # Composite Score
    df["Indicator_Score"] = (
        df["Momentum"] +
        df["Low_Volatility"].fillna(0) +
        df["Stable_Status"]
    )

    # Prediction Flag
    df["Expect_Target_gt_3"] = np.where(df["Indicator_Score"] >= 2, "YES", "NO")

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
# INPUT SECTION
# =========================
st.subheader("Enter Target Values")
user_input = st.text_area("Enter one value per line")

if st.button("Process Data"):
    if user_input.strip():
        values = [float(x) for x in user_input.split("\n")]

        df = load_data()
        new_df = pd.DataFrame({"Target": values})
        df = pd.concat([df, new_df], ignore_index=True)

        df = calculate_features(df)
        df.to_csv(CSV_FILE, index=False)
        st.success("Data processed successfully!")

# =========================
# LOAD & DISPLAY
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

    # Last 6 rows
    st.subheader("📋 Last 6 Records")
    st.dataframe(df.tail(6))

    # Download button
    st.download_button(
        label="⬇️ Download Full Dataset",
        data=df.to_csv(index=False),
        file_name=CSV_FILE,
        mime="text/csv"
    )
