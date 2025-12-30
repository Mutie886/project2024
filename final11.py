import csv
import os
import streamlit as st
import pandas as pd
import numpy as np

# Function to load dataset from CSV file
def load_dataset(filename):
    if not os.path.exists(filename):
        return [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

    df = pd.read_csv(filename)

    # Load relevant columns if they exist
    target_values = df['Target'].tolist() if 'Target' in df.columns else []
    mean_values = df['Mean'].tolist() if 'Mean' in df.columns else []
    variation_values = df['Variation'].tolist() if 'Variation' in df.columns else []
    variation_tv = df['Variation_Tv'].tolist() if 'Variation_Tv' in df.columns else []
    vstatus = df['Vstatus'].tolist() if 'Vstatus' in df.columns else []
    colors = df['Color'].tolist() if 'Color' in df.columns else []
    stddev_variation = df['StdDev_Variation'].tolist() if 'StdDev_Variation' in df.columns else []
    ave_mean = df['Ave_mean'].tolist() if 'Ave_mean' in df.columns else []
    std_tv = df['Std_Tv'].tolist() if 'Std_Tv' in df.columns else []
    remarks = df['Remarks'].tolist() if 'Remarks' in df.columns else []
    remarks2 = df['Remarks2'].tolist() if 'Remarks2' in df.columns else []
    remarks3 = df['Remarks3'].tolist() if 'Remarks3' in df.columns else []
    lag_target = df['lag_Target'].tolist() if 'lag_Target' in df.columns else []
    lag_mean = df['lag_Mean'].tolist() if 'lag_Mean' in df.columns else []
    lag_variation = df['lag_Variation'].tolist() if 'lag_Variation' in df.columns else []
    vstatus_lv = df['Vstatus_LV'].tolist() if 'Vstatus_LV' in df.columns else []

    return (target_values, mean_values, variation_values, colors, remarks, variation_tv, vstatus, std_tv,
            stddev_variation, ave_mean, lag_target, lag_mean, lag_variation, vstatus_lv, remarks2, remarks3)

# Updated Calculation function for mean
def calculate_mean(target_values):
    mean_values = [target_values[0] / 2] if target_values else []
    for i in range(1, len(target_values)):
        mean_values.append((target_values[i - 1] + target_values[i]) / 2)
    return mean_values

def calculate_variation(target_values, mean_values):
    return [target - mean for target, mean in zip(target_values, mean_values)]

def calculate_variation_tv(target_values):
    variation_tv = [0]
    for i in range(1, len(target_values)):
        variation_tv.append(target_values[i] - target_values[i - 1])
    return variation_tv

# No changes here, calculate vstatus based on Variation_Tv
def calculate_vstatus(variation_tv):
    return np.cumsum(variation_tv).tolist()

# Updated function to calculate Vstatus_LV based on lagged Variation_Tv (lag_variation)
def calculate_vstatus_lv(lag_variation):
    lag_variation = [0 if np.isnan(val) else val for val in lag_variation]
    return np.cumsum(lag_variation).tolist()

# Function to assign color based on target values
def assign_color(target_values):
    return ["Blue" if target < 2 else "Purple" if 2 <= target < 10 else "Pink" for target in target_values]

def calculate_std_dev(variation_values):
    return [np.std(variation_values[:i + 1], ddof=1) for i in range(len(variation_values))]

def calculate_ave_mean(target_values):
    cumulative_sum = 0
    ave_means = []
    for i, value in enumerate(target_values):
        cumulative_sum += value
        ave_means.append(cumulative_sum / (i + 1))
    return ave_means

def calculate_std_tv(variation_tv):
    return [np.std(variation_tv[:i + 1], ddof=1) for i in range(len(variation_tv))]

# Function to clear only target values
def clear_target_values():
    st.session_state['input_area'] = ""  # Reset input area in session state

# Ensure equal length of all columns by padding shorter ones
def equalize_column_lengths(lists):
    max_len = max(len(lst) for lst in lists)
    return [lst + [np.nan] * (max_len - len(lst)) for lst in lists]

# Function to calculate lagged values for multiple lists
def calculate_lag_values(*lists):
    lagged_values = []
    for lst in lists:
        lagged_values.append(lst[-2] if len(lst) > 1 else np.nan)
    return lagged_values

# Main function
def main():
    st.title("Data Processing and ARIMA Forecasting App")

    csv_filename= "my mg makex.csv"
    target_values, mean_values, variation_values, colors, remarks, variation_tv, vstatus, std_tv, stddev_variation, ave_mean, lag_target, lag_mean, lag_variation, vstatus_lv, remarks2, remarks3 = load_dataset(csv_filename)

    st.write("Enter the target values (press Enter after each value to store):")

    # Initialize session state for input area if it doesn't exist
    if 'input_area' not in st.session_state:
        st.session_state['input_area'] = ""

    col1, col2 = st.columns(2)

    # Handle the "Clear Input" button press before the text area is rendered
    with col2:
        if st.button("Clear Input"):
            clear_target_values()

    # Now create the text area after clearing input if needed
    new_targets = st.text_area("Enter target values separated by new lines:", 
                               value=st.session_state['input_area'], key="input_area")

    with col1:
        if st.button("Process"):
            if new_targets.strip():
                for user_input in new_targets.split("\n"):
                    try:
                        target = float(user_input)
                        target_values.append(target)

                        # Process the values
                        mean_values = calculate_mean(target_values)  # Updated to use the new mean calculation

                        # Calculate variations and statuses
                        variation_values = calculate_variation(target_values, mean_values)  # Updated variation calculation
                        variation_tv = calculate_variation_tv(target_values)
                        vstatus = calculate_vstatus(variation_tv)

                        # Maintain structure for lagged values
                        lag_target_val, lag_mean_val, lag_variation_val = calculate_lag_values(target_values, mean_values, variation_tv)  # Modified to use variation_tv for lag
                        lag_target.append(lag_target_val)
                        lag_mean.append(lag_mean_val)
                        lag_variation.append(lag_variation_val)

                        # Calculate Vstatus_LV
                        vstatus_lv = calculate_vstatus_lv(lag_variation)
                        colors = assign_color(target_values)
                        stddev_variation = calculate_std_dev(variation_values)
                        ave_mean = calculate_ave_mean(target_values)
                        std_tv = calculate_std_tv(variation_tv)

                        # Remarks handling
                        remarks.append("Vstatus < 0" if vstatus[-1] < 0 else "")
                        remarks2.append("Condition Met" if -0.30 <= vstatus[-1] <= 0.30 and -0.90 <= vstatus_lv[-1] <= 0.90 else "")
                        remarks3.append("CM" if -0.7 <= vstatus[-1] <= 0.7 and -0.9 <= vstatus_lv[-1] <= 0.9 else "")

                    except ValueError:
                        st.write("Invalid input! Please enter valid numbers.")

            # Save the processed data into a CSV file
            with open(csv_filename, mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=[ 
                    "Target", "Remarks", "Remarks2", "Remarks3", "Mean", "Variation", "Variation_Tv", "Vstatus", 
                    "Color", "StdDev_Variation", "Ave_mean", "Std_Tv", "lag_Target", "lag_Mean", 
                    "lag_Variation", "Vstatus_LV"
                ])
                writer.writeheader()

                # Ensure equal lengths of lists before saving
                all_lists = equalize_column_lengths([ 
                    target_values, remarks, remarks2, remarks3, mean_values, variation_values, variation_tv, 
                    vstatus, colors, stddev_variation, ave_mean, std_tv, lag_target, lag_mean, lag_variation, vstatus_lv
                ])

                for row in zip(*all_lists):
                    writer.writerow({
                        "Target": row[0], "Remarks": row[1], "Remarks2": row[2], "Remarks3": row[3], "Mean": row[4], 
                        "Variation": row[5], "Variation_Tv": row[6], "Vstatus": row[7], "Color": row[8], 
                        "StdDev_Variation": row[9], "Ave_mean": row[10], "Std_Tv": row[11], "lag_Target": row[12], 
                        "lag_Mean": row[13], "lag_Variation": row[14], "Vstatus_LV": row[15]
                    })

    # Display the last 6 rows of the processed data
    if target_values:
        df_display = pd.DataFrame({
            "Target": target_values, "Remarks": remarks, "Remarks2": remarks2, "Remarks3": remarks3, "Mean": mean_values,
            "Variation": variation_values, "Variation_Tv": variation_tv, "Vstatus": vstatus, "Color": colors,
            "StdDev_Variation": stddev_variation, "Ave_mean": ave_mean, "Std_Tv": std_tv, "lag_Target": lag_target,
            "lag_Mean": lag_mean, "lag_Variation": lag_variation, "Vstatus_LV": vstatus_lv
        })
        st.write("Last 6 entries:")
        st.dataframe(df_display.tail(6))  # Display only the last 6 rows

if __name__ == "__main__":
    main()
          #streamlit run final11.py