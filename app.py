import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# 1. Page Configuration Setup
st.set_page_config(page_title="SonRite HVAC Energy Auditor", layout="wide")
st.title("📊 HVAC Active Power Predictive Dashboard")
st.write("This application trains a champion Support Vector Regression (SVR) model on your industrial data to forecast live energy demand.")

# ==========================================
# 2. DATA INGESTION & PIPELINE CLEANING
# ==========================================
@st.cache_data
def load_and_clean_data():
    data = pd.read_parquet("hvac_data_cleaned.parquet")
    
    # Isolate active operating states only (on_off == 1.0)
    data_active = data[data['on_off'] == 1.0]
    return data_active

df_active = load_and_clean_data()

# ==========================================
# 3. LIVE MACHINE LEARNING ENGINE TRAIN
# ==========================================
@st.cache_resource 
def train_svr_engine(data):
    # Drop targets, data leak variables, and the constant on_off flag
    X = data.drop(columns=['active_power', 'active_energy', 'on_off'], errors='ignore')
    y = data['active_power']
    
    # Scale feature matrices
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit SVR with optimized hyperparameters
    model = SVR(kernel='rbf', C=100.0, epsilon=0.1)
    model.fit(X_scaled, y.values.ravel())
    
    return model, scaler, X.columns

model, scaler, feature_names = train_svr_engine(df_active)

# ==========================================
# 4. SIDEBAR USER SIMULATION SLIDERS
# ==========================================
st.sidebar.header("🔧 Live Simulation Variables")
st.sidebar.write("Adjust the environmental factors below to calculate live power demand:")

user_inputs = {}
for col in feature_names:
    min_val = float(df_active[col].min())
    max_val = float(df_active[col].max())
    mean_val = float(df_active[col].mean())
    
    # Only build a slider if the column actually varies
    if min_val != max_val:
        user_inputs[col] = st.sidebar.slider(f"{col}", min_val, max_val, mean_val)
    else:
        st.sidebar.text(f"🔒 {col} (Fixed): {mean_val}")
        user_inputs[col] = mean_val

# ==========================================
# 5. GENERATE LIVE REAL-TIME PREDICTIONS
# ==========================================
# Force the input dataframe to strictly contain only the features passed during fit
input_df = pd.DataFrame([user_inputs])[list(feature_names)]

# Standardize the user's manual choices using the trained scaler rules
input_scaled = scaler.transform(input_df)

# Run prediction
predicted_kw = model.predict(input_scaled)[0]

# Display results inside prominent UI metric blocks
col1, col2 = st.columns(2)
with col1:
    st.markdown("### 🎯 Predicted Power Demand")
    st.metric(label="Calculated SVR Target Output", value=f"{predicted_kw:.2f} kW")

with col2:
    st.markdown("### 📈 Historical Context")
    st.write(f"Dataset Range: **{df_active['active_power'].min():.2f} kW** to **{df_active['active_power'].max():.2f} kW**")

# ==========================================
# 6. RENDER DYNAMIC AUDIT PARITY CHART & DATA SHEET
# ==========================================
st.markdown("---")
st.markdown("### 📊 Model Validation Metric & Data Comparison")

# Create two side-by-side columns: left for the graph, right for the actual numbers spreadsheet
chart_col, table_col = st.columns([1, 1])

# Clean sample selection to eliminate name mismatch bug
sample_data = df_active.sample(n=min(500, len(df_active)), random_state=42)
X_sample = sample_data[list(feature_names)]
y_sample = sample_data['active_power'].values.ravel()

sample_scaled = scaler.transform(X_sample)
sample_preds = model.predict(sample_scaled)

with chart_col:
    st.markdown("#### 📈 Parity Visualization")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=y_sample, y=sample_preds, alpha=0.6, color='teal', ax=ax, label="SVR Boundaries")
    ax.plot([y_sample.min(), y_sample.max()], [y_sample.min(), y_sample.max()], 'r--', linewidth=2, label="Perfect Target Path")
    ax.set_xlabel("Actual Meter Readings (kW)")
    ax.set_ylabel("Model Predictions (kW)")
    ax.legend()
    st.pyplot(fig)

with table_col:
    st.markdown("#### 📋 Actual vs. Predicted Audit Log")
    
    # Construct a clean summary table matrix
    results_df = pd.DataFrame({
        'Actual Power (kW)': y_sample,
        'Predicted Power (kW)': np.round(sample_preds, 2)
    }, index=sample_data.index)
    
    # Calculate the absolute difference/error for each data row
    results_df['Error Margin (kW)'] = np.abs(results_df['Actual Power (kW)'] - results_df['Predicted Power (kW)'])
    results_df['Error Margin (kW)'] = results_df['Error Margin (kW)'].round(2)
    
    # Display the interactive, scrollable data grid on the screen
    st.dataframe(results_df, use_container_width=True, height=320)

# ==========================================
# 7. AUTOMATED FUTURE HORIZON CALENDAR PREDICTOR
# ==========================================
st.markdown("---")
st.markdown("### 📅 Automated Future Horizon Predictor")

# 1. Create a calendar input widget on the screen
future_date = st.date_input(
    "Select a future date to forecast:",
    min_value=datetime.date.today(),
    value=datetime.date.today() + datetime.timedelta(days=1) # Defaults to tomorrow
)

# 2. Extract Time-Engineering Parameters from the chosen date
future_day_of_week = future_date.weekday() # 0 = Monday, 6 = Sunday
future_month = future_date.month

st.write(f"Analyzing historical baselines for Month: **{future_month}** | Day of Week: **{future_day_of_week}**")

# 3. Fetch Historical Baseline Averages for that specific time bracket
matching_historical_data = df_active[
    (df_active.index.month == future_month) & 
    (df_active.index.dayofweek == future_day_of_week)
]

if not matching_historical_data.empty:
    # Calculate the average environmental conditions during that historical period
    baseline_features = matching_historical_data[list(feature_names)].mean()
    
    # Convert baseline features into a 2D dataframe for the model
    future_input_df = pd.DataFrame([baseline_features])[list(feature_names)]
    future_scaled = scaler.transform(future_input_df)
    
    # Run prediction using the SVR brain
    future_pred_kw = model.predict(future_scaled)[0]
    
    st.success(f"🔮 **Estimated Power Demand for {future_date}:** {future_pred_kw:.2f} kW")
    st.info(f"💡 *This prediction is built by combining historical trends for month {future_month} with your trained SVR thermodynamic model.*")
else:
    # Fallback if the dataset doesn't have records for that specific seasonal month yet
    overall_mean = df_active[list(feature_names)].mean()
    future_input_df = pd.DataFrame([overall_mean])[list(feature_names)]
    future_scaled = scaler.transform(future_input_df)
    future_pred_kw = model.predict(future_scaled)[0]
    
    st.warning(f"🔮 **Estimated Power Demand (Using Overall Asset Average):** {future_pred_kw:.2f} kW")
