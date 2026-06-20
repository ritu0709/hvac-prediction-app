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
    # Standardize data to extract the precise features you selected
    # We explicitly build the training matrix using ONLY these 5 columns
    X = pd.DataFrame(index=data.index)
    X['outside_temp'] = data['outside_temp']
    X['inlet_temp'] = data['inlet_temp']
    X['outlet_temp'] = data['outlet_temp']
    X['hour'] = data.index.hour
    X['day_of_week'] = data.index.dayofweek
    
    y = data['active_power']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = SVR(kernel='rbf', C=100.0, epsilon=0.1)
    model.fit(X_scaled, y.values.ravel())
    
    return model, scaler, X.columns

model, scaler, feature_names = train_svr_engine(df_active)

# ==========================================
# 4. SIDEBAR USER SIMULATION SLIDERS (CLEANED)
# ==========================================
# ==========================================
# 4. SIDEBAR USER SIMULATION SLIDERS (CLEANED & BOUNDED)
# ==========================================
st.sidebar.header("🔧 Live Simulation Variables")
st.sidebar.write("Adjust the primary thermal factors to calculate live power demand:")

user_inputs = {}

user_inputs['outside_temp'] = st.sidebar.slider(
    "Outside Temperature (°C)", 
    float(df_active['outside_temp'].min()), float(df_active['outside_temp'].max()), float(df_active['outside_temp'].mean())
)

# 🟢 FIXED: Set realistic Return water boundaries (Warm)
user_inputs['inlet_temp'] = st.sidebar.slider(
    "Inlet Chilled Water Temp (Return) (°C)", 
    min_value=12.0, 
    max_value=20.0, 
    value=14.0
)

# 🟢 FIXED: Set realistic Supply water boundaries (Cold)
user_inputs['outlet_temp'] = st.sidebar.slider(
    "Outlet Chilled Water Temp (Supply) (°C)", 
    min_value=6.0, 
    max_value=12.0, 
    value=7.0
)

# Automatically inject current time features into the calculation background
user_inputs['hour'] = float(datetime.datetime.now().hour)
user_inputs['day_of_week'] = float(datetime.datetime.now().weekday())

# ==========================================
# 5. GENERATE LIVE REAL-TIME PREDICTIONS
# ==========================================
input_df = pd.DataFrame([user_inputs])[list(feature_names)]
input_scaled = scaler.transform(input_df)
predicted_kw = model.predict(input_scaled)[0]

# Display results inside prominent UI metric blocks
col1, col2 = st.columns(2)
with col1:
    st.markdown("### 🎯 Predicted Power Demand (Sidebar Sliders)")
    st.metric(label="Calculated SVR Target Output", value=f"{predicted_kw:.2f} kW")
with col2:
    st.markdown("### 📈 Historical Context")
    st.write(f"Dataset Range: **{df_active['active_power'].min():.2f} kW** to **{df_active['active_power'].max():.2f} kW**")

# ==========================================
# 6. RENDER DYNAMIC AUDIT PARITY CHART & DATA SHEET
# ==========================================
st.markdown("---")
st.markdown("### 📊 Model Validation Metric & Data Comparison")

chart_col, table_col = st.columns([1, 1])

sample_data = df_active.sample(n=min(500, len(df_active)), random_state=42)
sample_data_features = pd.DataFrame(index=sample_data.index)
sample_data_features['outside_temp'] = sample_data['outside_temp']
sample_data_features['inlet_temp'] = sample_data['inlet_temp']
sample_data_features['outlet_temp'] = sample_data['outlet_temp']
sample_data_features['hour'] = sample_data_features.index.hour
sample_data_features['day_of_week'] = sample_data_features.index.dayofweek

X_sample = sample_data_features[list(feature_names)]
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
    results_df = pd.DataFrame({
        'Actual Power (kW)': y_sample,
        'Predicted Power (kW)': np.round(sample_preds, 2)
    }, index=sample_data.index)
    results_df['Error Margin (kW)'] = np.abs(results_df['Actual Power (kW)'] - results_df['Predicted Power (kW)']).round(2)
    st.dataframe(results_df, use_container_width=True, height=320)

# ==========================================
# 7. MANIPULATE A SPECIFIC FUTURE DATE TARGET
# ==========================================
st.markdown("---")
st.markdown("### 🔮 Manipulate Future Target Date Prediction")
st.write("Manually enter a custom future date and expected thermodynamic inputs to run a targeted prediction:")

input_col1, input_col2, input_col3 = st.columns(3)

with input_col1:
    future_date = st.date_input("Choose Future Date:", datetime.date.today() + datetime.timedelta(days=1))
    future_time = st.time_input("Choose Future Shift Time:", datetime.time(14, 0))

with input_col2:
    f_outside = st.number_input("Forecasted Outside Temp (°C):", min_value=10.0, max_value=50.0, value=35.0, step=0.5)
    # 🟢 FIXED BOUNDS
    f_inlet = st.number_input("Expected Inlet Chilled Water Temp (°C):", min_value=12.0, max_value=20.0, value=14.0, step=0.5)

with input_col3:
    # 🟢 FIXED BOUNDS
    f_outlet = st.number_input("Expected Outlet Chilled Water Temp (°C):", min_value=6.0, max_value=12.0, value=7.0, step=0.5)

if st.button("🚀 Calculate Future Date Power Demand"):
    combined_dt = datetime.datetime.combine(future_date, future_time)
    f_hour = combined_dt.hour
    f_day_of_week = combined_dt.weekday()
    
    future_scenario_df = pd.DataFrame([{
        'outside_temp': float(f_outside),
        'inlet_temp': float(f_inlet),
        'outlet_temp': float(f_outlet),
        'hour': float(f_hour),
        'day_of_week': float(f_day_of_week)
    }])[list(feature_names)]
    
    future_scenario_scaled = scaler.transform(future_scenario_df)
    future_prediction_kw = model.predict(future_scenario_scaled)[0]
    
    st.success(f"🎯 **SVR Forecast Result:** The predicted power demand for **{combined_dt.strftime('%B %d, %Y at %I:%M %p')}** is **{future_prediction_kw:.2f} kW**")
    st.info(f"💡 *Analysis Parameters Used — Outside: {f_outside}°C | Loop ΔT: {f_inlet - f_outlet:.1f}°C | Hour Key: {f_hour} | Day Code: {f_day_of_week}*")
