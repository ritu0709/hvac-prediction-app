import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Page Configuration Setup
st.set_page_config(page_title="SonRite HVAC Energy Auditor", layout="wide")
st.title("📊 HVAC Active Power Predictive Dashboard")
st.write("This application trains a champion Support Vector Regression (SVR) model on your industrial data to forecast live energy demand.")

# ==========================================
# 2. DATA INGESTION & PIPELINE CLEANING
# ==========================================
@st.cache_data # Cache data so it stays lightning-fast on reload
def load_and_clean_data():
    # Load the compressed parquet structure
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
    # 🟢 FIXED: Drop 'on_off' along with active_power and active_energy 
    # since it's a constant 1.0 and breaks sliders!
    X = data.drop(columns=['active_power', 'active_energy', 'on_off'], errors='ignore')
    y = data['active_power']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
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
    
    # 🟢 DEFENSIVE FIX: Only build a slider if the column actually varies!
    if min_val != max_val:
        user_inputs[col] = st.sidebar.slider(f"{col}", min_val, max_val, mean_val)
    else:
        # If a setting is locked constant (like a seasonal setpoint), show it as text instead of crashing
        st.sidebar.text(f"🔒 {col} (Fixed): {mean_val}")
        user_inputs[col] = mean_val

# ==========================================
# 5. GENERATE LIVE REAL-TIME PREDICTIONS
# ==========================================
# Convert user inputs into a single row matrix dataframe matching original order
input_df = pd.DataFrame([user_inputs])[feature_names]

# Standardize the user's manual choices using the trained scaler rules
# ==========================================
# 5. GENERATE LIVE REAL-TIME PREDICTIONS
# ==========================================
# 🟢 FIXED: Force the input dataframe to ONLY use columns listed in feature_names
input_df = pd.DataFrame([user_inputs])[list(feature_names)]

# Standardize the user's manual choices using the trained scaler rules
input_scaled = scaler.transform(input_df)

# Run prediction
predicted_kw = model.predict(input_scaled)[0]

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
# 6. RENDER DYNAMIC AUDIT PARITY CHART
# ==========================================
st.markdown("---")
st.markdown("### 📊 Model Validation Metric (Parity View)")

@st.cache_data
def load_and_clean_data():
    data = pd.read_parquet("hvac_data_cleaned.parquet")
    data_active = data[data['on_off'] == 1.0]
    
    # 🟢 ADD THIS LINE: Sample 3,000 rows randomly to make cloud training instant
    if len(data_active) > 3000:
        data_active = data_active.sample(n=3000, random_state=42)
        
    return data_active

fig, ax = plt.subplots(figsize=(10, 4))
# Plot a representative sample of active points to keep the layout lightweight
sample_data = df_active.sample(n=500, random_state=42)
X_sample = sample_data.drop(columns=['active_power', 'active_energy'])
y_sample = sample_data['active_power']

sample_scaled = scaler.transform(X_sample)
sample_preds = model.predict(sample_scaled)

sns.scatterplot(x=y_sample, y=sample_preds, alpha=0.5, color='teal', ax=ax, label="SVR Boundaries")
ax.plot([y_sample.min(), y_sample.max()], [y_sample.min(), y_sample.max()], 'r--', label="Perfect Target Path")
ax.set_xlabel("Actual Meter Readings (kW)")
ax.set_ylabel("Model Predictions (kW)")
ax.legend()
st.pyplot(fig)
