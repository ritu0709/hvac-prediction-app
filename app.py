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
# 6. RENDER DYNAMIC AUDIT PARITY CHART
# ==========================================
st.markdown("---")
st.markdown("### 📊 Model Validation Metric (Parity View)")

fig, ax = plt.subplots(figsize=(10, 4))

# Clean sample selection to eliminate the feature name mismatch bug
sample_data = df_active.sample(n=min(500, len(df_active)), random_state=42)
X_sample = sample_data[list(feature_names)] # 🟢 FIXED: Force sample to only use matched features
y_sample = sample_data['active_power']

sample_scaled = scaler.transform(X_sample)
sample_preds = model.predict(sample_scaled)

sns.scatterplot(x=y_sample, y=sample_preds, alpha=0.5, color='teal', ax=ax, label="SVR Boundaries")
ax.plot([y_sample.min(), y_sample.max()], [y_sample.min(), y_sample.max()], 'r--', label="Perfect Target Path")
ax.set_xlabel("Actual Meter Readings (kW)")
ax.set_ylabel("Model Predictions (kW)")
ax.legend()
st.pyplot(fig)
