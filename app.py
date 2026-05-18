import streamlit as st
import pandas as pd
import joblib

# Load dataset
df = pd.read_csv(r"C:\Users\Advance solution\Downloads\31. Laptop_CSV_File.csv")

# Load trained model
model = joblib.load("best_laptop_price_model.pkl")

st.title("Laptop Price Prediction")

# Inputs
company = st.selectbox("Brand", df['Company'].unique())

typename = st.selectbox("Type", df['TypeName'].unique())

ram = st.slider("RAM (GB)", 2, 64, 8)

os = st.selectbox("Operating System", df['OpSys'].unique())

weight = st.number_input("Weight", min_value=0.5, max_value=5.0, value=1.5)

touchscreen = st.selectbox("TouchScreen", ["No", "Yes"])

ips = st.selectbox("IPS Display", ["No", "Yes"])

screen_size = st.slider("Screen Size", 10.0, 18.0, 15.6)

resolution_x = st.selectbox(
    "Resolution Width",
    [1366, 1600, 1920, 2560, 3840]
)

resolution_y = st.selectbox(
    "Resolution Height",
    [768, 900, 1080, 1440, 2160]
)
cpu = st.selectbox("Cpu_brand", df['Cpu'].unique())


hdd = st.slider("HDD (GB)", 0, 2000, 0)

ssd = st.slider("SSD (GB)", 0, 2000, 256)

gpu = st.selectbox("GPU Brand", df['Gpu'].unique())


# Calculate PPI
ppi = ((resolution_x ** 2 + resolution_y ** 2) ** 0.5) / screen_size

# Convert Yes/No into 1/0
touchscreen = 1 if touchscreen == "Yes" else 0
ips = 1 if ips == "Yes" else 0

if st.button("Predict Price"):

    sample = pd.DataFrame({
        'Company': [company],
        'TypeName': [typename],
        'Ram': [ram],
        'OpSys': [os],
        'Weight': [weight],
        'TouchScreen': [touchscreen],
        'IPS': [ips],
        'PPI': [ppi],
        'CPU_name': [cpu],
        'HDD': [hdd],
        'SSD': [ssd],
        'Gpu brand': [gpu],
        
    })

    prediction = model.predict(sample)

    st.success(f"Estimated Laptop Price: ₹ {int(prediction[0])}")