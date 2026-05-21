import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide"
)

# ---------------- LOAD DATA ----------------
df = pd.read_csv(
    r"C:\Users\Advance solution\Downloads\31. Laptop_CSV_File.csv"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("best_laptop_price_model.pkl")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

body {
    background-color: #f3f6fb;
}

.hero {
    background: linear-gradient(135deg,#0f172a,#1e40af,#2563eb);
    padding: 40px;
    border-radius: 25px;
    text-align: center;
    color: white;
    margin-bottom: 30px;
}

.hero h1 {
    font-size: 55px;
}

.hero p {
    font-size: 20px;
}

.stButton>button {
    width: 100%;
    background: linear-gradient(135deg,#2563eb,#60a5fa);
    color: white;
    border-radius: 10px;
    height: 3em;
    font-size: 20px;
    border: none;
}

.result-box {
    background: linear-gradient(135deg,#16a34a,#22c55e);
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    color: white;
    margin-top: 30px;
}

.result-price {
    font-size: 50px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HERO SECTION ----------------
st.markdown("""
<div class="hero">
    <h1>💻 Laptop Price Prediction</h1>
    <p>AI Powered Machine Learning Web Application</p>
</div>
""", unsafe_allow_html=True)

# ---------------- INPUT SECTION ----------------

col1, col2 = st.columns(2)

with col1:

    company = st.selectbox(
        "🏢 Brand",
        df['Company'].unique()
    )

    typename = st.selectbox(
        "💼 Laptop Type",
        df['TypeName'].unique()
    )

    ram = st.slider(
        "🧠 RAM (GB)",
        2, 64, 8
    )

    os = st.selectbox(
        "🖥 Operating System",
        df['OpSys'].unique()
    )

    weight = st.number_input(
        "⚖ Weight (kg)",
        min_value=0.5,
        max_value=5.0,
        value=1.5
    )

    touchscreen = st.selectbox(
        "📱 TouchScreen",
        ["No", "Yes"]
    )

with col2:

    ips = st.selectbox(
        "🌈 IPS Display",
        ["No", "Yes"]
    )

    screen_size = st.slider(
        "📺 Screen Size",
        10.0, 18.0, 15.6
    )

    resolution_x = st.selectbox(
        "🖼 Resolution Width",
        [1366, 1600, 1920, 2560, 3840]
    )

    resolution_y = st.selectbox(
        "🖼 Resolution Height",
        [768, 900, 1080, 1440, 2160]
    )

    cpu = st.selectbox(
        "⚙ CPU",
        df['Cpu'].unique()
    )

    gpu = st.selectbox(
        "🎮 GPU Brand",
        df['Gpu'].unique()
    )

# STORAGE
hdd = st.slider(
    "💾 HDD Storage (GB)",
    0, 2000, 0
)

ssd = st.slider(
    "🚀 SSD Storage (GB)",
    0, 2000, 256
)

# ---------------- FEATURE ENGINEERING ----------------

ppi = ((resolution_x ** 2 + resolution_y ** 2) ** 0.5) / screen_size

touchscreen = 1 if touchscreen == "Yes" else 0
ips = 1 if ips == "Yes" else 0

# ---------------- PREDICTION ----------------

if st.button("🔍 Predict Laptop Price"):

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
        'Gpu brand': [gpu]
    })

    prediction = model.predict(sample)

    st.markdown(f"""
    <div class="result-box">

        <h2>💰 Estimated Laptop Price</h2>

        <div class="result-price">
            ₹ {int(prediction[0]):,}
        </div>

        <h3>{company} {typename}</h3>

    </div>
    """, unsafe_allow_html=True)
