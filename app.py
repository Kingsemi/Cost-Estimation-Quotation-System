import streamlit as st
import pandas as pd
import pickle
import numpy as np

# =====================================================
# App Configuration
# =====================================================
st.set_page_config(
    page_title="Electrical Installation Quotation System",
    page_icon="⚡",
    layout="centered"
)

# =====================================================
# Load Model & Feature Columns (Prediction-Only)
# =====================================================
@st.cache_resource
def load_assets():
    model = pickle.load(open("cost_estimation_model.pkl", "rb"))
    feature_columns = pickle.load(open("feature_columns.pkl", "rb"))
    return model, feature_columns

model, feature_columns = load_assets()

# =====================================================
# Header
# =====================================================
st.title("⚡ Electrical Installation Cost Quotation System")
st.markdown(
    """
    This system generates **instant cost estimates** for electrical installation projects  
    based on historical Nigerian project data.
    """
)

st.divider()

# =====================================================
# Client & Project Details
# =====================================================
st.subheader("📌 Project Information")

col1, col2 = st.columns(2)

with col1:
    client_name = st.text_input("Client Name")
    project_ref = st.text_input("Project Reference ID")

with col2:
    project_state = st.selectbox(
        "Project State",
        ["Lagos", "Abuja", "Oyo", "Ogun", "Rivers", "Kwara"]
    )
    building_type = st.selectbox(
        "Building Type",
        ["Residential", "Commercial", "Industrial"]
    )

labour_type = st.selectbox(
    "Labour Skill Level",
    ["Standard", "Skilled", "Highly Skilled"]
)

st.divider()

# =====================================================
# Technical Inputs
# =====================================================
st.subheader("🏗️ Technical Specifications")

col3, col4 = st.columns(2)

with col3:
    floor_area = st.number_input("Floor Area (m²)", min_value=10.0, step=10.0)
    num_rooms = st.number_input("Number of Rooms", min_value=1, step=1)

with col4:
    num_lights = st.number_input("Number of Lighting Points", min_value=1, step=1)
    num_sockets = st.number_input("Number of Socket Outlets", min_value=1, step=1)

st.divider()

# =====================================================
# Prediction Button
# =====================================================
if st.button("📊 Generate Quotation", use_container_width=True):

    # -----------------------------
    # Create Input DataFrame
    # -----------------------------
    input_df = pd.DataFrame([{
        "state": project_state,
        "building_type": building_type,
        "labour_type": labour_type,
        "floor_area_m2": floor_area,
        "num_rooms": num_rooms,
        "num_lights": num_lights,
        "num_sockets": num_sockets
    }])

    # -----------------------------
    # One-Hot Encoding
    # -----------------------------
    input_encoded = pd.get_dummies(input_df)

    # Align columns with training features
    for col in feature_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    input_encoded = input_encoded[feature_columns]

    # -----------------------------
    # Model Prediction
    # -----------------------------
    predicted_cost = model.predict(input_encoded)[0]

    # -----------------------------
    # Cost Breakdown Logic
    # -----------------------------
    MATERIAL_RATIO = 0.65
    LABOUR_RATIO = 0.35

    material_cost = predicted_cost * MATERIAL_RATIO
    labour_cost = predicted_cost * LABOUR_RATIO

    # =====================================================
    # Display Results
    # =====================================================
    st.success("✅ Quotation Generated Successfully")

    st.subheader("💰 Cost Summary")
    st.metric(
        label="Total Project Cost",
        value=f"₦{predicted_cost:,.2f}"
    )

    col5, col6 = st.columns(2)

    with col5:
        st.metric(
            label="🧱 Materials Cost",
            value=f"₦{material_cost:,.2f}"
        )

    with col6:
        st.metric(
            label="👷 Labour Cost",
            value=f"₦{labour_cost:,.2f}"
        )

    st.divider()

    # =====================================================
    # Quotation Summary Table
    # =====================================================
    st.subheader("📄 Quotation Details")

    quotation_table = pd.DataFrame({
        "Item": [
            "Materials Cost",
            "Labour Cost",
            "Total Estimated Cost"
        ],
        "Amount (₦)": [
            f"{material_cost:,.2f}",
            f"{labour_cost:,.2f}",
            f"{predicted_cost:,.2f}"
        ]
    })

    st.table(quotation_table)

    st.divider()

    # =====================================================
    # Disclaimer
    # =====================================================
    st.caption(
        "⚠️ This quotation is an estimate based on historical project data. "
        "Actual project costs may vary due to market conditions, design changes, or site constraints."
    )
