import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# =====================================================
# App Configuration
# =====================================================
st.set_page_config(
    page_title="Electrical Installation Quotation System",
    page_icon="⚡",
    layout="centered"
)

# =====================================================
# Load Model & Feature Columns
# =====================================================
@st.cache_resource
def load_assets():
    model = pickle.load(open("cost_estimation_model.pkl", "rb"))
    feature_columns = pickle.load(open("feature_columns.pkl", "rb"))
    return model, feature_columns

model, feature_columns = load_assets()

# =====================================================
# State Price Multipliers
# =====================================================
STATE_MULTIPLIERS = {
    "Lagos": 1.15,
    "Abuja": 1.12,
    "Rivers": 1.10,
    "Oyo": 1.00,
    "Ogun": 0.95,
    "Kwara": 0.92
}

# =====================================================
# Header
# =====================================================
st.title("⚡ Electrical Installation Cost Quotation System")
st.markdown("Transparent, data-driven electrical cost estimation.")

st.divider()

# =====================================================
# Project Information
# =====================================================
st.subheader("📌 Project Information")

col1, col2 = st.columns(2)

with col1:
    client_name = st.text_input("Client Name")
    project_reference = st.text_input("Project Reference")

with col2:
    state = st.selectbox("Project State", list(STATE_MULTIPLIERS.keys()))
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
st.subheader("🏗️ Building & Electrical Scope")

col3, col4 = st.columns(2)

with col3:
    floor_area_m2 = st.number_input("Floor Area (m²)", min_value=10.0, step=10.0)
    rooms = st.number_input("Number of Rooms", min_value=1, step=1)
    lighting_points = st.number_input("Lighting Points", min_value=1, step=1)
    socket_points = st.number_input("Socket Points", min_value=1, step=1)

with col4:
    switch_points = st.number_input("Switch Points", min_value=1, step=1)
    cable_length_m = st.number_input("Cable Length (m)", min_value=1.0, step=5.0)
    conduit_length_m = st.number_input("Conduit Length (m)", min_value=1.0, step=5.0)

st.divider()

# =====================================================
# Generate Quotation
# =====================================================
if st.button("📊 Generate Quotation", use_container_width=True):

    # -----------------------------
    # Input DataFrame
    # -----------------------------
    input_df = pd.DataFrame([{
        "state": state,
        "building_type": building_type,
        "floor_area_m2": floor_area_m2,
        "rooms": rooms,
        "lighting_points": lighting_points,
        "socket_points": socket_points,
        "switch_points": switch_points,
        "cable_length_m": cable_length_m,
        "conduit_length_m": conduit_length_m,
        "labour_type": labour_type
    }])

    # One-hot encoding
    input_encoded = pd.get_dummies(input_df)

    for col in feature_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    input_encoded = input_encoded[feature_columns]

    # -----------------------------
    # Base Prediction
    # -----------------------------
    base_cost = model.predict(input_encoded)[0]

    # Apply state multiplier
    multiplier = STATE_MULTIPLIERS[state]
    total_cost = base_cost * multiplier

    # -----------------------------
    # Cost Breakdown
    # -----------------------------
    material_cost = total_cost * 0.65
    labour_cost = total_cost * 0.35

    # =====================================================
    # Results
    # =====================================================
    st.success("✅ Quotation Generated Successfully")

    st.metric("Total Estimated Cost", f"₦{total_cost:,.2f}")
    st.caption(f"State price multiplier applied: ×{multiplier}")

    col5, col6 = st.columns(2)
    with col5:
        st.metric("🧱 Materials Cost", f"₦{material_cost:,.2f}")
    with col6:
        st.metric("👷 Labour Cost", f"₦{labour_cost:,.2f}")

    st.divider()

    # =====================================================
    # Feature Importance (Client Explanation)
    # =====================================================
    st.subheader("📊 What Drives This Cost?")

    importance = pd.Series(
        model.feature_importances_,
        index=feature_columns
    ).sort_values(ascending=False)

    # Group similar features for clarity
    importance_groups = {
        "Floor Area": importance.filter(like="floor_area").sum(),
        "Rooms": importance.filter(like="rooms").sum(),
        "Lighting Points": importance.filter(like="lighting_points").sum(),
        "Socket Points": importance.filter(like="socket_points").sum(),
        "Switch Points": importance.filter(like="switch_points").sum(),
        "Cable Length": importance.filter(like="cable_length").sum(),
        "Conduit Length": importance.filter(like="conduit_length").sum(),
        "Building Type": importance.filter(like="building_type").sum(),
        "Labour Skill": importance.filter(like="labour_type").sum(),
        "State Factor": importance.filter(like="state").sum()
    }

    importance_df = (
        pd.Series(importance_groups)
        .sort_values(ascending=True)
    )

    fig, ax = plt.subplots()
    importance_df.plot(kind="barh", ax=ax)
    ax.set_xlabel("Relative Importance")
    ax.set_title("Key Cost Drivers")

    st.pyplot(fig)

    st.caption(
        "This chart explains which project characteristics most influenced the estimated cost."
    )
