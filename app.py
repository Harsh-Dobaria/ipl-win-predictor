import streamlit as st
import pandas as pd
import shap
import pickle
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="IPL Predictor", layout="centered")

st.title("🏏 IPL Win Probability Predictor")
st.caption("Real-time match prediction using machine learning")

# ------------------ LOAD MODEL (CACHED) ------------------
@st.cache_resource
def load_model():
    model = pickle.load(open("model.pkl", "rb"))
    explainer = shap.TreeExplainer(model)
    return model, explainer

model_xgb, explainer = load_model()

# ------------------ INPUT SECTION ------------------
st.subheader("Match Situation")
st.info("Example: 120 runs, target 160, 15 overs 2 balls, 6 wickets lost")                             
col1, col2 = st.columns(2)

with col1:
    current_score = st.number_input("Current Score", min_value=0)
    target = st.number_input("Target Score", min_value=1)
    wickets_lost = st.number_input("Wickets Lost", min_value=0, max_value=10)

with col2:
    overs = st.number_input("Overs", min_value=0, max_value=20)
    balls = st.number_input("Balls (0–5)", min_value=0, max_value=5)

# ------------------ VALIDATION ------------------
if current_score > target:
    st.error("Score cannot exceed target")
    st.stop()

if st.button("Reset"):
    st.rerun()
    
# ------------------ PREDICTION ------------------
if st.button("Predict"):

    balls_bowled = overs * 6 + balls
    balls_remaining = 120 - balls_bowled
    if balls_remaining <= 0:
        st.error("Match already completed")
        st.stop()
    runs_remaining = target - current_score
    wickets_remaining = 10 - wickets_lost

    # Avoid divide by zero
    total_overs = overs + balls/6 if (overs + balls) > 0 else 1

    crr = current_score / total_overs
    rrr = runs_remaining / (balls_remaining / 6) if balls_remaining > 0 else 0

    pressure = rrr - crr
    wicket_pressure = wickets_remaining * balls_remaining
    runs_per_wicket = runs_remaining / (wickets_remaining + 1)

    # Phase
    if balls_remaining <= 36:
        phase_death, phase_middle, phase_powerplay = 1, 0, 0
    elif balls_remaining <= 90:
        phase_death, phase_middle, phase_powerplay = 0, 1, 0
    else:
        phase_death, phase_middle, phase_powerplay = 0, 0, 1

    input_df = pd.DataFrame({
        'runs_remaining': [runs_remaining],
        'balls_remaining': [balls_remaining],
        'wickets_remaining': [wickets_remaining],
        'crr': [crr],
        'rrr': [rrr],
        'pressure': [pressure],
        'wicket_pressure': [wicket_pressure],
        'runs_per_wicket': [runs_per_wicket],
        'phase_death': [phase_death],
        'phase_middle': [phase_middle],
        'phase_powerplay': [phase_powerplay]
    })

    prob = model_xgb.predict_proba(input_df)[0][1]

    # ------------------ OUTPUT ------------------
    st.subheader("Prediction")

    st.metric("Win Probability", f"{prob*100:.2f}%")

    if prob > 0.7:
        st.success("Strong chance of winning")
    elif prob > 0.4:
        st.warning("Match is balanced")
    else:
        st.error("Low chance of winning")

    # ------------------ MATCH SUMMARY ------------------
    st.subheader("Match Summary")

    st.write(f"Runs Remaining: {runs_remaining}")
    st.write(f"Balls Remaining: {balls_remaining}")
    st.write(f"Wickets Remaining: {wickets_remaining}")
    st.write(f"CRR: {crr:.2f}")
    st.write(f"RRR: {rrr:.2f}")

    # ------------------ SHAP ------------------
    st.subheader("Why this prediction?")

    shap_values = explainer.shap_values(input_df)

    fig, ax = plt.subplots()

    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=input_df.iloc[0],
            feature_names=input_df.columns
        ),
        show=False
    )

    st.pyplot(fig)
    

    vals = shap_values[0]
    features = input_df.columns

    top_idx = np.argmax(np.abs(vals))

    st.write("### Key Factor")
    st.write(f"Most impactful feature: **{features[top_idx]}**")

st.markdown("---")
st.markdown("Built using XGBoost + SHAP | IPL Win Predictor")