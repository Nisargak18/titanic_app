import streamlit as st
import joblib
import numpy as np

# Load the trained model using joblib
model = joblib.load("titanic_model.pkl")

# Streamlit UI
st.title("🚢 Titanic Survival Prediction")

st.write("Fill in the details below to predict if the passenger would have survived:")

# Input features
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.selectbox("Sex", ["Male", "Female"])
age = st.number_input("Age", min_value=0, max_value=100, value=30)
fare = st.number_input("Fare (ticket price)", min_value=0, value=10)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, value=0)
parch = st.number_input("Number of Parents/Children Aboard", min_value=0, value=0)

# Encode 'Sex' column
sex = 1 if sex == "Female" else 0

# Prepare input data for prediction
input_data = np.array([[pclass, sex, age, sibsp, parch, fare]])

# Make prediction when the button is clicked
if st.button("Predict"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.success("✅ The passenger would have **Survived**.")
    else:
        st.error("❌ The passenger would have **Not Survived**.")
