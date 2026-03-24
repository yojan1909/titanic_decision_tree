import joblib 
import pandas as pd 
import streamlit as st 
 
model = joblib.load("models/decision_tree_model.pkl") 
feature_columns = joblib.load("models/feature_columns.pkl") 
median_age = joblib.load("models/median_age.pkl") 
st.set_page_config(page_title="Titanic Survival Prediction", page_icon="🚢", layout="centered") 
st.title("🚢 Titanic Survival Prediction using Decision Tree") 
st.write("Enter passenger details to predict survival.") 
pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3]) 
sex = st.selectbox("Sex", ["male", "female"]) 
age = st.number_input("Age", min_value=0.0, max_value=100.0, value=float(median_age), step=1.0) 
sibsp = st.number_input("Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, value=0, 
step=1) 
parch = st.number_input("Parents/Children Aboard (Parch)", min_value=0, max_value=10, value=0, 
step=1) 
fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=32.0, step=1.0) 
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"]) 
if st.button("Predict Survival"): 
    input_data = { 
        "Pclass": pclass, 
        "Sex": sex, 
        "Age": age, 
        "SibSp": sibsp, 
        "Parch": parch, 
        "Fare": fare, 
        "Embarked": embarked 
    } 
    input_df = pd.DataFrame([input_data]) 
    input_encoded = pd.get_dummies(input_df, drop_first=True) 
    for col in feature_columns: 
        if col not in input_encoded.columns: 
            input_encoded[col] = 0 
    input_encoded = input_encoded[feature_columns] 
    prediction = model.predict(input_encoded)[0] 
    prediction_proba = model.predict_proba(input_encoded)[0] 
 
    if prediction == 1: 
        st.success("Prediction: Passenger would likely **Survive**") 
    else: 
        st.error("Prediction: Passenger would likely **Not Survive**") 
 
    st.subheader("Prediction Probabilities") 
    st.write(f"Did Not Survive: {prediction_proba[0]:.2%}") 
    st.write(f"Survived: {prediction_proba[1]:.2%}")
