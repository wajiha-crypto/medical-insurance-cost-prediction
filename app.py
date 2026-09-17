import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Insurance Cost Predictor", page_icon="💰")

st.title("Medical Insurance Cost Predictor")
st.write("Enter your details below to estimate your medical insurance cost.")

@st.cache_resource
def train_model():
    df = pd.read_csv("insurance.csv")
    df = df.drop_duplicates()

    df_encoded = df.copy()
    df_encoded['sex'] = df_encoded['sex'].map({'male': 0, 'female': 1})
    df_encoded['smoker'] = df_encoded['smoker'].map({'no': 0, 'yes': 1})
    df_encoded = pd.get_dummies(df_encoded, columns=['region'], drop_first=True)

    X = df_encoded.drop('charges', axis=1)
    y = df_encoded['charges']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, X.columns

model, feature_columns = train_model()

age = st.slider("Age", 18, 64, 30)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.slider("BMI", 15.0, 55.0, 25.0)
children = st.slider("Number of Children", 0, 5, 0)
smoker = st.selectbox("Smoker", ["no", "yes"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

if st.button("Predict Cost"):
    sex_val = 0 if sex == "male" else 1
    smoker_val = 1 if smoker == "yes" else 0
    region_northwest = 1 if region == "northwest" else 0
    region_southeast = 1 if region == "southeast" else 0
    region_southwest = 1 if region == "southwest" else 0

    input_data = pd.DataFrame([{
        "age": age,
        "sex": sex_val,
        "bmi": bmi,
        "children": children,
        "smoker": smoker_val,
        "region_northwest": region_northwest,
        "region_southeast": region_southeast,
        "region_southwest": region_southwest
    }])

    input_data = input_data[feature_columns]
    prediction = model.predict(input_data)[0]

    st.success(f"Estimated Insurance Cost: ${prediction:,.2f}")

st.caption("Built as part of the Medical Insurance Cost Prediction project — Big Brains internship.")
