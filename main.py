import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import streamlit as st

df = pd.read_csv("Bank Customer Churn Prediction.csv")
df = df[['credit_score', 'age', 'balance', 'estimated_salary']].dropna()


scaler = StandardScaler()
scaled_features = scaler.fit_transform(df)


kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(scaled_features)


df['Cluster'] = kmeans.labels_
avg_scores = df.groupby('Cluster')['credit_score'].mean().sort_values()
risk_labels = ['Very High Risk', 'High Risk', 'Medium Risk', 'Low Risk']
risk_map = {cluster: risk for cluster, risk in zip(avg_scores.index, risk_labels)}


st.set_page_config(page_title="Customer Risk Prediction", page_icon="⚡", layout="centered")

st.markdown(
    """
    <style>
    /* Dark background for whole page */
    .stApp {
        background-color: #595959;
        color: #f5f5f5;
    }
    
    /* Title color */
    h1, h2, h3 {
        color: #f5f5f5;
    }

    /* Light-colored input fields */
    .stNumberInput input, .stTextInput input {
        background-color: #ffffff;
        color: #000000;
        border-radius: 6px;
        border: 1px solid #ccc;
        padding: 6px;
    }
    /* Change label font color */
    label[data-testid="stWidgetLabel"] > div {
        color: #CEB88A; /* Gold color */
        font-size: 40px;
        font-weight: bold;
    }

    /* Button styling */
    .stButton>button {
        background-color: #1f77b4;
        color: #ffffff;
        border-radius: 6px;
        padding: 10px 20px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #135e96;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🧾 Customer Risk Prediction")

st.markdown("### 👥 Enter Customer Details:")

credit_score = st.number_input("Credit Score", min_value=300, max_value=850, step=1)
age = st.number_input("Age", min_value=18, max_value=100, step=1)
balance = st.number_input("Balance", min_value=0.0, step=100.0)
salary = st.number_input("Estimated Salary", min_value=0.0, step=100.0)


if st.button("🔍 Predict Risk"):
    input_data = np.array([[credit_score, age, balance, salary]])
    scaled_input = scaler.transform(input_data)
    cluster = kmeans.predict(scaled_input)[0]
    risk = risk_map[cluster]

   
    colors = {
        "Very High Risk": "#ff4d4d",   # Red
        "High Risk": "#ff944d",       # Orange
        "Medium Risk": "#ffd11a",     # Yellow
        "Low Risk": "#4dff88"         # Green
    }

    st.markdown(
        f"<div style='background-color:#1e1e1e; padding:15px; border-radius:10px; text-align:center;'>"
        f"<h3 style='color:{colors[risk]};'>🎯 Predicted Customer Risk Level: {risk}</h3>"
        f"</div>",
        unsafe_allow_html=True
    )
