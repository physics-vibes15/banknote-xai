import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from lime import lime_tabular
import matplotlib.pyplot as plt

# --- LOAD DATA ---
@st.cache_data
def load_data():
    # UCI banknote dataset
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/"
        "data_banknote_authentication.txt"
    )
    df = pd.read_csv("data_banknote_authentication.txt", header=None)
    df.columns = ['variance', 'skewness', 'kurtosis', 'entropy', 'class']
    return df

df = load_data()

# --- TRAIN MODEL ---
X = df.drop('class', axis=1)
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# --- LIME EXPLAINER ---
explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X.columns,
    class_names=['genuine', 'counterfeit'],
    mode='classification'
)

# --- STREAMLIT UI ---
st.title("💵 Banknote Counterfeit Detector with Explainable AI")

st.write(
    f"Model trained with accuracy: **{acc:.2%}**"
)

st.write("Upload features manually to test or pick a random sample from the dataset.")

option = st.radio(
    "Choose input mode",
    ("Use random test sample", "Enter manually")
)

if option == "Use random test sample":
    index = st.slider("Test sample index", 0, len(X_test)-1, 0)
    sample = X_test.iloc[index]
else:
    v = st.number_input("Variance", -10.0, 10.0, step=0.1)
    s = st.number_input("Skewness", -10.0, 10.0, step=0.1)
    k = st.number_input("Kurtosis", -10.0, 10.0, step=0.1)
    e = st.number_input("Entropy", -10.0, 10.0, step=0.1)
    sample = pd.Series([v, s, k, e], index=X.columns)

# --- PREDICT ---
sample_array = sample.values.reshape(1, -1)
pred = model.predict(sample_array)[0]
pred_proba = model.predict_proba(sample_array)[0]

label = "Genuine" if pred == 0 else "Counterfeit"
st.subheader(f"Prediction: **{label}**")
st.write(f"Probability of being genuine: **{pred_proba[0]:.2%}**")

# --- EXPLAIN WITH LIME ---
if st.button("Explain Prediction"):
    exp = explainer.explain_instance(
        data_row=sample.values,
        predict_fn=model.predict_proba
    )
    st.write("Explanation:")
    fig = exp.as_pyplot_figure()
    st.pyplot(fig)

