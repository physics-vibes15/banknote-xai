# 💵 Banknote Counterfeit Detector with Explainable AI

This Streamlit app is an advanced, fully explainable ML project that detects whether a banknote is genuine or counterfeit using classical machine learning combined with LIME interpretability.

## 🚀 Features

- Trained a RandomForest classifier on the UCI Banknote Authentication dataset
- Achieved 99% test accuracy
- Supports user input or random test samples
- Generates local explanations for predictions using LIME
- Fully interactive, deployed on Streamlit Cloud

## 📊 Dataset

- [UCI Banknote Authentication Data](https://archive.ics.uci.edu/ml/datasets/banknote+authentication)
- 4 numerical features (variance, skewness, kurtosis, entropy)
- Label: genuine (0) or counterfeit (1)

## 🛠️ Tech Stack

- Python
- scikit-learn
- pandas
- matplotlib
- LIME
- Streamlit

## ⚙️ How to Run Locally

```bash
git clone https://github.com/physics-vibes15/banknote-xai.git
cd banknote-xai
pip install -r requirements.txt
streamlit run banknote_xai.py
