import streamlit as st
import pandas as pd
import seaborn as sns
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Page title
st.markdown("""
# Translation Initation Rate Prediction App 

This app allows you to predict Translation Initation Rate in Saccharomyces cerevisiae using mRNA features using Machine Learning methods

**Credits**
- App built in `Python` + `Streamlit` by Sulagno Chakraborty, Inayat Ullah Irshad and Dr. Ajeet K. Sharma
[[Read the Paper]]().
---
""")

def load_data(file):
    df = pd.read_csv(file)
    df = df[df['initiation_rate'] < 0.25]  # Filter values less than 0.25
    return df


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    return y_pred, r2


def plot_results(y_test, y_pred):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs. Predicted Values")
    plt.grid(True)
    st.pyplot()


def main():

    # File Upload
    file = st.file_uploader("Upload CSV", type=["csv"])
    if not file:
        st.warning("Please upload a CSV file.")
        return

    # Load Data
    df = load_data(file)
    st.write("Data:")
    st.write(df)

    # Split Data into Features and Target
    X = df.drop('initiation_rate', axis=1)
    y = df['initiation_rate']

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load Models
    rf_model_path = "tir_rf_model.pkl"
    xgb_model_path = "tir_xgb_model.pkl"

    with open(rf_model_path, 'rb') as f:
        rf_model = pickle.load(f)

    with open(xgb_model_path, 'rb') as f:
        xgb_model = pickle.load(f)

    # Evaluate Random Forest Model
    rf_y_pred, rf_r2 = evaluate_model(rf_model, X_test, y_test)
    st.write("Random Forest R-squared Value:", rf_r2)

    # Plot Random Forest Results
    st.subheader("Random Forest Results")
    plot_results(y_test, rf_y_pred)

    # Evaluate XGBoost Model
    xgb_y_pred, xgb_r2 = evaluate_model(xgb_model, X_test, y_test)
    st.write("XGBoost R-squared Value:", xgb_r2)

    # Plot XGBoost Results
    st.subheader("XGBoost Results")
    plot_results(y_test, xgb_y_pred)


if __name__ == '__main__':
    main()

