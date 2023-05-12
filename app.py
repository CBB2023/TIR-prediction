import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from scipy.stats import pearsonr
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
    return df


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    r = pearsonr(y_test, y_pred)
    return y_pred, r


def plot_results(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred)
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title("Actual vs. Predicted Values")
    ax.grid(True)
    st.pyplot(fig)



def main():

    # Example input file
    example_file = 'demo_file.csv'


    # File Upload
    file = st.file_uploader("Upload CSV", type=["csv"], key='file_uploader')
    if not file:
        st.warning("Please upload a CSV file or use the example file.")
        st.write("Example File:")
        df_example = pd.read_csv(example_file)
        st.write(df_example)
        return

    # Load Data
    df = load_data(file)
    st.write("Data:")
    st.write(df)
    


    # Split Data into Features and Target
    X = df.drop('initiation_rate', axis=1)
    y = df['initiation_rate']

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Load Models
    rf_model_path = "tir_rf_model.pkl"
    xgb_model_path = "tir_xgb_model.pkl"

    with open(rf_model_path, 'rb') as f:
        rf_model = pickle.load(f)

    with open(xgb_model_path, 'rb') as f:
        xgb_model = pickle.load(f)

    # Evaluate Random Forest Model
    rf_y_pred, rf_r = evaluate_model(rf_model, X_test, y_test)
    st.write("Random Forest R Value:", rf_r)

    # Plot Random Forest Results
    st.subheader("Random Forest Results")
    plot_results(y_test, rf_y_pred)

    # Evaluate XGBoost Model
    xgb_y_pred, xgb_r = evaluate_model(xgb_model, X_test, y_test)
    st.write("XGBoost R Value:", xgb_r)

    # Plot XGBoost Results
    st.subheader("XGBoost Results")
    plot_results(y_test, xgb_y_pred)


if __name__ == '__main__':
    main()
