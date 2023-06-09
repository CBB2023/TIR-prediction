import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import base64

# Page title
st.markdown("""
# TIR Predictor 

This allows user to predict Translation Initation Rate in Saccharomyces cerevisiae using mRNA features using Machine Learning methods.

### Introduction: 

Translation initiation, which is the rate-limiting step in protein synthesis, can vary significantly and have a profound impact on cellular protein levels. 
Multiple molecular factors, such as mRNA structure stability, coding sequence length, and specific motifs in mRNA, influence the translation initiation rate, 
allowing precise control of protein synthesis. Despite the crucial role of translation initiation rate, accurately predicting its absolute values based on mRNA 
sequence features remains challenging. To address this issue, we developed a machine learning model specifically trained to predict the in vivo 
initiation rate in S. cerevisiae transcripts. This has been developed on python 3.9 

### How to use:

1. Upload your input file with specified features as in "Example file".
2. Click on the "Start Prediction" to initiate the analysis.
3. After completion download the output file by clicking "Download Predictions".

Note: The output file will contain the perdcited translation initiation rate of the input given for specific given genes.It works properly with one or more genes.

**Credits**
- Built in `Python` + `Streamlit` by Sulagno Chakraborty, Inayat Ullah Irshad, Mahima and Ajeet K. Sharma
[[Read the Paper]]().
---
""")

def load_data(file):
    df = pd.read_csv(file)
    return df


#def train_model(model, X_train, y_train):
    #model.fit(X_train, y_train)
    #return model


def evaluate_model(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred



def main():

    # Example input file
    example_file = 'demo_file.csv'
   
    # File Upload
    file = st.file_uploader("Upload the file", type=["csv","xlsx"], key='file_uploader')
    if not file:
        st.warning("Please upload a file in excel or csv format")
        st.write("Example File:")
        df_example = pd.read_csv(example_file)
        st.write(df_example)
        return


    # Load Data
    df = load_data(file)
    st.write("Data:")
    st.write(df)
    
    

    # Start Prediction Button
    if st.button("Start Prediction"):  
        # Load Models
        rf_model_path = "tir_rf_model.pkl"
        xgb_model_path = "tir_xgb_model.pkl"

        with open(rf_model_path, 'rb') as f:
            rf_model = pickle.load(f)

        with open(xgb_model_path, 'rb') as f:
            xgb_model = pickle.load(f)

        # Evaluate Random Forest Model
        rf_y_pred = evaluate_model(rf_model, df)


        # Evaluate XGBoost Model
        xgb_y_pred = evaluate_model(xgb_model, df)


        # Create a DataFrame with predictions
        df_predictions = pd.DataFrame({
            'Random Forest Predictions': rf_y_pred,
            'XGBoost Predictions': xgb_y_pred
        })

        # Provide a download link for predictions
        csv = df_predictions.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # Convert DataFrame to base64 encoding
        href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Predictions</a>'
        st.markdown("Download Predictions:")
        st.markdown(href, unsafe_allow_html=True)
   

if __name__ == '__main__':
    main()

