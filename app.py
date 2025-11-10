import streamlit as st
import joblib
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

#load assets and data
try:
    xgb_model=joblib.load("loan_prediction_xgb_model.pkl")
    with open('model_metrics.json', 'r') as f:
        model_metrics = json.load(f)
    df_full = pd.read_csv('loan_data_cleaned.csv')
    df_full[' loan_status'] = df_full[' loan_status'].replace({ ' Approved':'low risk',
    ' Rejected':'high risk'
})
    #loading test data and predictions
    x_test_plot=pd.read_csv('X_test.csv')
    y_test_plot=pd.read_csv('y_test.csv').squeeze()
    y_pred_plot=pd.read_csv('y_pred.csv').squeeze()

except FileNotFoundError:
    st.error("Required files or data files are missing. Please ensure all necessary files are in the working directory.")
    st.stop()

# All continuous features that remained in your DataFrame.
CONTINUOUS_FEATURES = [' no_of_dependents', ' income_annum', ' loan_amount', ' loan_term',
       ' cibil_score', 'Age']

CATEGORICAL_FEATURES = [
    'Gender_Encoded', 
    'Self_Employed_Encoded', 
    'education_encoded'
]

model_features = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES

#streamlit app
st.set_page_config(layout="wide")
st.title('XGBoost Loan Risk Classifier')

#creating tabs, 1; visualizations, 2; risk prediction, 3; model metrics
tab1, tab2, tab3 = st.tabs(["Visualizations (EDA)", "Risk Prediction", "Performance Metrics"])


with tab1:
    st.header("Exploratory Data Analysis (EDA) Visualizations of Training Data")

    original_categorical_features = [' education', 'Gender', ' self_employed', ' no_of_dependents']
    continuous_features_to_plot = [' income_annum', ' loan_amount',' loan_term',' cibil_score', 'Age']
    target_column = ' loan_status'

    # Display pairplot for continuous features
    st.subheader("1. Risk Distribution by Categorical Features")
    for feature in original_categorical_features:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.countplot(x=feature, hue=target_column, data=df_full, 
                      palette={'low risk': 'skyblue', 'high risk': 'salmon'}, ax=ax)
        ax.set_title(f'Risk Distribution by {feature.strip()}')
        st.pyplot(fig)
        plt.close(fig)
    st.subheader("2. Distribution of Continuous Features by Risk Status")
    for feature in continuous_features_to_plot:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(x=target_column, y=feature, data=df_full, 
                    palette={'low risk': 'lightgreen', 'high risk': 'lightcoral'}, ax=ax)
        ax.set_title(f'{feature} Distribution by Risk Status')
        st.pyplot(fig)
        plt.close(fig)
    st.subheader("3. Correlation Heatmap: Income vs. Loan")
    corr_matrix = df_full[[' income_annum', ' loan_amount']].corr()
    fig_corr, ax_corr = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax_corr)
    ax_corr.set_title("Income per Annum and Loan Amount Correlation")
    st.pyplot(fig_corr)
    plt.close(fig_corr)


    #RISK PREDICTION TAB
with tab2:
    st.header("Predict Loan Risk Based on Applicant Features")
    with st.expander('Enter Applicant Details', expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.slider('Age', 18,90,30)
            cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=750)
            loan_term = st.number_input("Loan Term (Months)", min_value=12, max_value=360, value=120)
        
        with col2:
            income = st.number_input("Income per Annum", min_value=100, value=100000000)
            loan_amount = st.number_input("Loan Amount", min_value=100, value=100000000)
            # Numeric Input for Dependents
            no_of_dependents = st.number_input("No. of Dependents", min_value=0, max_value=10, value=0, step=1)
        
        with col3:
            gender = st.selectbox("Gender", ["Female", "Male"])
            self_employed = st.selectbox("Self Employed", ["No", "Yes"])
            education = st.selectbox("Education", ["Not Graduate", "Graduate"])
    
    predict_button = st.button("Calculate Risk Score", type="primary")

    if predict_button:
        # Data preparation logic (Encoding user input)
        raw_data = {
            ' no_of_dependents': no_of_dependents,
            ' income_annum': income, 
            ' loan_amount': loan_amount,
            ' loan_term': loan_term,
            ' cibil_score': cibil_score,
            'Age': age,
            # Numeric, used directly
            
            # Encoded Features
            'Gender_Encoded': 1 if gender == "Male" else 0, 
            'Self_Employed_Encoded': 1 if self_employed == " Yes" else 0, 
            'education_encoded': 1 if education == " Graduate" else 0, 
        }

        input_df = pd.DataFrame([raw_data])
        final_input_df = input_df[model_features]
        final_input_array = final_input_df.values

        # Prediction
        prediction = xgb_model.predict(final_input_array)
        prediction_proba = xgb_model.predict_proba(final_input_array)[:, 1]

        risk_status = "High Risk" if prediction[0] == 1 else "Low Risk"

        # Display Results
        st.subheader("Prediction Outcome")
        if prediction[0] == 1:
            st.error(f"Predicted Status: **{risk_status}**")
            st.metric("Probability of High Risk", f"{prediction_proba[0]*100:.2f}%")
        else:
            st.success(f"Predicted Status: **{risk_status}**")
            st.metric("Probability of Low Risk", f"{(1 - prediction_proba[0])*100:.2f}%")

with tab3:
    st.header("Model Performance Summary")
    
    col_a, col_b = st.columns(2)
    col_a.metric("Overall Accuracy", f"{model_metrics['accuracy']:.4f}")
    col_b.metric("ROC AUC Score", f"{model_metrics['roc_auc']:.4f}")
    
    st.subheader("Classification Report")
    report_df = pd.DataFrame(model_metrics['classification_report']).transpose().round(4)
    st.dataframe(report_df)
    
    st.subheader("Visual Assessment of Performance")
    col_plot1, col_plot2 = st.columns(2)

    # 1. Confusion Matrix Plot
    with col_plot1:
        st.write("#### Confusion Matrix")
        labels = ['low risk (0)', 'high risk (1)']
        
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        cm_display = ConfusionMatrixDisplay.from_predictions(
            y_test_plot, y_pred_plot, display_labels=labels, cmap='Blues', colorbar=False, ax=ax_cm
        )
        ax_cm.set_title("Confusion Matrix")
        st.pyplot(fig_cm)
        plt.close(fig_cm)

    # 2. ROC AUC Curve Plot
    with col_plot2:
        st.write("#### ROC AUC Curve")
        fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_estimator(
            xgb_model, 
            x_test_plot, 
            y_test_plot, 
            name=f'XGBoost (AUC={model_metrics["roc_auc"]:.2f})', 
            ax=ax_roc
        )
        ax_roc.plot([0, 1], [0, 1], linestyle='--', color='red', label='Baseline (AUC = 0.50)')
        ax_roc.set_title("ROC AUC Curve")
        ax_roc.legend(loc='lower right')
        st.pyplot(fig_roc)
        plt.close(fig_roc)
