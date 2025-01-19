from fastapi import FastAPI
import streamlit as st
import json
import pandas as pd
import plotly.express as px
from datetime import datetime

app = FastAPI()

def create_quality_plot(drift_share, missing_values_share):
    df = pd.DataFrame({
        'Metric': ['Drift Share', 'Missing Values Share'],
        'Value': [drift_share, missing_values_share]
    })
    fig = px.bar(df, x='Metric', y='Value', title='Dataset Quality')
    return fig

def main():
    st.title("Data Drift Monitoring Dashboard")
    
    try:
        with open("/app/reporting/report.json", "r") as f:
            report_data = json.load(f)
        
        # Dataset Overview
        st.header("Dataset Analysis")
        
        # Model Statistics
        col1, col2 = st.columns(2)
        with col1:
            total_columns = report_data['metrics'][0]['result']['number_of_columns']
            st.metric("Total Features", total_columns)
        
        with col2:
            drift_share = report_data['metrics'][0]['result']['share_of_drifted_columns']
            st.metric("Drift Share", f"{drift_share:.2%}")
        
        # Dataset Quality Plot
        st.header("Dataset Quality")
        missing_values = report_data['metrics'][1]['result']['current']['share_of_missing_values']
        quality_fig = create_quality_plot(drift_share, missing_values)
        st.plotly_chart(quality_fig)
        
        # Column Drift Analysis
        st.header("Feature Drift Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            drift_score_0 = report_data['metrics'][2]['result']['drift_score']
            st.metric("Feature 0 Drift Score", f"{drift_score_0:.4f}")
            
        with col2:
            drift_score_1 = report_data['metrics'][3]['result']['drift_score']
            st.metric("Feature 1 Drift Score", f"{drift_score_1:.4f}")
        
        # Timestamp
        st.sidebar.info(f"Report generated on: {report_data.get('timestamp', 'N/A')}")
        
    except Exception as e:
        st.error(f"Error loading dashboard: {str(e)}")

if __name__ == "__main__":
    main()