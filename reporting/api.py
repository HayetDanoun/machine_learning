from fastapi import FastAPI
import streamlit as st
import json
import pandas as pd

app = FastAPI()

def process_report_metrics(report_data):
    """Extract and process key metrics from the report"""
    summary = {
        'Dataset Statistics': [],
        'Column Drift Details': []
    }
    
    # Process main dataset drift metrics
    for metric in report_data.get('metrics', []):
        if metric['metric'] == 'DatasetDriftMetric':
            result = metric['result']
            summary['Dataset Statistics'] = [{
                'Metric': 'Total Columns',
                'Value': result['number_of_columns']
            }, {
                'Metric': 'Drifted Columns',
                'Value': result['number_of_drifted_columns']
            }, {
                'Metric': 'Drift Share',
                'Value': f"{result['drift_share']*100:.2f}%"
            }]
        
        # Process column-level drift details
        elif metric['metric'] == 'DataDriftTable':
            drift_columns = []
            for col, details in metric['result'].get('drift_by_columns', {}).items():
                if isinstance(details, dict):  # Skip any malformed entries
                    drift_columns.append({
                        'Column': col,
                        'Drift Score': f"{details['drift_score']:.4f}",
                        'Drift Detected': '🔴 Yes' if details['drift_detected'] else '🟢 No'
                    })
            summary['Column Drift Details'] = drift_columns[:10]  # Show only top 10 columns
            
    return summary

@app.get("/")
async def serve_report():
    with open("/app/reporting/report.json", "r") as f:
        report_data = json.load(f)
    return report_data

def main():
    st.title("Data Drift Analysis Dashboard")
    
    try:
        with open("/app/reporting/report.json", "r") as f:
            report_data = json.load(f)
        
        # Process and display metrics
        summary = process_report_metrics(report_data)
        
        # Display dataset statistics
        st.header("Dataset Overview")
        st.table(pd.DataFrame(summary['Dataset Statistics']))
        
        # Display column drift details
        st.header("Column Drift Analysis (Top 10 Columns)")
        st.dataframe(pd.DataFrame(summary['Column Drift Details']))
        
        # Add timestamp
        st.sidebar.info(f"Report generated on: {report_data.get('timestamp', 'N/A')}")
        
    except Exception as e:
        st.error(f"Error loading report: {str(e)}")

if __name__ == "__main__":
    main()