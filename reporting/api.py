from fastapi import FastAPI
import streamlit as st
import json

# Create FastAPI app instance
app = FastAPI()

# Define a route to serve the report
@app.get("/")
async def serve_report():
    with open("/app/reporting/report.json", "r") as f:
        report_data = json.load(f)
    return report_data

# Streamlit UI code
def main():
    st.title("Evidently Report")
    
    # Load the report data
    with open("/app/reporting/report.json", "r") as f:
        report_data = json.load(f)
    
    # Display the report data
    st.write(report_data)

if __name__ == "__main__":
    main()