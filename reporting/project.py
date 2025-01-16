print("Starting the project script...")

import sys
print(sys.executable)

import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

try:
    # Load reference and production data
    print("Loading data...")
    ref_data = pd.read_csv('/app/data/ref_data.csv')
    prod_data = pd.read_csv('/app/data/prod_data.csv')

    print("Reference Data:")
    print(ref_data.head())
    print("\nProduction Data:")
    print(prod_data.head())

    # Step 1: Create a mapping for production data columns
    print("\nProcessing production data...")
    prod_cols = {}
    for col in prod_data.columns:
        try:
            # Try to convert column name to float (for handling cases like '162.1')
            col_num = float(col)
            # Round to nearest integer for mapping
            col_idx = int(round(col_num))
            if col_idx < 1024:  # Only include if it's within our range
                prod_cols[str(col_idx)] = col
        except ValueError:
            continue

    # Step 2: Create new production DataFrame with correct structure
    new_prod_data = pd.DataFrame()
    
    # Copy numeric data using the mapping
    for i in range(1024):
        col_str = str(i)
        if col_str in prod_cols:
            # If we have matching data, copy it
            new_prod_data[col_str] = pd.to_numeric(prod_data[prod_cols[col_str]], errors='coerce')
        else:
            # If no matching data, use mean of surrounding columns or 0
            new_prod_data[col_str] = 0

    # Fill NaN values with 0
    new_prod_data = new_prod_data.fillna(0)

    # Add label column
    new_prod_data['label'] = 'cat'  # Set all labels to 'cat' to match reference data

    # Ensure reference data columns are correctly named and typed
    ref_cols = {str(i): str(i) for i in range(1024)}
    ref_cols['label'] = 'label'
    ref_data = ref_data.rename(columns=ref_cols)
    
    # Convert numeric columns to float64
    numeric_cols = [str(i) for i in range(1024)]
    ref_data[numeric_cols] = ref_data[numeric_cols].astype(float)
    new_prod_data[numeric_cols] = new_prod_data[numeric_cols].astype(float)

    print("\nVerifying data shapes:")
    print(f"Reference data shape: {ref_data.shape}")
    print(f"Production data shape: {new_prod_data.shape}")
    
    # Take a small subset of the data for testing
    ref_data_sample = ref_data.head(10)
    new_prod_data_sample = new_prod_data.head(10)

    print("\nVerifying data shapes:")
    print(f"Reference data shape: {ref_data_sample.shape}")
    print(f"Production data shape: {new_prod_data_sample.shape}")
    
    print("\nVerifying column alignment:")
    print("First 5 columns of reference data:", list(ref_data_sample.columns[:5]))
    print("First 5 columns of production data:", list(new_prod_data_sample.columns[:5]))

    # Create and generate the report
    print("\nGenerating report...")
    report = Report(metrics=[DataDriftPreset()])

    # Convert report to dictionary and print
    report_dict = report.as_dict()
    print("\nReport Results:")
    print(report_dict)
    
    # Calculate report with processed data
    print("Calculating report...")
    report.run(reference_data=ref_data_sample, current_data=new_prod_data_sample)
    print("Report completed.")
    report.save_html("/app/reporting/report.html")
    print("Report saved.")

    # Append custom HTML content
    with open("/app/reporting/report.html", "a") as report_file:
        report_file.write("<hr>")
        report_file.write('<a href="/">Back to main page</a>')
        report_file.write("<p>Custom content added to the report.</p>")
    print("Custom content added to the report.")
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()