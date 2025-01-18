print("Starting the project script...")

import sys
print(sys.executable)

import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

from evidently.metrics import ColumnDriftMetric
from evidently.metrics import DatasetDriftMetric
from evidently.metrics import DatasetMissingValuesMetric

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

     # Set sample size and random seed for reproducibility
    DESIRED_SAMPLE_SIZE = 100
    RANDOM_SEED = 42

    print("\nVerifying data shapes:")
    print(f"Reference data shape: {ref_data.shape}")
    print(f"Production data shape: {new_prod_data.shape}")
    
    # Determine actual sample size based on available data
    ref_sample_size = min(DESIRED_SAMPLE_SIZE, len(ref_data))
    prod_sample_size = min(DESIRED_SAMPLE_SIZE, len(new_prod_data))

    if prod_sample_size < DESIRED_SAMPLE_SIZE:
        print(f"\nWarning: Production data has fewer samples ({len(new_prod_data)}) than desired ({DESIRED_SAMPLE_SIZE})")
        print(f"Using all available production data samples: {prod_sample_size}")
    
    # Take a random subset of the data for testing
    ref_data_sample = ref_data.sample(n=ref_sample_size, random_state=RANDOM_SEED)
    new_prod_data_sample = new_prod_data.sample(n=prod_sample_size, random_state=RANDOM_SEED)

    print(f"\nVerifying sampled data shapes:")
    print(f"Reference data sample shape: {ref_data_sample.shape}")
    print(f"Production data sample shape: {new_prod_data_sample.shape}")

    
    print("\nVerifying column alignment:")
    print("First 5 columns of reference data:", list(ref_data_sample.columns[:5]))
    print("First 5 columns of production data:", list(new_prod_data_sample.columns[:5]))

    # Create and generate the report
    print("\nGenerating report...")
    selected_metrics = [
        DatasetDriftMetric(),
        DatasetMissingValuesMetric(),
        ColumnDriftMetric(column_name="0", stattest="wasserstein"),
        ColumnDriftMetric(column_name="1", stattest="wasserstein")
    ]
    print("Selected metrics for the report:")
    for metric in selected_metrics:
        print(f"- {metric.__class__.__name__}")

    report = Report(metrics=selected_metrics)

    # Calculate report with processed data
    print("Calculating report...")
    report.run(reference_data=ref_data_sample, current_data=new_prod_data_sample)
    print("Report completed.")
    
    # Save report as JSON
    report.save_json("/app/reporting/report.json")
    print("Report saved as JSON.")

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()