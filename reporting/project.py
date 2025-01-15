import sys
print(sys.executable)

import pandas as pd
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab, ClassificationPerformanceTab

# Load reference and production data
ref_data = pd.read_csv('/app/data/ref_data.csv')
prod_data = pd.read_csv('/app/data/prod_data.csv')

# Print the first few rows of the datasets to inspect the data
print("Reference Data:")
print(ref_data.head())
print("Production Data:")
print(prod_data.head())

# Check data types of the columns
print("Reference Data Types:")
print(ref_data.dtypes)
print("Production Data Types:")
print(prod_data.dtypes)

# Clean production data by converting non-numerical columns to numerical or dropping them
prod_data = prod_data.apply(pd.to_numeric, errors='coerce')
prod_data = prod_data.dropna(axis=1, how='any')

# Align columns of production data with reference data
prod_data = prod_data.reindex(columns=ref_data.columns)

# Print cleaned production data types
print("Cleaned Production Data Types:")
print(prod_data.dtypes)

# Create and generate the dashboard
dashboard = Dashboard(tabs=[DataDriftTab(), ClassificationPerformanceTab()])
dashboard.calculate(ref_data, prod_data)
dashboard.save('/app/reporting/report.html')