import pandas as pd

# Path to the Excel file you want to read
file_path = 'clean_data/MedDRA_PT_SOC_25_0.xlsx'

# Column to check for duplicates (change this to your target column name)
column_name = 'PT_Code'

# Read the Excel file into a pandas DataFrame
df = pd.read_excel(file_path)

# Filter rows where the selected column has duplicate values
# keep=False ensures all occurrences of duplicates are marked (not just the first or last)
duplicates = df[df.duplicated(subset=[column_name], keep=False)]

# Save the filtered rows with duplicates to a new Excel file
duplicates.to_excel('clean_data/duplicates_pt_soc.xlsx', index=False)

# Print summary
print(f"{len(duplicates)} rows with duplicated values in column '{column_name}' saved to 'duplicates_pt_soc.xlsx'.")
