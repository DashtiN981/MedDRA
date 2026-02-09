import pandas as pd

# Path to your input Excel file
file_path = './data/PT_SOC_25_0.xlsx'

# Column to check for duplicates (change as needed)
column_name = 'PT_Code'

# Read the Excel file
df = pd.read_excel(file_path)

# Drop duplicate rows based on the selected column, keeping the first occurrence
df_unique = df.drop_duplicates(subset=[column_name], keep='first')

# Save the cleaned data to a new Excel file
df_unique.to_excel('./data/PT_SOC_without_duplicates.xlsx', index=False)

# Print confirmation
print(f"{len(df_unique)} unique rows saved to 'PT_SOC_without_duplicates.xlsx'.")
