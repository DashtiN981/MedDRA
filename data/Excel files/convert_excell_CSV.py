import pandas as pd

# Load Excel file with delimiter handling
df = pd.read_excel('./data/Excel files/PT2_SOC_25_0.xlsx')

# Save as CSV
df.to_csv('./data/PT2_SOC_25_0.csv', sep=';', index=False)