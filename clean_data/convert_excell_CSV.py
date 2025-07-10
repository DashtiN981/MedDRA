import pandas as pd

# Load Excel file with delimiter handling
df = pd.read_excel('./clean_data/KI_Projekt_Mosaic_AE_Codierung_2024_07_03.xlsx')

# Save as CSV
df.to_csv('./clean_data/KI_Projekt_Mosaic_AE_Codierung_2024_07_03.csv', sep=';', index=False)