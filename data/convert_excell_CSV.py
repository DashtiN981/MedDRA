import pandas as pd

# Load Excel file with delimiter handling
df = pd.read_excel('./data/KI_Projekt_Dauno_AE_Codierung_2022_10_20.xlsx')

# Save as CSV
df.to_csv('./data/KI_Projekt_Dauno_AE_Codierung_2022_10_20.csv', sep=';', index=False)