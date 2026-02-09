import pandas as pd

# Load Excel file with delimiter handling
df = pd.read_excel('/home/naghmedashti/MedDRA-LLM/data/Excel files/soc_translation_de_en.xlsx')

# Save as CSV
df.to_csv('/home/naghmedashti/MedDRA-LLM/data/soc_translation_de_en.csv', sep=';', index=False)