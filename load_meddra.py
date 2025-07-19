import pandas as pd

# Try loading the file with a different delimiter and encoding
llt_df = pd.read_csv("/home/naghmedashti/MedDRA-LLM/data/LLT_Code_Deutsch_25_0.csv", sep=';', encoding='latin1', engine='python')
pt_df = pd.read_csv("/home/naghmedashti/MedDRA-LLM/data/PT_SOC_25_0.csv", sep=';', encoding='latin1', engine='python')

# Display the first few rows of the dataset
print("Sample rows from the MedDRA LLT dataset:")
print(llt_df.head())

# Print the total number of LLT entries
print(f"\nTotal number of LLT entries: {len(llt_df)}")

# Display the first few rows of the dataset
print("Sample rows from the MedDRA PT dataset:")
print(pt_df.head())

# Print the total number of LLT entries
print(f"\nTotal number of PT entries: {len(pt_df)}")