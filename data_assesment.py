import pandas as pd

# Load the uploaded datasets
ae_data_path = "/home/naghmedashti/MedDRA-LLM/data/KI_Projekt_Mosaic_AE_Codierung_2024_07_03.csv"
llt_data_path = "/home/naghmedashti/MedDRA-LLM/data/MedDRA1_LLT_Code_25_0.csv"

# Load the data
ae_df = pd.read_csv(ae_data_path, sep=";", encoding="latin1")
llt_df = pd.read_csv(llt_data_path, sep=";", encoding="latin1")

# Display the basic information and a few sample rows from each
ae_df_info = ae_df.info()
llt_df_info = llt_df.info()
ae_df_head = ae_df.head()
llt_df_head = llt_df.head()

ae_df_info, llt_df_info, ae_df_head, llt_df_head
print(ae_df.head())
print(llt_df.head())
