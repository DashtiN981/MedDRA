from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class RAGConfig:
    # LLM client settings
    api_key: str
    base_url: str
    llm_api_name: str
    llm_temp: float
    llm_token: int

    # Retrieval / evaluation settings
    top_k: int
    max_rows: int | None
    emb_dim: int
    run_seed: int

    # Dataset / dictionary names
    dataset_name: str
    dataset_emb_name: str
    llt_dictionary_name: str
    llt_dictionary_emb_name: str
    pt_dictionary_name: str
    output_file_name: str

    # Paths
    ae_csv_file: str
    ae_emb_file: str
    llt_csv_file: str
    llt_emb_file: str
    pt_csv_file: str


def load_config() -> RAGConfig:
    # Parameters
    top_k = 100
    max_rows = None  # e.g., set an int to limit rows; None = all
    emb_dim = 384
    run_seed = 44

    # Datasets
    dataset_name = "KI_Projekt_Mosaic_AE_Codierung_2024_07_03"
    dataset_emb_name = "ae_embeddings_Mosaic"

    # Dictionaries and output
    llt_dictionary_name = "LLT2_Code_English_25_0"   # includes LLT_Code, LLT_Term, PT_Code
    llt_dictionary_emb_name = "llt2_embeddings"
    pt_dictionary_name = "PT2_SOC_25_0"              # supports PT_Code,SOC_Code; PT_Term,SOC_Term optional; Ist_Primary_SOC & Primary_SOC_Code optional
    output_file_name = "Mosaic_output"

    # Paths
    ae_csv_file = f"/home/naghmedashti/MedDRA-LLM/data/{dataset_name}.csv"
    ae_emb_file = f"/home/naghmedashti/MedDRA-LLM/embedding/{dataset_emb_name}.json"
    llt_csv_file = f"/home/naghmedashti/MedDRA-LLM/data/{llt_dictionary_name}.csv"
    llt_emb_file = f"/home/naghmedashti/MedDRA-LLM/embedding/{llt_dictionary_emb_name}.json"
    pt_csv_file = f"/home/naghmedashti/MedDRA-LLM/data/{pt_dictionary_name}.csv"

    # LLM client settings (API key comes from environment)
    api_key = os.getenv("MEDDRA_LLM_API_KEY", "")
    base_url = "http://pluto/v1/"
    llm_api_name = "nvidia-llama-3.3-70b-instruct-fp8"  # or "Llama-3.3-70B-Instruct" llama-3.3-70b-instruct-awq  or GPT-OSS-120B
    llm_temp = 0.0
    llm_token = 250

    return RAGConfig(
        api_key=api_key,
        base_url=base_url,
        llm_api_name=llm_api_name,
        llm_temp=llm_temp,
        llm_token=llm_token,
        top_k=top_k,
        max_rows=max_rows,
        emb_dim=emb_dim,
        run_seed=run_seed,
        dataset_name=dataset_name,
        dataset_emb_name=dataset_emb_name,
        llt_dictionary_name=llt_dictionary_name,
        llt_dictionary_emb_name=llt_dictionary_emb_name,
        pt_dictionary_name=pt_dictionary_name,
        output_file_name=output_file_name,
        ae_csv_file=ae_csv_file,
        ae_emb_file=ae_emb_file,
        llt_csv_file=llt_csv_file,
        llt_emb_file=llt_emb_file,
        pt_csv_file=pt_csv_file,
    )
