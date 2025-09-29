import streamlit as st
import json
import os

# --- File paths ---
INPUT_FILE = r"./RAG_Models/rag_prompting_reasoning_v3_final.json"
OUTPUT_FILE = r"./RAG_Models/output_checked_naghmeV3.json"



# --- Load input data ---
if not os.path.exists(INPUT_FILE):
    st.error(f"Input file '{INPUT_FILE}' not found.")
    st.stop()

with open(INPUT_FILE, "r") as f:
    data = json.load(f)

# --- Load or initialize results ---
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r") as f:
        results = json.load(f)
else:
    results = []

# --- Helper: check if an entry was already processed ---
def entry_already_processed(entry, results):
    return any(
        entry["AE_text"] == r["AE_text"]
        and entry["true_term"] == r["true_term"]
        and entry["predicted"] == r["predicted"]
        for r in results
    )

# --- Step 1: Handle and save all exact matches automatically ---
new_results = results.copy()
for entry in data:
    if entry_already_processed(entry, results):
        continue
    if entry.get("exact_match", False) is True:
        entry["manual check"] = True
        new_results.append(entry)

# Save if we added any new auto-matches
if len(new_results) > len(results):
    with open(OUTPUT_FILE, "w") as f:
        json.dump(new_results, f, indent=2)
    results = new_results

# --- Step 2: Create list of remaining manual entries ---
remaining = [
    entry for entry in data
    if not entry_already_processed(entry, results)
    and not entry.get("exact_match", False)
]

# --- Step 3: If all entries handled ---
if len(remaining) == 0:
    st.success("All entries have been reviewed or auto-checked.")
    st.download_button(" Download results", json.dumps(results, indent=2), file_name=OUTPUT_FILE)
    st.stop()

# --- Step 4: Show current manual entry ---
current_entry = remaining[0]
progress = len(results) + 1
total = len(data)

st.title("Manual Review for AE Terms")
st.markdown(f"### Reviewing entry {progress} of {total} (manual checks only)")

# Display entry
for key, value in current_entry.items():
    st.write(f"**{key}**: {value}")

# Manual check input
manual_check = st.radio("Manual Check:", options=["true", "false"], index=0, horizontal=True)

# Submit button
if st.button(" Save & Next"):
    current_entry["manual check"] = manual_check.lower() == "true"
    results.append(current_entry)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    st.rerun()

# to run:
# # streamlit run C:\Users\IW5\Documents\OrgaDreden\LLM_Projects\Meddra\helpers\resultscheck.py