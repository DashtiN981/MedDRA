import json
import matplotlib.pyplot as plt

# Load the final checked results
file_path = "/home/naghmedashti/MedDRA-LLM/RAG_Models/manual_checks/Dauno_output_NewRAG__full__reviewer1.json"
with open(file_path, "r", encoding="latin1") as f:
    data = json.load(f)

# Total number of AE entries
total_entries = len(data)

# Compute the number of true values for each accuracy type
exact_match_count = sum(1 for d in data if d.get("exact_LLT_match") is True)
fuzzy_match_count = sum(1 for d in data if d.get("LLT_fuzzy_match") is True)
manual_checked_correct = sum(1 for d in data if d.get("manual check") is True)

# Compute accuracy percentages
exact_accuracy = 100 * exact_match_count / total_entries
fuzzy_accuracy = 100 * fuzzy_match_count / total_entries
manual_accuracy = 100 * manual_checked_correct / total_entries

# Create bar chart
labels = ["Exact Match", "Fuzzy Match", "Manual Check"]
accuracies = [exact_accuracy, fuzzy_accuracy, manual_accuracy]

plt.figure(figsize=(8, 6))
bars = plt.bar(labels, accuracies, color=["#4CAF50", "#2196F3", "#FF9800"])
plt.ylim(0, 100)
plt.ylabel("Accuracy (%)")
plt.title("Accuracy Comparison Mosaic Dataset – Model Reasoning v3")

# Annotate bars
for bar, acc in zip(bars, accuracies):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{acc:.1f}%", ha='center', va='bottom')

# Save plot
output_path = "/home/naghmedashti/MedDRA-LLM/RAG_Models/manual_checks/Dauno_output_NewRAG_accuracy_comparison.png"
plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.show()
