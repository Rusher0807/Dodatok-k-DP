import pandas as pd
import matplotlib.pyplot as plt
import re

# Load your Excel results
df = pd.read_excel("./results/AR_CleanMitra7AChunkBatchMalicious.xlsx")

# Optional: Derive true label from filename
def extract_true_label(filename):
    return "malicious"  

#def extract_true_label(filename):
#    return "benign"  

# Extract model label and true label
def parse_model_label(text):
    match = re.search(r"Final Label: (\w+)", text)
    return match.group(1).lower() if match else "unknown"

df["true_label"] = df["Filename"].apply(extract_true_label)
df["predicted_label"] = df["Analysis Result"].apply(parse_model_label)

# Confusion matrix
confusion = pd.crosstab(df["true_label"], df["predicted_label"], rownames=["Actual"], colnames=["Predicted"])
print(confusion)

# Accuracy
correct = (df["true_label"] == df["predicted_label"]).sum()
total = len(df)
accuracy = correct / total
print(f"\n✅ Accuracy: {accuracy:.2%} ({correct}/{total})")

# Plot confusion matrix
confusion.plot(kind="bar", stacked=True)
plt.title("Prediction Distribution per True Class")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
