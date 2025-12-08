import pandas as pd
import joblib
import torch
from sklearn.metrics import classification_report, confusion_matrix


# 0. DEFINE THE WRAPPER CLASS (MUST MATCH NAME)
# ------------------------------------------------
class EmailClassifierWrapper:
    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer



print("[INFO] Loading model...")
wrapper = joblib.load("bert_email_classifier.pkl")
model = wrapper.model
tokenizer = wrapper.tokenizer
model.eval()

# ------------------------------------------------
# 2. LOAD EVALUATION CSV/EXCEL WITH RAW EMAILS
# ------------------------------------------------
# MODIFY THIS LINE to point to whatever file you evaluate
eval_df = pd.read_excel("/Users/aswinbalajitr/Desktop/NLP-Project copy/Data/gmail_subject_body_date.xlsx")

# Ensure columns exist
required_cols = {"subject", "body"}
if not required_cols.issubset(eval_df.columns):
    raise ValueError(f"Input file must contain: {required_cols}")

# Merge text fields
eval_df["subject"] = eval_df["subject"].fillna("")
eval_df["body"] = eval_df["body"].fillna("")
eval_df["text"] = eval_df["subject"] + " " + eval_df["body"]

# If labels exist, map them
if "label" in eval_df.columns:
    eval_df["label"] = eval_df["label"].map({"non_job": 0, "job": 1})

print(f"[INFO] Loaded {len(eval_df)} evaluation samples.")

# ------------------------------------------------
# 3. RUN MODEL PREDICTIONS
# ------------------------------------------------
preds = []
probs = []

print("[INFO] Running predictions...")

for text in eval_df["text"]:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)

    with torch.no_grad():
        logits = model(**inputs).logits
        prob = torch.softmax(logits, dim=1)[0][1].item()
        pred = int(torch.argmax(logits))

    preds.append(pred)
    probs.append(prob)

eval_df["job_label"] = preds
eval_df["prob_job"] = probs

print("[INFO] Predictions complete.")

# ------------------------------------------------
# 4. CLASSIFICATION REPORT (only if true labels exist)
# ------------------------------------------------
if "label" in eval_df.columns:
    print("\n===== CLASSIFICATION REPORT =====\n")
    print(classification_report(eval_df["label"], eval_df["job_label"], target_names=["non_job", "job"]))

    print("\n===== CONFUSION MATRIX =====\n")
    print(confusion_matrix(eval_df["label"], eval_df["job_label"]))
else:
    print("[INFO] No ground-truth labels found. Skipping evaluation metrics.")

# ------------------------------------------------
# 5. SAVE PREDICTIONS
# ------------------------------------------------
output_path = "/Users/aswinbalajitr/Desktop/NLP-Project copy/Data/mail_classified3.xlsx"
eval_df.to_excel(output_path, index=False)
print(f"[INFO] Predictions saved to {output_path}")
