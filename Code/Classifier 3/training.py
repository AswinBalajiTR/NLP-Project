import pandas as pd
import joblib
import torch
from datasets import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import TrainingArguments, Trainer

# ------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------
df = pd.read_csv("/Users/aswinbalajitr/Desktop/NLP-Project copy/Data/train.csv", encoding="latin1")

# Merge fields
df["subject"] = df["subject"].fillna("")
df["email_body"] = df["email_body"].fillna("")
df["text"] = df["subject"] + " " + df["email_body"]

# Encode labels
df["label"] = df["label"].map({"job": 1, "non_job": 0})

# ------------------------------------------------
# 2. BALANCE DATASET (DOWNSAMPLE MAJORITY CLASS)
# ------------------------------------------------
job_df = df[df["label"] == 1]
nonjob_df = df[df["label"] == 0]

min_count = min(len(job_df), len(nonjob_df))

job_df = job_df.sample(min_count, random_state=42)
nonjob_df = nonjob_df.sample(min_count, random_state=42)

df_balanced = pd.concat([job_df, nonjob_df]).sample(frac=1, random_state=42)
print(f"[INFO] Balanced dataset size: {len(df_balanced)}")

# ------------------------------------------------
# 3. CONVERT TO HUGGINGFACE DATASET
# ------------------------------------------------
dataset = Dataset.from_pandas(df_balanced[["text", "label"]])

# ------------------------------------------------
# 4. TOKENIZER & MODEL
# ------------------------------------------------
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

dataset = dataset.map(tokenize, batched=True)

# ------------------------------------------------
# 5. TRAINING CONFIG
# ------------------------------------------------
args = TrainingArguments(
    output_dir="bert_email_classifier",
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=50,
    save_strategy="no",   # No checkpoints
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
)

# ------------------------------------------------
# 6. TRAIN ON 100% OF DATASET
# ------------------------------------------------
trainer.train()

# ------------------------------------------------
# 7. SAVE MODEL + TOKENIZER (REAL MODEL FILES)
# ------------------------------------------------
trainer.save_model("bert_email_classifier")
tokenizer.save_pretrained("bert_email_classifier")

print("[INFO] Saved HuggingFace model to bert_email_classifier/")

# ------------------------------------------------
# 8. SAVE PICKLE WRAPPER FOR EASY LOADING
# ------------------------------------------------
class EmailClassifierWrapper:
    def __init__(self, model_path="bert_email_classifier"):
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.model.eval()

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        prob = torch.softmax(logits, dim=1)[0][1].item()
        label = int(torch.argmax(logits))
        return label, prob

# Save wrapper object as pickle
wrapper = EmailClassifierWrapper()
joblib.dump(wrapper, "bert_email_classifier.pkl")

print("[INFO] Pickle wrapper saved as bert_email_classifier.pkl")
