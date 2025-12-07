import pandas as pd
from zero_shot_classifier import ZeroShotJobClassifier

INPUT_FILE = "gmail_subject_body_date.xlsx"
OUTPUT_FILE = "gmail_with_job_labels.xlsx"

def load_data(file_path):
    return pd.read_excel(file_path)

def build_text_column(df):
    df["text"] = df["subject"].fillna("") + " " + df["body"].fillna("")
    return df

def apply_classification(df, model):
    print("Applying zero-shot job classification...")
    df["predicted_label"] = df["text"].apply(model.predict)
    return df

def save_output(df, file_path):
    df.to_excel(file_path, index=False)
    print(f"[✓] Saved predictions to: {file_path}")

if __name__ == "__main__":
    # Step 1: Load Gmail data from Excel
    df = load_data(INPUT_FILE)

    # Step 2: Build unified text field
    df = build_text_column(df)

    # Step 3: Initialize zero-shot classifier
    clf = ZeroShotJobClassifier()

    # Step 4: Apply predictions
    df = apply_classification(df, clf)

    # Step 5: Save results
    save_output(df, OUTPUT_FILE)