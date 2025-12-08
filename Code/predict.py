import os
import sys
import joblib
import pandas as pd


class EmailJobClassifier:

    def __init__(self, excel_filename: str, model_filename: str, output_filename: str):

        code_dir = os.getcwd()                           # e.g., /.../Code
        project_root = os.path.dirname(code_dir)         # /... (project root)

        self.data_dir = os.path.join(project_root, "Data")
        os.makedirs(self.data_dir, exist_ok=True)

        self.model_dir = code_dir                        # model in Code/

        # Input Gmail export (fresh emails)
        self.excel_path = os.path.join(self.data_dir, excel_filename)    # Data/
        # Output classified file (persistent, incremental)
        self.output_path = os.path.join(self.data_dir, output_filename)  # Data/
        self.model_path = os.path.join(self.model_dir, model_filename)   # Code/

    # ---------------------------------------------------------
    @staticmethod
    def load_excel_safely(path: str):
        try:
            print(f"[INFO] Loading Excel from: {path}")
            return pd.read_excel(path)
        except Exception as e:
            print(f"[ERROR] Unable to read Excel file: {e}")
            sys.exit(1)

    @staticmethod
    def load_model_safely(path: str):
        try:
            print(f"[INFO] Loading model from: {path}")
            return joblib.load(path)
        except Exception as e:
            print(f"[ERROR] Unable to load model: {e}")
            sys.exit(1)

    # ---------------------------------------------------------
    def classify(self):
        # Check files
        if not os.path.exists(self.excel_path):
            print(f"[ERROR] Excel NOT FOUND: {self.excel_path}")
            sys.exit(1)

        if not os.path.exists(self.model_path):
            print(f"[ERROR] Model NOT FOUND: {self.model_path}")
            sys.exit(1)

        # -------------------------------------------------
        # 1) Load latest Gmail export (source of truth)
        # -------------------------------------------------
        df_src = self.load_excel_safely(self.excel_path)

        # Required columns in source
        required_cols = {"subject", "body"}
        if not required_cols.issubset(df_src.columns):
            print(f"[ERROR] Excel missing required columns {required_cols}")
            print("Found:", df_src.columns.tolist())
            sys.exit(1)

        # Ensure 'id' exists (for matching old classifications)
        if "id" not in df_src.columns:
            print("[WARN] 'id' column not found in source Excel.")
            print("Incremental classification works best if 'id' is present.")
            # We can still proceed, but then every run will reclassify everything.
            use_id = False
        else:
            use_id = True
            df_src["id"] = df_src["id"].astype(str)

        # -------------------------------------------------
        # 2) If classified file already exists, load it
        #    and reuse old classifications where possible
        # -------------------------------------------------
        if os.path.exists(self.output_path):
            df_old = self.load_excel_safely(self.output_path)

            # Make sure id is string for matching
            if use_id and "id" in df_old.columns:
                df_old["id"] = df_old["id"].astype(str)

            print(f"[INFO] Loaded existing classified file: {self.output_path}")
            print(f"[INFO] Old classified rows: {len(df_old)}")
        else:
            df_old = None
            print(f"[INFO] No existing classified file found. Will create a new one.")

        # -------------------------------------------------
        # 3) Prepare base df with all current emails
        # -------------------------------------------------
        # Start from source df
        df = df_src.copy()

        # Initialize columns job_label/prob_job if not present
        if "job_label" not in df.columns:
            df["job_label"] = pd.NA
        if "prob_job" not in df.columns:
            df["prob_job"] = pd.NA

        # If we have previous classifications, merge them in
        if df_old is not None and use_id and "id" in df_old.columns:
            # Keep only columns we care about from old file
            cols_to_pull = ["id", "job_label", "prob_job"]
            cols_to_pull = [c for c in cols_to_pull if c in df_old.columns]

            df_old_small = df_old[cols_to_pull].drop_duplicates(subset=["id"])

            # Merge old classifications into current df by id
            df = df.merge(
                df_old_small,
                on="id",
                how="left",
                suffixes=("", "_old"),
            )

            # If job_label_old/prob_job_old exist, fill missing new ones
            if "job_label_old" in df.columns:
                df["job_label"] = df["job_label"].fillna(df["job_label_old"])
                df.drop(columns=["job_label_old"], inplace=True)

            if "prob_job_old" in df.columns:
                df["prob_job"] = df["prob_job"].fillna(df["prob_job_old"])
                df.drop(columns=["prob_job_old"], inplace=True)

        # -------------------------------------------------
        # 4) Identify rows that still need classification
        # -------------------------------------------------
        # Prepare text
        df["subject"] = df["subject"].fillna("")
        df["body"] = df["body"].fillna("")
        df["text"] = df["subject"] + " " + df["body"]

        # Rows where job_label is still missing → new or never classified
        mask_new = df["job_label"].isna()

        num_to_classify = mask_new.sum()
        print(f"[INFO] Total emails in source: {len(df)}")
        print(f"[INFO] Rows needing classification: {num_to_classify}")

        if num_to_classify == 0:
            print("[INFO] Nothing new to classify. Exiting.")
            # Still save df (in case structure changed)
            df.to_excel(self.output_path, index=False)
            print(f"[INFO] Updated file saved to: {self.output_path}")
            return df

        # -------------------------------------------------
        # 5) Load model and classify ONLY new rows
        # -------------------------------------------------
        model = self.load_model_safely(self.model_path)

        print("[INFO] Running predictions on NEW rows only...")
        texts_new = df.loc[mask_new, "text"]

        preds = model.predict(texts_new)
        probs = model.predict_proba(texts_new)[:, 1]

        # Assign back
        df.loc[mask_new, "job_label"] = preds
        df.loc[mask_new, "prob_job"] = probs

        # -------------------------------------------------
        # 6) Save back to SAME output Excel (incremental)
        # -------------------------------------------------
        df.to_excel(self.output_path, index=False)

        print(f"[INFO] Incremental classification complete.")
        print(f"[INFO] Saved updated classifications to: {self.output_path}")
        print(df.loc[mask_new, ["subject", "job_label", "prob_job"]].head())

        return df


def main():
    classifier = EmailJobClassifier(
        excel_filename="gmail_subject_body_date.xlsx",   # source from Gmail export
        model_filename="job_classifier_baseline.pkl",    # your trained model
        output_filename="mail_classified.xlsx",          # persistent classified file
    )
    classifier.classify()


if __name__ == "__main__":
    main()
