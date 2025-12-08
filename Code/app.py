# app.py
import os
from datetime import datetime

import pandas as pd
import streamlit as st

# ---- import your existing modules (NO CHANGES to them) ----
import gmail_read
from gmail_read import GmailLiveReader, QUERY, OUTPUT_EXCEL  # reuse your constants
import predict
import ner
import rag  # this gives us rag.ask()


# ========== helpers ==========

def _safe_read_excel(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            return pd.read_excel(path)
        except Exception as e:
            st.error(f"Error reading {path}: {e}")
            return pd.DataFrame()
    return pd.DataFrame()


def _last_modified(path: str) -> str:
    if not os.path.exists(path):
        return "No file yet"
    ts = os.path.getmtime(path)
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


# ========== actions that call YOUR code ==========

def fetch_new_emails_once():
    """
    Run ONE fetch cycle using your GmailLiveReader, instead of the infinite while loop.
    This reuses the same OUTPUT_EXCEL and QUERY that gmail_read.py uses.
    """
    st.write("Starting one-time Gmail fetch using GmailLiveReader...")

    reader = GmailLiveReader(
        credentials_path="../credentials.json",
        token_path="../token.json",
        gmail_account_index=0,
    )

    # Load existing Excel if present
    if os.path.exists(OUTPUT_EXCEL):
        master_df = pd.read_excel(OUTPUT_EXCEL)
        if "id" in master_df.columns:
            seen_ids = set(master_df["id"].astype(str).tolist())
        else:
            seen_ids = set()
        st.write(f"Loaded {len(master_df)} rows from {OUTPUT_EXCEL}")
    else:
        master_df = pd.DataFrame(
            columns=[
                "id",
                "sender_name",
                "sender_email",
                "subject",
                "body",
                "date_received",
                "gmail_link",
            ]
        )
        seen_ids = set()
        st.write(f"No existing Excel found. Will create {OUTPUT_EXCEL}")

    # Fetch only NEW emails for the same query used in gmail_read.py
    df_new = reader.fetch_new_as_dataframe(QUERY, seen_ids)

    if df_new.empty:
        st.info("No new emails found for this query.")
        return master_df

    # Append and save
    master_df = pd.concat([master_df, df_new], ignore_index=True)
    master_df.to_excel(OUTPUT_EXCEL, index=False)

    st.success(f"Appended {len(df_new)} new email(s) to {OUTPUT_EXCEL}")
    st.dataframe(df_new[["date_received", "sender_email", "subject"]])

    return master_df


def run_classifier():
    """
    Call predict.main() exactly like your CLI.
    This reads gmail_subject_body_date.xlsx and writes mail_classified.xlsx.
    """
    st.write("Running classifier (predict.py)...")
    try:
        predict.main()
        st.success("Classification completed. mail_classified.xlsx updated.")
    except Exception as e:
        st.error(f"Error running classifier: {e}")


def run_ner():
    """
    Call ner.main() exactly like your CLI.
    This reads mail_classified.xlsx and writes mail_classified_llm_parsed.xlsx.
    """
    st.write("Running NER parser (ner.py)...")
    try:
        ner.main()
        st.success("NER completed. mail_classified_llm_parsed.xlsx updated.")
    except Exception as e:
        st.error(f"Error running NER: {e}")


# ========== Streamlit pages ==========

def page_fetch_and_classify():
    st.header("1️⃣ Gmail Fetch + Job Classification")

    st.markdown("This page uses your existing **gmail_read.py** + **predict.py** logic.")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Fetch NEW emails from Gmail (one batch)"):
            fetch_new_emails_once()

    with col_btn2:
        if st.button("Run job/non-job classifier on emails"):
            run_classifier()

    st.markdown("---")
    st.subheader("Raw Gmail Export (gmail_subject_body_date.xlsx)")

    gmail_df = _safe_read_excel("gmail_subject_body_date.xlsx")
    st.caption(f"Last updated: {_last_modified('gmail_subject_body_date.xlsx')}")
    if gmail_df.empty:
        st.info("No Gmail Excel yet or file is empty.")
    else:
        st.dataframe(gmail_df.head(200))

    st.markdown("---")
    st.subheader("Classified Emails (mail_classified.xlsx)")
    class_df = _safe_read_excel("mail_classified.xlsx")
    st.caption(f"Last updated: {_last_modified('mail_classified.xlsx')}")
    if class_df.empty:
        st.info("No classified file yet. Run the classifier first.")
    else:
        # small summary: job vs non_job counts
        if "job_label" in class_df.columns:
            counts = class_df["job_label"].value_counts()
            st.write("Job vs Non-Job counts:")
            st.bar_chart(counts)
        st.dataframe(
            class_df[
                [
                    c
                    for c in [
                        "date_received",
                        "sender_email",
                        "subject",
                        "job_label",
                        "prob_job",
                    ]
                    if c in class_df.columns
                ]
            ].head(200)
        )


def page_ner_view():
    st.header("2️⃣ Job Entity Extraction (NER)")

    st.markdown("This page uses your **ner.py** to parse job emails into structured fields.")

    if st.button("Run NER on classified job emails"):
        run_ner()

    st.markdown("---")
    st.subheader("Parsed Job Records (mail_classified_llm_parsed.xlsx)")

    parsed_df = _safe_read_excel("mail_classified_llm_parsed.xlsx")
    st.caption(f"Last updated: {_last_modified('mail_classified_llm_parsed.xlsx')}")

    if parsed_df.empty:
        st.info("No parsed job records yet. Run NER first.")
    else:
        # show basic metrics
        st.write(f"Total parsed rows: {len(parsed_df)}")
        cols = [
            "company_name",
            "position_applied",
            "application_date",
            "status",
            "mail_link",
        ]
        show_cols = [c for c in cols if c in parsed_df.columns]
        st.dataframe(parsed_df[show_cols].head(200))


def page_rag():
    st.header("3️⃣ RAG Assistant – Ask About Your Applications")

    st.markdown(
        """
This uses your **rag.py**:

- Loads `mail_classified_llm_parsed.xlsx`
- Uses ChromaDB + embeddings
- Calls `rag.ask(question)` with your local Ollama Llama 3.1
        """
    )

    question = st.text_area(
        "Ask something like:",
        "What is the latest status of my ASML application?",
        height=80,
    )

    if st.button("Ask RAG"):
        if not question.strip():
            st.warning("Type a question first.")
            return

        try:
            with st.spinner("Thinking..."):
                answer = rag.ask(question)
            st.subheader("Answer")
            st.write(answer)
        except Exception as e:
            st.error(
                "RAG failed. Make sure `mail_classified_llm_parsed.xlsx` exists "
                "and Chroma store is set up. Error:\n\n" + str(e)
            )


# ========== main ==========

def main():
    st.set_page_config(page_title="Job Email Tracker (Existing Pipeline)", layout="wide")

    st.title("Job Email Tracker – Streamlit Wrapper")
    st.caption(
        "This UI only orchestrates your existing gmail_read.py, predict.py, ner.py, and rag.py. "
        "It does not change their logic."
    )

    tab1, tab2, tab3 = st.tabs(
        ["1. Fetch + Classify", "2. NER Parsed Jobs", "3. RAG Assistant"]
    )

    with tab1:
        page_fetch_and_classify()
    with tab2:
        page_ner_view()
    with tab3:
        page_rag()


if __name__ == "__main__":
    main()
