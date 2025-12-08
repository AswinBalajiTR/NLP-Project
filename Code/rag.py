import os
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from typing import TypedDict, List, Any

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

from langgraph.graph import StateGraph, END


# ==========================================================
# PROJECT PATH SETUP (Code / Data / chroma_store)
# ==========================================================

# Assume we run this from inside project_root/Code
code_dir = os.getcwd()                          # e.g. /.../NLP-Project/Code
project_root = os.path.dirname(code_dir)        # /.../NLP-Project

DATA_DIR = os.path.join(project_root, "Data")
os.makedirs(DATA_DIR, exist_ok=True)

CHROMA_DIR = os.path.join(project_root, "chroma_store")
os.makedirs(CHROMA_DIR, exist_ok=True)

EXCEL_FILENAME = "mail_classified_llm_parsed.xlsx"
EXCEL_PATH = os.path.join(DATA_DIR, EXCEL_FILENAME)

# ==========================================================
# LOAD EMAILS FROM EXCEL
# ==========================================================

print(f">>> Loading parsed job emails from: {EXCEL_PATH}")
df = pd.read_excel(EXCEL_PATH)

required_cols = [
    "mailcontent",
    "company_name",
    "position_applied",
    "application_date",
    "status",
    "mail_link",
]

for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# Use a STABLE ID per email, based on mail_link
# (so we can detect what is already in Chroma)
df["mail_link"] = df["mail_link"].fillna("").astype(str)
df["doc_id"] = df["mail_link"]

# Fallback if mail_link is empty for some reason
mask_empty = df["doc_id"] == ""
df.loc[mask_empty, "doc_id"] = [
    f"row_{i}" for i in df.index[mask_empty]
]

# ==========================================================
# INIT CHROMA CLIENT + COLLECTION
# ==========================================================

print(f">>> Initializing Chroma vector database at: {CHROMA_DIR}")
client = chromadb.PersistentClient(path=CHROMA_DIR)

collection = client.get_or_create_collection(
    name="job_email_collection",
    metadata={"hnsw:space": "cosine"},
)

# ==========================================================
# INCREMENTAL ADD: ONLY EMBED NEW DOCS
# ==========================================================

# Get existing IDs already stored in Chroma
existing = collection.get(limit=1000000)   # large limit to fetch all
existing_ids = set(existing.get("ids", []))

print(f">>> Existing vectors in Chroma: {len(existing_ids)}")

# Find new rows whose doc_id is not in Chroma yet
df_new = df[~df["doc_id"].isin(existing_ids)].copy()

if df_new.empty:
    print(">>> No new rows to embed. Chroma store is up to date.\n")
else:
    print(f">>> New rows to embed: {len(df_new)}")

    # Prepare docs and metadata for ONLY new rows
    ids = df_new["doc_id"].tolist()
    documents = df_new["mailcontent"].astype(str).tolist()
    metadatas = df_new[
        ["company_name", "position_applied", "application_date", "status", "mail_link"]
    ].astype(str).to_dict(orient="records")

    print("\n>>> Loading embedding model (BGE-Large)...")
    embedder = SentenceTransformer("BAAI/bge-large-en-v1.5")

    print(">>> Embedding email texts (new only)...")
    embeddings = embedder.encode(documents, convert_to_numpy=True)

    print(">>> Storing new embeddings in Chroma...")
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    print("✔ Embedding + storage for new rows complete.\n")

# ==========================================================
# LANGCHAIN RETRIEVER + LLM (Ollama Llama 3.1)
# ==========================================================

# Re-wrap embeddings for LangChain
lc_embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

vectorstore = Chroma(
    client=client,
    collection_name="job_email_collection",
    embedding_function=lc_embedder,
    persist_directory=CHROMA_DIR,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

print(">>> Using Ollama Llama 3.1 model...")

# This calls your local Ollama server (http://localhost:11434)
llm = Ollama(
    model="llama3.1",   # change if you use another model, e.g. "qwen2.5:14b-instruct"
    temperature=0.2,
)

prompt_template = PromptTemplate(
    input_variables=["question", "context"],
    template="""
You are an assistant specializing in understanding job application emails.

Use ONLY the context given. Do NOT guess or hallucinate.

QUESTION:
{question}

CONTEXT FROM EMAILS:
{context}

Provide a concise, accurate status update:
"""
)

# ==========================================================
# LANGGRAPH STATE + NODES
# ==========================================================

class RAGState(TypedDict):
    question: str
    retrieved_docs: List[Any]
    answer: str


def retrieve_node(state: RAGState):
    """Retrieve top matching email records."""
    docs = retriever.invoke(state["question"])
    state["retrieved_docs"] = docs
    return state


def llm_node(state: RAGState):
    """Generate a factual answer using retrieved emails."""
    if not state["retrieved_docs"]:
        state["answer"] = "I couldn't find any matching job application emails for this question."
        return state

    context = "\n\n".join(
        f"EMAIL:\n{doc.page_content}\nMETADATA: {doc.metadata}"
        for doc in state["retrieved_docs"]
    )

    final_prompt = prompt_template.format(
        question=state["question"],
        context=context,
    )

    output = llm.invoke(final_prompt)

    if isinstance(output, str):
        state["answer"] = output.strip()
    else:
        try:
            state["answer"] = output.content.strip()
        except Exception:
            state["answer"] = str(output).strip()

    return state


# ==========================================================
# BUILD LANGGRAPH PIPELINE
# ==========================================================

workflow = StateGraph(RAGState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", llm_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

rag_app = workflow.compile()

# ==========================================================
# PUBLIC API FOR ASKING QUESTIONS
# ==========================================================

def ask(question: str) -> str:
    """Ask anything about your job applications."""
    initial_state: RAGState = {
        "question": question,
        "retrieved_docs": [],
        "answer": "",
    }
    final_state = rag_app.invoke(initial_state)
    return final_state["answer"]


# ==========================================================
# COMMAND LINE INTERFACE
# ==========================================================
if __name__ == "__main__":
    print("\n Job Application RAG System Ready.\n")
    print(f"(Using data from: {EXCEL_PATH})")
    print(f"(Chroma store at: {CHROMA_DIR})\n")

    while True:
        q = input("Ask about your job applications → ")
        if q.lower() in ["exit", "quit"]:
            break
        print("\n--- ANSWER ---")
        print(ask(q))
        print("--------------\n")
