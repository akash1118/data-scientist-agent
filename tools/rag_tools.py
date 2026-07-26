# tools/rag_tools.py
# --------------------
# WHY: RAG (Retrieval-Augmented Generation) needs raw text to search through.
#      A pandas DataFrame is structured (rows/columns), but our vector store
#      (FAISS) works with plain text. This file converts one into the other.
# WHAT: Turns each DataFrame row into a simple sentence, e.g.
#       "Employee Name is John. Department is Sales. Salary is 45000."
# HOW: Simple string formatting - no advanced NLP needed. This keeps the
#      RAG pipeline easy to understand for beginners.

import pandas as pd


def dataframe_to_documents(df: pd.DataFrame, max_rows: int = 500) -> list:
    """
    WHY: This is the very first step of RAG - turning structured data into
         text "documents" that can later be embedded and searched.
    WHAT: Converts each row of the DataFrame into one text string.
    HOW: Loops over the rows and joins "column_name is value" pairs together.

    Example output for one row:
        "Employee Name is John Smith. Age is 28. Department is Sales.
         Salary is 45000. Experience is 5. City is Mumbai."
    """
    documents = []

    # We cap the number of rows to keep embedding fast and cheap for a classroom demo.
    limited_df = df.head(max_rows)

    for _, row in limited_df.iterrows():
        # Build one sentence per row by joining "column is value" for every column.
        parts = [f"{column} is {row[column]}" for column in df.columns]
        row_text = ". ".join(parts) + "."
        documents.append(row_text)

    return documents


def chunk_documents(documents: list, chunk_size: int = 5) -> list:
    """
    WHY: Searching one row at a time is fine, but grouping a few rows together
         ("chunking") often gives the LLM more context to answer questions like
         "what trends do you observe?" that need to compare multiple rows.
    WHAT: Groups the list of row-documents into bigger text chunks.
    HOW: Simple slicing - takes `chunk_size` rows at a time and joins them
         with newlines into one bigger chunk.
    """
    chunks = []
    for i in range(0, len(documents), chunk_size):
        chunk_group = documents[i:i + chunk_size]
        combined_chunk = "\n".join(chunk_group)
        chunks.append(combined_chunk)
    return chunks


def build_dataset_summary_document(df: pd.DataFrame) -> str:
    """
    WHY: Some questions (e.g. "what trends do you observe?") are about the
         WHOLE dataset, not a single row. Adding one "summary document" to
         our vector store helps the RAG agent answer those big-picture questions.
    WHAT: Creates one text block describing dataset-level facts.
    HOW: Uses pandas' describe() and dtypes to build a short summary string.
    """
    numeric_df = df.select_dtypes(include=["number"])
    lines = [f"This dataset has {df.shape[0]} rows and {df.shape[1]} columns."]
    lines.append(f"The columns are: {', '.join(df.columns)}.")

    for column in numeric_df.columns:
        lines.append(
            f"The average {column} is {round(numeric_df[column].mean(), 2)}, "
            f"minimum is {numeric_df[column].min()}, "
            f"maximum is {numeric_df[column].max()}."
        )

    return " ".join(lines)
