# rag/vector_store.py
# ---------------------
# WHY: To answer natural-language questions about the dataset, we need a way
#      to quickly find the text chunks that are most relevant to a question.
#      FAISS (Facebook AI Similarity Search) is a fast library for exactly
#      this - it stores "embeddings" (number vectors) and finds the closest
#      matches to a query.
# WHAT: Two functions - one to BUILD a FAISS index from text chunks, and one
#       to SEARCH that index for the most relevant chunks to a question.
# HOW: Uses LangChain's FAISS wrapper, which handles the embedding + storage
#      + search steps for us so we don't have to write low-level vector math.

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from utils.llm import get_embedding_model


def build_vector_store(text_chunks: list) -> FAISS:
    """
    WHY: This is Step 2 and 3 of RAG: "Create embeddings" + "Store in FAISS".
    WHAT: Takes a list of plain text strings and returns a searchable FAISS index.
    HOW:
        1. Wrap each text string in a LangChain Document object.
        2. Get our Gemini embedding model (turns text into number vectors).
        3. Ask FAISS to embed every document and build a searchable index.
    """
    # Step 1: LangChain's FAISS wrapper expects a list of Document objects,
    # not plain strings, so we wrap each chunk.
    documents = [Document(page_content=chunk) for chunk in text_chunks]

    # Step 2: Get the embedding model that will convert text -> vectors.
    embedding_model = get_embedding_model()

    # Step 3: FAISS.from_documents() does the heavy lifting:
    # it calls the embedding model on every document and builds the index.
    vector_store = FAISS.from_documents(documents, embedding_model)

    return vector_store


def retrieve_relevant_chunks(vector_store: FAISS, question: str, top_k: int = 4) -> list:
    """
    WHY: This is Step 4 of RAG: "Retrieve relevant chunks".
    WHAT: Given a user's question, finds the `top_k` most similar text chunks
          from the FAISS index.
    HOW: FAISS embeds the question the same way it embedded the documents,
         then compares vector distances to find the closest matches.
    """
    matching_documents = vector_store.similarity_search(question, k=top_k)

    # Extract just the text content from each matching Document.
    relevant_chunks = [doc.page_content for doc in matching_documents]
    return relevant_chunks
