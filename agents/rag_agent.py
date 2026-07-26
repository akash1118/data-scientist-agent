# agents/rag_agent.py
# ----------------------
# WHY: This agent demonstrates RAG (Retrieval-Augmented Generation) - the
#      technique of finding relevant information FIRST, then asking the LLM
#      to answer using ONLY that information (instead of guessing/hallucinating).
# WHAT: The third node in our LangGraph workflow. Converts the DataFrame into
#       text, stores it in FAISS, retrieves the most relevant chunks for the
#       user's question, and asks Gemini to answer using those chunks.
# HOW: Uses tools/rag_tools.py to build text chunks, rag/vector_store.py to
#      embed + search them, and utils/llm.py to generate the final answer.

from graph.state import AgentState
from tools.rag_tools import dataframe_to_documents, chunk_documents, build_dataset_summary_document
from rag.vector_store import build_vector_store, retrieve_relevant_chunks
from utils.llm import get_llm


RAG_ANSWER_PROMPT_TEMPLATE = """
You are a helpful data analyst assistant. Answer the user's question using
ONLY the context below, which was retrieved from the dataset. If the answer
cannot be found in the context, say so honestly instead of guessing.

Context (retrieved from the dataset):
{context}

Question: {question}

Give a short, clear, direct answer.
"""


def run_rag_agent(state: AgentState) -> AgentState:
    """
    WHY: This is the LangGraph node that answers natural-language questions
         about the uploaded dataset using Retrieval-Augmented Generation.
    WHAT: Reads uploaded_dataframe + user_question from state, retrieves the
          most relevant rows, and writes retrieved_documents + rag_answer
          back into state.
    HOW: Follows the classic RAG pipeline step by step:
         1. DataFrame -> text documents
         2. Chunk the documents
         3. Embed + store them in FAISS
         4. Retrieve the top-k most relevant chunks for the question
         5. Ask the LLM to answer using only those chunks
    """
    df = state["uploaded_dataframe"]
    question = state.get("user_question")

    if not question:
        # No question was asked, so there's nothing for the RAG agent to do.
        state["rag_answer"] = None
        state["retrieved_documents"] = []
        return state

    # Step 1: Turn each row into a plain-text sentence.
    row_documents = dataframe_to_documents(df)

    # Step 2: Group rows into slightly bigger chunks, and add one summary
    # document so "big picture" questions can be answered too.
    text_chunks = chunk_documents(row_documents, chunk_size=5)
    text_chunks.append(build_dataset_summary_document(df))

    # Step 3: Embed all chunks and store them in a FAISS vector index.
    vector_store = build_vector_store(text_chunks)

    # Step 4: Retrieve the chunks most relevant to the user's question.
    relevant_chunks = retrieve_relevant_chunks(vector_store, question, top_k=4)
    state["retrieved_documents"] = relevant_chunks

    # Step 5: Ask the LLM to answer using ONLY the retrieved context.
    context_text = "\n".join(relevant_chunks)
    prompt = RAG_ANSWER_PROMPT_TEMPLATE.format(context=context_text, question=question)

    llm = get_llm()
    response = llm.invoke(prompt)
    state["rag_answer"] = response.content

    # Log this step for the execution trace.
    messages = state.get("messages", [])
    messages.append({"role": "agent", "content": f"RAG Agent: answered question '{question}'."})
    state["messages"] = messages

    return state
