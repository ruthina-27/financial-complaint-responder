import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
from transformers import pipeline

"""
RAG Pipeline Module
------------------
Provides functions for loading the vector store, embedding queries, retrieving relevant complaint chunks,
formatting prompts, generating answers with an LLM, and running the full RAG pipeline for Q&A.
"""

# Paths (adjust as needed)
VECTOR_STORE_PATH = os.path.join('vector_store', 'faiss_index.bin')
METADATA_PATH = os.path.join('vector_store', 'metadata.pkl')
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

# Load embedding model
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Load LLM (using Hugging Face pipeline, can be replaced with LangChain integration)
generator = pipeline('text-generation', model='gpt2')  # Replace with your preferred model

def load_vector_store():
    """
    Load the FAISS vector store and associated metadata.
    Returns:
        index (faiss.Index): The FAISS index for similarity search.
        metadata (list): List of metadata dicts for each chunk.
    """
    index = faiss.read_index(VECTOR_STORE_PATH)
    with open(METADATA_PATH, 'rb') as f:
        metadata = pickle.load(f)
    return index, metadata

def embed_query(query: str):
    """
    Embed a user query using the same model as the document chunks.
    Args:
        query (str): The user's question.
    Returns:
        np.ndarray: The embedding vector for the query.
    """
    return embedding_model.encode([query])

def retrieve(query: str, k: int = 5) -> List[Tuple[str, dict]]:
    """
    Retrieve the top-k most relevant text chunks for a given query.
    Args:
        query (str): The user's question.
        k (int): Number of chunks to retrieve.
    Returns:
        List of (chunk_text, metadata) tuples.
    """
    index, metadata = load_vector_store()
    query_vec = embed_query(query)
    D, I = index.search(query_vec, k)
    results = []
    for idx in I[0]:
        chunk_text = metadata[idx]['text']
        meta = metadata[idx]
        results.append((chunk_text, meta))
    return results

def format_prompt(context_chunks: List[str], question: str) -> str:
    """
    Format the prompt for the LLM, including context and the user's question.
    Args:
        context_chunks (List[str]): Retrieved complaint excerpts.
        question (str): The user's question.
    Returns:
        str: The formatted prompt.
    """
    context = '\n---\n'.join(context_chunks)
    prompt = (
        "You are a financial analyst assistant for CrediTrust. "
        "Your task is to answer questions about customer complaints. "
        "Use the following retrieved complaint excerpts to formulate your answer. "
        "If the context doesn't contain the answer, state that you don't have enough information.\n"
        f"Context: {context}\nQuestion: {question}\nAnswer:"
    )
    return prompt

def generate_answer(prompt: str) -> str:
    """
    Generate an answer from the LLM given a prompt.
    Args:
        prompt (str): The prompt including context and question.
    Returns:
        str: The generated answer.
    """
    response = generator(prompt, max_length=256, do_sample=True, temperature=0.7)
    return response[0]['generated_text'].split('Answer:')[-1].strip()

def rag_answer(question: str, k: int = 5):
    """
    Run the full RAG pipeline: retrieve context, format prompt, generate answer.
    Args:
        question (str): The user's question.
        k (int): Number of chunks to retrieve.
    Returns:
        dict: Contains the answer and the retrieved sources.
    """
    retrieved = retrieve(question, k)
    context_chunks = [chunk for chunk, meta in retrieved]
    prompt = format_prompt(context_chunks, question)
    answer = generate_answer(prompt)
    return {
        'answer': answer,
        'sources': retrieved
    } 