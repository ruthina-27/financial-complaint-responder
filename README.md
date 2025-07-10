# Financial Complaint Responder (RAG Pipeline)

This project builds a complaint-answering chatbot using Retrieval-Augmented Generation (RAG) on the CFPB consumer complaints dataset. The chatbot retrieves relevant complaint excerpts and generates answers to user questions, with full transparency of sources.

## Project Overview
- **Goal:** Build a trustworthy, explainable chatbot for financial complaint Q&A using RAG.
- **Data:** CFPB consumer complaints (filtered for 5 key products).
- **Pipeline:** EDA → Cleaning → Chunking/Embedding → Vector Store → Retrieval → LLM Generation → Evaluation → Interactive App.

## Project Structure
```
financial-complaint-responder/
├── data/                # Raw and processed datasets
├── notebooks/           # Jupyter Notebooks (EDA, preprocessing, evaluation)
├── src/                 # Python scripts (embedding, indexing, RAG pipeline)
├── vector_store/        # Vector database files
├── requirements.txt     # Python dependencies
├── app.py               # Gradio app for chatbot
└── README.md            # Project documentation
```

## Setup
1. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```
2. **Download the CFPB complaints dataset** and place it in the project root as `complaints.csv` (already present).

## Usage
### 1. EDA & Preprocessing
- Open and run `notebooks/eda_and_preprocessing.ipynb` to:
  - Analyze the dataset
  - Filter and clean the data
  - Save the result to `data/filtered_complaints.csv`

### 2. Chunking, Embedding, and Indexing
- Run the script to create the vector store:
  ```
  python src/embed_and_index.py
  ```
- This will save the FAISS index and metadata in `vector_store/`.

### 3. RAG Pipeline & Evaluation
- The core RAG logic is in `src/rag_pipeline.py`.
- To evaluate the system, see `notebooks/evaluation.md` for a table of sample questions, answers, and analysis.

### 4. Interactive Chatbot App
- Launch the chatbot with:
  ```
  python app.py
  ```
- Enter your question, view the AI's answer, and see the retrieved complaint sources for transparency.
- Use the "Clear" button to reset.
- *(Optional: If streaming is enabled, answers will appear token-by-token for a better user experience.)*

## Deliverables
- Cleaned and filtered dataset: `data/filtered_complaints.csv`
- Vector store: `vector_store/faiss.index` and `vector_store/metadata.pkl`
- EDA and preprocessing notebook: `notebooks/eda_and_preprocessing.ipynb`
- RAG pipeline: `src/rag_pipeline.py`
- Evaluation table: `notebooks/evaluation.md`
- Gradio app: `app.py`

## Notes
- The pipeline uses `sentence-transformers/all-MiniLM-L6-v2` for embeddings and FAISS for vector search.
- Adjust chunk size and overlap in `src/embed_and_index.py` as needed.
- All code is documented and organized for clarity and reproducibility.

---

For questions or improvements, please open an issue or PR. 