# Financial Complaint Responder (RAG Pipeline)

This project builds a complaint-answering chatbot using Retrieval-Augmented Generation (RAG) on the CFPB consumer complaints dataset.

## Project Structure

```
financial-complaint-responder/
├── data/                # Raw and processed datasets
├── notebooks/           # Jupyter Notebooks (EDA, preprocessing)
├── src/                 # Python scripts (embedding, indexing)
├── vector_store/        # Vector database files
├── requirements.txt     # Python dependencies
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

## Deliverables
- Cleaned and filtered dataset: `data/filtered_complaints.csv`
- Vector store: `vector_store/faiss.index` and `vector_store/metadata.pkl`
- EDA and preprocessing notebook: `notebooks/eda_and_preprocessing.ipynb`

## Notes
- The pipeline uses `sentence-transformers/all-MiniLM-L6-v2` for embeddings and FAISS for vector search.
- Adjust chunk size and overlap in `src/embed_and_index.py` as needed.

---

For questions or improvements, please open an issue or PR. 