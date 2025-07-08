import os
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Parameters
CLEANED_DATA_PATH = os.path.join('data', 'filtered_complaints.csv')
VECTOR_STORE_DIR = 'vector_store'
CHUNK_SIZE = 256  # words
CHUNK_OVERLAP = 32  # words
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

def chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - chunk_overlap):
        chunk = ' '.join(words[i:i+chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

# Load data
print('Loading cleaned data...')
df = pd.read_csv(CLEANED_DATA_PATH)

# Prepare chunks and metadata
print('Chunking narratives...')
chunks = []
metadatas = []
for idx, row in tqdm(df.iterrows(), total=len(df)):
    text_chunks = chunk_text(row['Cleaned narrative'])
    for chunk in text_chunks:
        chunks.append(chunk)
        metadatas.append({
            'complaint_id': row.get('Complaint ID', idx),
            'product': row['Product'],
            'original_index': idx
        })

# Load embedding model
print('Loading embedding model...')
model = SentenceTransformer(EMBEDDING_MODEL)

# Embed chunks
print('Embedding chunks...')
embeddings = model.encode(chunks, show_progress_bar=True, batch_size=64, normalize_embeddings=True)

# Create FAISS index
print('Creating FAISS index...')
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings, dtype=np.float32))

# Save index and metadata
faiss.write_index(index, os.path.join(VECTOR_STORE_DIR, 'faiss.index'))
with open(os.path.join(VECTOR_STORE_DIR, 'metadata.pkl'), 'wb') as f:
    pickle.dump(metadatas, f)

print(f"Saved FAISS index and metadata for {len(chunks)} chunks.") 