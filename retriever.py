import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

# Load and preprocess dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    df.fillna("NA", inplace=True)
    df['text'] = df.apply(lambda row: ' '.join([f"{col}: {row[col]}" for col in df.columns]), axis=1)
    return df

# Build the FAISS index
def build_index(texts, model):
    embeddings = model.encode(texts, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

# Retrieve top K relevant texts
def retrieve(query, model, index, texts, k=5):
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k)
    return [texts[i] for i in I[0]]

# Main retrieval setup
def setup_retriever(data_path):
    df = load_data(data_path)
    texts = df['text'].tolist()
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index, embeddings = build_index(texts, model)
    return texts, model, index
