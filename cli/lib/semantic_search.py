import numpy as np
import os
from sentence_transformers import SentenceTransformer

from lib.search_utils import load_movies

class SemanticSearch:
    def __init__(self, model_name='all-MiniLM-L6-v2') -> None:
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}
    
    def generate_embedding(self, text):
        if len(text.strip()) == 0:
            raise ValueError("Please enter words")
        embeddings = self.model.encode([text])
        return embeddings[0]
    
    def build_embeddings(self, documents):
        self.documents = documents
        doc_strings = []
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
            doc_strings.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(doc_strings, show_progress_bar=True)
        np.save("cache/movie_embeddings.npy", self.embeddings)
        return self.embeddings
    
    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc['id']] =  doc
        if os.path.exists("cache/movie_embeddings.npy"):
            self.embeddings = np.load("cache/movie_embeddings.npy")
            if len(self.embeddings) == len(documents):
                return self.embeddings
        self.embeddings = self.build_embeddings(documents)
        return self.embeddings
        
    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        embedded_query = self.generate_embedding(query)
        sim_scores = []
        for i, doc in enumerate(self.embeddings):
            sim_scores.append((cosine_similarity(embedded_query, doc), self.documents[i]))
        sorted_sim_scores = sorted(sim_scores, key=lambda t: t[0], reverse=True)
        list_top_scores = []
        for tup in sorted_sim_scores[:limit]:
            d = {"score": tup[0], "title": tup[1]["title"], "description": tup[1]["description"]}
            list_top_scores.append(d)
        return list_top_scores

def search_term(term, limit):
    search_instance = SemanticSearch()
    movies = load_movies()
    embeddings = search_instance.load_or_create_embeddings(movies)
    results = search_instance.search(term, limit)
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['title']} (score: {r['score']:.4f})\n {r['description'][:100]+"..."}")


def verify_model():
    search_instance = SemanticSearch()
    model = search_instance.model
    print(f"Model loaded: {model}")
    print(f"Max sequence length: {model.max_seq_length}")

def embed_text(text):
    sem_search = SemanticSearch()
    embedding = sem_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    search_instance = SemanticSearch()
    movies = load_movies()
    embeddings = search_instance.load_or_create_embeddings(movies)
    print(f"Number of docs:   {len(movies)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query):
    search_instance = SemanticSearch()
    embedding = search_instance.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 ==  0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)