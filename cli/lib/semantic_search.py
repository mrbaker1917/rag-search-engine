
from sentence_transformers import SentenceTransformer

class SemanticSearch:
    def __init__(self, model_name='all-MiniLM-L6-v2') -> None:
        self.model = SentenceTransformer(model_name)
    
    def generate_embedding(self, text):
        if len(text.strip()) == 0:
            raise ValueError("Please enter words")
        embeddings = self.model.encode([text])
        return embeddings[0]


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