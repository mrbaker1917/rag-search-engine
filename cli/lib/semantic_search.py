
from sentence_transformers import SentenceTransformer

class SemanticSearch:
    def __init__(self, model_name='all-MiniLM-L6-v2') -> None:
        self.model = SentenceTransformer(model_name)


def verify_model():
    search_instance = SemanticSearch()
    model = search_instance.model
    print(f"Model loaded: {model}")
    print(f"Max sequence length: {model.max_seq_length}")

    # model.encode(text)