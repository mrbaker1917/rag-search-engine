from PIL import Image
from sentence_transformers import SentenceTransformer
from lib.search_utils import load_movies

class MultimodalSearch:
    def __init__(self,  documents: list[object] | None = None, model_name='clip-ViT-B-32'):
        self.model = SentenceTransformer(model_name)
        self.documents = documents or []
        self.texts = [f"{doc['title']}: {doc['description']}" for doc in self.documents]
        self.texts_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def encode_text(self, text):
        return self.model.encode(text)

    def encode_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return self.model.encode(image)

    def search_with_image(self, image_path):
        image_embedding = self.encode_image(image_path)
        similarities = [self.cosine_similarity(image_embedding, text_emb) for text_emb in self.texts_embeddings]
        ranked_docs = sorted(zip(self.documents, similarities), key=lambda x: x[1], reverse=True)
        return [{"id:": doc["id"], "title": doc["title"], "description": doc["description"], "similarity": sim} for doc, sim in ranked_docs][:5]

    @staticmethod
    def cosine_similarity(vec1, vec2):
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm_vec1 = sum(a * a for a in vec1) ** 0.5
        norm_vec2 = sum(b * b for b in vec2) ** 0.5
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
        return dot_product / (norm_vec1 * norm_vec2)

def verify_image_embedding(image_path):
    search = MultimodalSearch()
    embedding = search.encode_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")

def image_search_command(image_path):
    movies = load_movies()
    search = MultimodalSearch(documents=movies)
    results = search.search_with_image(image_path)
    return results