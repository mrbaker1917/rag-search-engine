from PIL import Image
from sentence_transformers import SentenceTransformer

class MultimodalSearch:
    def __init__(self, model_name='clip-ViT-B-32'):
        self.model = SentenceTransformer(model_name)

    def encode_text(self, text):
        return self.model.encode(text)

    def encode_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return self.model.encode(image)

    def search(self, query, data):
        query_embedding = self.encode_text(query)
        results = []
        for item in data:
            if 'text' in item:
                item_embedding = self.encode_text(item['text'])
            elif 'image' in item:
                item_embedding = self.encode_image(item['image'])
            else:
                continue
            similarity = self.cosine_similarity(query_embedding, item_embedding)
            results.append((item, similarity))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

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