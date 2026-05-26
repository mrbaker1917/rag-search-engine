import os
from google import genai
import time

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies


api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        bm25_results = self._bm25_search(query, limit=limit * 500)

        chunked_sem_results = self.semantic_search.search_chunks(
            query, limit=limit * 500
        )
        bm25_scores = [s[1] for s in bm25_results]
        ch_sem_scores = [float(c["score"]) for c in chunked_sem_results]
        normed_bm25 = normalize_scores(bm25_scores)
        normed_chunk_sems = normalize_scores(ch_sem_scores)
        combo = {}
        for result, norm_score in zip(bm25_results, normed_bm25):
            doc = self.idx.docmap[result[0]]
            combo[result[0]] = {
                "title": doc["title"],
                "description": doc["description"],
                "BM25": norm_score,
                "Semantic": 0.0,
                "Hybrid": 0.0,
            }

        for res, norm_score in zip(chunked_sem_results, normed_chunk_sems):
            if res["id"] in combo:
                combo[res["id"]]["Semantic"] = norm_score
            else:
                doc = self.idx.docmap[res["id"]]
                combo[res["id"]] = {
                    "title": doc["title"],
                    "description": doc["description"],
                    "BM25": 0.0,
                    "Semantic": norm_score,
                    "Hybrid": 0.0,
                }

        for k, v in combo.items():
            h_score = hybrid_score(v["BM25"], v["Semantic"], alpha)
            combo[k]["Hybrid"] = h_score

        return sorted(combo.items(), key=lambda item: item[1]["Hybrid"], reverse=True)[
            :limit
        ]

    def rrf_search(self, query, k, limit=10):
        bm25_results = self._bm25_search(query, limit=limit * 500)

        chunked_sem_results = self.semantic_search.search_chunks(
            query, limit=limit * 500
        )
        rrf_dic = {}
        for i, res in enumerate(bm25_results, 1):
            doc = self.idx.docmap[res[0]]
            rrf_dic[res[0]] = {
                "title": doc["title"],
                "description": doc["description"],
                "BM25": i,
                "Semantic": limit * 500 + 1,
                "rrf": 0.0,
            }

        for j, r in enumerate(chunked_sem_results, 1):
            if r["id"] in rrf_dic:
                rrf_dic[r["id"]]["Semantic"] = j
            else:
                doc = self.idx.docmap[r["id"]]
                rrf_dic[r["id"]] = {
                    "title": doc["title"],
                    "description": doc["description"],
                    "BM25": limit * 500 + 1,
                    "Semantic": j,
                    "rrf": 0.0,
                }
        for i, d in rrf_dic.items():

            rrf_score = 1 / (k + d["BM25"]) + 1 / (k + d["Semantic"])
            rrf_dic[i]["rrf"] = rrf_score
        return sorted(rrf_dic.items(), key=lambda item: item[1]["rrf"], reverse=True)[
            :limit
        ]


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [1.0] * len(scores)

    return [(s - min_score) / (max_score - min_score) for s in scores]


def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score


def weighted_search(query, alpha=0.5, limit=5):
    movies = load_movies()
    hybrid_search = HybridSearch(movies)
    results = hybrid_search.weighted_search(query, alpha, limit)
    for i, r in enumerate(results, 1):
        print(
            f"{i}. {r[1]['title']}\n Hybrid Score: {r[1]['Hybrid']:.3f}\n BM25: {r[1]['BM25']:.3f}, Semantic: {r[1]['Semantic']:.3f}\n {r[1]['description'][:100]}"
        )

def rrf_search_rerank(query, k, limit):
    movies = load_movies()
    hybrid_search = HybridSearch(movies)
    results = hybrid_search.rrf_search(query, k, limit)
    client = genai.Client(api_key=api_key)
    model="gemma-4-31b-it"
    new_results = []
    for i, r in enumerate(results, 1):
        contents=f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {r[1]["title"]} - {r[1]["description"]}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Output ONLY the number in your response, no other text or explanation.

Score:"""
        response = client.models.generate_content(model=model, contents=contents, config=None)
        new_results.append((float(response.text), r[1]))
        if i < len(results)-1:
            time.sleep(3)
    # print(new_results)
    ranked_llm_scores = sorted(new_results, key=lambda t: t[0], reverse=True)[:int(limit/5)]
    print("Re-ranking top 3 results using individual method...")
    print("Reciprocal Rank Fusion Results for 'family movie about bears in the woods' (k=60):")
    
    for j, t in enumerate(ranked_llm_scores, 1):
        print(f"{j}. {t[1]["title"]}")
        print(f"Re-rank Score: {float(t[0]):.3f}")
        print(f"RRF Score: {t[1]['rrf']:.3f}")
        print(f"BM25: {t[1]['BM25']:.3f}, Semantic: {t[1]['Semantic']:.3f}") 
        print(f"{t[1]['description'][:100]}")

def rrf_search(query, k, limit):
    movies = load_movies()
    hybrid_search = HybridSearch(movies)
    results = hybrid_search.rrf_search(query, k, limit)
    for i, r in enumerate(results, 1):
        print(
            f"{i}. {r[1]['title']}\n RRF Score: {r[1]['rrf']:.3f}\n BM25: {r[1]['BM25']:.3f}, Semantic: {r[1]['Semantic']:.3f}\n {r[1]['description'][:100]}"
        )
