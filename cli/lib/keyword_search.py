from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies, load_stop_words

import string


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []
    for movie in movies:
        p_query_arr = preprocess_text(query)
        p_title_arr = preprocess_text(movie["title"])
        for w in p_query_arr:
            for t in p_title_arr:
                if w in t:
                    results.append(movie)
            if len(results) >= limit:
                break
    return results

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("","", string.punctuation))
    arr = [w for w in text.split(" ") if w != ""]
    stop_words = load_stop_words()
    arr2 = []
    for w in arr:
        if w not in stop_words:
            arr2.append(w)
    return arr2
