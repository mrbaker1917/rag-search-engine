from pickle import dump
import os
from typing import Dict
from collections import defaultdict
from .search_utils import (
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    load_movies,
    load_stop_words
)

class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.index_path = os.path.joint(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.joint(CACHE_DIR, "docmap.pkl")

    def build(self):
        movies = load_movies()
        for m in movies:
            doc_id = m["id"]
            doc_description = f"{m['title']} {m['description']}"
            self.docmap[doc_id] = m
            self.__add_document(doc_id, doc_description)

    def save(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.index_path, 'wb') as file:
            dump(self.index, file)
        with open(self.docmap_path,"wb") as file: 
            dump(self.docmap, file)
            
    def get_documents(self, term):
        return sorted(list(self.index[term.lower()]))

    def __add_document(self, doc_id, text):
        tokens = [w for w in text.split(" ") if w != ""]
        for t in tokens:
            t = t.lower()
            if t not in self.index.keys():
                self.index[t] = set([doc_id])
            else:
                self.index[t].add(doc_id)
    
    
    