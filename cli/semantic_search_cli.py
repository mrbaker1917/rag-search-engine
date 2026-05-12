#!/usr/bin/env python3

import argparse
from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, search_term
from lib.search_utils import DEFAULT_SEARCH_LIMIT

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify", help="Verify that the embedding model is loaded.")
    subparsers.add_parser("verify_embeddings", help="verifies that embeddings have loaded.")
    embed_text_parser = subparsers.add_parser("embed_text", help="provides embedding for text")
    embed_text_parser.add_argument("text", type=str)
    embed_query_parser = subparsers.add_parser("embed_query", help="Embeds query text.")
    embed_query_parser.add_argument("query", type=str)
    search_parser = subparsers.add_parser("search", help="enter a term to search movie database")
    search_parser.add_argument("search_term", type=str)
    search_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Search limit")
    chunk_parser = subparsers.add_parser("chunk", help="divides string into chunks")
    chunk_parser.add_argument("text", type=str, help="text to be chunked")
    chunk_parser.add_argument("--chunk-size", type=int, default=200)
    
    args = parser.parse_args()

    match args.command:
        case "chunk":
            arr = args.text.split(" ")
            n = args.chunk_size
            arr_len = len(arr)
            print(f"Chunking {len(args.text)} characters")
            chunks = []
            for i in range(0, arr_len, n):
                chunks.append(arr[i:i+n])
            for i, chunk in enumerate(chunks, 1):
                print(f"{i}. {" ".join(chunk)}")

        case "search":
            search_term(args.search_term, args.limit)
        case "embed_query":
            embed_query_text(args.query)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_text":
            embed_text(args.text)
        case "verify":
            verify_model()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()