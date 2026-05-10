#!/usr/bin/env python3

import argparse
from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify", help="Verify that the embedding model is loaded.")
    subparsers.add_parser("verify_embeddings", help="verifies that embeddings have loaded.")
    embed_text_parser = subparsers.add_parser("embed_text", help="provides embedding for text")
    embed_text_parser.add_argument("text", type=str)
    embed_query_parser = subparsers.add_parser("embed_query", help="Embeds query text.")
    embed_query_parser.add_argument("query", type=str)
    args = parser.parse_args()

    match args.command:
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