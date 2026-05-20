import argparse

from lib.hybrid_search import normalize_scores, weighted_search

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    normalize_parser = subparsers.add_parser("normalize", help="Normalizes a list of scores")
    normalize_parser.add_argument("scores", nargs="*", type=float, help="data to normalize")
    weighted_search_parser = subparsers.add_parser("weighted-search", help="Allows weighting of keyword vs semantic search.")
    weighted_search_parser.add_argument("query", type=str, help="A string to search.")
    weighted_search_parser.add_argument("--alpha", type=float, default=0.5, help="Determines weighting of searches.")
    weighted_search_parser.add_argument("--limit", type=int, default=5, help="Determines how many results you want.")
    
    args = parser.parse_args()

    match args.command:
        case "weighted-search":
            weighted_search(args.query, args.alpha, args.limit)
        case "normalize":
            normalized = normalize_scores(args.scores)
            for score in normalized:
                print(f"* {score:.4f}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()

