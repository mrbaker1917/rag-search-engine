import argparse, json
from unittest import case

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    data = json.load(open("data/movies.json", "r"))
        
    match args.command:
        case "search":
            found_movies = []
            print(f"Searching for: {args.query}")
            for movie in data["movies"]:
                if args.query.lower() in movie["title"].lower():
                    found_movies.append(movie)
            for i, movie in enumerate(found_movies[:5]):
                print(f"{i+1}. {movie['title']}")


        case _:
            parser.print_help()

if __name__ == "__main__":
    main()