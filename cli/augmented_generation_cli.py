import argparse

from lib.augmented_generation import rag_command, summarize_command, citations_command


def main() -> None:
     parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
     subparsers = parser.add_subparsers(dest="command", help="Available commands")
     rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
     rag_parser.add_argument("query", type=str, help="Search query for RAG")
     summarize_parser = subparsers.add_parser(
         "summarize", help="Generates multi-document summary."
     )
     summarize_parser.add_argument("query", type=str, help="Terms to search and summarize")
     summarize_parser.add_argument("--limit", type=int, default=5, help="Limit the number of search results to summarize.")

     citations_parser = subparsers.add_parser("citations", help="Provides citations for claims in results.")
     citations_parser.add_argument("query", type=str, help="Terms to search for citations")
     citations_parser.add_argument("--limit", type=int, default=5, help="Limit the number of search results with citations.")

     
     args = parser.parse_args()
     
     match args.command:
        case "rag":
            result = rag_command(args.query)
            print("Search Results:")
            for doc in result["search_results"]:
                print(f"  - {doc['title']}")
            print()
            print("RAG Response:")
            print(result['answer'])
        case "summarize":
            result = summarize_command(args.query, args.limit)
            print("Search Results:")
            for doc in result["search_results"]:
                print(f"  - {doc['title']}")
            print()
            print("LLM Summary:")
            print(result['answer'])
        case "citations":
            result = citations_command(args.query, args.limit)
            print("Search Results:")
            for doc in result["search_results"]:
                print(f"  - {doc['title']}")
            print()
            print("LLM Answer:")
            print(result['answer'])
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()