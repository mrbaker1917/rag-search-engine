import argparse

from lib.describe_image import describe_command


def main() -> None:
     parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
     subparsers = parser.add_subparsers(dest="command", help="Available commands")
     describe_parser = subparsers.add_parser("describe", help="Describe an image using the Gemini API")
     
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
        case "question":
            result = question_command(args.query, args.limit)
            print("Search Results:")
            for doc in result["search_results"]:
                print(f"  - {doc['title']}")
            print()
            print("Answer:")
            print(result['answer'])
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()