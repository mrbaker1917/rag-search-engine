import argparse

from lib.multimodal_search import verify_image_embedding, image_search_command

def main() -> None:
     parser = argparse.ArgumentParser(description="Multimodal Search CLI")
     subparsers = parser.add_subparsers(dest="command", help="Available commands")
     mm_parser = subparsers.add_parser(
        "verify_image_embedding", help="Verify that image embeddings are generated correctly"
    )
     mm_parser.add_argument("image_path", type=str, help="Path to the image to verify embedding")
     image_search_parser = subparsers.add_parser(
        "image_search", help="Search for movies using an image"
    )
     image_search_parser.add_argument("image_path", type=str, help="Path to the image to search with")
     args = parser.parse_args()
     
     match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image_path)
        case "image_search":
            results = image_search_command(args.image_path)
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['title']} (similarity: {result['similarity']:.3f})\n   {result['description']}\n")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()