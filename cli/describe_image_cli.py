import argparse

from lib.describe_image import describe_command


def main() -> None:
     parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
     subparsers = parser.add_subparsers(dest="command", help="Available commands")
     parser.add_argument("--image", type=str, help="Provide file path to an image to describe")
     parser.add_argument("--query", type=str, help="Terms to search along with the image")
     
     args = parser.parse_args()
     
     results = describe_command(args.image, args.query)    

if __name__ == "__main__":
    main()