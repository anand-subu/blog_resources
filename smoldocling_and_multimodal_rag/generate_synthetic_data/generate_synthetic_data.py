import os
import argparse
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator


def initialize_generator(model_name: str) -> TestsetGenerator:
    """
    Initializes the testset generator with the specified LLM model and embedding model.

    Args:
        model_name (str): The name of the language model to use.

    Returns:
        TestsetGenerator: Configured testset generator instance.
    """
    llm_wrapper = LangchainLLMWrapper(ChatOpenAI(model=model_name))
    embedding_wrapper = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
    return TestsetGenerator(llm=llm_wrapper, embedding_model=embedding_wrapper)


def generate_testsets(base_dir: Path, output_dir: Path, testset_size: int, model_name: str):
    """
    Generates testsets for each markdown subfolder in the base directory.

    Args:
        base_dir (Path): Path to the root folder containing markdown subfolders.
        output_dir (Path): Path to the directory where output testsets will be saved.
        testset_size (int): Number of questions to generate for each testset.
        model_name (str): Name of the language model to use.
    """
    generator = initialize_generator(model_name)
    output_dir.mkdir(exist_ok=True)

    for subfolder in base_dir.iterdir():
        if subfolder.is_dir():
            print(f"Processing folder: {subfolder.name}")

            loader = DirectoryLoader(str(subfolder), glob="**/*.md", show_progress=True)
            docs = loader.load()

            if not docs:
                print(f"No markdown files found in {subfolder.name}, skipping.")
                continue

            dataset = generator.generate_with_langchain_docs(docs, testset_size=testset_size)

            output_path = output_dir / f"{subfolder.name}_testset.json"
            dataset.to_pandas().to_json(output_path, orient="records", indent=2)

            print(f"Saved dataset for {subfolder.name} to {output_path}\n")


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Generate testsets from markdown folders using RAGAS.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("markdown_vlm"),
        help="Base directory containing markdown subfolders."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("testsets"),
        help="Directory where testset JSON files will be saved."
    )
    parser.add_argument(
        "--testset-size",
        type=int,
        default=20,
        help="Number of questions to generate per testset."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Name of the OpenAI model to use (e.g., gpt-4o-mini, gpt-3.5-turbo)."
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    generate_testsets(args.base_dir, args.output_dir, args.testset_size, args.model)


if __name__ == "__main__":
    main()
