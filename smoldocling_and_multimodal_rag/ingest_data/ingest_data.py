import os
import argparse
from tqdm import tqdm
from langchain.document_loaders import TextLoader
from langchain.text_splitter import MarkdownTextSplitter
from langchain.docstore.document import Document
import weaviate
import weaviate.classes as wvc
from weaviate.classes.init import Auth


# Initialize the Markdown text splitter
text_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=100)


def process_markdown_file(file_path, document_name, page_number):
    """
    Loads and splits a markdown file into chunks, attaching metadata.

    Args:
        file_path (str): Path to the markdown file.
        document_name (str): Name of the document.
        page_number (str): Page number extracted from the filename.

    Returns:
        list: List of chunked Document objects with metadata.
    """
    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()
    chunks = text_splitter.split_documents(documents)

    for chunk in chunks:
        chunk.metadata['document_name'] = document_name
        chunk.metadata['page_number'] = page_number

    return chunks


def process_directory(root_dir):
    """
    Walks through a directory of markdown files organized by document name.

    Args:
        root_dir (str): Root directory containing subfolders of markdown files.

    Returns:
        list: List of all chunked Document objects.
    """
    all_chunks = []
    for document_name in os.listdir(root_dir):
        document_path = os.path.join(root_dir, document_name)
        if not os.path.isdir(document_path):
            continue
        for filename in os.listdir(document_path):
            if not filename.endswith(".md"):
                continue
            file_path = os.path.join(document_path, filename)
            page_number = os.path.splitext(filename)[0]
            try:
                chunks = process_markdown_file(file_path, document_name, page_number)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    return all_chunks


def convert_document_to_json(doc):
    """
    Converts a LangChain document object to JSON format.

    Args:
        doc (dict): LangChain document JSON object.

    Returns:
        dict: Dictionary with content and metadata.
    """
    return {**doc["kwargs"]["metadata"], "content": doc["kwargs"]["page_content"]}


def connect_to_weaviate():
    """
    Connects to Weaviate Cloud using environment variables.

    Returns:
        weaviate.Client: Connected Weaviate client instance.
    """
    return weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=Auth.api_key(api_key=os.getenv("WEAVIATE_API_KEY")),
        headers={'X-OpenAI-Api-key': os.getenv("OPENAI_API_KEY")}
    )


def create_schema_if_needed(client):
    """
    Creates the 'DocumentChunk' schema in Weaviate if it doesn't exist.

    Args:
        client (weaviate.Client): Connected Weaviate client.
    """
    try:
        client.collections.create(
            name="DocumentChunk",
            vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_openai(),
            reranker_config=wvc.config.Configure.Reranker.voyageai(model="rerank-2"),
            properties=[
                wvc.config.Property(name="content", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="document_name", data_type=wvc.config.DataType.TEXT, vectorize_property_name=False, tokenization=wvc.config.Tokenization.FIELD),
                wvc.config.Property(name="page_number", data_type=wvc.config.DataType.TEXT, vectorize_property_name=False, tokenization=wvc.config.Tokenization.FIELD),
                wvc.config.Property(name="source", data_type=wvc.config.DataType.TEXT, vectorize_property_name=False, tokenization=wvc.config.Tokenization.FIELD),
            ]
        )
    except Exception as e:
        print(f"Schema creation error: {e}")


def upload_documents(client, docs):
    """
    Uploads document chunks to the Weaviate collection.

    Args:
        client (weaviate.Client): Connected Weaviate client.
        docs (list): List of Document objects to upload.
    """
    collection = client.collections.get("DocumentChunk")

    with collection.batch.dynamic() as batch:
        for data_row in tqdm(docs, desc="Uploading to Weaviate"):
            json_doc = convert_document_to_json(data_row.to_json())
            batch.add_object(properties=json_doc)

            if batch.number_errors > 10:
                print("Batch import stopped due to excessive errors.")
                break


def main():
    parser = argparse.ArgumentParser(description="Ingest Markdown files into Weaviate.")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory containing markdown files")
    args = parser.parse_args()

    print("Processing markdown files...")
    all_docs = process_directory(args.root_dir)

    print("Connecting to Weaviate...")
    client = connect_to_weaviate()

    print("Creating schema if needed...")
    create_schema_if_needed(client)

    print("Uploading documents...")
    upload_documents(client, all_docs)


if __name__ == "__main__":
    main()
