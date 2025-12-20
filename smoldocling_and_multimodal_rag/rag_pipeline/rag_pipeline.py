import os
from typing import List, Dict

import weaviate
from weaviate.classes.init import Auth
import voyageai
from openai import OpenAI


class Retriever:
    """
    Handles retrieval of relevant documents using Weaviate hybrid search and VoyageAI reranking.
    """

    def __init__(self,
                 weaviate_url: str,
                 weaviate_api_key: str,
                 openai_api_key: str,
                 voyage_api_key: str,
                 collection_name: str = "DocumentChunk"):
        """
        Initialize the Retriever with required API keys and connection info.
        """
        self.client = weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=Auth.api_key(api_key=weaviate_api_key),
            headers={
                'X-OpenAI-Api-key': openai_api_key,
                'X-Voyageai-Api-Key': voyage_api_key
            }
        )

        self.collection = self.client.collections.get(collection_name)
        self.voyage_client = voyageai.Client(api_key=voyage_api_key)

    def hybrid_query(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Performs a hybrid search on the Weaviate collection.
        Returns a list of documents with their metadata.
        """
        response = self.collection.query.hybrid(query=query, limit=limit)
        results = []
        for obj in response.objects:
            prop = obj.properties
            results.append({
                'content': prop.get('content', ''),
                'document_name': prop.get('document_name', ''),
                'page_number': prop.get('page_number', ''),
                'source': prop.get('source', '')
            })
        return results

    def rerank(self, query: str, documents: List[Dict], top_k: int = 3, model: str = "rerank-2") -> List[Dict]:
        """
        Reranks the given list of documents using VoyageAI based on the query.
        Returns the top_k most relevant documents.
        """
        contents = [doc['content'] for doc in documents]
        reranked = self.voyage_client.rerank(query, contents, model=model, top_k=top_k)

        ranked_docs = []
        for result in reranked.results:
            original_doc = next((doc for doc in documents if doc['content'] == result.document), None)
            if original_doc:
                ranked_docs.append({
                    'content': original_doc['content'],
                    'document_name': original_doc['document_name'],
                    'page_number': original_doc['page_number'],
                    'source': original_doc['source'],
                    'relevance_score': result.relevance_score
                })

        return ranked_docs

    def retrieve(self, query: str, initial_limit: int = 10, rerank_top_k: int = 3) -> List[Dict]:
        """
        Retrieves and reranks relevant documents for a given query.
        """
        initial_docs = self.hybrid_query(query, limit=initial_limit)
        return self.rerank(query, initial_docs, top_k=rerank_top_k)


class LLMGenerator:
    """
    Handles interaction with the OpenAI API to generate answers based on document context.
    """

    def __init__(self, openai_api_key: str, model: str = "gpt-4o"):
        """
        Initialize the LLMGenerator with the OpenAI API key and desired model.
        """
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model

    def generate_answer(self,
                        query: str,
                        context_docs: List[Dict],
                        temperature: float = 0.0,
                        max_tokens: int = 500) -> str:
        """
        Generates an answer to the query using only the provided context documents.
        Includes citations with document names and page numbers.
        """
        context = "\n\n".join([
            f"Document: {doc['document_name']} (Page: {doc['page_number']})\nContent: {doc['content']}"
            for doc in context_docs
        ])

        system_prompt = (
            "Answer the following query strictly using only the provided context. "
            "Do not include any information that is not explicitly stated in the context. "
            "If the context does not contain sufficient information to answer the query, "
            "respond explicitly stating you do not have sufficient information. "
            "When providing an answer, include a citation for each piece of information used, "
            "indicating the document name and page number (e.g., [Document Name: <DocumentName>, Page No: 3])."
        )

        user_prompt = (
            f"Query: {query}\n\n"
            f"Context:\n{context}\n\n"
            f"Answer:"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )

        return response.choices[0].message.content.strip()


# Example usage
if __name__ == "__main__":
    retriever = Retriever(
        weaviate_url=os.environ["WEAVIATE_URL"],
        weaviate_api_key=os.environ["WEAVIATE_API_KEY"],
        openai_api_key=os.environ["OPENAI_API_KEY"],
        voyage_api_key=os.environ["VOYAGE_API_KEY"]
    )

    answerer = LLMGenerator(openai_api_key=os.environ["OPENAI_API_KEY"])

    query = "What were the neonatal outcomes assessed in relation to maternal gout diagnosis?"
    context_docs = retriever.retrieve(query, initial_limit=50, rerank_top_k=25)
    answer = answerer.generate_answer(query, context_docs)

    print(answer)
