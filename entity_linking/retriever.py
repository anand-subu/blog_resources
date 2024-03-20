from rank_bm25 import BM25Okapi
from typing import List, Tuple, Dict
from nltk.tokenize import word_tokenize
from tqdm import tqdm

class BM25Retriever:
    """
    A class for retrieving documents using the BM25 algorithm, optimized for documents stored in a dictionary.
    
    Attributes:
        index (List[int, str]): A dictionary with document IDs as keys and document texts as values.
        tokenized_docs (List[List[str]]): Tokenized version of the documents in `processed_index`.
        bm25 (BM25Okapi): An instance of the BM25Okapi model from the rank_bm25 package.
    """
    
    def __init__(self, docs_with_ids: Dict[int, str]):
        """
        Initializes the BM25Retriever with a dictionary of documents.
        
        Args:
            docs_with_ids (Dict[int, str]): A dictionary with document IDs as keys and document texts as values.
        """
        self.index = docs_with_ids
        self.tokenized_docs = self._tokenize_docs([x[1] for x in self.index])
        self.bm25 = BM25Okapi(self.tokenized_docs)
            
    def _tokenize_docs(self, docs: List[str]) -> List[List[str]]:
        """
        Tokenizes the documents using NLTK's word_tokenize.
        
        Args:
            docs (List[str]): A list of documents to be tokenized.
        
        Returns:
            List[List[str]]: A list of tokenized documents.
        """
        return [word_tokenize(doc.lower()) for doc in docs]
    
    def query(self, query: str, top_n: int = 10) -> List[Tuple[int, float]]:
        """
        Queries the BM25 model and retrieves the top N documents with their scores.
        
        Args:
            query (str): The query string.
            top_n (int): The number of top documents to retrieve.
        
        Returns:
            List[Tuple[int, float]]: A list of tuples, each containing a document ID and its BM25 score.
        """
        tokenized_query = word_tokenize(query.lower())
        scores = self.bm25.get_scores(tokenized_query)
        doc_scores_with_ids = [(doc_id, scores[i]) for i, (doc_id, _) in enumerate(self.index)]
        top_doc_ids_and_scores = sorted(doc_scores_with_ids, key=lambda x: x[1], reverse=True)[:top_n]
        return [x[0] for x in top_doc_ids_and_scores]
