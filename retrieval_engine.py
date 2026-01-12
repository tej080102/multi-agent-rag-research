"""
Hybrid Search Retrieval Engine for RAG Systems.

This module implements a hybrid search approach that combines:
- **Dense Search (FAISS)**: Uses semantic embeddings to find documents based on meaning.
  Dense retrievers excel at understanding synonyms, paraphrases, and conceptual similarity.
  
- **Sparse Search (BM25)**: Uses traditional keyword-based matching with TF-IDF weighting.
  BM25 excels at exact keyword matching and handles rare terms better than dense methods.

Why Hybrid Search?
------------------
Neither dense nor sparse retrieval is universally superior. Dense methods struggle with:
- Rare keywords (out-of-vocabulary terms)
- Exact phrase matching
- Highly specific technical terminology

Sparse methods struggle with:
- Semantic understanding (synonyms, paraphrases)
- Context-based similarity
- Cross-lingual retrieval

By combining both approaches, we get the best of both worlds:
1. BM25 captures exact lexical matches that embeddings might miss
2. FAISS captures semantic similarity that keyword matching misses
3. Reranking with a cross-encoder (Cohere) provides final relevance ordering

This hybrid approach has been shown to significantly outperform either method alone
in production RAG systems.
"""

import os
from typing import List, Optional, Tuple

import cohere
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


def load_and_chunk_pdf(
    pdf_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Load a PDF file and split it into chunks for processing.
    
    This helper function handles the document ingestion pipeline:
    1. Loads the PDF using PyPDFLoader (extracts text from each page)
    2. Splits the text into overlapping chunks using RecursiveCharacterTextSplitter
    
    Why RecursiveCharacterTextSplitter?
    -----------------------------------
    - It tries to split on natural boundaries (paragraphs, sentences, words)
    - Maintains semantic coherence within chunks
    - The recursive approach ensures we don't split mid-word
    
    Why overlapping chunks?
    -----------------------
    - Prevents information loss at chunk boundaries
    - Ensures context is preserved for sentences spanning multiple chunks
    - Improves retrieval recall for edge-case queries
    
    Args:
        pdf_path: Path to the PDF file to load.
        chunk_size: Maximum size of each chunk in characters (default: 1000).
        chunk_overlap: Number of overlapping characters between chunks (default: 200).
        
    Returns:
        List of Document objects, each containing a text chunk and metadata.
        
    Raises:
        FileNotFoundError: If the PDF file does not exist.
        ValueError: If chunk_overlap >= chunk_size.
        
    Example:
        >>> documents = load_and_chunk_pdf("research_paper.pdf", chunk_size=500)
        >>> print(f"Created {len(documents)} chunks")
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")
    
    # Load PDF - each page becomes a Document
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    
    # Split into smaller chunks for better retrieval granularity
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]  # Try to split on natural boundaries
    )
    
    documents = text_splitter.split_documents(pages)
    
    # Add chunk index to metadata for traceability
    for idx, doc in enumerate(documents):
        doc.metadata["chunk_index"] = idx
        doc.metadata["source_file"] = pdf_path
    
    return documents


class HybridRetriever:
    """
    A hybrid retrieval system combining dense (FAISS) and sparse (BM25) search.
    
    This class implements a two-stage retrieval pipeline:
    
    Stage 1: Parallel Retrieval
    ---------------------------
    - FAISS (Dense): Converts query to embedding and finds semantically similar documents
    - BM25 (Sparse): Tokenizes query and matches based on term frequency statistics
    
    Stage 2: Reranking
    ------------------
    - Combines results from both retrievers
    - Uses Cohere's cross-encoder reranker to produce final relevance scores
    - Cross-encoders consider query-document interaction, providing superior ranking
    
    Why this architecture?
    ----------------------
    1. **Recall Boost**: Each retriever catches documents the other might miss
    2. **Precision via Reranking**: Cross-encoder reranking corrects initial ranking errors
    3. **Efficiency**: Bi-encoders (FAISS) enable fast initial retrieval; cross-encoder
       (Cohere) is only applied to the candidate set
    
    Attributes:
        documents: List of Document objects to search over.
        vector_store: FAISS index for dense retrieval.
        bm25: BM25Okapi index for sparse retrieval.
        embeddings: Embedding model for query encoding.
        cohere_client: Cohere API client for reranking.
        
    Example:
        >>> from retrieval_engine import HybridRetriever, load_and_chunk_pdf
        >>> 
        >>> # Load and chunk documents
        >>> docs = load_and_chunk_pdf("manual.pdf")
        >>> 
        >>> # Initialize retriever
        >>> retriever = HybridRetriever(docs, cohere_api_key="your-api-key")
        >>> 
        >>> # Search and rerank
        >>> results = retriever.search("How do I reset my password?", k=10)
        >>> final_results = retriever.rerank(results, "How do I reset my password?", top_n=5)
    """
    
    def __init__(
        self,
        documents: List[Document],
        cohere_api_key: Optional[str] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize the hybrid retriever with FAISS and BM25 indexes.
        
        This constructor builds both retrieval indexes:
        
        1. FAISS Index (Dense Retrieval):
           - Generates embeddings for all documents using the specified model
           - Builds an efficient similarity search index
           - Default model: all-MiniLM-L6-v2 (fast, good quality, 384 dimensions)
        
        2. BM25 Index (Sparse Retrieval):
           - Tokenizes all documents (simple whitespace tokenization)
           - Builds inverted index with BM25 scoring capabilities
           - Uses BM25Okapi variant (recommended for most use cases)
        
        Args:
            documents: List of Document objects to index. Each document should have
                       a `page_content` attribute containing the text.
            cohere_api_key: API key for Cohere reranking. If None, will try to read
                           from COHERE_API_KEY environment variable.
            embedding_model: HuggingFace model ID for generating embeddings.
                           Default is all-MiniLM-L6-v2 for balance of speed/quality.
                           
        Raises:
            ValueError: If documents list is empty.
            RuntimeError: If Cohere API key is not provided and not in environment.
            
        Note:
            The embedding model is downloaded on first use. Ensure you have internet
            connectivity for the initial run.
        """
        if not documents:
            raise ValueError("Documents list cannot be empty")
        
        self.documents = documents
        
        # Initialize Cohere client for reranking
        api_key = cohere_api_key or os.getenv("COHERE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Cohere API key required. Pass cohere_api_key or set COHERE_API_KEY env var."
            )
        self.cohere_client = cohere.Client(api_key)
        
        # Initialize embeddings model for dense retrieval
        # all-MiniLM-L6-v2 offers excellent speed/quality tradeoff
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}  # For cosine similarity
        )
        
        # Build FAISS vector store
        # FAISS uses L2 distance by default; normalized embeddings make this equivalent to cosine
        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        
        # Build BM25 index
        # Tokenize documents for BM25 (simple whitespace tokenization)
        # In production, consider more sophisticated tokenization (e.g., spacy)
        tokenized_corpus = [
            self._tokenize(doc.page_content) for doc in documents
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        print(f"✓ Initialized HybridRetriever with {len(documents)} documents")
        print(f"  - FAISS index: {self.vector_store.index.ntotal} vectors")
        print(f"  - BM25 corpus: {len(tokenized_corpus)} documents")
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization for BM25.
        
        Converts text to lowercase and splits on whitespace.
        In production, consider using a proper tokenizer (spaCy, NLTK)
        for better handling of punctuation and stop words.
        
        Args:
            text: Text to tokenize.
            
        Returns:
            List of lowercase tokens.
        """
        return text.lower().split()
    
    def search(
        self,
        query: str,
        k: int = 10
    ) -> List[Tuple[Document, float, str]]:
        """
        Perform hybrid search using both FAISS and BM25.
        
        This method executes parallel retrieval from both indexes:
        
        FAISS Search:
        - Encodes query using the embedding model
        - Finds k nearest neighbors using L2 distance
        - Returns documents with similarity scores
        
        BM25 Search:
        - Tokenizes query using the same tokenizer as indexing
        - Computes BM25 scores against all documents
        - Returns top k documents by BM25 score
        
        Result Combination:
        - Results are tagged with their source ("faiss" or "bm25")
        - Duplicates are NOT removed at this stage (reranker handles this)
        - This preserves signal about which retriever found each document
        
        Args:
            query: The search query string.
            k: Number of documents to retrieve from each retriever (default: 10).
               Total results will be up to 2*k (before deduplication in rerank).
               
        Returns:
            List of tuples, each containing:
            - Document: The retrieved document
            - float: The relevance score (FAISS distance or BM25 score)
            - str: Source identifier ("faiss" or "bm25")
            
        Example:
            >>> results = retriever.search("machine learning optimization", k=5)
            >>> for doc, score, source in results:
            ...     print(f"[{source}] Score: {score:.4f} - {doc.page_content[:50]}...")
        """
        results = []
        
        # Dense retrieval with FAISS
        # similarity_search_with_score returns (Document, distance) tuples
        # Lower distance = higher similarity for L2
        faiss_results = self.vector_store.similarity_search_with_score(query, k=k)
        for doc, score in faiss_results:
            results.append((doc, score, "faiss"))
        
        # Sparse retrieval with BM25
        # BM25 returns scores for ALL documents, we take top k
        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k indices by BM25 score
        top_k_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )[:k]
        
        for idx in top_k_indices:
            doc = self.documents[idx]
            score = bm25_scores[idx]
            results.append((doc, score, "bm25"))
        
        print(f"✓ Retrieved {len(faiss_results)} from FAISS, {k} from BM25")
        return results
    
    def rerank(
        self,
        results: List[Tuple[Document, float, str]],
        query: str,
        top_n: int = 5
    ) -> List[Tuple[Document, float]]:
        """
        Rerank combined results using Cohere's cross-encoder reranker.
        
        Why Reranking?
        --------------
        Initial retrieval uses bi-encoders (separate query/doc encoding) which are fast
        but less accurate. Cross-encoders process query and document together, enabling:
        - Better understanding of query-document interaction
        - More nuanced relevance assessment
        - Correction of initial ranking errors
        
        The tradeoff is speed: cross-encoders are slower, so we only apply them to
        the candidate set from initial retrieval (typically 10-20 documents).
        
        Deduplication:
        --------------
        Documents appearing in both FAISS and BM25 results are deduplicated by
        content hash before reranking, preventing wasted API calls.
        
        Args:
            results: List of (Document, score, source) tuples from search().
            query: The original query string.
            top_n: Number of top results to return after reranking (default: 5).
            
        Returns:
            List of (Document, relevance_score) tuples, sorted by relevance.
            The relevance_score is from Cohere's reranker (0-1 scale).
            
        Raises:
            cohere.CohereError: If the Cohere API request fails.
            
        Example:
            >>> results = retriever.search("how to train a model", k=10)
            >>> reranked = retriever.rerank(results, "how to train a model", top_n=3)
            >>> for doc, score in reranked:
            ...     print(f"Score: {score:.4f} - {doc.page_content[:100]}...")
        """
        if not results:
            return []
        
        # Deduplicate by content (same document might appear in both FAISS and BM25)
        seen_content = set()
        unique_docs = []
        
        for doc, score, source in results:
            content_hash = hash(doc.page_content)
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)
        
        print(f"✓ Deduped: {len(results)} → {len(unique_docs)} unique documents")
        
        if not unique_docs:
            return []
        
        # Prepare documents for Cohere reranking
        doc_texts = [doc.page_content for doc in unique_docs]
        
        # Call Cohere Rerank API
        # rerank-v3.5 is the latest recommended model (as of 2024)
        response = self.cohere_client.rerank(
            model="rerank-v3.5",
            query=query,
            documents=doc_texts,
            top_n=min(top_n, len(unique_docs))
        )
        
        # Extract reranked results
        reranked_results = []
        for hit in response.results:
            doc = unique_docs[hit.index]
            relevance_score = hit.relevance_score
            reranked_results.append((doc, relevance_score))
        
        print(f"✓ Reranked to top {len(reranked_results)} results")
        return reranked_results


def main():
    """
    Example usage of the HybridRetriever.
    
    This demonstrates the complete pipeline:
    1. Load and chunk a PDF document
    2. Initialize the hybrid retriever
    3. Perform search and reranking
    """
    import sys
    
    # Check if PDF path is provided
    if len(sys.argv) < 2:
        print("Usage: python retrieval_engine.py <pdf_path> [query]")
        print("Example: python retrieval_engine.py document.pdf 'What is machine learning?'")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    query = sys.argv[2] if len(sys.argv) > 2 else "What is the main topic of this document?"
    
    print(f"\n{'='*60}")
    print("Hybrid Search Retrieval Engine Demo")
    print(f"{'='*60}\n")
    
    # Step 1: Load and chunk the PDF
    print(f"[1/4] Loading PDF: {pdf_path}")
    documents = load_and_chunk_pdf(pdf_path, chunk_size=500, chunk_overlap=100)
    print(f"      Created {len(documents)} chunks\n")
    
    # Step 2: Initialize the retriever
    print("[2/4] Initializing HybridRetriever...")
    retriever = HybridRetriever(documents)
    print()
    
    # Step 3: Perform search
    print(f"[3/4] Searching for: '{query}'")
    results = retriever.search(query, k=10)
    print()
    
    # Step 4: Rerank results
    print("[4/4] Reranking results with Cohere...")
    final_results = retriever.rerank(results, query, top_n=5)
    print()
    
    # Display results
    print(f"{'='*60}")
    print("Top Results:")
    print(f"{'='*60}\n")
    
    for i, (doc, score) in enumerate(final_results, 1):
        print(f"Result {i} (Score: {score:.4f})")
        print(f"Source: {doc.metadata.get('source_file', 'unknown')}, "
              f"Page: {doc.metadata.get('page', 'N/A')}, "
              f"Chunk: {doc.metadata.get('chunk_index', 'N/A')}")
        print(f"Content: {doc.page_content[:300]}...")
        print("-" * 40)
        print()


if __name__ == "__main__":
    main()
