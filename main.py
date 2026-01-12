import os
from dotenv import load_dotenv
from retrieval_engine import HybridRetriever, load_and_chunk_pdf

# ---------------------------------------------------------
# SETUP: Make sure you have your API Keys ready!
# Create a .env file with:
# COHERE_API_KEY=your_cohere_api_key_here
# ---------------------------------------------------------

# Load environment variables from .env file
load_dotenv()

def main():
    # 1. Load Data
    pdf_path = "test_doc.pdf"  # <--- Make sure this file exists!
    print(f"📄 Loading {pdf_path}...")
    
    try:
        docs = load_and_chunk_pdf(pdf_path)
        print(f"✅ Loaded {len(docs)} chunks.")
    except Exception as e:
        print(f"❌ Error loading PDF: {e}")
        return

    # 2. Initialize Retriever
    print("\n⚙️ Initializing Hybrid Retriever (FAISS + BM25)...")
    try:
        retriever = HybridRetriever(docs)
    except RuntimeError as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you have a .env file with COHERE_API_KEY set!")
        return

    # 3. Test Query
    query = "What is the main optimization technique mentioned?"
    print(f"\n🔍 Testing Query: '{query}'")

    # 4. Run Hybrid Search + Rerank
    search_results = retriever.search(query, k=5)
    results = retriever.rerank(
        results=search_results, 
        query=query, 
        top_n=3
    )

    # 5. Print Results
    # Note: rerank returns list of (Document, relevance_score) tuples
    print("\n🏆 Top Results:")
    print("=" * 60)
    for idx, (doc, score) in enumerate(results, 1):
        print(f"\n{idx}. [Score: {score:.4f}]")
        print(f"   Source: Page {doc.metadata.get('page', 'N/A')}, Chunk {doc.metadata.get('chunk_index', 'N/A')}")
        print(f"   Content: {doc.page_content[:200]}...")
        print("-" * 60)

if __name__ == "__main__":
    main()