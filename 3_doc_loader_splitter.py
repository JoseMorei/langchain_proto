"""
Working with Document Loaders and Text Splitters
A workflow to load documents, split them, and prepare them for retrieval.

Instructions:
- Import the necessary document loaders to work with both PDF and web content.
- Load the provided paper about LangChain architecture.
- Create two different text splitters with varying parameters.
- Compare the resulting chunks from different splitters.
- Examine the metadata preservation across splitting.
- Create a simple function to display statistics about your document chunks.

!pip install --force-reinstall --no-cache-dir tenacity==8.2.3 --user
!pip install "ibm-watsonx-ai==1.0.8" --user
!pip install "ibm-watson-machine-learning==1.0.367" --user
!pip install "langchain-ibm==0.1.7" --user
!pip install "langchain-community==0.2.10" --user
!pip install "langchain-experimental==0.0.62" --user
!pip install "langchainhub==0.1.18" --user
!pip install "langchain==0.2.11" --user
!pip install "pypdf==4.2.0" --user
!pip install "chromadb==0.4.24" --user

restart kernel
import os
os._exit(00)

"""

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

# Load the LangChain paper
paper_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/96-FDF8f7coh0ooim7NyEQ/langchain-paper.pdf"
pdf_loader = PyPDFLoader(paper_url)
pdf_document = pdf_loader.load()

# Load content from LangChain website
web_url = "https://python.langchain.com/v0.2/docs/introduction/"
web_loader = WebBaseLoader(web_url)
web_document = web_loader.load()

# Create two different text splitters
splitter_1 = CharacterTextSplitter(chunk_size=300, chunk_overlap=30, separator="\n")
splitter_2 = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", ". ", " ", ""])

# Apply both splitters to the PDF document
chunks_1 = splitter_1.split_documents(pdf_document)
chunks_2 = splitter_2.split_documents(pdf_document)

# Define a function to display document statistics
def display_document_stats(docs, name):
    """Display statistics about a list of document chunks"""
    total_chunks = len(docs)
    total_chars = sum(len(doc.page_content) for doc in docs)
    avg_chunk_size = total_chars / total_chunks if total_chunks > 0 else 0
    
    # Count unique metadata keys across all documents
    all_metadata_keys = set()
    for doc in docs:
        all_metadata_keys.update(doc.metadata.keys())
    
    # Print the statistics
    print(f"\n=== {name} Statistics ===")
    print(f"Total number of chunks: {total_chunks}")
    print(f"Average chunk size: {avg_chunk_size:.2f} characters")
    print(f"Metadata keys preserved: {', '.join(all_metadata_keys)}")
    
    if docs:
        print("\nExample chunk:")
        example_doc = docs[min(5, total_chunks-1)]  # Get the 5th chunk or the last one if fewer
        print(f"Content (first 150 chars): {example_doc.page_content[:150]}...")
        print(f"Metadata: {example_doc.metadata}")
        
        # Calculate length distribution
        lengths = [len(doc.page_content) for doc in docs]
        min_len = min(lengths)
        max_len = max(lengths)
        print(f"Min chunk size: {min_len} characters")
        print(f"Max chunk size: {max_len} characters")

# Display stats for both chunk sets
display_document_stats(chunks_1, "Splitter 1")
display_document_stats(chunks_2, "Splitter 2")