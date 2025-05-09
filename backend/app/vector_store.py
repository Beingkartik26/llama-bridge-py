from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from typing import List, Any
import os
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()

class VectorStore:
    def __init__(self, collection_name: str = "langchain_demo"):
        print("Initializing VectorStore...")
        self.collection_name = collection_name
        self.initialize_chroma()
    
    def initialize_chroma(self):
        """
        Initialize Chroma vector store
        """
        try:
            print("Initializing Chroma...")
            # Chroma will automatically create a persistent directory
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                persist_directory="./chroma_db"
            )
            print("✓ VectorStore initialization complete")
            
        except Exception as e:
            print(f"❌ Error in initialize_chroma: {str(e)}")
            logger.error(f"Error in initialize_chroma: {str(e)}")
            raise
    
    def store_documents(self, documents: List[Document], embedding_function: Any) -> None:
        """
        Store documents in Chroma
        """
        print(f"\nStoring {len(documents)} documents in vector store...")
        try:
            # Create a new vectorstore instance with the embedding function
            vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=embedding_function,
                persist_directory="./chroma_db"
            )
            
            # Add documents to Chroma
            vectorstore.add_documents(documents)
            vectorstore.persist()  # Save to disk
            print("✓ Documents stored successfully")
        except Exception as e:
            print(f"❌ Error storing documents: {str(e)}")
            raise

    def get_relevant_documents(self, query: str, embedding_function: Any, k: int = 4) -> List[Document]:
        """
        Retrieve relevant documents from Chroma
        """
        print(f"\nSearching for relevant documents for query: {query}")
        try:
            # Create a new vectorstore instance with the embedding function
            vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=embedding_function,
                persist_directory="./chroma_db"
            )
            
            # Search for similar documents
            docs = vectorstore.similarity_search(query, k=k)
            print(f"✓ Found {len(docs)} relevant documents")
            return docs
        except Exception as e:
            print(f"❌ Error retrieving documents: {str(e)}")
            raise