from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from typing import List, Any, Optional
import os
from dotenv import load_dotenv
import logging
import chromadb
from chromadb import Settings
from chromadb.utils.batch_utils import create_batches
import uuid

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()

class VectorStore:
    def __init__(self, collection_name: str = None):
        print("Initializing VectorStore...")
        # Generate a unique collection name if none provided
        self.collection_name = collection_name or f"collection_{uuid.uuid4()}"
        self.initialize_chroma()
    
    def initialize_chroma(self):
        """
        Initialize Chroma vector store with improved settings
        """
        try:
            print("Initializing Chroma...")
            # Initialize with better settings
            settings = Settings(
                allow_reset=True,
                anonymized_telemetry=False,
                is_persistent=True
            )
            
            # Create persistent client
            self.client = chromadb.PersistentClient(
                path="./chroma_db",
                settings=settings
            )
            
            # Delete existing collection if it exists
            try:
                self.client.delete_collection(name=self.collection_name)
                print(f"Deleted existing collection: {self.collection_name}")
            except Exception as e:
                print(f"No existing collection to delete: {str(e)}")
            
            # Create new collection with cosine similarity
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            print(f"✓ VectorStore initialization complete with collection: {self.collection_name}")
            
        except Exception as e:
            print(f"❌ Error in initialize_chroma: {str(e)}")
            logger.error(f"Error in initialize_chroma: {str(e)}")
            raise
    
    def store_documents(self, documents: List[Document], embedding_function: Any) -> None:
        """
        Store documents in Chroma with improved batching
        """
        print(f"\nStoring {len(documents)} documents in vector store...")
        try:
            # Prepare documents for storage
            items = []
            for i, doc in enumerate(documents):
                # Generate embedding for the document
                embedding = embedding_function.embed_query(doc.page_content)
                
                # Create a vector item with unique ID
                item = {
                    "id": f"doc_{uuid.uuid4()}",
                    "text": doc.page_content,
                    "vector": embedding,
                    "metadata": doc.metadata
                }
                items.append(item)
                print(f"\nPrepared document {i+1}:")
                print(f"Content preview: {doc.page_content[:200]}...")
                print(f"Metadata: {doc.metadata}")
            
            # Store in batches
            ids = [item["id"] for item in items]
            documents = [item["text"] for item in items]
            embeddings = [item["vector"] for item in items]
            metadatas = [item["metadata"] for item in items]
            
            print("\nStoring documents in batches...")
            for batch in create_batches(
                api=self.client,
                documents=documents,
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas,
            ):
                self.collection.add(*batch)
            
            # Verify documents were stored
            count = self.collection.count()
            print(f"\n✓ Documents stored successfully. Collection now contains {count} documents.")
            
            # Print a sample of stored documents
            print("\nVerifying stored documents:")
            result = self.collection.get()
            if result and result["documents"]:
                print(f"\nFound {len(result['documents'])} documents in collection:")
                for i, doc in enumerate(result["documents"][:3]):  # Show first 3 documents
                    print(f"\nDocument {i+1}:")
                    print(f"Content preview: {doc[:200]}...")
                    if result["metadatas"] and result["metadatas"][i]:
                        print(f"Metadata: {result['metadatas'][i]}")
            else:
                print("No documents found in collection after storage!")
                
        except Exception as e:
            print(f"❌ Error storing documents: {str(e)}")
            raise

    def get_relevant_documents(self, query: str, embedding_function: Any, k: int = 12) -> List[Document]:
        """
        Retrieve relevant documents with flexible search capabilities
        """
        print(f"\nSearching for relevant documents for query: {query}")
        try:
            # Check collection status
            count = self.collection.count()
            print(f"Collection contains {count} documents")
            
            if count == 0:
                print("Warning: Collection is empty! No documents to search.")
                return []
            
            # Generate query embedding
            query_embedding = embedding_function.embed_query(query)
            print("Generated query embedding")
            
            # First try semantic search without filters
            print("Performing semantic search...")
            result = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k
            )
            
            # Log the found documents
            if result and result["documents"]:
                print("\nFound documents:")
                for i, doc in enumerate(result["documents"][0]):
                    print(f"\nDocument {i+1}:")
                    print(f"Content: {doc[:200]}...")  # Print first 200 chars
                    if result["metadatas"] and result["metadatas"][0][i]:
                        print(f"Metadata: {result['metadatas'][0][i]}")
                    if result["distances"] and result["distances"][0]:
                        print(f"Similarity score: {result['distances'][0][i]}")
            else:
                print("No documents found in search results!")
            
            # Convert results to Documents
            documents = []
            if result and result["documents"]:
                for i in range(len(result["documents"][0])):
                    doc = Document(
                        page_content=result["documents"][0][i],
                        metadata=result["metadatas"][0][i] if result["metadatas"] else {}
                    )
                    documents.append(doc)
            
            print(f"\n✓ Found {len(documents)} relevant documents")
            return documents
            
        except Exception as e:
            print(f"❌ Error retrieving documents: {str(e)}")
            raise
    
    def delete_collection(self):
        """
        Delete the collection
        """
        try:
            self.client.delete_collection(name=self.collection_name)
            print(f"✓ Collection {self.collection_name} deleted")
        except Exception as e:
            print(f"❌ Error deleting collection: {str(e)}")
            raise
        
     