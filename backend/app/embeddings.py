from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    def __init__(self):
        try:
            logger.info("Initializing HuggingFaceEmbeddings...")
            # Using a smaller model for faster download and inference
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            logger.info("HuggingFaceEmbeddings initialized successfully")
        except ImportError as e:
            logger.error(f"Error importing transformers: {str(e)}")
            logger.error("Please ensure you have installed the required packages:")
            logger.error("pip install torch==2.1.0")
            logger.error("pip install transformers==4.35.0")
            logger.error("pip install sentence-transformers==2.2.2")
            raise e
        except Exception as e:
            logger.error(f"Unexpected error during initialization: {str(e)}")
            raise e
    
    def generate_embeddings(self, documents: List[Document]) -> List[Document]:
        """
        Generate embeddings for the documents
        """
        return documents  # The embeddings will be generated when storing in Pinecone 