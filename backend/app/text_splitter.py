from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List
import re

class TextSplitter:
    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 400):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            # Prioritize natural breaks in the text
            separators=[
                "\n\n",  # Double newline (paragraph break)
                "\n",    # Single newline
                ". ",    # Sentence end
                "! ",    # Exclamation
                "? ",    # Question
                "; ",    # Semicolon
                ": ",    # Colon
                ", ",    # Comma
                " ",     # Space
                ""       # Character
            ],
            is_separator_regex=False
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks while preserving semantic meaning
        """
        final_docs = []
        for doc in documents:
            # Clean the text
            content = doc.page_content.strip()
            
            # Split into chunks
            chunks = self.text_splitter.split_text(content)
            
            # Create documents with metadata
            for chunk in chunks:
                if chunk.strip():  # Only add non-empty chunks
                    chunk_doc = Document(
                        page_content=chunk.strip(),
                        metadata=doc.metadata.copy()
                    )
                    final_docs.append(chunk_doc)
        
        return final_docs 