from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.schema import Document

class DataLoader:
    def __init__(self):
        self.supported_extensions = {
            '.pdf': PyPDFLoader,
            '.txt': TextLoader
        }
    
    def load_document(self, file_path: str) -> List[Document]:
        """
        Load a document based on its file extension
        """
        file_extension = file_path.lower().split('.')[-1]
        if f'.{file_extension}' not in self.supported_extensions:
            raise ValueError(f"Unsupported file extension: {file_extension}")
        
        loader_class = self.supported_extensions[f'.{file_extension}']
        loader = loader_class(file_path)
        return loader.load() 