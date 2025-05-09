from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional
import os
import tempfile
import logging
from .data_loader import DataLoader
from .text_splitter import TextSplitter
from .embeddings import EmbeddingGenerator
from .vector_store import VectorStore
from .rag_chain import RAGChain

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

print("Starting Llama Bridge API...")

app = FastAPI(
    title="Llama Bridge API",
    description="Backend API for Llama Bridge application",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG components
print("Initializing RAG components...")
try:
    data_loader = DataLoader()
    print("✓ DataLoader initialized")
    
    text_splitter = TextSplitter()
    print("✓ TextSplitter initialized")
    
    embedding_generator = EmbeddingGenerator()
    print("✓ EmbeddingGenerator initialized")
    
    vector_store = VectorStore()
    print("✓ VectorStore initialized")
    
    rag_chain = RAGChain()
    print("✓ RAGChain initialized")
    print("All components initialized successfully!")
except Exception as e:
    print(f"❌ Error initializing components: {str(e)}")
    logger.error(f"Error initializing components: {str(e)}")
    raise

class Query(BaseModel):
    question: str

@app.get("/")
async def root():
    return JSONResponse(
        content={
            "message": "Welcome to Llama Bridge API",
            "status": "active"
        }
    )

@app.get("/health")
async def health_check():
    return JSONResponse(
        content={
            "status": "healthy",
            "version": "1.0.0"
        }
    )

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a document (PDF or TXT)
    """
    try:
        print(f"\n📄 Processing new document: {file.filename}")
        logger.debug(f"Received file: {file.filename}")
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
            print(f"✓ Created temporary file: {temp_file_path}")
            logger.debug(f"Created temporary file: {temp_file_path}")

        # Process the document
        print("Loading document...")
        documents = data_loader.load_document(temp_file_path)
        print(f"✓ Document loaded: {len(documents)} pages/sections")
        logger.debug(f"Document loaded, got {len(documents)} pages/sections")
        
        print("Splitting documents into chunks...")
        chunks = text_splitter.split_documents(documents)
        print(f"✓ Created {len(chunks)} chunks")
        logger.debug(f"Created {len(chunks)} chunks")
        
        print("Storing documents in vector store...")
        vector_store.store_documents(chunks, embedding_generator.embeddings)
        print("✓ Documents stored successfully")
        logger.debug("Documents stored successfully")

        # Clean up
        os.unlink(temp_file_path)
        print("✓ Temporary file cleaned up")
        logger.debug("Temporary file cleaned up")

        return JSONResponse(
            content={
                "message": "Document processed successfully",
                "chunks": len(chunks)
            }
        )
    except Exception as e:
        print(f"❌ Error in upload_document: {str(e)}")
        logger.error(f"Error in upload_document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_documents(query: Query):
    """
    Query the documents using RAG with streaming response
    """
    try:
        print(f"\n🔍 Processing query: {query.question}")
        
        # Retrieve relevant documents
        print("Retrieving relevant documents...")
        relevant_docs = vector_store.get_relevant_documents(
            query.question,
            embedding_generator.embeddings
        )
        print(f"✓ Found {len(relevant_docs)} relevant documents")
        
        # Log the retrieved documents for debugging
        print("\nRetrieved Documents:")
        for i, doc in enumerate(relevant_docs):
            print(f"\nDocument {i+1}:")
            print(f"Content: {doc.page_content[:200]}...")  # Print first 200 chars
        
        # Return streaming response
        return StreamingResponse(
            rag_chain.generate_streaming_response(relevant_docs, query.question),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "Content-Type": "text/event-stream"
            }
        )
    except Exception as e:
        print(f"❌ Error in query_documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("\n🚀 Starting server...")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 