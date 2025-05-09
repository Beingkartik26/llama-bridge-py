from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from typing import List, AsyncGenerator
from langchain.schema import Document
import json

class RAGChain:
    def __init__(self, model_name: str = "llama3.2"):
        self.llm = Ollama(model=model_name)
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful AI assistant. Use the following pieces of context to answer the question at the end. 
            The context contains relevant excerpts from the document. Analyze the context carefully and provide a detailed answer.
            
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            Make sure to use the specific information from the context to answer the question.
            
            Context: {context}
            
            Question: {question}
            
            Answer: """
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
    
    async def generate_streaming_response(self, context: List[Document], question: str) -> AsyncGenerator[str, None]:
        """
        Generate streaming response using the RAG chain
        """
        context_str = "\n\n".join([doc.page_content for doc in context])
        
        # Log the context for debugging
        print("\nContext being sent to LLM:")
        print(context_str)
        
        # Get the prompt
        prompt = self.prompt_template.format(context=context_str, question=question)
        
        # Create the payload for Ollama
        payload = {
            "model": self.llm.model,
            "prompt": prompt,
            "stream": True
        }
        
        # Use the Ollama client directly for streaming
        async for chunk in self.llm._acreate_stream(
            api_url="http://localhost:11434/api/generate",
            payload=payload
        ):
            if chunk:  # chunk is already a string
                yield f"data: {json.dumps({'text': chunk})}\n\n"

    def generate_response(self, context: List[Document], question: str) -> str:
        """
        Generate response using the RAG chain
        """
        context_str = "\n\n".join([doc.page_content for doc in context])
        return self.chain.invoke({"context": context_str, "question": question})["text"] 