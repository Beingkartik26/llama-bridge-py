from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from typing import List
from langchain.schema import Document

class RAGChain:
    def __init__(self, model_name: str = "llama3.2"):
        self.llm = Ollama(model=model_name)
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""Use the following pieces of context to answer the question at the end. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            
            Context: {context}
            
            Question: {question}
            
            Answer: """
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
    
    def generate_response(self, context: List[Document], question: str) -> str:
        """
        Generate response using the RAG chain
        """
        context_str = "\n\n".join([doc.page_content for doc in context])
        return self.chain.run(context=context_str, question=question) 