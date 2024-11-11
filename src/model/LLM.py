import os
from typing import Dict, Any, Union
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chat_models.openai import ChatOpenAI
from langchain_community.llms import SambaStudio
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

class LLMChatbot:

    def __init__(
        self,
        cloud_service: str = "groq",
        temperature: float = 0.7,
        model_name: str = "llama-3.2-3b-preview",
        system_message: str = ""
    ):
        # Load environment variables
        load_dotenv()

        self.SAMBASTUDIO_URL="https://api.sambanova.ai/v1/chat/completions"
        
        # Initialize the appropriate chat model
        if cloud_service == "groq":
            self.llm = ChatGroq(
                temperature=temperature,
                groq_api_key=os.environ.get("GROQ_API_KEY"),
                model_name=model_name
            )
        elif cloud_service == "openai":
            self.llm = ChatOpenAI(
                temperature=temperature,
                openai_api_key=os.environ.get("OPENAI_API_KEY"),
                model_name=model_name
            )
        elif cloud_service == "sambanova":
            self.llm = SambaStudio(
                temperature=temperature,
                sambastudio_api_key=os.environ.get("SAMBANOVA_API_KEY"),
                sambastudio_url=self.SAMBASTUDIO_URL,
                model_name=model_name
            )
        else:
            raise ValueError("Invalid cloud_service. Choose 'groq', 'openai', or 'sambanova'.")
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            return_messages=True
        )
        
        self.set_system_message(system_message)

    def set_system_message(self, system_message: str):
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        self.chain = self.prompt | self.llm
    
    def get_response(self, user_prompt: str) -> str:
        try:
            # Get conversation history
            history = self.memory.load_memory_variables({})
            
            # Get response from the model
            response = self.chain.invoke({
                "input": user_prompt,
                "history": history.get("history", [])
            })
            
            # Save the conversation
            self.memory.save_context(
                {"input": user_prompt},
                {"output": response.content}
            )
            
            return response.content
                
        except Exception as e:
            return f"Error procesando el mensaje: {str(e)}"
