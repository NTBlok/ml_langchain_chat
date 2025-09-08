from typing import Dict, List, Any, Optional
from datetime import datetime
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import LLMChain
from pydantic import BaseModel

# Define data models
class ChatMessage(BaseModel):
    content: str
    sender: str  # 'user' or 'ai'
    timestamp: str

class ChatSession:
    def __init__(self, session_id: str, max_history: int = 5):
        self.session_id = session_id
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=max_history,
            return_messages=True
        )
        self.llm = ChatOllama(
            model="mistral",
            base_url="http://llm:11434",
            temperature=0.7
        )
        self.setup_prompt_templates()
        self.chain = self.create_chain()
    
    def setup_prompt_templates(self):
        """Initialize the system and human message templates."""
        self.system_prompt = """
        You are a helpful AI assistant for data analysis. Your responses should be:
        - Clear and concise
        - Focused on data interpretation
        - Honest about limitations
        - Supportive and friendly
        
        If asked about data, you can provide insights based on available information.
        """
        
        self.prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(self.system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{input}")
            ]
        )
    
    def create_chain(self) -> LLMChain:
        """Create the LangChain with memory integration."""
        return LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=self.memory,
            verbose=True
        )
    
    def get_response(self, user_input: str) -> Dict[str, Any]:
        """Generate a response to user input with context from memory."""
        try:
            response = self.chain({"input": user_input})
            return {
                "content": response["text"],
                "timestamp": datetime.now().isoformat(),
                "session_id": self.session_id
            }
        except Exception as e:
            return {
                "content": f"I encountered an error: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "session_id": self.session_id,
                "error": True
            }

# Session management
class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, ChatSession] = {}
    
    def get_session(self, session_id: str) -> ChatSession:
        """Get or create a chat session."""
        if session_id not in self.sessions:
            self.sessions[session_id] = ChatSession(session_id)
        return self.sessions[session_id]

# Initialize session manager
session_manager = SessionManager()

def get_llm_response(user_input: str, session_id: str = "default") -> Dict[str, Any]:
    """
    Get a response from the LLM with conversation context.
    
    Args:
        user_input: The user's message
        session_id: Unique identifier for the conversation session
        
    Returns:
        Dictionary containing the response and metadata
    """
    session = session_manager.get_session(session_id)
    return session.get_response(user_input)