import os
from dotenv import load_dotenv
from .agents.gemini_agent import GeminiAgent
load_dotenv()

def get_agent(agent_name: str):

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")
    
    return GeminiAgent(model="gemini-2.5-flash", api_key=api_key)