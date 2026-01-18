import os
from dotenv import load_dotenv
from .agents.openai_agent import OpenAIAgent
from .agents.anthropic_agent import AnthropicAgent
from .agents.gemini_agent import GeminiAgent
from .agents.grok_agent import GrokAgent

load_dotenv()

def get_agent(agent_name: str):
    """Factory to get the correct agent instance based on name."""
    if agent_name == "gpt-4":
        return OpenAIAgent(model="gpt-4-turbo", api_key=os.getenv("OPENAI_API_KEY"))
    elif agent_name == "claude":
        return AnthropicAgent(model="claude-3-opus-20240229", api_key=os.getenv("ANTHROPIC_API_KEY"))
    elif agent_name == "gemini":
        return GeminiAgent(model="gemini-1.5-pro", api_key=os.getenv("GEMINI_API_KEY"))
    elif agent_name == "grok":
        return GrokAgent(model="grok-beta", api_key=os.getenv("XAI_API_KEY"))
    else:
        raise ValueError(f"Unknown agent: {agent_name}")