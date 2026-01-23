from abc import ABC, abstractmethod
import logging

class BaseAgent(ABC):
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key
        self.logger = logging.getLogger(f"agent.{model}")

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.7, response_schema: type = None) -> str:
        pass