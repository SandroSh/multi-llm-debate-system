from anthropic import Anthropic
from tenacity import retry, stop_after_attempt, wait_exponential
from .base import BaseAgent

class AnthropicAgent(BaseAgent):
    def __init__(self, model: str, api_key: str):
        super().__init__(model, api_key)
        self.client = Anthropic(api_key=api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
        try:
            response = self.client.messages.create(
                model=self.model,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=4096,
                temperature=temperature
            )
            return response.content[0].text
        except Exception as e:
            self.logger.error(f"error calling Anthropic: {e}")
            raise