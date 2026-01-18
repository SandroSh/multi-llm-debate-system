from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from .base import BaseAgent

class OpenAIAgent(BaseAgent):
    def __init__(self, model: str, api_key: str):
        super().__init__(model, api_key)
        self.client = OpenAI(api_key=api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"error calling OpenAI: {e}")
            raise