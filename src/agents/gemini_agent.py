from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential
from .base import BaseAgent

class GeminiAgent(BaseAgent):
    def __init__(self, model: str, api_key: str):
        super().__init__(model, api_key)
        self.client = genai.Client(api_key=api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=temperature
                )
            )

            return response.text
            
        except Exception as e:
            self.logger.error(f"error calling Gemini: {e}")
            raise