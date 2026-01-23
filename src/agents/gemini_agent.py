from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

logger = logging.getLogger(__name__)

class GeminiAgent:
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.client = genai.Client(api_key=api_key)

    def _prepare_schema(self, schema_class):
        if schema_class is None:
            return None
        
        schema = schema_class.model_json_schema()
        
        def remove_additional_props(obj):
            if isinstance(obj, dict):
                obj.pop('additionalProperties', None)
                obj.pop('title', None)
                for v in obj.values():
                    remove_additional_props(v)
            elif isinstance(obj, list):
                for item in obj:
                    remove_additional_props(item)
        
        remove_additional_props(schema)
        return schema

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=3))
    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.7, response_schema: type = None):
        try:
            cleaned_schema = self._prepare_schema(response_schema)
            
            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                response_mime_type="application/json",
                response_schema=cleaned_schema, 
            )
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=user_prompt,
                config=config
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini Error: {e}")
            raise e