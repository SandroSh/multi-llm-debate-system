from typing import Dict
from src.agents.base import BaseAgent
from src.core.schemas import RolePreferences
from src.core.role_manager import RoleManager

class DebateOrchestrator:
    def __init__(self, agents: Dict[str, BaseAgent]):
        self.agents = agents
        self.role_map = {}
        self.reverse_role_map = {} 
        self.history = {}

    def run_stage_0(self, question: str):
        assessments = {}
        
        system_prompt = (
            "You are a participant in a multi-LLM debate"
            "Analyze the following question and determine if you should be a 'Solver' "
            "or a 'Judge'. Respond ONLY in JSON format."
        )
        
        user_prompt = f"Question: {question}\n\nProvide your role preferences and confidence."

        for name, agent in self.agents.items():
            raw_response = agent.generate(system_prompt, user_prompt)
            assessments[name] = RolePreferences.model_validate_json(self._extract_json(raw_response))

        self.role_map = RoleManager.assign_roles(assessments)
        self.reverse_role_map = {v: k for k, v in self.role_map.items()}
        
        return self.role_map

    def _extract_json(self, text: str) -> str:
        if "```json" in text:
            return text.split("```json")[1].split("```")[0].strip()
        return text.strip()