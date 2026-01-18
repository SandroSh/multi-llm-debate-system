from typing import Dict, List
from .schemas import RolePreferences

class RoleManager:
    @staticmethod
    def assign_roles(assessments: Dict[str, RolePreferences]) -> Dict[str, str]:
        """
        assign roles based on LLM self-assessments
        Returns a mapping of {agent_id: role_name}
        """
        sorted_for_judge = sorted(
            assessments.items(), 
            key=lambda x: x[1].confidence_by_role.get("Judge", 0), 
            reverse=True
        )
        
        assignments = {}
        judge_id = sorted_for_judge[0][0]
        assignments[judge_id] = "Judge"
        
        # Solver_1, Solver_2, Solver_3
        solver_count = 1
        for agent_id, _ in sorted_for_judge[1:]:
            assignments[agent_id] = f"Solver_{solver_count}"
            solver_count += 1
            
        return assignments