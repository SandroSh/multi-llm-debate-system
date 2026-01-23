from typing import Dict
from .schemas import RolePreferences

class RoleManager:
    @staticmethod
    def assign_roles(assessments: Dict[str, RolePreferences]) -> Dict[str, str]:
        """
        assign roles based on LLM self-assessments
        
        highest confidence  in Judge role becomes Judge
        remaining  solver
        
        Returns
            Dict mapping agent_id to role_name
        """
        sorted_for_judge = sorted(
            assessments.items(), 
            key=lambda x: x[1].confidence_judge, 
            reverse=True
        )
        
        assignments = {}
        
        judge_id = sorted_for_judge[0][0]
        assignments[judge_id] = "Judge"
        
        remaining = sorted_for_judge[1:]
        remaining_sorted = sorted(
            remaining,
            key=lambda x: x[1].confidence_solver,
            reverse=True
        )
    
        for idx, (agent_id, _) in enumerate(remaining_sorted, start=1):
            assignments[agent_id] = f"Solver_{idx}"
            
        return assignments