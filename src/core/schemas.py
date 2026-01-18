from typing import List, Dict, Optional
from pydantic import BaseModel, Field

class RolePreferences(BaseModel):
    role_preferences: List[str] = Field(
        ..., 
        description="A list of roles the model prefers, ordered from most to least preferred (e.g., ['Solver', 'Judge'])."
    )
    confidence_by_role: Dict[str, float] = Field(
        ..., 
        description="How confident the model feels in each role, scored from 0 to 1 (e.g., Solver_1, Judge)."
    )
    reasoning: str = Field(
        ..., 
        description="Explanation of why these roles match the model’s strengths."
    )

class SolverSolution(BaseModel):
    reasoning: str = Field(
        ..., 
        description="Step-by-step explanation of how the solution was reached."
    )
    refined_solution: Optional[str] = Field(
        None, 
        description="The full final version of the solution, written clearly."
    )
    refined_answer: str = Field(
        ..., 
        description="The final short answer (for example: '42' or 'Solver 1')."
    )
    confidence: float = Field(
        ..., 
        description="How confident the solver is that the answer is correct (0 to 1)."
    )
    changes_made: Optional[List[Dict[str, str]]] = Field(
        None, 
        description="Notes describing what was changed after receiving feedback."
    )

class ErrorDetail(BaseModel):
    location: str = Field(
        ..., 
        description="Where the mistake happens (which step or section)."
    )
    error_type: str = Field(
        ..., 
        description="Type of mistake, such as logic error, math error, or made-up information."
    )
    description: str = Field(
        ..., 
        description="Clear explanation of what the mistake is."
    )
    severity: str = Field(
        ..., 
        description="How serious the error is: critical, minor, or just a suggestion."
    )

class PeerReview(BaseModel):
    solution_id: str = Field(
        ..., 
        description="The ID of the solution being reviewed (e.g., 'solver_2')."
    )
    strengths: List[str]
    weaknesses: List[str]
    errors: List[ErrorDetail]
    suggested_changes: List[str]
    overall_assessment: str = Field(
        ..., 
        description="Overall judgment of the solution, such as correct, partly correct, or incorrect."
    )

class FinalVerdict(BaseModel):
    winner: str = Field(
        ..., 
        description="The ID of the solver with the best solution."
    )
    confidence: float = Field(
        ..., 
        description="How confident the judge is in this final choice (0 to 1)."
    )
    reasoning: str = Field(
        ..., 
        description="Explanation of why this solver was chosen over the others."
    )
