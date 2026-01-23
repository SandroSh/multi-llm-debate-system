from typing import List, Dict, Optional
from pydantic import BaseModel, Field, ConfigDict



class ChangeRecord(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    critique: str = Field(
        ...,
        description="The specific critique or feedback received"
    )
    response: str = Field(
        ...,
        description="How the solver responded to this critique"
    )
    accepted: bool = Field(
        ...,
        description="Whether the critique was accepted (True) or rejected (False)"
    )


class RolePreferences(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    role_preferences: List[str] = Field(
        ..., 
        description="A list of roles the model prefers, ordered from most to least preferred (e.g., ['Solver', 'Judge'])."
    )
    
    confidence_solver: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in performing as a Solver (0.0 to 1.0)"
    )
    
    confidence_judge: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in performing as a Judge (0.0 to 1.0)"
    )
    
    reasoning: str = Field(
        ..., 
        description="Explanation of why these roles match the model's strengths for this specific question."
    )
    
    @property
    def confidence_by_role(self) -> Dict[str, float]:
        return {
            "Solver": self.confidence_solver,
            "Judge": self.confidence_judge
        }


class SolverSolution(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
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
        ge=0.0,
        le=1.0,
        description="How confident the solver is that the answer is correct (0 to 1)."
    )
    
    changes_made: Optional[List[ChangeRecord]] = Field(
        None, 
        description="Notes describing what was changed after receiving feedback."
    )


class ErrorDetail(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
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
    model_config = ConfigDict(extra='forbid')
    
    solution_id: str = Field(
        ..., 
        description="The ID of the solution being reviewed (e.g., 'solver_2')."
    )
    
    strengths: List[str] = Field(
        ...,
        description="List of strengths identified in the solution"
    )
    
    weaknesses: List[str] = Field(
        ...,
        description="List of weaknesses identified in the solution"
    )
    
    errors: List[ErrorDetail] = Field(
        ...,
        description="Detailed list of errors found"
    )
    
    suggested_changes: List[str] = Field(
        ...,
        description="List of suggested improvements"
    )
    
    overall_assessment: str = Field(
        ..., 
        description="Overall judgment of the solution, such as correct, partly correct, or incorrect."
    )


class FinalVerdict(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    winner: str = Field(
        ..., 
        description="The ID of the solver with the best solution."
    )
    winning_answer: str = Field(
        ..., description="The final short answer produced by the winning solver."
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="How confident the judge is in this final choice (0 to 1)."
    )
    
    reasoning: str = Field(
        ..., 
        description="Explanation of why this solver was chosen over the others."
    )