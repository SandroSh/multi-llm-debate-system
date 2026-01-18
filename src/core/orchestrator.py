from typing import Dict, List
from src.agents.base import BaseAgent
from src.core.schemas import RolePreferences, SolverSolution, PeerReview, FinalVerdict
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

        user_prompt = (
            f"Question: {question}\n\nProvide your role preferences and confidence."
        )

        for name, agent in self.agents.items():
            raw_response = agent.generate(system_prompt, user_prompt)
            assessments[name] = RolePreferences.model_validate_json(
                self._extract_json(raw_response)
            )

        self.role_map = RoleManager.assign_roles(assessments)
        self.reverse_role_map = {v: k for k, v in self.role_map.items()}

        return self.role_map

    def _extract_json(self, text: str) -> str:
        if "```json" in text:
            return text.split("```json")[1].split("```")[0].strip()
        return text.strip()

    def run_stage_1(self, question: str) -> Dict[str, SolverSolution]:
        """
        solvers generate independent solutions
        returns  a dict of {solver_id: SolverSolution_object}
        """
        solutions = {}

        system_prompt = (
            "You are an expert mathematical and logical reasoner"
            "Solve the given problem step-by-step"
            "Your output must be strict JSON following the schema provided"
        )

        solver_ids = [aid for aid, role in self.role_map.items() if "Solver" in role]

        for agent_id in solver_ids:
            role_name = self.role_map[agent_id]
            print(f"solution from {role_name} ({agent_id})...")

            user_prompt = (
                f"question: {question}\n\n"
                f"you are acting as {role_name}. provide a detailed solution"
            )

            agent = self.agents[agent_id]
            raw_response = agent.generate(system_prompt, user_prompt)

            try:
                solution = SolverSolution.model_validate_json(
                    self._extract_json(raw_response)
                )
                solutions[agent_id] = solution
            except Exception as e:
                print(f"error {agent_id}: {e}")

        self.history["stage_1_solutions"] = solutions
        return solutions

    def run_stage_2(
        self, question: str, solutions: Dict[str, SolverSolution]
    ) -> Dict[str, List[PeerReview]]:
        """
        each solver critiques the other two
        returns a dict of {reviewer_id: [PeerReview_1, PeerReview_2]}
        """
        reviews = {}

        system_prompt = (
            "You are a critical reviewer in a debate system"
            "Analyze the provided solution for logical gaps, calculation errors, or missed constraints"
            "Be harsh but fair. Output strict JSON"
        )

        solver_ids = [aid for aid, role in self.role_map.items() if "Solver" in role]

        for reviewer_id in solver_ids:
            reviewer_role = self.role_map[reviewer_id]
            reviews[reviewer_id] = []

            peers = [sid for sid in solver_ids if sid != reviewer_id]

            for peer_id in peers:
                peer_role = self.role_map[peer_id]
                peer_solution = solutions[peer_id]

                print(f"{reviewer_role} reviewing {peer_role}")

                user_prompt = (
                    f"Original Question: {question}\n"
                    f"Solution to Review (from {peer_role}):\n"
                    f"Answer: {peer_solution.refined_answer}\n"
                    f"Reasoning: {peer_solution.reasoning}\n\n"
                    f"Evaluate this solution."
                )

                agent = self.agents[reviewer_id]
                raw_response = agent.generate(system_prompt, user_prompt)

                try:
                    review = PeerReview.model_validate_json(
                        self._extract_json(raw_response)
                    )
                    review.solution_id = peer_role
                    reviews[reviewer_id].append(review)
                except Exception as e:
                    print(f"error {reviewer_id}: {e}")

        self.history["stage_2_reviews"] = reviews
        return reviews
    def run_stage_3(self, question: str, 
                    initial_solutions: Dict[str, SolverSolution], 
                    reviews: Dict[str, List[PeerReview]]) -> Dict[str, SolverSolution]:
        """
        solvers update their work based on peer feedback
        returns a dict of {solver_id: SolverSolution_object}
        """
        refined_solutions = {}
        
        system_prompt = (
            "You are a flexible problem solver. "
            "Review the critiques from your peers. "
            "If they are correct, fix your solution. If they are wrong, defend your stance. "
            "Output your FINAL updated solution in strict JSON."
        )
        
        solver_ids = [aid for aid, role in self.role_map.items() if "Solver" in role]
        
        for agent_id in solver_ids:
            role_name = self.role_map[agent_id]
            print(f"{role_name} is refining their solution...")
            
            incoming_critiques = []
            for reviewer_id, review_list in reviews.items():
                for review in review_list:
                    if review.solution_id == role_name:
                        incoming_critiques.append(review)
            
            critiques_text = ""
            for i, c in enumerate(incoming_critiques):
                critiques_text += f"critique {i+1}:\n{c.description}\nErrors noted: {c.errors}\n\n"

            user_prompt = (
                f"Original Question: {question}\n"
                f"Your Original Answer: {initial_solutions[agent_id].refined_answer}\n"
                f"Your Original Reasoning: {initial_solutions[agent_id].reasoning}\n\n"
                f"Peer Reviews Received:\n{critiques_text}\n"
                f"Based on this feedback, provide your verified, final solution."
            )
            
            agent = self.agents[agent_id]
            raw_response = agent.generate(system_prompt, user_prompt)
            
            try:
                refined = SolverSolution.model_validate_json(self._extract_json(raw_response))
                refined_solutions[agent_id] = refined
            except Exception as e:
                print(f"error parsing refined solution from {agent_id}: {e}")
                refined_solutions[agent_id] = initial_solutions[agent_id]

        self.history['stage_3_refined'] = refined_solutions
        return
    
    def run_stage_4(self, question: str, refined_solutions: Dict[str, SolverSolution]) -> FinalVerdict:
        """
        judge picks winner
        """
        
        judge_id = [aid for aid, role in self.role_map.items() if role == "Judge"][0]
        judge_agent = self.agents[judge_id]
        
        system_prompt = (
            "You are the Final Judge in a multi-agent debate. "
            "You must evaluate the final solutions provided by three solvers. "
            "Select the single correct answer and explain why others might be wrong. "
            "Output strict JSON."
        )
        
        candidates_text = ""
        for agent_id, sol in refined_solutions.items():
            role = self.role_map[agent_id]
            candidates_text += (
                f"--- {role} ---\n"
                f"Answer: {sol.refined_answer}\n"
                f"Confidence: {sol.confidence}\n"
                f"Reasoning: {sol.reasoning}\n\n"
            )
            
        user_prompt = (
            f"Question: {question}\n\n"
            f"Final Solutions Submitted:\n{candidates_text}\n"
            f"Determine the winner."
        )
        
        raw_response = judge_agent.generate(system_prompt, user_prompt)
        
        try:
            verdict = FinalVerdict.model_validate_json(self._extract_json(raw_response))
            self.history['stage_4_verdict'] = verdict
            return verdict
        except Exception as e:
            print(f"error {e}")
            return FinalVerdict(winner="Error", confidence=0.0, reasoning="parsing failed")
        
    def run_full_debate(self, question: str):
        self.history = {}
        
        # assign roles
        self.run_stage_0(question)
        
        # independent solutions
        initial_solutions = self.run_stage_1(question)
        
        # rreview
        reviews = self.run_stage_2(question, initial_solutions)
        
        # refinement
        refined_solutions = self.run_stage_3(question, initial_solutions, reviews)
        
        # judgment
        verdict = self.run_stage_4(question, refined_solutions)
        
        return verdict, self.history