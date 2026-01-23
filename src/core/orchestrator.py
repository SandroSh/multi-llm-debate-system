from typing import Dict, List
from src.agents.base import BaseAgent
from src.core.schemas import RolePreferences, SolverSolution, PeerReview, FinalVerdict
from src.core.role_manager import RoleManager


class DebateOrchestrator:
    PERSONAS = {
        "Solver_1": "You are a First-Principles Thinker. Break the problem into its most basic logical components and build the solution from the ground up. Avoid assumptions.",
        "Solver_2": "You are a Skeptical Critic. Assume there is a hidden trap or a common misconception in the problem. Look for edge cases and verify every calculation twice.",
        "Solver_3": "You are a Creative Strategist. Look for elegant shortcuts, symmetries, or unconventional logical paths that others might miss, while maintaining strict mathematical rigor.",
    }

    def __init__(self, agents: Dict[str, BaseAgent]):

        self.agents = agents
        self.role_map = {}
        self.reverse_role_map = {}
        self.history = {}

    def _extract_json(self, text: str) -> str:

        if "```json" in text:
            return text.split("```json")[1].split("```")[0].strip()
        return text.strip()

    def run_stage_0(self, question: str) -> Dict[str, str]:
        """
        self-assessment and role assignment.
        rach agent evaluates which role suits them best for the given question
        Parameters:
            question
        Returns:
            Dictionary mapping agent IDs to assigned roles
        """
        assessments = {}

        system_prompt = (
            "You are participating in a multi-LLM debate system. "
            "You will be assigned one of two role types:\n"
            "1. SOLVER: Independently solve the problem, receive peer critiques, and refine your solution\n"
            "2. JUDGE: Evaluate all final solutions after peer review and select the best one\n\n"
            "Assess your suitability for each role based on the given question."
        )

        user_prompt = (
            f"Question: {question}\n\n"
            "For this specific question, provide:\n"
            "1. confidence_solver: Your confidence (0.0 to 1.0) in solving this problem independently\n"
            "2. confidence_judge: Your confidence (0.0 to 1.0) in evaluating and comparing solutions\n"
            "3. role_preferences: Your preferred roles in order, e.g., ['Solver', 'Judge']\n"
            "4. reasoning: Explain why you'd be good at each role for THIS question\n\n"
            "Consider the problem type, your strengths, and what each role requires."
        )

        for agent_id, agent in self.agents.items():
            print(f"\n[{agent_id}] Requesting self-assessment...")
            try:
                raw_response = agent.generate(
                    system_prompt,
                    user_prompt,
                    temperature=0.1,
                    response_schema=RolePreferences,
                )

                json_str = self._extract_json(raw_response)
                assessment = RolePreferences.model_validate_json(json_str)
                assessments[agent_id] = assessment

                print(f"solver confidence: {assessment.confidence_solver:.2f}")
                print(f"judge confidence:  {assessment.confidence_judge:.2f}")
                print(f"preferences: {assessment.role_preferences}")

            except Exception as e:
                assessments[agent_id] = RolePreferences(
                    role_preferences=["Solver", "Judge"],
                    confidence_solver=0.5,
                    confidence_judge=0.5,
                    reasoning=f"Fallback due to error: {str(e)}",
                )
        try:
            self.role_map = RoleManager.assign_roles(assessments)
            self.reverse_role_map = {v: k for k, v in self.role_map.items()}

            roles = list(self.role_map.values())
            judge_count = sum(1 for r in roles if r == "Judge")
            solver_count = sum(1 for r in roles if r.startswith("Solver_"))

            if judge_count != 1 or solver_count != 3:
                raise ValueError("invalid role distribution")

            print("role assignment successful:")
            for agent_id, role in sorted(self.role_map.items(), key=lambda x: x[1]):
                assessment = assessments[agent_id]
                confidence = (
                    assessment.confidence_judge
                    if role == "Judge"
                    else assessment.confidence_solver
                )
                print(f"{agent_id:12} - {role:10} (confidence: {confidence:.2f})")

            return self.role_map

        except Exception as e:
            print("fallback role distribution...\n")
            return self._default_role_assignment()

    def run_stage_1(self, question: str) -> Dict[str, SolverSolution]:
        """
        independent solution generation.
        each solver generates their solution independently with their assigned persona
        parameters:
            question: The problem to be solved
        returns:
            dictionary mapping agent IDs to their SolverSolution objects
        """
        solutions = {}
        solver_ids = [aid for aid, role in self.role_map.items() if "Solver" in role]

        for agent_id in solver_ids:
            role_name = self.role_map[agent_id]
            persona_instruction = self.PERSONAS.get(
                role_name, "You are an expert reasoner."
            )

            system_prompt = (
                f"{persona_instruction} "
                "solve the given problem step-by-step. "
                "your output must be strict JSON following the schema provided."
            )

            print(f"requesting solution from {role_name}...")
            user_prompt = f"question: {question}\n\nyou are acting as {role_name}. provide a detailed solution."

            agent = self.agents[agent_id]
            raw_response = agent.generate(
                system_prompt, user_prompt, response_schema=SolverSolution
            )

            try:
                solution = SolverSolution.model_validate_json(
                    self._extract_json(raw_response)
                )
                solutions[agent_id] = solution
            except Exception as e:
                print(f"Error parsing solution from {role_name}: {e}")

        self.history["stage_1_solutions"] = solutions
        return solutions

    def run_stage_2(
        self, question: str, solutions: Dict[str, SolverSolution]
    ) -> Dict[str, List[PeerReview]]:
        """
        peer review
        each solver reviews the solutions of the othes
        parameters:
            question: The original problem
            solutions: Dictionary of solutions from stage 1

        Returns:
            Dictionary mapping reviewer agent IDs to lists of PeerReview objects
        """
        reviews = {}
        solver_ids = [aid for aid, role in self.role_map.items() if "Solver" in role]

        for reviewer_id in solver_ids:
            reviewer_role = self.role_map[reviewer_id]
            reviews[reviewer_id] = []
            peers = [sid for sid in solver_ids if sid != reviewer_id]

            system_prompt = (
                f"You are {reviewer_role}. You are now acting as a Peer Reviewer. "
                "Analyze the provided solution for logical gaps, calculation errors, or missed constraints. "
                "Be harsh but fair. Output strict JSON."
            )

            for peer_id in peers:
                peer_role = self.role_map[peer_id]
                peer_solution = solutions[peer_id]
                print(f"{reviewer_role} reviewing {peer_role}...")

                user_prompt = (
                    f"Original Question: {question}\n"
                    f"Solution to Review (from {peer_role}):\n"
                    f"Answer: {peer_solution.refined_answer}\n"
                    f"Reasoning: {peer_solution.reasoning}\n\n"
                    "Evaluate this solution."
                )

                agent = self.agents[reviewer_id]
                raw_response = agent.generate(
                    system_prompt, user_prompt, response_schema=PeerReview
                )

                try:
                    review = PeerReview.model_validate_json(
                        self._extract_json(raw_response)
                    )
                    review.solution_id = peer_role
                    reviews[reviewer_id].append(review)
                except Exception as e:
                    print(f"Error parsing review from {reviewer_role}: {e}")

        self.history["stage_2_reviews"] = reviews
        return reviews

    def run_stage_3(
        self,
        question: str,
        initial_solutions: Dict[str, SolverSolution],
        reviews: Dict[str, List[PeerReview]],
    ) -> Dict[str, SolverSolution]:
        """
        solution refinement
        each solver reviews peer feedback and refines their solution accordingly
        parameters:
            question: The original problem
            initial_solutions: Original solutions from stage 1
            reviews: Peer reviews from stage 2
        Returns:
            Dictionary mapping agent IDs to refined SolverSolution objects
        """
        refined_solutions = {}
        solver_ids = [aid for aid, role in self.role_map.items() if "Solver" in role]

        for agent_id in solver_ids:
            role_name = self.role_map[agent_id]
            persona_instruction = self.PERSONAS.get(role_name, "")

            system_prompt = (
                f"{persona_instruction} You are a flexible problem solver. "
                "Review the critiques from your peers carefully. "
                "For each critique:\n"
                "- If it's valid, incorporate the fix into your solution\n"
                "- If it's invalid, explain why you reject it\n\n"
                "Output your FINAL updated solution in strict JSON format.\n"
                "In 'changes_made', provide a list of objects with:\n"
                "  - critique: the feedback you received\n"
                "  - response: how you addressed it\n"
                "  - accepted: true if you accepted it, false if you rejected it"
            )

            print(f"\n[{role_name}] Refining solution based on peer feedback...")

            incoming_critiques = []
            for reviewer_id, review_list in reviews.items():
                for review in review_list:
                    if review.solution_id == role_name:
                        incoming_critiques.append(review)

            critiques_text = ""
            for i, c in enumerate(incoming_critiques, 1):
                reviewer_role = self.role_map[
                    [rid for rid, rlist in reviews.items() if c in rlist][0]
                ]
                critiques_text += f"\n--- Critique {i} (from {reviewer_role}) ---\n"
                critiques_text += f"Overall: {c.overall_assessment}\n"
                critiques_text += f"Strengths: {', '.join(c.strengths)}\n"
                critiques_text += f"Weaknesses: {', '.join(c.weaknesses)}\n"

                if c.errors:
                    critiques_text += "Errors identified:\n"
                    for err in c.errors:
                        critiques_text += (
                            f"  - [{err.severity}] {err.location}: {err.description}\n"
                        )

                if c.suggested_changes:
                    critiques_text += "Suggested changes:\n"
                    for change in c.suggested_changes:
                        critiques_text += f"  - {change}\n"
                critiques_text += "\n"

            user_prompt = (
                f"Original Question: {question}\n\n"
                f"Your Original Answer: {initial_solutions[agent_id].refined_answer}\n"
                f"Your Original Reasoning:\n{initial_solutions[agent_id].reasoning}\n\n"
                f"{'=' * 70}\n"
                f"PEER REVIEWS RECEIVED:\n"
                f"{'=' * 70}\n"
                f"{critiques_text}\n"
                f"{'=' * 70}\n\n"
                "Based on these reviews, provide your verified, final solution.\n"
                "Address each significant critique explicitly in your 'changes_made' field."
            )

            agent = self.agents[agent_id]
            try:
                raw_response = agent.generate(
                    system_prompt,
                    user_prompt,
                    temperature=0.3,
                    response_schema=SolverSolution,
                )

                refined = SolverSolution.model_validate_json(
                    self._extract_json(raw_response)
                )
                refined_solutions[agent_id] = refined

                print(f"Refined answer: {refined.refined_answer}")
                print(f"Confidence: {refined.confidence:.2f}")
                if refined.changes_made:
                    print(f"Changes made: {len(refined.changes_made)}")

            except Exception as e:
                print(f"Error refining solution: {e}")
                refined_solutions[agent_id] = initial_solutions[agent_id]

        self.history["stage_3_refined"] = refined_solutions
        print("=" * 70 + "\n")
        return refined_solutions

    def run_stage_4(
        self,
        question: str,
        initial_solutions: Dict[str, SolverSolution],
        reviews: Dict[str, List[PeerReview]],
        refined_solutions: Dict[str, SolverSolution],
    ) -> FinalVerdict:
        """
        final judgment
        the judge evaluates all solutions and selects the best one
        Parameters:
            question: The original problem
            initial_solutions: Original solutions from stage 1
            reviews: Peer reviews from stage 2
            refined_solutions: Refined solutions from stage 3
        Returns
            FinalVerdict object containing the winner and reasoning
        """
        judge_id = [aid for aid, role in self.role_map.items() if role == "Judge"][0]
        judge_agent = self.agents[judge_id]

        system_prompt = (
            "You are the Final Judge in a multi-LLM debate system. "
            "You will receive:\n"
            "1. Three original solutions from independent Solvers\n"
            "2. Peer reviews each Solver received\n"
            "3. Three refined solutions after incorporating feedback\n\n"
            "Your task: Select the BEST final solution based on:\n"
            "- Logical correctness\n"
            "- How well critiques were addressed\n"
            "- Mathematical rigor\n"
            "- Clarity of reasoning\n\n"
            "Be objective and analytical. Output strict JSON."
        )

        context = f"ORIGINAL QUESTION:\n{question}\n\n"
        context += "=" * 70 + "\n\n"

        for agent_id in initial_solutions.keys():
            role = self.role_map[agent_id]

            context += f"{'=' * 70}\n"
            context += f"{role.upper()} - ORIGINAL SOLUTION\n"
            context += f"{'=' * 70}\n"
            orig = initial_solutions[agent_id]
            context += f"Answer: {orig.refined_answer}\n"
            context += f"Confidence: {orig.confidence}\n"
            context += f"Reasoning:\n{orig.reasoning}\n\n"

            context += f"PEER REVIEWS RECEIVED BY {role}:\n"
            context += "-" * 70 + "\n"

            review_count = 0
            for reviewer_id, review_list in reviews.items():
                for review in review_list:
                    if review.solution_id == role:
                        reviewer_role = self.role_map[reviewer_id]
                        review_count += 1
                        context += f"\nReview #{review_count} (from {reviewer_role}):\n"
                        context += f"  Overall: {review.overall_assessment}\n"
                        context += f"  Strengths: {', '.join(review.strengths)}\n"
                        context += f"  Weaknesses: {', '.join(review.weaknesses)}\n"
                        if review.errors:
                            context += f"  Errors found: {len(review.errors)}\n"
                            for err in review.errors:
                                context += f"    - [{err.severity}] {err.description}\n"

            context += "\n"

            context += f"{role.upper()} - REFINED SOLUTION\n"
            refined = refined_solutions[agent_id]
            context += f"Final Answer: {refined.refined_answer}\n"
            context += f"Confidence: {refined.confidence}\n"
            context += f"Reasoning:\n{refined.reasoning}\n"

            if refined.changes_made:
                context += "\nChanges Made:\n"
                for change in refined.changes_made:
                    status = "Accepted" if change.accepted else "Rejected"
                    context += f"  [{status}] {change.critique}\n"
                    context += f"      Response: {change.response}\n"

        user_prompt = (
            f"{context}\n"
            "Based on all the above information, determine which Solver has the BEST final solution. "
            "Consider the quality of their original work, how they responded to critiques, "
            "and the correctness of their refined answer."
        )

        raw_response = judge_agent.generate(
            system_prompt, user_prompt, temperature=0.2, response_schema=FinalVerdict
        )

        try:
            verdict = FinalVerdict.model_validate_json(self._extract_json(raw_response))

            winner_agent_id = None
            for agent_id, role in self.role_map.items():
                if role.lower() == verdict.winner.lower():
                    winner_agent_id = agent_id
                    break

            if winner_agent_id is None:
                raise ValueError(
                    f"Could not find agent for winner role: {verdict.winner}"
                )

            winning_answer = refined_solutions[winner_agent_id].refined_answer

            verdict = FinalVerdict(
                winner=verdict.winner,
                winning_answer=winning_answer,
                confidence=verdict.confidence,
                reasoning=verdict.reasoning,
            )

            self.history["stage_4_verdict"] = verdict

            print(f"\n{'=' * 70}")
            print(f"VERDICT: {verdict.winner}")
            print(f"WINNING ANSWER: {verdict.winning_answer}")
            print(f"Confidence: {verdict.confidence:.2f}")
            print(f"Reasoning: {verdict.reasoning}")
            print(f"{'=' * 70}\n")

            return verdict

        except Exception as e:
            print(f"Error parsing verdict: {e}")
            best_solver = max(refined_solutions.items(), key=lambda x: x[1].confidence)
            role = self.role_map[best_solver[0]]
            winning_answer = best_solver[1].refined_answer

            return FinalVerdict(
                winner=role,
                winning_answer=winning_answer,
                confidence=best_solver[1].confidence,
                reasoning=f"Fallback selection: highest confidence solver after parsing error: {str(e)}",
            )

    def run_full_debate(self, question: str):
        """
        Parameters:
            question: The problem to be solved
        Returns:
            Tuple of (verdict, history) where verdict is the FinalVerdict and
            history contains all intermediate results from each stage
        """
        self.history = {}

        self.run_stage_0(question)

        initial_solutions = self.run_stage_1(question)

        reviews = self.run_stage_2(question, initial_solutions)

        refined_solutions = self.run_stage_3(question, initial_solutions, reviews)

        verdict = self.run_stage_4(
            question, initial_solutions, reviews, refined_solutions
        )

        return verdict, self.history
