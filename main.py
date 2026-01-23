from src import get_agent
from src.core.orchestrator import DebateOrchestrator


def main():
    agents = {
        "gemini_1": get_agent("gemini"),
        "gemini_2": get_agent("gemini"),
        "gemini_3": get_agent("gemini"),
        "gemini_4": get_agent("gemini"),
    }

    orchestrator = DebateOrchestrator(agents)

    test_question = "In how many ways can you tile a 3x8 rectangle with 2x1 dominoes?"

    print("Role Assignment-------------------------------------------------------")
    roles = orchestrator.run_stage_0(test_question)

    for agent, role in roles.items():
        
        print(f"agent {agent} assigned to: {role}")


if __name__ == "__main__":
    main()
