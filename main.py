from src import get_agent
from src.core.orchestrator import DebateOrchestrator

def main():
    agents = {
        "gpt4": get_agent("gpt-4"),
        "claude": get_agent("claude"),
        "gemini": get_agent("gemini"),
        "grok": get_agent("grok")
    }

    orchestrator = DebateOrchestrator(agents)
    
    test_question = "In how many ways can you tile a 3x8 rectangle with 2x1 dominoes?"
    
    print("Role Assignment-------------------------------------------------------")
    roles = orchestrator.run_stage_0(test_question)
    
    for agent, role in roles.items():
        print(f"Agent {agent} assigned to: {role}")

if __name__ == "__main__":
    main()