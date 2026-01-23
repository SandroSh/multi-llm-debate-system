# Gemini Multi-Agent Debate System

This project is an experimental setup to see how multiple LLMs (using Google Gemini) can work together to solve complex math and logic problems. Instead of just asking one AI, we let them debate, review each other's work, and come to a final "judged" conclusion.

## Project Aims
* **Self-Selection:** Agents look at a problem and decide who should be the "Manager" (Judge) and who should be the "Solvers."
* **Structured Reasoning:** Forcing the LLMs to use strict JSON schemas (via Pydantic) so the data is actually usable.
* **Debate & Refinement:** Solvers propose solutions, others peer-review them, and the Judge picks the best answer.
* **Accuracy Tracking:** Scripts to run multiple problems and plot the success rate automatically.

---

## Folder Structure

```text
llm-systems-project/
├── data/               # JSON results and evaluation logs
├── plots/              # Generated accuracy graphs
├── scripts/
│   ├── evaluate_results.py  # Runs the 25-problem test bench
├── src/
│   ├── agents/
│   │   └── gemini_agent.py  # Gemini API wrapper with retry logic
│   ├── core/
│   │   ├── orchestrator.py  # The "brain" that manages the debate stages
│   │   └── schemas.py       # Pydantic models 
│   └── __init__.py
├── main.py             # Entry point for a single test run
└── requirements.txt   