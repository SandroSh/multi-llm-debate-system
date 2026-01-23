import json
import os
import logging
import sys
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src import get_agent
from src.core.orchestrator import DebateOrchestrator

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

DATA_DIR = "data"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

plt.style.use('default')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3


def check_correctness(judge_agent, system_answer: str, correct_answer: str) -> bool:
   
    system_prompt = (
        "You are an impartial answer grader. Compare the system's answer with the ground truth.\n"
        "Answers are considered correct if they are semantically equivalent, even if formatted differently.\n"
        "For example:\n"
        "  - '153' and '153.0' are equivalent\n"
        "  - 'Solver_3' and 'solver 3' are equivalent\n"
        "  - '42' and 'forty-two' are equivalent\n\n"
        "Reply with ONLY a single word: 'YES' if correct, 'NO' if incorrect.\n"
        "Do not include any explanation or additional text."
    )
    
    user_prompt = (
        f"Ground Truth: {correct_answer}\n"
        f"System Answer: {system_answer}\n\n"
        f"Are these answers equivalent?"
    )
    
    try:
        response = judge_agent.generate(
            system_prompt, 
            user_prompt, 
            temperature=0.0
        )
        
        cleaned_response = response.strip().upper()
        
        print(f"ground Truth: {correct_answer}")
        print(f"system Answer: {system_answer}")
        print(f"grader Response: {cleaned_response}")
        
        if "YES" in cleaned_response:
            print("correct")
            return True
        elif "NO" in cleaned_response:
            print("incorrect")
            return False
        else:
            result = system_answer.strip().lower() == correct_answer.strip().lower()
            print(f"{result}")
            return result
            
    except Exception as e:
        result = system_answer.strip().lower() == correct_answer.strip().lower()
        print(f"fallback comparison: {result}")
        return result


def create_plots(df):
   
    plt.figure(figsize=(10, 6))
    correct_conf = df[df["is_correct"]]["confidence"]
    incorrect_conf = df[~df["is_correct"]]["confidence"]
    
    plt.hist(correct_conf, bins=10, alpha=0.6, label=f"correct (n={len(correct_conf)})", color="#10b981", edgecolor='black')
    plt.hist(incorrect_conf, bins=10, alpha=0.6, label=f"incorrect (n={len(incorrect_conf)})", color="#ef4444", edgecolor='black')
    
    plt.xlabel("Judge Confidence", fontsize=12, fontweight='bold')
    plt.ylabel("Count", fontsize=12, fontweight='bold')
    plt.title("Confidence Distribution by Correctness", fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/1_confidence_distribution.png")
    plt.close()
 
    if len(df) > 1 and 'category' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        category_stats = df.groupby('category').agg({
            'is_correct': ['mean', 'count']
        }).reset_index()
        category_stats.columns = ['category', 'accuracy', 'count']
        category_stats['accuracy'] = category_stats['accuracy'] * 100
        
        colors = ['#10b981' if acc >= 50 else '#ef4444' for acc in category_stats['accuracy']]
        bars = ax.bar(category_stats['category'], category_stats['accuracy'], color=colors, alpha=0.7, edgecolor='black')
    
        for bar, count in zip(bars, category_stats['count']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'n={int(count)}', ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel("accuracy ", fontsize=12, fontweight='bold')
        ax.set_xlabel("category", fontsize=12, fontweight='bold')
        ax.set_title("accuracy by Problem Category", fontsize=14, fontweight='bold')
        ax.set_ylim(0, 110)
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% threshold')
        ax.legend()
        plt.xticks(rotation=45, ha='right')
        plt.grid(alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f"{PLOTS_DIR}/2_accuracy_by_category.png")
        plt.close()
       
    if len(df) > 1:
        plt.figure(figsize=(10, 6))
        winner_counts = df['winner_role'].value_counts()
        
        n_colors = len(winner_counts)
        colors = plt.cm.Set3(np.linspace(0, 1, n_colors))
        
        plt.pie(winner_counts.values, labels=winner_counts.index, autopct='%1.1f%%',
               colors=colors, startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
        plt.title("Distribution of Winning Solvers", fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(f"{PLOTS_DIR}/3_winner_distribution.png")
        plt.close()
     
 
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
  
    accuracy = df["is_correct"].mean() * 100
    ax1.text(0.5, 0.6, f"{accuracy:.1f}%", ha='center', va='center', 
            fontsize=60, fontweight='bold', 
            color='#10b981' if accuracy >= 50 else '#ef4444')
    ax1.text(0.5, 0.3, "Overall Accuracy", ha='center', va='center', 
            fontsize=16, fontweight='bold')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    correct_count = df["is_correct"].sum()
    incorrect_count = len(df) - correct_count
    ax2.bar(['Correct', 'Incorrect'], [correct_count, incorrect_count],
           color=['#10b981', '#ef4444'], alpha=0.7, edgecolor='black')
    ax2.set_ylabel("Count", fontweight='bold')
    ax2.set_title("Correct vs Incorrect Answers", fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')

    avg_conf_correct = df[df["is_correct"]]["confidence"].mean() if correct_count > 0 else 0
    avg_conf_incorrect = df[~df["is_correct"]]["confidence"].mean() if incorrect_count > 0 else 0
    
    bars = ax3.bar(['Correct Answers', 'Incorrect Answers'], 
                  [avg_conf_correct, avg_conf_incorrect],
                  color=['#10b981', '#ef4444'], alpha=0.7, edgecolor='black')
    
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    ax3.set_ylabel("Average Confidence", fontweight='bold')
    ax3.set_title("Mean Confidence by Correctness", fontweight='bold')
    ax3.set_ylim(0, 1.1)
    ax3.grid(alpha=0.3, axis='y')
    
    ax4.axis('off')
    stats_data = [
        ["Total Problems", f"{len(df)}"],
        ["Correct", f"{correct_count}"],
        ["Incorrect", f"{incorrect_count}"],
        ["Accuracy", f"{accuracy:.2f}%"],
        ["Avg Confidence", f"{df['confidence'].mean():.3f}"],
        ["Confidence Std", f"{df['confidence'].std():.3f}"],
    ]
    
    table = ax4.table(cellText=stats_data, cellLoc='left',
                     colWidths=[0.6, 0.4], loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    for i in range(len(stats_data)):
        table[(i, 0)].set_facecolor('#f0f0f0')
        table[(i, 0)].set_text_props(weight='bold')
        table[(i, 1)].set_facecolor('#ffffff')
    
    ax4.set_title("Summary Statistics", fontweight='bold', fontsize=12, pad=20)
    
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/5_performance_dashboard.png")
    plt.close()


def evaluate_and_plot():
    agents = {
        "gemini_1": get_agent("gemini"),
        "gemini_2": get_agent("gemini"),
        "gemini_3": get_agent("gemini"),
        "gemini_4": get_agent("gemini"),
    }

    orchestrator = DebateOrchestrator(agents)
    grader_agent = agents["gemini_1"]

    with open(f"{DATA_DIR}/input_problems.json", "r") as f:
        problems = json.load(f)

    results = []

    for problem in tqdm(problems, desc="Evaluating problems"):
        problem_id = problem["id"]
        question = problem["question"]
        correct_answer = problem["correct_answer"]
        category = problem["category"]
        
        print(f"PROBLEM {problem_id}: {category}")
        
        try:
            verdict, history = orchestrator.run_full_debate(question)
            
            is_correct = check_correctness(
                grader_agent, 
                verdict.winning_answer, 
                correct_answer
            )
            
            results.append({
                "id": problem_id,
                "category": category,
                "question": question,
                "correct_answer": correct_answer,
                "system_answer": verdict.winning_answer,
                "winner_role": verdict.winner,
                "confidence": verdict.confidence,
                "is_correct": is_correct,
                "judge_reasoning": verdict.reasoning,
            })
            
        except Exception as e:
            print(f"ERROR processing problem {problem_id}: {e}") 
            results.append({
                "id": problem_id,
                "category": category,
                "question": question,
                "correct_answer": correct_answer,
                "system_answer": "ERROR",
                "winner_role": "Error",
                "confidence": 0.0,
                "is_correct": False,
                "judge_reasoning": str(e),
            })

  
    with open(f"{DATA_DIR}/results_raw.json", "w") as f:
        json.dump(results, f, indent=2)

    df = pd.DataFrame(results)
    accuracy = df["is_correct"].mean() * 100
    
    print('FINAL RESULTS')
    print(f"total Problems: {len(results)}")
    print(f"correct: {df['is_correct'].sum()}")
    print(f"incorrect: {(~df['is_correct']).sum()}")
    print(f"accuracy: {accuracy:.2f}%")
    print(f"average confidence: {df['confidence'].mean():.3f}")
  
    if len(df) > 0:
        print("\nAccuracy by Category:")
        category_accuracy = df.groupby('category')['is_correct'].agg(['mean', 'count'])
        category_accuracy['mean'] = category_accuracy['mean'] * 100
        category_accuracy.columns = ['Accuracy (%)', 'Count']
        print(category_accuracy)
        print()

    if len(df) > 0:
        print("\nWinner Distribution:")
        winner_stats = df['winner_role'].value_counts()
        for winner, count in winner_stats.items():
            pct = (count / len(df)) * 100
            print(f"  {winner}: {count} ({pct:.1f}%)")
        print()
    
    create_plots(df)



if __name__ == "__main__":
    evaluate_and_plot()