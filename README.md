# AI Prompt Evolution System

> **Automatically evolves LLM prompts using evolutionary algorithms** — mutation, selection, and ranking — to optimize for accuracy, reasoning quality, and coherence.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live-orange)](https://pranaymahendrakar.github.io/ai-prompt-evolution-system/)

---

## Overview

The **AI Prompt Evolution System** applies evolutionary computation principles to automatically discover high-performing prompts for Large Language Models (LLMs). Instead of manually crafting prompts, the system iteratively evolves a population of prompt candidates through selection, mutation, and fitness evaluation — converging toward prompts that maximize model output quality.

---

## How It Works

The system implements a **Genetic Algorithm (GA)** loop for prompt optimization:

```
Initial Population
       |
       v
[1] Generate Prompts  ──► Template, Few-Shot, CoT, Structured
       |
       v
[2] LLM Inference     ──► Run each prompt through the model
       |
       v
[3] Evaluate Quality  ──► Accuracy + Reasoning + Coherence
       |
       v
[4] Rank & Select     ──► Elitism + Tournament Selection
       |
       v
[5] Mutate            ──► Prefix Injection, Paraphrase, Crossover
       |
       v
Repeat for N Generations  ──► Convergence to Best Prompt
```

---

## Architecture

```
ai-prompt-evolution-system/
├── prompt_generator.py      # Generates initial diverse prompt populations
├── evaluation_engine.py     # Multi-metric evaluation (accuracy, reasoning, coherence)
├── mutation_engine.py       # Evolutionary mutation and crossover operators
├── ranking_system.py        # Selection, ranking, and evolution history tracking
├── evolution_pipeline.py    # Orchestrates the full evolutionary loop
├── docs/
│   └── index.html           # GitHub Pages: evolution dashboard
└── README.md
```

---

## Modules

### prompt_generator.py
Generates the initial population of prompt candidates using diverse strategies:
- **Template-based**: Combines instruction prefixes with task templates and reasoning modifiers
- **Few-shot**: Prepends worked examples before the target task
- **Chain-of-Thought (CoT)**: Scaffolds step-by-step reasoning structure
- **Structured**: Uses markdown headers and bullet-point formatting

### evaluation_engine.py
Scores each LLM response across three metrics:

| Metric | Description | Weight |
|--------|-------------|--------|
| **Accuracy** | Keyword overlap and reference answer similarity | 40% |
| **Reasoning Quality** | Logical connectives, step structure, response depth | 35% |
| **Coherence** | Transition words, sentence diversity, paragraph structure | 25% |

The weighted sum produces a **composite fitness score** in [0, 1].

### mutation_engine.py
Applies evolutionary operators to produce offspring prompts:
- **Prefix Injection**: Prepend new instruction prefixes
- **Suffix Modification**: Append or replace trailing instructions
- **Instruction Paraphrase**: Substitute instruction words with synonyms
- **Structure Mutation**: Toggle between plain and structured formatting
- **Temperature Word Swap**: Randomly replace words based on temperature parameter
- **Crossover**: Single-point crossover between two parent prompts

Supports **adaptive mutation rates**: high early (exploration), low later (exploitation).

### ranking_system.py
Manages selection and evolution tracking:
- **Greedy Ranking**: Sort by composite fitness score
- **Tournament Selection**: Randomly sample k candidates, pick the best
- **Elitist Selection**: Preserve top N% + tournament fill
- **Diversity-Preserving Ranking**: Penalize prompts too similar to higher-ranked ones
- **Evolution History**: Records per-generation statistics and all-time best prompts
- **Convergence Data**: Best/average/worst scores per generation for visualization

### evolution_pipeline.py
The main orchestrator that wires all modules together and runs the evolutionary loop for N generations.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/PranayMahendrakar/ai-prompt-evolution-system.git
cd ai-prompt-evolution-system

# No external dependencies required for the mock pipeline
# For real LLM integration, install your preferred API client:
pip install openai  # or anthropic, together, etc.
```

---

## Quick Start

```python
from evolution_pipeline import EvolutionPipeline

pipeline = EvolutionPipeline(
    task="Explain quantum entanglement in simple terms.",
    n_generations=10,
    population_size=8,
    expected_keywords=["quantum", "entanglement", "particles", "correlation"],
    reference_answer="Quantum entanglement links particles so measuring one instantly affects the other.",
    seed=42,
)

results = pipeline.run()
print(f"Best prompt score: {results['best_prompt']['composite_score']:.4f}")
print(f"Best prompt text: {results['best_prompt']['prompt_text']}")
```

### Using a Real LLM

```python
import openai

def openai_llm(prompt_text: str) -> str:
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt_text}],
        max_tokens=300,
    )
    return response.choices[0].message.content

pipeline = EvolutionPipeline(
    task="Explain machine learning.",
    n_generations=5,
    population_size=6,
    llm_fn=openai_llm,
)
results = pipeline.run()
```

---

## Evolutionary Prompt Optimization Techniques

### Genetic Algorithms for Prompts
Standard genetic algorithms (GAs) maintain a population of candidate solutions and iteratively improve them via selection, crossover, and mutation. Applied to prompts, each candidate is a text string, fitness is measured by LLM output quality, and operators modify prompt structure and wording.

### Selection Strategies
- **Elitism** ensures the best prompts always survive, preventing regression
- **Tournament selection** applies selective pressure without the variance of roulette-wheel selection
- **Rank-based selection** normalizes against outlier scores

### Mutation Operators
Prompt mutation differs from binary GA mutation: operators are semantics-aware, replacing instruction phrases with paraphrases rather than flipping bits. Crossover concatenates sentence-level halves of two parent prompts.

### Adaptive Exploration vs Exploitation
Early generations use high mutation rates to explore the prompt space broadly. As generations progress, the mutation rate decays, focusing on refining promising candidates rather than random exploration.

### Evaluation as Fitness Function
The composite fitness function (accuracy × 0.40 + reasoning × 0.35 + coherence × 0.25) defines the selection pressure. Better-calibrated fitness functions lead to better-evolved prompts.

---

## Output Files

After running the pipeline, the following are saved to `evolution_output/`:

- `evolution_history.json` — Full per-generation statistics
- `best_prompts.json` — Top 5 prompts across all generations

---

## GitHub Pages

The evolution dashboard is live at:

**[https://pranaymahendrakar.github.io/ai-prompt-evolution-system/](https://pranaymahendrakar.github.io/ai-prompt-evolution-system/)**

It displays:
- Evolution convergence chart (best/avg/worst scores per generation)
- Leaderboard of best-performing prompts
- Mutation operator effectiveness comparison
- Documentation and technique explanations

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

**Pranay M Mahendrakar** · [GitHub](https://github.com/PranayMahendrakar)
