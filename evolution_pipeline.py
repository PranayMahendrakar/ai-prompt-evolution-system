"""
evolution_pipeline.py
----------------------
Orchestrates the full AI Prompt Evolution pipeline:
  1. Generate initial prompt population
  2. Simulate LLM responses (mock or real API)
  3. Evaluate output quality
  4. Rank and select survivors
  5. Mutate survivors to produce next generation
  6. Repeat for N generations
  7. Save evolution history and best prompts
"""

import json
import os
import random
import time
from typing import List, Dict, Optional, Callable
from dataclasses import asdict

from prompt_generator import Prompt, PromptGenerator
from evaluation_engine import EvaluationEngine, EvaluationResult
from mutation_engine import MutationEngine
from ranking_system import RankingSystem


# ------------------------------------------------------------------ #
#  Mock LLM Interface (replace with real API call in production)       #
# ------------------------------------------------------------------ #

class MockLLM:
    """
    Simulates an LLM for testing the pipeline without API costs.
    Generates deterministic mock responses based on prompt content.
    """

    QUALITY_RESPONSES = [
        (
            "Machine learning is a branch of AI that enables systems to learn "
            "from data. First, a model is trained on examples. Then, it identifies "
            "patterns and generalizes to new inputs. For example, image classifiers "
            "learn to distinguish cats from dogs by analyzing thousands of labeled photos. "
            "Therefore, ML models improve with more data and better training."
        ),
        (
            "Step 1: Understand the concept. Machine learning uses algorithms to "
            "parse data and learn from it. Step 2: Apply the knowledge. Models are "
            "trained on datasets and tested on unseen data. Step 3: Evaluate performance. "
            "Metrics like accuracy and F1-score measure how well the model generalizes."
        ),
        (
            "As a knowledgeable assistant, I can explain that machine learning is "
            "fundamentally about enabling computers to learn without explicit programming. "
            "Supervised learning uses labeled data, unsupervised learning finds hidden "
            "patterns, and reinforcement learning trains agents via rewards. "
            "Consequently, ML powers applications from spam filters to self-driving cars."
        ),
        (
            "Machine learning. Data. Patterns. Predictions."
        ),
        (
            "OK so like machine learning is basically when computers learn stuff. "
            "Its pretty cool. You give it data and it figures things out. "
            "Thats basically it I think."
        ),
    ]

    def __init__(self, seed: int = 42):
        random.seed(seed)

    def generate(self, prompt_text: str) -> str:
        """
        Generate a mock response. Quality varies based on prompt structure.
        Prompts with more structure get higher-quality responses.
        """
        score_hint = 0
        if "step by step" in prompt_text.lower():
            score_hint += 2
        if "expert" in prompt_text.lower():
            score_hint += 1
        if "###" in prompt_text:
            score_hint += 1
        if len(prompt_text) > 200:
            score_hint += 1

        # Map score_hint to response quality
        idx = min(score_hint, len(self.QUALITY_RESPONSES) - 1)
        # Add some noise
        if random.random() < 0.3:
            idx = max(0, idx + random.choice([-1, 0, 1]))
        idx = max(0, min(idx, len(self.QUALITY_RESPONSES) - 1))
        return self.QUALITY_RESPONSES[idx]


# ------------------------------------------------------------------ #
#  Evolution Pipeline                                                  #
# ------------------------------------------------------------------ #

class EvolutionPipeline:
    """
    Runs the full evolutionary prompt optimization loop.

    Pipeline:
      GeneratePopulation -> LLM(response) -> Evaluate -> Rank/Select -> Mutate -> Repeat
    """

    def __init__(
        self,
        task: str,
        n_generations: int = 10,
        population_size: int = 8,
        elite_fraction: float = 0.25,
        mutation_rate: float = 0.5,
        expected_keywords: Optional[List[str]] = None,
        reference_answer: Optional[str] = None,
        llm_fn: Optional[Callable[[str], str]] = None,
        output_dir: str = "evolution_output",
        seed: int = 42,
    ):
        """
        Args:
            task: The task/question the prompts should address.
            n_generations: Number of evolutionary cycles.
            population_size: Number of prompts per generation.
            elite_fraction: Fraction of top prompts preserved each generation.
            mutation_rate: Base probability of mutating each prompt.
            expected_keywords: Keywords for accuracy evaluation.
            reference_answer: Reference answer for accuracy evaluation.
            llm_fn: Callable that takes prompt text and returns response text.
                    Defaults to MockLLM if not provided.
            output_dir: Directory to save results.
            seed: Random seed.
        """
        self.task = task
        self.n_generations = n_generations
        self.population_size = population_size
        self.expected_keywords = expected_keywords
        self.reference_answer = reference_answer
        self.output_dir = output_dir

        # Initialize components
        self.generator = PromptGenerator(seed=seed)
        self.evaluator = EvaluationEngine()
        self.mutator = MutationEngine(seed=seed, mutation_rate=mutation_rate)
        self.ranker = RankingSystem(
            elite_fraction=elite_fraction,
            tournament_size=3,
            seed=seed,
        )
        self.llm = llm_fn or MockLLM(seed=seed).generate

        os.makedirs(output_dir, exist_ok=True)

    def run(self) -> Dict:
        """
        Execute the full evolutionary optimization pipeline.

        Returns:
            Dict with 'best_prompt', 'evolution_history', and 'convergence' data.
        """
        print("\n" + "="*70)
        print("  AI PROMPT EVOLUTION PIPELINE")
        print(f"  Task: {self.task[:60]}{'...' if len(self.task) > 60 else ''}")
        print(f"  Generations: {self.n_generations} | Population: {self.population_size}")
        print("="*70)

        # ---- Step 1: Generate Initial Population ---- #
        population = self.generator.generate_population(
            self.task, population_size=self.population_size
        )

        for gen in range(1, self.n_generations + 1):
            print(f"\n--- Generation {gen}/{self.n_generations} ---")
            start_time = time.time()

            # ---- Step 2: Run Prompts Through LLM ---- #
            responses = []
            for prompt in population:
                response = self.llm(prompt.text)
                responses.append({"id": prompt.id, "response": response})

            # ---- Step 3: Evaluate Output Quality ---- #
            eval_results = self.evaluator.evaluate_batch(
                responses,
                expected_keywords=self.expected_keywords,
                reference_answer=self.reference_answer,
            )

            # Update prompt fitness scores
            eval_map = {r.prompt_id: r for r in eval_results}
            prompt_texts = {}
            for prompt in population:
                if prompt.id in eval_map:
                    prompt.fitness_score = eval_map[prompt.id].composite_score
                    prompt.metrics = {
                        "accuracy": eval_map[prompt.id].accuracy,
                        "reasoning_quality": eval_map[prompt.id].reasoning_quality,
                        "coherence": eval_map[prompt.id].coherence,
                    }
                prompt_texts[prompt.id] = prompt.text

            # ---- Step 4: Record Generation & Rank ---- #
            self.ranker.record_generation(gen, eval_results, prompt_texts)
            survivors_eval = self.ranker.elitist_selection(eval_results, self.population_size)
            survivor_ids = {r.prompt_id for r in survivors_eval}
            survivors = [p for p in population if p.id in survivor_ids]

            # If we have fewer survivors than needed, fill with top-scored
            if len(survivors) < self.population_size:
                sorted_pop = sorted(population, key=lambda p: p.fitness_score, reverse=True)
                existing_ids = {p.id for p in survivors}
                for p in sorted_pop:
                    if p.id not in existing_ids:
                        survivors.append(p)
                    if len(survivors) >= self.population_size:
                        break

            elapsed = time.time() - start_time
            best = max(eval_results, key=lambda r: r.composite_score)
            print(f"  Gen {gen} complete in {elapsed:.2f}s | "
                  f"Best score: {best.composite_score:.4f} ({best.prompt_id})")

            # ---- Step 5: Mutate for Next Generation ---- #
            if gen < self.n_generations:
                population = self.mutator.adaptive_mutate(
                    survivors,
                    generation=gen,
                    max_generations=self.n_generations,
                )
                # Maintain population size
                while len(population) < self.population_size:
                    new_prompt = self.generator.generate_template_prompt(self.task)
                    new_prompt.generation = gen
                    population.append(new_prompt)
                population = population[:self.population_size]

        # ---- Step 6: Save Results ---- #
        history_path = os.path.join(self.output_dir, "evolution_history.json")
        self.ranker.save_history(history_path)

        # Save best prompts
        top_prompts = self.ranker.get_top_prompts(n=5)
        best_path = os.path.join(self.output_dir, "best_prompts.json")
        with open(best_path, "w") as f:
            json.dump(top_prompts, f, indent=2)

        # Print final leaderboard
        self.ranker.print_leaderboard(top_n=5)

        return {
            "best_prompt": self.ranker.all_time_best,
            "evolution_history": [r.to_dict() for r in self.ranker.evolution_history],
            "convergence": self.ranker.get_convergence_data(),
            "top_prompts": top_prompts,
        }


# ------------------------------------------------------------------ #
#  Entry Point                                                         #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    pipeline = EvolutionPipeline(
        task="Explain the concept of machine learning in simple terms, including how it works and real-world applications.",
        n_generations=5,
        population_size=6,
        elite_fraction=0.33,
        mutation_rate=0.6,
        expected_keywords=["machine learning", "data", "model", "training", "patterns"],
        reference_answer=(
            "Machine learning is a subset of AI that trains models on data to "
            "recognize patterns and make predictions without explicit programming."
        ),
        output_dir="evolution_output",
        seed=42,
    )

    results = pipeline.run()

    print("\n=== FINAL RESULTS ===")
    print(f"Best Prompt (Generation {results['best_prompt']['generation']}):")
    print(f"  Score: {results['best_prompt']['composite_score']:.4f}")
    print(f"  Text: {results['best_prompt']['prompt_text'][:200]}...")

    print("\nConvergence (best scores per generation):")
    for g, s in zip(results['convergence']['generations'], results['convergence']['best_scores']):
        bar = '#' * int(s * 30)
        print(f"  Gen {g:3d}: [{bar:<30}] {s:.4f}")
