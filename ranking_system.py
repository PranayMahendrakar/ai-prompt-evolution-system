"""
ranking_system.py
------------------
Ranks and selects prompt candidates based on fitness scores.
Implements tournament selection, elitism, and diversity-preserving ranking.
Also tracks evolution history and identifies the best-performing prompts.
"""

import json
import math
import random
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime


try:
    from evaluation_engine import EvaluationResult
except ImportError:
    @dataclass
    class EvaluationResult:
        prompt_id: str
        accuracy: float
        reasoning_quality: float
        coherence: float
        composite_score: float
        details: Dict = field(default_factory=dict)

        def to_dict(self):
            return asdict(self)


@dataclass
class GenerationRecord:
    """Snapshot of a single generation in the evolution history."""
    generation: int
    timestamp: str
    population_size: int
    best_score: float
    average_score: float
    worst_score: float
    best_prompt_id: str
    best_prompt_text: str
    score_distribution: List[float]

    def to_dict(self):
        return asdict(self)


class RankingSystem:
    """
    Ranks prompt populations using fitness scores and selection strategies.
    Supports: greedy ranking, tournament selection, rank-based selection,
    and NSGA-II-inspired diversity preservation.
    """

    def __init__(
        self,
        elite_fraction: float = 0.2,
        tournament_size: int = 3,
        diversity_weight: float = 0.1,
        seed: int = 42,
    ):
        """
        Args:
            elite_fraction: Top fraction preserved via elitism.
            tournament_size: Number of competitors in tournament selection.
            diversity_weight: Weight given to prompt diversity in ranking.
            seed: Random seed for reproducibility.
        """
        self.elite_fraction = elite_fraction
        self.tournament_size = tournament_size
        self.diversity_weight = diversity_weight
        random.seed(seed)
        self.evolution_history: List[GenerationRecord] = []
        self.all_time_best: Optional[Dict] = None

    # ------------------------------------------------------------------ #
    #  Core Ranking Methods                                                #
    # ------------------------------------------------------------------ #

    def rank_by_score(
        self,
        evaluation_results: List[EvaluationResult],
        descending: bool = True,
    ) -> List[EvaluationResult]:
        """
        Simple greedy ranking by composite fitness score.

        Returns:
            Sorted list of evaluation results.
        """
        ranked = sorted(
            evaluation_results,
            key=lambda r: r.composite_score,
            reverse=descending,
        )
        print(f"[RankingSystem] Ranked {len(ranked)} prompts. "
              f"Best: {ranked[0].prompt_id} ({ranked[0].composite_score:.4f}), "
              f"Worst: {ranked[-1].prompt_id} ({ranked[-1].composite_score:.4f})")
        return ranked

    def rank_with_diversity(
        self,
        prompts_and_scores: List[Tuple[str, str, float]],
    ) -> List[Tuple[str, str, float]]:
        """
        Rank prompts considering both fitness score and textual diversity.
        Penalizes prompts that are too similar to higher-ranked ones.

        Args:
            prompts_and_scores: List of (prompt_id, prompt_text, composite_score).

        Returns:
            Re-ranked list with diversity bonus applied.
        """
        if not prompts_and_scores:
            return []

        # Sort by score first
        sorted_prompts = sorted(prompts_and_scores, key=lambda x: x[2], reverse=True)
        selected = [sorted_prompts[0]]

        for candidate in sorted_prompts[1:]:
            cand_id, cand_text, cand_score = candidate
            # Compute average similarity to already-selected prompts
            avg_sim = sum(
                self._text_similarity(cand_text, sel[1]) for sel in selected
            ) / len(selected)
            # Apply diversity bonus: reward dissimilarity
            diversity_bonus = (1.0 - avg_sim) * self.diversity_weight
            adjusted_score = cand_score + diversity_bonus
            selected.append((cand_id, cand_text, adjusted_score))

        # Re-sort by adjusted scores
        selected.sort(key=lambda x: x[2], reverse=True)
        return selected

    def tournament_selection(
        self,
        evaluation_results: List[EvaluationResult],
        n_select: int,
    ) -> List[EvaluationResult]:
        """
        Select prompts using tournament selection.
        Randomly sample tournament_size candidates and pick the best.

        Args:
            evaluation_results: Pool of evaluated prompts.
            n_select: Number of prompts to select.

        Returns:
            Selected prompt evaluations.
        """
        selected = []
        pool = list(evaluation_results)

        for _ in range(n_select):
            if len(pool) < self.tournament_size:
                tournament = pool
            else:
                tournament = random.sample(pool, self.tournament_size)
            winner = max(tournament, key=lambda r: r.composite_score)
            selected.append(winner)

        print(f"[RankingSystem] Tournament selection: {n_select} selected "
              f"from {len(evaluation_results)} (tournament_size={self.tournament_size})")
        return selected

    def elitist_selection(
        self,
        evaluation_results: List[EvaluationResult],
        population_size: int,
    ) -> List[EvaluationResult]:
        """
        Preserve the top elite_fraction of prompts unchanged,
        fill remaining slots using tournament selection.

        Args:
            evaluation_results: Current generation's evaluated prompts.
            population_size: Target population size for next generation.

        Returns:
            Selected prompts for next generation.
        """
        ranked = self.rank_by_score(evaluation_results)
        n_elite = max(1, int(len(ranked) * self.elite_fraction))
        elite = ranked[:n_elite]

        # Fill rest with tournament selection from full pool
        n_tournament = population_size - n_elite
        if n_tournament > 0:
            tournament_picks = self.tournament_selection(
                evaluation_results, n_select=n_tournament
            )
        else:
            tournament_picks = []

        survivors = elite + tournament_picks
        print(f"[RankingSystem] Elitist selection: {n_elite} elite + "
              f"{len(tournament_picks)} tournament = {len(survivors)} total")
        return survivors

    def rank_based_selection(
        self,
        evaluation_results: List[EvaluationResult],
        n_select: int,
    ) -> List[EvaluationResult]:
        """
        Assign selection probability proportional to rank (not raw score).
        Reduces selection pressure from outlier high/low scores.

        Args:
            evaluation_results: Evaluated prompts.
            n_select: Number to select.

        Returns:
            Selected prompts.
        """
        ranked = self.rank_by_score(evaluation_results)
        n = len(ranked)
        # Rank probabilities: best gets weight n, worst gets weight 1
        weights = [n - i for i in range(n)]
        total_weight = sum(weights)
        probs = [w / total_weight for w in weights]

        selected_indices = random.choices(range(n), weights=probs, k=n_select)
        selected = [ranked[i] for i in selected_indices]
        return selected

    # ------------------------------------------------------------------ #
    #  Evolution History Tracking                                          #
    # ------------------------------------------------------------------ #

    def record_generation(
        self,
        generation: int,
        evaluation_results: List[EvaluationResult],
        prompt_texts: Dict[str, str],
    ) -> GenerationRecord:
        """
        Record statistics for a completed generation.

        Args:
            generation: Generation number.
            evaluation_results: All evaluated prompts this generation.
            prompt_texts: Mapping of prompt_id -> prompt_text.

        Returns:
            GenerationRecord for this generation.
        """
        scores = [r.composite_score for r in evaluation_results]
        best_result = max(evaluation_results, key=lambda r: r.composite_score)
        best_text = prompt_texts.get(best_result.prompt_id, "N/A")

        record = GenerationRecord(
            generation=generation,
            timestamp=datetime.now().isoformat(),
            population_size=len(evaluation_results),
            best_score=max(scores),
            average_score=sum(scores) / len(scores),
            worst_score=min(scores),
            best_prompt_id=best_result.prompt_id,
            best_prompt_text=best_text,
            score_distribution=sorted(scores, reverse=True),
        )
        self.evolution_history.append(record)

        # Update all-time best
        if (self.all_time_best is None or
                best_result.composite_score > self.all_time_best["composite_score"]):
            self.all_time_best = {
                "generation": generation,
                "prompt_id": best_result.prompt_id,
                "prompt_text": best_text,
                "composite_score": best_result.composite_score,
                "accuracy": best_result.accuracy,
                "reasoning_quality": best_result.reasoning_quality,
                "coherence": best_result.coherence,
            }
            print(f"[RankingSystem] New all-time best! Gen {generation}: "
                  f"{best_result.prompt_id} score={best_result.composite_score:.4f}")

        print(f"[RankingSystem] Gen {generation}: "
              f"best={record.best_score:.4f}, "
              f"avg={record.average_score:.4f}, "
              f"worst={record.worst_score:.4f}")
        return record

    def get_top_prompts(self, n: int = 5) -> List[Dict]:
        """
        Return the top N prompts across all recorded generations.

        Returns:
            List of dicts with prompt info and scores.
        """
        all_bests = [
            {
                "generation": r.generation,
                "prompt_id": r.best_prompt_id,
                "prompt_text": r.best_prompt_text,
                "score": r.best_score,
            }
            for r in self.evolution_history
        ]
        all_bests.sort(key=lambda x: x["score"], reverse=True)
        return all_bests[:n]

    def get_convergence_data(self) -> Dict:
        """
        Return data describing how scores evolved over generations.
        Useful for plotting evolution curves.

        Returns:
            Dict with generation numbers and score series.
        """
        return {
            "generations": [r.generation for r in self.evolution_history],
            "best_scores": [r.best_score for r in self.evolution_history],
            "avg_scores": [r.average_score for r in self.evolution_history],
            "worst_scores": [r.worst_score for r in self.evolution_history],
        }

    def save_history(self, filepath: str):
        """Serialize full evolution history to JSON."""
        data = {
            "evolution_history": [r.to_dict() for r in self.evolution_history],
            "all_time_best": self.all_time_best,
            "convergence": self.get_convergence_data(),
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[RankingSystem] Saved history ({len(self.evolution_history)} generations) to {filepath}")

    def load_history(self, filepath: str):
        """Load evolution history from JSON file."""
        with open(filepath) as f:
            data = json.load(f)
        self.evolution_history = [
            GenerationRecord(**r) for r in data.get("evolution_history", [])
        ]
        self.all_time_best = data.get("all_time_best")
        print(f"[RankingSystem] Loaded {len(self.evolution_history)} generation records.")

    def print_leaderboard(self, top_n: int = 5):
        """Print a formatted leaderboard of best prompts."""
        print("\n" + "=" * 60)
        print(f"  PROMPT EVOLUTION LEADERBOARD (Top {top_n})")
        print("=" * 60)
        if self.all_time_best:
            print(f"  All-Time Best: {self.all_time_best['prompt_id']} "
                  f"(gen {self.all_time_best['generation']}) "
                  f"score={self.all_time_best['composite_score']:.4f}")
        print("-" * 60)
        top = self.get_top_prompts(top_n)
        for rank, entry in enumerate(top, 1):
            print(f"  #{rank:2d} | Gen {entry['generation']:3d} | "
                  f"Score {entry['score']:.4f} | {entry['prompt_id']}")
            print(f"       {entry['prompt_text'][:70]}...")
        print("=" * 60)

    # ------------------------------------------------------------------ #
    #  Utility                                                             #
    # ------------------------------------------------------------------ #

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Compute token Jaccard similarity between two texts."""
        t1 = set(text1.lower().split())
        t2 = set(text2.lower().split())
        if not t1 or not t2:
            return 0.0
        return len(t1 & t2) / len(t1 | t2)


if __name__ == "__main__":
    # Demo: simulate evaluation results and run ranking
    import random
    random.seed(42)

    mock_results = [
        EvaluationResult(
            prompt_id=f"prompt_{i:04d}",
            accuracy=random.uniform(0.3, 0.9),
            reasoning_quality=random.uniform(0.2, 0.8),
            coherence=random.uniform(0.4, 0.95),
            composite_score=random.uniform(0.3, 0.9),
            details={},
        )
        for i in range(10)
    ]

    prompt_texts = {r.prompt_id: f"Sample prompt text for {r.prompt_id}" for r in mock_results}

    ranker = RankingSystem(elite_fraction=0.2, tournament_size=3, seed=42)

    # Record generation
    record = ranker.record_generation(
        generation=1,
        evaluation_results=mock_results,
        prompt_texts=prompt_texts,
    )

    # Elitist selection for next generation
    survivors = ranker.elitist_selection(mock_results, population_size=8)
    print(f"\nSurvivors: {[r.prompt_id for r in survivors]}")

    # Print leaderboard
    ranker.print_leaderboard(top_n=5)

    # Convergence data
    convergence = ranker.get_convergence_data()
    print(f"\nConvergence data: {convergence}")
