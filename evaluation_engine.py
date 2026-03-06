"""
evaluation_engine.py
---------------------
Evaluates prompt quality using multiple metrics:
  - Accuracy: keyword/expected-answer overlap
  - Reasoning Quality: structural depth and logical connectives
  - Coherence: readability and sentence flow
"""

import re
import math
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    prompt_id: str
    accuracy: float
    reasoning_quality: float
    coherence: float
    composite_score: float
    details: Dict[str, str]

    def to_dict(self):
        return {
            "prompt_id": self.prompt_id,
            "accuracy": round(self.accuracy, 4),
            "reasoning_quality": round(self.reasoning_quality, 4),
            "coherence": round(self.coherence, 4),
            "composite_score": round(self.composite_score, 4),
            "details": self.details,
        }


class EvaluationEngine:
    """
    Multi-metric evaluation engine for LLM prompt responses.
    Computes accuracy, reasoning quality, and coherence scores
    and combines them into a weighted composite fitness score.
    """

    # Weights for composite score
    WEIGHTS = {
        "accuracy": 0.40,
        "reasoning_quality": 0.35,
        "coherence": 0.25,
    }

    # Logical connectives that indicate reasoning depth
    REASONING_CONNECTIVES = [
        "because", "therefore", "thus", "hence", "so",
        "consequently", "as a result", "this means",
        "in other words", "for example", "specifically",
        "first", "second", "third", "finally", "in conclusion",
        "step", "given that", "it follows", "we can see",
    ]

    # Transition words that indicate coherence
    COHERENCE_TRANSITIONS = [
        "however", "moreover", "furthermore", "additionally",
        "in contrast", "on the other hand", "similarly",
        "for instance", "in summary", "to summarize",
        "next", "then", "previously", "subsequently",
    ]

    def __init__(
        self,
        accuracy_weight: float = 0.40,
        reasoning_weight: float = 0.35,
        coherence_weight: float = 0.25,
    ):
        total = accuracy_weight + reasoning_weight + coherence_weight
        self.WEIGHTS = {
            "accuracy": accuracy_weight / total,
            "reasoning_quality": reasoning_weight / total,
            "coherence": coherence_weight / total,
        }

    def evaluate_accuracy(
        self,
        response: str,
        expected_keywords: Optional[List[str]] = None,
        reference_answer: Optional[str] = None,
    ) -> float:
        """
        Score accuracy based on keyword overlap and/or reference answer similarity.

        Args:
            response: The LLM output text.
            expected_keywords: List of keywords expected in a correct answer.
            reference_answer: Reference answer for similarity comparison.

        Returns:
            Accuracy score in [0, 1].
        """
        if not response.strip():
            return 0.0

        score = 0.0
        components = 0

        if expected_keywords:
            resp_lower = response.lower()
            hits = sum(1 for kw in expected_keywords if kw.lower() in resp_lower)
            score += hits / len(expected_keywords)
            components += 1

        if reference_answer:
            score += self._token_overlap(response, reference_answer)
            components += 1

        if components == 0:
            # Fallback: reward length and structure
            words = len(response.split())
            score = min(1.0, words / 100.0)
            return score

        return score / components

    def evaluate_reasoning_quality(self, response: str) -> float:
        """
        Evaluate the depth and quality of reasoning in a response.

        Looks for: logical connectives, step-by-step structure,
        enumerated points, length, and question answering completeness.

        Returns:
            Reasoning quality score in [0, 1].
        """
        if not response.strip():
            return 0.0

        lower = response.lower()
        words = response.split()
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip()]

        # 1. Connective density
        connective_count = sum(1 for c in self.REASONING_CONNECTIVES if c in lower)
        connective_score = min(1.0, connective_count / 5.0)

        # 2. Structural indicators (numbered steps, bullets)
        structural_score = 0.0
        if re.search(r'(step \d|\d+\.\s|•|\-\s)', lower):
            structural_score = 0.8
        elif re.search(r'(first|second|third|finally|lastly)', lower):
            structural_score = 0.6

        # 3. Length adequacy (target: 50-300 words for a quality response)
        word_count = len(words)
        if word_count < 10:
            length_score = 0.1
        elif word_count < 30:
            length_score = 0.4
        elif word_count <= 300:
            length_score = min(1.0, word_count / 100.0)
        else:
            length_score = max(0.5, 1.0 - (word_count - 300) / 500.0)

        # 4. Sentence diversity
        avg_sent_len = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        diversity_score = min(1.0, avg_sent_len / 15.0) if avg_sent_len >= 5 else 0.3

        return (
            0.30 * connective_score +
            0.30 * structural_score +
            0.25 * length_score +
            0.15 * diversity_score
        )

    def evaluate_coherence(self, response: str) -> float:
        """
        Evaluate the coherence and readability of a response.

        Looks for: transition words, paragraph structure,
        sentence count variety, and readability proxies.

        Returns:
            Coherence score in [0, 1].
        """
        if not response.strip():
            return 0.0

        lower = response.lower()
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip()]

        # 1. Transition word density
        transition_count = sum(1 for t in self.COHERENCE_TRANSITIONS if t in lower)
        transition_score = min(1.0, transition_count / 3.0)

        # 2. Sentence count (more sentences = more structured)
        sent_count = len(sentences)
        if sent_count < 2:
            structure_score = 0.2
        elif sent_count <= 10:
            structure_score = min(1.0, sent_count / 5.0)
        else:
            structure_score = 0.9

        # 3. Paragraph structure
        paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
        para_score = min(1.0, len(paragraphs) / 3.0) if len(paragraphs) > 1 else 0.5

        # 4. Repetition penalty
        words = response.lower().split()
        unique_ratio = len(set(words)) / max(len(words), 1)
        repetition_score = unique_ratio

        return (
            0.30 * transition_score +
            0.30 * structure_score +
            0.20 * para_score +
            0.20 * repetition_score
        )

    def evaluate(
        self,
        prompt_id: str,
        response: str,
        expected_keywords: Optional[List[str]] = None,
        reference_answer: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Run all evaluation metrics and return an EvaluationResult.

        Args:
            prompt_id: Identifier of the prompt being evaluated.
            response: LLM response text.
            expected_keywords: Optional keywords for accuracy evaluation.
            reference_answer: Optional reference for accuracy evaluation.

        Returns:
            EvaluationResult with all metric scores.
        """
        accuracy = self.evaluate_accuracy(response, expected_keywords, reference_answer)
        reasoning = self.evaluate_reasoning_quality(response)
        coherence = self.evaluate_coherence(response)

        composite = (
            self.WEIGHTS["accuracy"] * accuracy +
            self.WEIGHTS["reasoning_quality"] * reasoning +
            self.WEIGHTS["coherence"] * coherence
        )

        details = {
            "response_length": f"{len(response.split())} words",
            "accuracy_note": "keyword/reference overlap" if expected_keywords or reference_answer else "length heuristic",
            "reasoning_note": "connective + structural analysis",
            "coherence_note": "transition + sentence diversity",
        }

        result = EvaluationResult(
            prompt_id=prompt_id,
            accuracy=accuracy,
            reasoning_quality=reasoning,
            coherence=coherence,
            composite_score=composite,
            details=details,
        )

        print(f"[EvaluationEngine] {prompt_id}: accuracy={accuracy:.3f}, "
              f"reasoning={reasoning:.3f}, coherence={coherence:.3f}, "
              f"composite={composite:.3f}")
        return result

    def evaluate_batch(
        self,
        prompts_and_responses: List[Dict],
        expected_keywords: Optional[List[str]] = None,
        reference_answer: Optional[str] = None,
    ) -> List[EvaluationResult]:
        """
        Evaluate a batch of (prompt_id, response) pairs.

        Args:
            prompts_and_responses: List of dicts with 'id' and 'response' keys.
            expected_keywords: Shared keywords for accuracy evaluation.
            reference_answer: Shared reference for accuracy evaluation.

        Returns:
            List of EvaluationResult objects.
        """
        results = []
        for item in prompts_and_responses:
            result = self.evaluate(
                prompt_id=item["id"],
                response=item.get("response", ""),
                expected_keywords=expected_keywords,
                reference_answer=reference_answer,
            )
            results.append(result)
        return results

    def _token_overlap(self, text1: str, text2: str) -> float:
        """Compute token-level Jaccard similarity between two texts."""
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        if not tokens1 or not tokens2:
            return 0.0
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        return len(intersection) / len(union)


if __name__ == "__main__":
    engine = EvaluationEngine()

    test_response = (
        "Machine learning is a subset of artificial intelligence that enables "
        "computers to learn from data. First, a model is trained on examples. "
        "Second, it identifies patterns. Therefore, it can make predictions on new data. "
        "For example, spam filters learn to distinguish spam from legitimate email."
    )

    result = engine.evaluate(
        prompt_id="prompt_0001",
        response=test_response,
        expected_keywords=["machine learning", "data", "model", "patterns", "predictions"],
        reference_answer="Machine learning uses data to train models that make predictions.",
    )

    print("\n=== Evaluation Result ===")
    import json
    print(json.dumps(result.to_dict(), indent=2))
