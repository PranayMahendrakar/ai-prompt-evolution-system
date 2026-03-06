"""
mutation_engine.py
-------------------
Mutates prompt candidates using multiple strategies:
  - Prefix injection
  - Suffix modification
  - Instruction paraphrase
  - Temperature-based word substitution
  - Crossover between two parent prompts
"""

import random
import re
from typing import List, Tuple, Optional
from dataclasses import replace as dc_replace


# Import Prompt dataclass - in production use shared module
# For standalone use, define minimal version here
try:
    from prompt_generator import Prompt
except ImportError:
    from dataclasses import dataclass, field
    from typing import Dict

    @dataclass
    class Prompt:
        id: str
        text: str
        generation: int = 0
        parent_ids: List[str] = field(default_factory=list)
        fitness_score: float = 0.0
        metrics: Dict[str, float] = field(default_factory=dict)


class MutationEngine:
    """
    Applies evolutionary mutation operators to prompt candidates.
    Supports single-point mutations, crossover, and adaptive mutation rates.
    """

    # Instruction prefix pool for injection mutations
    PREFIX_POOL = [
        "You are an expert in this domain. ",
        "Think carefully and step by step. ",
        "As a highly knowledgeable assistant, ",
        "Use logical reasoning and evidence. ",
        "Consider multiple perspectives before answering. ",
        "Provide a thorough and well-structured response. ",
        "Be precise, accurate, and comprehensive. ",
        "Draw on deep expertise to answer: ",
        "Approach this analytically: ",
        "With careful thought and clarity, ",
    ]

    # Suffix pool for suffix mutation
    SUFFIX_POOL = [
        " Explain your reasoning clearly.",
        " Provide specific examples.",
        " Be concise but complete.",
        " Show your chain of thought.",
        " Justify your answer.",
        " Summarize key points at the end.",
        " Break your answer into clear steps.",
        " Highlight the most important insight.",
        " Consider edge cases and exceptions.",
        " Use structured formatting if helpful.",
    ]

    # Paraphrase substitutions for instruction words
    PARAPHRASE_MAP = {
        "answer": ["respond to", "address", "reply to", "explain"],
        "solve": ["work through", "resolve", "tackle", "figure out"],
        "explain": ["describe", "elaborate on", "clarify", "discuss"],
        "analyze": ["examine", "evaluate", "assess", "inspect"],
        "provide": ["give", "offer", "present", "share"],
        "think": ["reason", "reflect", "consider", "ponder"],
        "carefully": ["thoroughly", "precisely", "attentively", "methodically"],
        "step by step": ["systematically", "in sequence", "one step at a time", "methodically"],
        "expert": ["specialist", "authority", "professional", "knowledgeable"],
        "detailed": ["comprehensive", "thorough", "in-depth", "complete"],
    }

    def __init__(self, seed: int = 42, mutation_rate: float = 0.3):
        random.seed(seed)
        self.mutation_rate = mutation_rate
        self._counter = 0

    def _next_id(self, base_id: str) -> str:
        self._counter += 1
        return f"{base_id}_mut{self._counter:03d}"

    # ------------------------------------------------------------------ #
    #  Mutation Operators                                                  #
    # ------------------------------------------------------------------ #

    def prefix_injection(self, prompt: Prompt) -> Prompt:
        """Prepend a new instruction prefix to the prompt text."""
        new_prefix = random.choice(self.PREFIX_POOL)
        # Avoid doubling if same prefix already exists
        if prompt.text.startswith(new_prefix):
            new_prefix = random.choice([p for p in self.PREFIX_POOL if p != new_prefix])
        new_text = new_prefix + prompt.text
        return Prompt(
            id=self._next_id(prompt.id),
            text=new_text,
            generation=prompt.generation + 1,
            parent_ids=[prompt.id],
        )

    def suffix_modification(self, prompt: Prompt) -> Prompt:
        """Append or replace the trailing instruction in the prompt."""
        new_suffix = random.choice(self.SUFFIX_POOL)
        # Remove existing suffix if it matches any known suffix
        text = prompt.text
        for known_suffix in self.SUFFIX_POOL:
            if text.rstrip().endswith(known_suffix.strip()):
                text = text.rstrip()[: -len(known_suffix.strip())].rstrip()
                break
        new_text = text + new_suffix
        return Prompt(
            id=self._next_id(prompt.id),
            text=new_text,
            generation=prompt.generation + 1,
            parent_ids=[prompt.id],
        )

    def instruction_paraphrase(self, prompt: Prompt) -> Prompt:
        """
        Replace instruction words with semantically equivalent alternatives
        using the paraphrase map.
        """
        text = prompt.text
        words_replaced = 0
        for original, alternatives in self.PARAPHRASE_MAP.items():
            if original in text.lower() and random.random() < 0.5:
                replacement = random.choice(alternatives)
                # Case-insensitive replacement
                pattern = re.compile(re.escape(original), re.IGNORECASE)
                text = pattern.sub(replacement, text, count=1)
                words_replaced += 1
        if words_replaced == 0:
            # Fallback: inject prefix
            return self.prefix_injection(prompt)
        return Prompt(
            id=self._next_id(prompt.id),
            text=text,
            generation=prompt.generation + 1,
            parent_ids=[prompt.id],
        )

    def structure_mutation(self, prompt: Prompt) -> Prompt:
        """
        Convert unstructured prompt to a structured format or vice versa.
        Adds markdown headers and bullet points to plain text prompts.
        """
        text = prompt.text
        has_structure = bool(re.search(r'(###|\*\*|^\d+\.|^-\s)', text, re.MULTILINE))

        if has_structure:
            # Strip structure markers
            text = re.sub(r'#+\s', '', text)
            text = re.sub(r'^\*\*|\*\*$', '', text, flags=re.MULTILINE)
            text = re.sub(r'^\d+\.\s', '', text, flags=re.MULTILINE)
        else:
            # Add structure
            lines = [l.strip() for l in text.split('\n') if l.strip()]
            if len(lines) >= 3:
                structured_lines = [f"### Prompt\n{lines[0]}\n"]
                if len(lines) > 1:
                    structured_lines.append("### Instructions")
                    for line in lines[1:]:
                        structured_lines.append(f"- {line}")
                structured_lines.append("\n### Response")
                text = "\n".join(structured_lines)

        return Prompt(
            id=self._next_id(prompt.id),
            text=text,
            generation=prompt.generation + 1,
            parent_ids=[prompt.id],
        )

    def temperature_word_swap(self, prompt: Prompt, temperature: float = 0.5) -> Prompt:
        """
        Randomly swap words with synonyms from the paraphrase pool based on temperature.
        Higher temperature = more aggressive mutation.
        """
        text = prompt.text
        words = text.split()
        mutated = []
        for word in words:
            clean = word.lower().strip('.,!?;:')
            if clean in self.PARAPHRASE_MAP and random.random() < temperature:
                replacement = random.choice(self.PARAPHRASE_MAP[clean])
                # Preserve capitalization
                if word[0].isupper():
                    replacement = replacement.capitalize()
                word = word.replace(clean, replacement, 1)
            mutated.append(word)
        return Prompt(
            id=self._next_id(prompt.id),
            text=' '.join(mutated),
            generation=prompt.generation + 1,
            parent_ids=[prompt.id],
        )

    def crossover(self, parent1: Prompt, parent2: Prompt) -> Tuple[Prompt, Prompt]:
        """
        Perform single-point crossover between two parent prompts.
        Splits each prompt at a midpoint and combines the halves.

        Returns:
            Two offspring prompts.
        """
        sentences1 = re.split(r'(?<=[.!?])\s+', parent1.text)
        sentences2 = re.split(r'(?<=[.!?])\s+', parent2.text)

        # Midpoint split
        mid1 = max(1, len(sentences1) // 2)
        mid2 = max(1, len(sentences2) // 2)

        child1_text = ' '.join(sentences1[:mid1] + sentences2[mid2:])
        child2_text = ' '.join(sentences2[:mid2] + sentences1[mid1:])

        child1 = Prompt(
            id=self._next_id(parent1.id),
            text=child1_text,
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[parent1.id, parent2.id],
        )
        child2 = Prompt(
            id=self._next_id(parent2.id),
            text=child2_text,
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[parent1.id, parent2.id],
        )
        return child1, child2

    # ------------------------------------------------------------------ #
    #  Population-level Mutation                                           #
    # ------------------------------------------------------------------ #

    def mutate_population(
        self,
        population: List[Prompt],
        mutation_rate: Optional[float] = None,
        crossover_rate: float = 0.2,
    ) -> List[Prompt]:
        """
        Apply mutations to a population of prompts.

        For each prompt, randomly apply one of: prefix_injection,
        suffix_modification, instruction_paraphrase, structure_mutation,
        or temperature_word_swap. Also applies crossover between pairs.

        Args:
            population: List of parent prompts.
            mutation_rate: Override default mutation rate.
            crossover_rate: Fraction of population to undergo crossover.

        Returns:
            List of mutated offspring prompts.
        """
        rate = mutation_rate if mutation_rate is not None else self.mutation_rate
        operators = [
            self.prefix_injection,
            self.suffix_modification,
            self.instruction_paraphrase,
            self.structure_mutation,
            lambda p: self.temperature_word_swap(p, temperature=0.4),
        ]

        offspring = []

        # Single-point mutations
        for prompt in population:
            if random.random() < rate:
                op = random.choice(operators)
                mutant = op(prompt)
                offspring.append(mutant)
            else:
                offspring.append(prompt)

        # Crossover
        random.shuffle(offspring)
        n_cross = int(len(offspring) * crossover_rate)
        for i in range(0, n_cross - 1, 2):
            c1, c2 = self.crossover(offspring[i], offspring[i + 1])
            offspring[i] = c1
            offspring[i + 1] = c2

        print(f"[MutationEngine] Produced {len(offspring)} offspring "
              f"(rate={rate:.2f}, crossover_rate={crossover_rate:.2f})")
        return offspring

    def adaptive_mutate(
        self,
        population: List[Prompt],
        generation: int,
        max_generations: int = 20,
    ) -> List[Prompt]:
        """
        Adaptively adjust mutation rate based on generation progress.
        Higher mutation rate early (exploration), lower rate later (exploitation).

        Args:
            population: Current prompt population.
            generation: Current generation number.
            max_generations: Total planned generations.

        Returns:
            Mutated offspring list.
        """
        # Decay mutation rate from 0.7 to 0.1 over generations
        adaptive_rate = 0.7 - 0.6 * (generation / max(max_generations, 1))
        adaptive_rate = max(0.1, min(0.7, adaptive_rate))
        print(f"[MutationEngine] Adaptive mutation rate: {adaptive_rate:.3f} (gen {generation}/{max_generations})")
        return self.mutate_population(population, mutation_rate=adaptive_rate)


if __name__ == "__main__":
    # Demo: create sample prompts and mutate them
    try:
        from prompt_generator import Prompt, PromptGenerator
        gen = PromptGenerator(seed=42)
        task = "Explain quantum computing."
        population = gen.generate_population(task, population_size=4)
    except ImportError:
        population = [
            Prompt(id="prompt_0001", text="Explain quantum computing step by step."),
            Prompt(id="prompt_0002", text="You are an expert. Answer: What is quantum computing?"),
        ]

    engine = MutationEngine(seed=42, mutation_rate=0.8)
    print("\n=== Original Population ===")
    for p in population:
        print(f"[{p.id}] {p.text[:80]}...")

    offspring = engine.mutate_population(population, mutation_rate=0.9)
    print("\n=== Mutated Offspring ===")
    for p in offspring:
        print(f"[{p.id}] gen={p.generation} parents={p.parent_ids}")
        print(f"  {p.text[:100]}...")
