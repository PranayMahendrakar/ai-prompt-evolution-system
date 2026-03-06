"""
prompt_generator.py
Generate initial prompt population for evolutionary optimization.
"""

import random
import json
from typing import List, Dict
from dataclasses import dataclass, field, asdict


@dataclass
class Prompt:
    id: str
    text: str
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    fitness_score: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


class PromptGenerator:
    INSTRUCTION_PREFIXES = [
        "You are an expert assistant. ",
        "Think step by step and ",
        "As a knowledgeable AI, ",
        "Carefully analyze the following and ",
        "Using logical reasoning, ",
        "Provide a detailed and accurate response: ",
    ]

    REASONING_MODIFIERS = [
        "Explain your reasoning.",
        "Show your chain of thought.",
        "Break down your answer into steps.",
        "Be concise yet thorough.",
        "Use examples where applicable.",
        "Summarize your conclusion at the end.",
    ]

    TASK_TEMPLATES = [
        "Answer the following question: {task}",
        "Solve this problem: {task}",
        "Analyze and respond to: {task}",
        "Provide an expert response to: {task}",
        "Evaluate and answer: {task}",
    ]

    FEW_SHOT_EXAMPLES = [
        {"input": "What is the capital of France?", "output": "The capital of France is Paris."},
        {"input": "Explain gravity.", "output": "Gravity is a force attracting objects with mass."}
    ]

    def __init__(self, seed=42):
        random.seed(seed)
        self._counter = 0

    def _next_id(self):
        self._counter += 1
        return f"prompt_{self._counter:04d}"

    def generate_template_prompt(self, task):
        prefix = random.choice(self.INSTRUCTION_PREFIXES)
        template = random.choice(self.TASK_TEMPLATES)
        modifier = random.choice(self.REASONING_MODIFIERS)
        return Prompt(id=self._next_id(), text=f"{prefix}{template.format(task=task)} {modifier}")

    def generate_few_shot_prompt(self, task, n_examples=1):
        examples = random.sample(self.FEW_SHOT_EXAMPLES, min(n_examples, len(self.FEW_SHOT_EXAMPLES)))
        parts = [f"Input: {e['input']}\nOutput: {e['output']}" for e in examples]
        parts.append(f"Input: {task}\nOutput:")
        return Prompt(id=self._next_id(), text="\n\n".join(parts))

    def generate_cot_prompt(self, task):
        text = (
            f"Let's think through this step by step.\n\n"
            f"Question: {task}\n\n"
            f"Step 1: Understand the question.\n"
            f"Step 2: Identify key information.\n"
            f"Step 3: Apply relevant knowledge.\n"
            f"Step 4: Formulate the answer.\n\nAnswer:"
        )
        return Prompt(id=self._next_id(), text=text)

    def generate_structured_prompt(self, task):
        text = (
            f"### Task\n{task}\n\n"
            f"### Instructions\n"
            f"- Read carefully.\n- Be accurate and complete.\n\n"
            f"### Response"
        )
        return Prompt(id=self._next_id(), text=text)

    def generate_population(self, task, population_size=10):
        generators = [
            self.generate_template_prompt,
            self.generate_few_shot_prompt,
            self.generate_cot_prompt,
            self.generate_structured_prompt,
        ]
        population = [generators[i % len(generators)](task) for i in range(population_size)]
        print(f"[PromptGenerator] Generated {len(population)} prompts.")
        return population

    def save_population(self, population, filepath):
        with open(filepath, "w") as f:
            json.dump([p.to_dict() for p in population], f, indent=2)

    def load_population(self, filepath):
        with open(filepath) as f:
            return [Prompt.from_dict(d) for d in json.load(f)]


if __name__ == "__main__":
    gen = PromptGenerator(seed=42)
    task = "Explain machine learning in simple terms."
    population = gen.generate_population(task, population_size=8)
    for p in population:
        print(f"[{p.id}] {p.text[:100]}...")
