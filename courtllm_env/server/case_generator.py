# server/case_generator.py — Parametric case factory

from dataclasses import dataclass
from typing import List, Dict
import random
import json

DOMAINS = ["biomedical", "historical", "legal", "financial", "scientific"]

HALLUCINATION_TYPES = [
    "fabricated_statistic",      # Real metric, wrong number
    "attribution_error",         # Real quote, wrong speaker
    "date_shift",                # Real event, wrong year
    "causal_inversion",          # A causes B → B causes A
    "entity_substitution",       # Real fact, wrong named entity
]

@dataclass
class Case:
    query: str
    flagged_claims: List[Dict]
    evidence_corpus: List[Dict]
    ground_truth: List[Dict]
    difficulty: int

class CaseGenerator:
    """
    Generates parametric courtroom cases with planted hallucinations.
    Each episode is unique to prevent memorization.
    """

    def __init__(self, evidence_db=None):
        self.evidence_db = evidence_db or []
        self.case_templates = self._load_templates()

    def generate(self, difficulty: int = 1, seed: int = None) -> Case:
        """
        Generate a new case with planted hallucinations.

        Args:
            difficulty: 0-3, controls number of claims and hallucinations
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)

        domain = random.choice(DOMAINS)

        # Sample true facts from evidence database
        true_facts = self._sample_true_facts(domain, n=difficulty + 2)

        # Inject hallucinations
        planted_hallucinations = self._inject_hallucinations(
            true_facts,
            n=max(1, difficulty)
        )

        # Build evidence corpus with red herrings
        evidence_corpus = self._build_corpus(
            true_facts,
            red_herrings=difficulty,
            size=20 + difficulty * 10
        )

        # Generate query
        query = self._template_query(domain, true_facts, planted_hallucinations)

        return Case(
            query=query,
            flagged_claims=planted_hallucinations,
            evidence_corpus=evidence_corpus,
            ground_truth=true_facts,
            difficulty=difficulty
        )

    def _sample_true_facts(self, domain: str, n: int) -> List[Dict]:
        """Sample N true facts from the evidence database for given domain"""
        domain_facts = [
            fact for fact in self.evidence_db
            if fact.get("domain") == domain
        ]

        if len(domain_facts) < n:
            # Fallback to synthetic facts if database is empty
            return self._generate_synthetic_facts(domain, n)

        return random.sample(domain_facts, min(n, len(domain_facts)))

    def _generate_synthetic_facts(self, domain: str, n: int) -> List[Dict]:
        """Generate synthetic facts when database is empty (for testing)"""
        templates = {
            "biomedical": [
                "Aspirin reduces the risk of heart attacks by approximately 25%",
                "The human genome contains approximately 20,000-25,000 genes",
                "Penicillin was discovered by Alexander Fleming in 1928"
            ],
            "historical": [
                "World War II ended in 1945",
                "The Declaration of Independence was signed in 1776",
                "The Berlin Wall fell in 1989"
            ],
            "legal": [
                "The US Constitution has 27 amendments",
                "Miranda rights were established in 1966",
                "The Supreme Court has 9 justices"
            ],
            "financial": [
                "The 2008 financial crisis was triggered by subprime mortgages",
                "The Federal Reserve was established in 1913",
                "The Dow Jones Industrial Average tracks 30 companies"
            ],
            "scientific": [
                "The speed of light is approximately 299,792 km/s",
                "DNA has a double helix structure",
                "Water boils at 100°C at sea level"
            ]
        }

        facts = templates.get(domain, templates["scientific"])
        selected = random.sample(facts, min(n, len(facts)))

        return [
            {
                "source_id": f"{domain.upper()}_{i:03d}",
                "title": f"{domain.title()} Fact {i}",
                "domain": domain,
                "snippet": fact,
                "is_true": True
            }
            for i, fact in enumerate(selected)
        ]

    def _inject_hallucinations(self, true_facts: List[Dict], n: int) -> List[Dict]:
        """Inject N hallucinations based on true facts"""
        hallucinations = []

        for i in range(min(n, len(true_facts))):
            fact = true_facts[i]
            halluc_type = random.choice(HALLUCINATION_TYPES)

            hallucinated_claim = self._apply_hallucination(
                fact["snippet"],
                halluc_type
            )

            hallucinations.append({
                "claim_id": f"claim_{i:03d}",
                "claim_text": hallucinated_claim,
                "suspicion_reason": f"Potential {halluc_type.replace('_', ' ')}",
                "original_fact": fact["snippet"],
                "hallucination_type": halluc_type
            })

        return hallucinations

    def _apply_hallucination(self, fact: str, halluc_type: str) -> str:
        """Apply a specific hallucination type to a fact"""
        if halluc_type == "fabricated_statistic":
            # Change numbers by ±30%
            import re
            numbers = re.findall(r'\d+', fact)
            if numbers:
                old_num = numbers[0]
                new_num = int(int(old_num) * random.uniform(0.7, 1.3))
                return fact.replace(old_num, str(new_num), 1)

        elif halluc_type == "date_shift":
            # Shift years by ±10%
            import re
            years = re.findall(r'\b(19|20)\d{2}\b', fact)
            if years:
                old_year = years[0]
                new_year = str(int(old_year) + random.randint(-10, 10))
                return fact.replace(old_year, new_year, 1)

        elif halluc_type == "entity_substitution":
            # Replace named entities (simplified)
            words = fact.split()
            if len(words) > 3:
                # Find capitalized words (likely entities)
                entities = [w for w in words if w[0].isupper() and len(w) > 2]
                if entities:
                    old_entity = random.choice(entities)
                    new_entity = random.choice(["Smith", "Johnson", "Williams", "Brown"])
                    return fact.replace(old_entity, new_entity, 1)

        # Default: add "approximately" or change verb tense
        return fact.replace("is", "was", 1) if "is" in fact else f"Approximately, {fact}"

    def _build_corpus(self, true_facts: List[Dict], red_herrings: int, size: int) -> List[Dict]:
        """Build evidence corpus with true facts and red herrings"""
        corpus = list(true_facts)

        # Add red herrings (misleading but non-supporting sources)
        for i in range(red_herrings):
            corpus.append({
                "source_id": f"RED_HERRING_{i:03d}",
                "title": f"Misleading Source {i}",
                "domain": random.choice(DOMAINS),
                "snippet": "This source contains partially related but ultimately non-supporting information.",
                "is_true": False
            })

        # Pad to target size with neutral sources
        while len(corpus) < size:
            corpus.append({
                "source_id": f"NEUTRAL_{len(corpus):03d}",
                "title": f"Neutral Source {len(corpus)}",
                "domain": random.choice(DOMAINS),
                "snippet": "This is a neutral source with general information.",
                "is_true": True
            })

        random.shuffle(corpus)
        return corpus[:size]

    def _template_query(self, domain: str, true_facts: List[Dict], hallucinations: List[Dict]) -> str:
        """Generate a query that incorporates both true facts and hallucinations"""
        templates = {
            "biomedical": "What are the key medical facts about {}?",
            "historical": "Explain the historical context of {}.",
            "legal": "What are the legal principles regarding {}?",
            "financial": "Describe the financial aspects of {}.",
            "scientific": "What are the scientific facts about {}?"
        }

        template = templates.get(domain, "Explain the facts about {}.")

        # Extract topic from first fact
        topic = "this topic"
        if true_facts:
            snippet = true_facts[0]["snippet"]
            words = snippet.split()
            if len(words) > 3:
                topic = " ".join(words[:5])

        return template.format(topic)

    def _load_templates(self) -> Dict:
        """Load case templates (placeholder)"""
        return {}
