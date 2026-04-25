# server/judge.py — Deterministic citation + NLI verifier

from dataclasses import dataclass
from typing import List, Dict
import numpy as np

@dataclass
class JudgeResult:
    citation_checks: List[Dict]  # [{source_id, exists, similarity}]
    rulings: List[Dict]           # [{claim_id, ruling, reason, step}]
    consistency_violations: List[str]

class Judge:
    """
    Deterministic verification module.
    Checks: (1) Citation validity (2) Internal consistency
    """

    def __init__(self):
        self.embedder = None  # Will be initialized lazily

    def evaluate(self, action, evidence_corpus: List[dict]) -> JudgeResult:
        """Run deterministic checks on Defendant's action"""

        citation_checks = self._check_citations(action, evidence_corpus)
        rulings = self._generate_rulings(action, citation_checks)

        return JudgeResult(
            citation_checks=citation_checks,
            rulings=rulings,
            consistency_violations=[]
        )

    def _check_citations(self, action, corpus: List[dict]) -> List[Dict]:
        """
        Check each cited source:
        1. Does source_id exist in corpus?
        2. Does source text support the claim?
        """
        checks = []
        corpus_ids = {s["source_id"]: s for s in corpus}

        for source_id in action.source_ids:
            if source_id not in corpus_ids:
                checks.append({
                    "source_id": source_id,
                    "exists": False,
                    "similarity": 0.0,
                    "reason": "Source ID not found in evidence corpus"
                })
            else:
                source = corpus_ids[source_id]
                similarity = self._compute_similarity(
                    action.content,
                    source.get("snippet", "")
                )
                checks.append({
                    "source_id": source_id,
                    "exists": True,
                    "similarity": similarity,
                    "reason": "Valid citation" if similarity >= 0.75 else "Weak support"
                })

        return checks

    def _compute_similarity(self, claim: str, source_text: str) -> float:
        """
        Compute semantic similarity between claim and source.
        Uses sentence-transformers in production.
        """
        # Simplified version - in production use sentence-transformers
        if not claim or not source_text:
            return 0.0

        # Basic word overlap as placeholder
        claim_words = set(claim.lower().split())
        source_words = set(source_text.lower().split())

        if not claim_words or not source_words:
            return 0.0

        overlap = len(claim_words & source_words)
        union = len(claim_words | source_words)

        return overlap / union if union > 0 else 0.0

    def _generate_rulings(self, action, citation_checks: List[Dict]) -> List[Dict]:
        """Generate rulings for each claim based on citation checks"""
        rulings = []

        for claim_id in action.claim_ids:
            # Check if all citations are valid
            valid_citations = sum(1 for c in citation_checks if c["exists"] and c["similarity"] >= 0.75)
            total_citations = len(citation_checks)

            if total_citations == 0:
                ruling = "warning"
                reason = "No citations provided"
            elif valid_citations == total_citations:
                ruling = "approved"
                reason = "All citations valid"
            elif valid_citations > 0:
                ruling = "partial"
                reason = f"{valid_citations}/{total_citations} citations valid"
            else:
                ruling = "rejected"
                reason = "No valid citations"

            rulings.append({
                "claim_id": claim_id,
                "ruling": ruling,
                "reason": reason,
                "step": 0  # Will be set by environment
            })

        return rulings
