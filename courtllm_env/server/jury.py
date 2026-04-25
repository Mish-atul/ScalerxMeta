# server/jury.py — 3-juror consensus panel

from dataclasses import dataclass
from typing import List, Dict, Tuple
import random

@dataclass
class JuryResult:
    tally: Dict[str, Dict[str, int]]  # {claim_id: {accept: N, reject: M}}
    clarifying_questions: List[str]
    accepted: bool  # Overall verdict

class JuryPanel:
    """
    3-juror consensus panel:
    - Juror A: Frozen LLM (temperature 0.3)
    - Juror B: Frozen LLM (temperature 0.9)
    - Juror C: Wikipedia fact-checker (deterministic)
    """

    def __init__(self, n_jurors=3):
        self.n_jurors = n_jurors
        # In production, initialize actual LLM jurors
        # For now, use rule-based placeholders

    def deliberate(self, action, corpus: List[dict]) -> JuryResult:
        """Run 3-juror deliberation on Defendant's testimony"""

        votes = []
        questions = []

        # Juror A: Conservative LLM (strict)
        vote_a, q_a = self._juror_a_evaluate(action, corpus)
        votes.append(vote_a)
        if q_a:
            questions.append(q_a)

        # Juror B: Liberal LLM (lenient)
        vote_b, q_b = self._juror_b_evaluate(action, corpus)
        votes.append(vote_b)
        if q_b:
            questions.append(q_b)

        # Juror C: Wikipedia fact-checker (deterministic)
        vote_c, q_c = self._juror_c_evaluate(action, corpus)
        votes.append(vote_c)
        if q_c:
            questions.append(q_c)

        # Tally votes for each claim
        tally = {}
        for claim_id in action.claim_ids:
            tally[claim_id] = {
                "accept": sum(1 for v in votes if v == "accept"),
                "reject": sum(1 for v in votes if v == "reject")
            }

        # Overall verdict: supermajority (2/3) required
        accept_votes = sum(1 for v in votes if v == "accept")
        accepted = accept_votes >= 2

        return JuryResult(
            tally=tally,
            clarifying_questions=questions,
            accepted=accepted
        )

    def _juror_a_evaluate(self, action, corpus: List[dict]) -> Tuple[str, str]:
        """
        Juror A: Conservative evaluator (strict on citations)
        In production: use frozen LLM with temp=0.3
        """
        # Placeholder: strict rule-based evaluation
        has_citations = len(action.source_ids) > 0
        high_confidence = action.confidence >= 0.7

        if has_citations and high_confidence:
            return "accept", None
        elif not has_citations:
            return "reject", "Where are your sources?"
        else:
            return "reject", "Confidence too low for this claim"

    def _juror_b_evaluate(self, action, corpus: List[dict]) -> Tuple[str, str]:
        """
        Juror B: Liberal evaluator (lenient, considers uncertainty)
        In production: use frozen LLM with temp=0.9
        """
        # Placeholder: lenient rule-based evaluation
        has_some_evidence = len(action.source_ids) > 0 or action.confidence < 0.5

        if has_some_evidence:
            return "accept", None
        else:
            return "reject", "Need at least some supporting evidence"

    def _juror_c_evaluate(self, action, corpus: List[dict]) -> Tuple[str, str]:
        """
        Juror C: Wikipedia fact-checker (deterministic, ungameable)
        In production: use Wikipedia API for entity verification
        """
        # Placeholder: check if sources exist in corpus
        corpus_ids = {s["source_id"] for s in corpus}
        valid_sources = all(sid in corpus_ids for sid in action.source_ids)

        if not action.source_ids:
            return "reject", "No verifiable sources provided"
        elif valid_sources:
            return "accept", None
        else:
            return "reject", "Some sources not found in evidence database"

class FrozenLLMJuror:
    """Placeholder for frozen LLM juror (to be implemented with actual model)"""
    def __init__(self, model: str, temp: float):
        self.model = model
        self.temp = temp

    def evaluate(self, content: str, corpus: List[dict]) -> Tuple[str, str]:
        # In production: call actual LLM
        return "accept", None

class WikipediaJuror:
    """Placeholder for Wikipedia API juror (to be implemented)"""
    def evaluate(self, content: str, corpus: List[dict]) -> Tuple[str, str]:
        # In production: query Wikipedia API
        return "accept", None
