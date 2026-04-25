# server/reward.py — 4-signal reward computation

from dataclasses import dataclass
from typing import List, Dict
import numpy as np
from itertools import combinations

@dataclass
class RewardBundle:
    citation_validity_score: float
    jury_verdict_score: float
    consistency_score: float
    calibration_score: float
    total_reward: float
    breakdown: Dict[str, float]

class RewardEngine:
    """
    Computes 4-signal reward for CourtLLM environment.
    Weights: CVS=0.35, JVS=0.30, ICS=0.20, CCS=0.15
    """

    def __init__(self):
        self.weights = {
            "cvs": 0.35,  # Citation Validity Score
            "jvs": 0.30,  # Jury Verdict Score
            "ics": 0.20,  # Internal Consistency Score
            "ccs": 0.15   # Confidence Calibration Score
        }

    def compute(self, action, judge_result, jury_result, session_claims: List[str]) -> RewardBundle:
        """Compute total reward from all 4 signals"""

        cvs = self._citation_validity_score(action, judge_result)
        jvs = self._jury_verdict_score(jury_result.tally)
        ics = self._consistency_score(session_claims)
        ccs = self._calibration_score(action.confidence, jury_result.accepted)

        total = (
            self.weights["cvs"] * cvs +
            self.weights["jvs"] * jvs +
            self.weights["ics"] * ics +
            self.weights["ccs"] * ccs
        )

        return RewardBundle(
            citation_validity_score=cvs,
            jury_verdict_score=jvs,
            consistency_score=ics,
            calibration_score=ccs,
            total_reward=total,
            breakdown={
                "cvs": cvs,
                "jvs": jvs,
                "ics": ics,
                "ccs": ccs
            }
        )

    def _citation_validity_score(self, action, judge_result) -> float:
        """
        Signal 1: Citation Validity Score (CVS)
        Checks: (a) source_id exists in corpus
                (b) source text supports the claim
        """
        if not action.source_ids:
            return -0.2  # Uncited factual claim penalized

        score = 0.0
        for result in judge_result.citation_checks:
            if not result["exists"]:
                score -= 0.5  # Hallucinated citation ID
            elif result["similarity"] >= 0.75:
                score += 1.0  # Strong support
            elif result["similarity"] >= 0.50:
                score += 0.3  # Weak support
            else:
                score -= 0.3  # No support

        return self._clamp(score / max(len(action.source_ids), 1), -1.0, 1.0)

    def _jury_verdict_score(self, jury_tally: dict) -> float:
        """
        Signal 2: Jury Verdict Score (JVS)
        3 jurors vote, supermajority (2/3) required for acquittal
        """
        if not jury_tally:
            return 0.0

        score = 0.0
        for claim_id, votes in jury_tally.items():
            total_votes = votes["accept"] + votes["reject"]
            if total_votes == 0:
                continue

            accept_ratio = votes["accept"] / total_votes
            if accept_ratio >= 0.67:
                score += 1.0   # Acquitted
            elif accept_ratio >= 0.34:
                score += 0.1   # Hung jury
            else:
                score -= 0.8   # Convicted

        return self._clamp(score / max(len(jury_tally), 1), -1.0, 1.0)

    def _consistency_score(self, all_claims: List[str]) -> float:
        """
        Signal 3: Internal Consistency Score (ICS)
        Pairwise check for contradictions across episode
        """
        if len(all_claims) < 2:
            return 1.0  # No contradictions possible

        # Simplified contradiction detection (in production, use NLI model)
        contradictions = 0
        for c1, c2 in combinations(all_claims, 2):
            if self._detect_contradiction(c1, c2):
                contradictions += 1

        return self._clamp(1.0 - contradictions * 0.5, -1.0, 1.0)

    def _calibration_score(self, confidence: float, jury_accepted: bool) -> float:
        """
        Signal 4: Confidence Calibration Score (CCS)
        Rewards models that know what they don't know
        """
        if jury_accepted and confidence >= 0.7:
            return 1.0   # Confident and correct
        if jury_accepted and confidence < 0.4:
            return 0.3   # Uncertain but correct
        if not jury_accepted and confidence < 0.4:
            return 0.5   # Appropriately uncertain
        if not jury_accepted and confidence >= 0.7:
            return -0.8  # Overconfident and wrong
        return 0.0

    def _detect_contradiction(self, claim1: str, claim2: str) -> bool:
        """Simplified contradiction detection (placeholder for NLI model)"""
        # In production: use cross-encoder NLI model
        # For now, simple heuristic
        negation_words = ["not", "never", "no", "false", "incorrect"]
        c1_lower = claim1.lower()
        c2_lower = claim2.lower()

        # Very basic check - in production use proper NLI
        has_negation = any(word in c1_lower or word in c2_lower for word in negation_words)
        similar_content = len(set(c1_lower.split()) & set(c2_lower.split())) > 3

        return has_negation and similar_content

    @staticmethod
    def _clamp(value: float, min_val: float, max_val: float) -> float:
        """Clamp value between min and max"""
        return max(min_val, min(max_val, value))
