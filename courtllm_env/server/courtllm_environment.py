# server/courtllm_environment.py — Main environment class

import time
from uuid import uuid4
from typing import Optional
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import CourtAction, CourtObservation, CourtState
from server.case_generator import CaseGenerator, Case
from server.judge import Judge
from server.jury import JuryPanel
from server.reward import RewardEngine
from server.evidence_db import EvidenceDatabase

class CourtLLMEnvironment:
    """
    OpenEnv-compatible courtroom environment for hallucination reduction.
    Implements the 3 required methods: reset(), step(), state()
    """

    def __init__(self, stage: int = 0):
        self.stage = stage
        self.evidence_db = EvidenceDatabase()
        self.generator = CaseGenerator(evidence_db=self.evidence_db.get_all())
        self.judge = Judge()
        self.jury = JuryPanel(n_jurors=3)
        self.reward_engine = RewardEngine()
        self._episode_state: Optional[CourtState] = None
        self._case: Optional[Case] = None
        self._session_claims: list = []

    def reset(self) -> CourtObservation:
        """Start a fresh episode — generate a new case."""
        case = self.generator.generate(difficulty=self.stage + 1)

        self._episode_state = CourtState(
            episode_id=str(uuid4()),
            step_count=0,
            stage=self.stage,
            active_claims=[c["claim_id"] for c in case.flagged_claims],
            total_convictions=0,
            total_acquittals=0,
            timestamp=time.time()
        )
        self._case = case
        self._session_claims = []  # tracks all Defendant claims for consistency check

        return CourtObservation(
            case_id=self._episode_state.episode_id,
            plaintiff_query=case.query,
            flagged_claims=case.flagged_claims,
            evidence_corpus=case.evidence_corpus,
            jury_questions=[],
            prior_rulings=[],
            verdict_tally={},
            step_count=0,
            done=False,
            reward=0.0
        )

    def step(self, action: CourtAction) -> CourtObservation:
        """Execute one Defendant action — run it through Judge + Jury."""
        if self._episode_state is None or self._case is None:
            raise RuntimeError("Must call reset() before step()")

        self._session_claims.append(action.content)
        self._episode_state.step_count += 1

        # 1. Judge runs deterministic checks
        judge_result = self.judge.evaluate(action, self._case.evidence_corpus)

        # 2. Jury deliberates
        jury_result = self.jury.deliberate(action, self._case.evidence_corpus)

        # 3. Compute 4-signal reward
        reward_bundle = self.reward_engine.compute(
            action=action,
            judge_result=judge_result,
            jury_result=jury_result,
            session_claims=self._session_claims
        )

        # 4. Update conviction/acquittal counts
        if jury_result.accepted:
            self._episode_state.total_acquittals += 1
        else:
            self._episode_state.total_convictions += 1

        # 5. Update active claims
        for claim_id in action.claim_ids:
            if claim_id in self._episode_state.active_claims:
                self._episode_state.active_claims.remove(claim_id)

        # 6. Check if episode is done
        done = (
            self._episode_state.step_count >= self._max_steps() or
            len(self._episode_state.active_claims) == 0
        )

        return CourtObservation(
            case_id=self._episode_state.episode_id,
            plaintiff_query=self._case.query,
            flagged_claims=self._case.flagged_claims,
            evidence_corpus=self._case.evidence_corpus,
            jury_questions=jury_result.clarifying_questions,
            prior_rulings=judge_result.rulings,
            verdict_tally=jury_result.tally,
            step_count=self._episode_state.step_count,
            done=done,
            reward=reward_bundle.total_reward
        )

    def state(self) -> CourtState:
        """Return current episode metadata."""
        if self._episode_state is None:
            raise RuntimeError("Must call reset() before state()")
        return self._episode_state

    def _max_steps(self) -> int:
        """Maximum steps per episode based on curriculum stage"""
        return {0: 3, 1: 4, 2: 5, 3: 6}.get(self.stage, 6)

    def set_stage(self, stage: int):
        """Update curriculum stage"""
        self.stage = stage
        if self._episode_state:
            self._episode_state.stage = stage
