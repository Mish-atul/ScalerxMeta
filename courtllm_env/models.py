# models.py — follows OpenEnv's "Type-Safe by Design" pattern

from dataclasses import dataclass, field
from typing import List, Optional, Literal

# ─────────────────────────────────────────────
# ACTION (what the Defendant can do)
# ─────────────────────────────────────────────
@dataclass
class CourtAction:
    action_type: Literal[
        "generate_testimony",   # Respond to query with citations
        "cite_source",          # Add an additional citation to a claim
        "concede_claim",        # Proactively withdraw a weak claim
        "request_clarification" # Ask for query clarification
    ]
    content: str                        # Testimony text
    claim_ids: List[str]                # Claims this action addresses
    confidence: float                   # Self-reported confidence [0.0, 1.0]
    source_ids: Optional[List[str]] = field(default_factory=list)

# ─────────────────────────────────────────────
# OBSERVATION (what the Defendant sees)
# ─────────────────────────────────────────────
@dataclass
class CourtObservation:
    case_id: str
    plaintiff_query: str
    flagged_claims: List[dict]          # [{claim_id, claim_text, suspicion_reason}]
    evidence_corpus: List[dict]         # [{source_id, title, domain, snippet}]
    jury_questions: List[str]
    prior_rulings: List[dict]           # [{claim_id, ruling, reason, step}]
    verdict_tally: dict                 # {claim_id: {accept: N, reject: M}}
    step_count: int
    done: bool
    reward: float

# ─────────────────────────────────────────────
# STATE (episode metadata)
# ─────────────────────────────────────────────
@dataclass
class CourtState:
    episode_id: str
    step_count: int
    stage: int                          # Curriculum stage: 0, 1, 2, 3
    active_claims: List[str]
    total_convictions: int
    total_acquittals: int
    timestamp: float
