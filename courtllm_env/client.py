# client.py — Public API (Server-Independent)

import httpx
from typing import Optional
from .models import CourtAction, CourtObservation, CourtState

class CourtLLMClient:
    """
    Client for the CourtLLM environment.

    Usage:
        # Remote Space
        client = CourtLLMClient("https://mishatul-courtllm-openenv.hf.space")
        obs = client.reset()
        result = client.step(CourtAction(...))
        client.close()

        # Or use as context manager
        with CourtLLMClient("http://localhost:8000") as client:
            obs = client.reset()
            result = client.step(action)
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=30.0)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close the HTTP client"""
        self.client.close()

    def reset(self) -> CourtObservation:
        """Reset environment and return initial observation"""
        response = self.client.post(f"{self.base_url}/reset")
        response.raise_for_status()
        data = response.json()

        return CourtObservation(
            case_id=data["case_id"],
            plaintiff_query=data["plaintiff_query"],
            flagged_claims=data["flagged_claims"],
            evidence_corpus=data["evidence_corpus"],
            jury_questions=data["jury_questions"],
            prior_rulings=data["prior_rulings"],
            verdict_tally=data["verdict_tally"],
            step_count=data["step_count"],
            done=data["done"],
            reward=data["reward"]
        )

    def step(self, action: CourtAction) -> CourtObservation:
        """Execute action and return observation + reward"""
        payload = {
            "action_type": action.action_type,
            "content": action.content,
            "claim_ids": action.claim_ids,
            "confidence": action.confidence,
            "source_ids": action.source_ids or []
        }

        response = self.client.post(f"{self.base_url}/step", json=payload)
        response.raise_for_status()
        data = response.json()

        return CourtObservation(
            case_id=data["case_id"],
            plaintiff_query=data["plaintiff_query"],
            flagged_claims=data["flagged_claims"],
            evidence_corpus=data["evidence_corpus"],
            jury_questions=data["jury_questions"],
            prior_rulings=data["prior_rulings"],
            verdict_tally=data["verdict_tally"],
            step_count=data["step_count"],
            done=data["done"],
            reward=data["reward"]
        )

    def state(self) -> CourtState:
        """Get current episode state"""
        response = self.client.get(f"{self.base_url}/state")
        response.raise_for_status()
        data = response.json()

        return CourtState(
            episode_id=data["episode_id"],
            step_count=data["step_count"],
            stage=data["stage"],
            active_claims=data["active_claims"],
            total_convictions=data["total_convictions"],
            total_acquittals=data["total_acquittals"],
            timestamp=data["timestamp"]
        )

    def set_stage(self, stage: int):
        """Update curriculum stage"""
        response = self.client.post(f"{self.base_url}/set_stage/{stage}")
        response.raise_for_status()
        return response.json()

    def health(self) -> dict:
        """Check server health"""
        response = self.client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
