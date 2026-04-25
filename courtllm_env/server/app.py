# server/app.py — FastAPI server

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from server.courtllm_environment import CourtLLMEnvironment
from models import CourtAction, CourtObservation, CourtState

# Create environment instance
env = CourtLLMEnvironment(stage=0)

# Create FastAPI app
app = FastAPI(
    title="CourtLLM Environment",
    description="Multi-agent courtroom environment for LLM hallucination reduction",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API
class ActionRequest(BaseModel):
    action_type: str
    content: str
    claim_ids: list[str]
    confidence: float
    source_ids: Optional[list[str]] = []

class ObservationResponse(BaseModel):
    case_id: str
    plaintiff_query: str
    flagged_claims: list[dict]
    evidence_corpus: list[dict]
    jury_questions: list[str]
    prior_rulings: list[dict]
    verdict_tally: dict
    step_count: int
    done: bool
    reward: float

class StateResponse(BaseModel):
    episode_id: str
    step_count: int
    stage: int
    active_claims: list[str]
    total_convictions: int
    total_acquittals: int
    timestamp: float

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "CourtLLM Environment",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/reset")
async def reset() -> ObservationResponse:
    """Reset environment and return initial observation"""
    try:
        obs = env.reset()
        return ObservationResponse(
            case_id=obs.case_id,
            plaintiff_query=obs.plaintiff_query,
            flagged_claims=obs.flagged_claims,
            evidence_corpus=obs.evidence_corpus,
            jury_questions=obs.jury_questions,
            prior_rulings=obs.prior_rulings,
            verdict_tally=obs.verdict_tally,
            step_count=obs.step_count,
            done=obs.done,
            reward=obs.reward
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/step")
async def step(action_req: ActionRequest) -> ObservationResponse:
    """Execute action and return observation + reward"""
    try:
        action = CourtAction(
            action_type=action_req.action_type,
            content=action_req.content,
            claim_ids=action_req.claim_ids,
            confidence=action_req.confidence,
            source_ids=action_req.source_ids or []
        )

        obs = env.step(action)

        return ObservationResponse(
            case_id=obs.case_id,
            plaintiff_query=obs.plaintiff_query,
            flagged_claims=obs.flagged_claims,
            evidence_corpus=obs.evidence_corpus,
            jury_questions=obs.jury_questions,
            prior_rulings=obs.prior_rulings,
            verdict_tally=obs.verdict_tally,
            step_count=obs.step_count,
            done=obs.done,
            reward=obs.reward
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/state")
async def get_state() -> StateResponse:
    """Get current episode state"""
    try:
        state = env.state()
        return StateResponse(
            episode_id=state.episode_id,
            step_count=state.step_count,
            stage=state.stage,
            active_claims=state.active_claims,
            total_convictions=state.total_convictions,
            total_acquittals=state.total_acquittals,
            timestamp=state.timestamp
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/set_stage/{stage}")
async def set_stage(stage: int):
    """Update curriculum stage"""
    try:
        env.set_stage(stage)
        return {"status": "success", "stage": stage}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
