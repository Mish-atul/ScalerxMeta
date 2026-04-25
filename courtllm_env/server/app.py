# server/app.py — FastAPI server for CourtLLM Environment

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
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

# ─── Web UI ───────────────────────────────────────────────────
WEB_UI_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CourtLLM — Courtroom for LLM Hallucination Reduction</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        :root {
            --bg-dark: #0a0a0f;
            --bg-card: #12121a;
            --bg-card-hover: #1a1a25;
            --border: #2a2a3a;
            --text-primary: #e8e8f0;
            --text-secondary: #8888a0;
            --text-muted: #55556a;
            --accent: #6c63ff;
            --accent-glow: rgba(108, 99, 255, 0.3);
            --green: #2ecc71;
            --red: #e74c3c;
            --yellow: #f1c40f;
            --gradient-1: linear-gradient(135deg, #6c63ff 0%, #a855f7 100%);
            --gradient-2: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
            --gradient-3: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        }
        
        body {
            font-family: 'Inter', -apple-system, sans-serif;
            background: var(--bg-dark);
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
        }
        
        .bg-pattern {
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: 
                radial-gradient(circle at 20% 20%, rgba(108,99,255,0.06) 0%, transparent 50%),
                radial-gradient(circle at 80% 80%, rgba(168,85,247,0.04) 0%, transparent 50%);
            pointer-events: none;
            z-index: 0;
        }
        
        .container {
            max-width: 1100px;
            margin: 0 auto;
            padding: 2rem 1.5rem;
            position: relative;
            z-index: 1;
        }
        
        header {
            text-align: center;
            margin-bottom: 2.5rem;
        }
        
        .logo {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }
        
        h1 {
            font-size: 2rem;
            font-weight: 800;
            background: var(--gradient-1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        
        .subtitle {
            color: var(--text-secondary);
            font-size: 0.95rem;
            font-weight: 400;
        }
        
        .badge-row {
            display: flex;
            gap: 0.5rem;
            justify-content: center;
            margin-top: 1rem;
            flex-wrap: wrap;
        }
        
        .badge {
            padding: 0.25rem 0.75rem;
            border-radius: 99px;
            font-size: 0.75rem;
            font-weight: 600;
            border: 1px solid var(--border);
            background: var(--bg-card);
            color: var(--text-secondary);
        }
        
        .badge.live {
            border-color: var(--green);
            color: var(--green);
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { box-shadow: 0 0 0 0 rgba(46,204,113,0.3); }
            50% { box-shadow: 0 0 0 6px rgba(46,204,113,0); }
        }
        
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 1.5rem; }
        
        .card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1.5rem;
            transition: all 0.3s ease;
        }
        
        .card:hover {
            border-color: var(--accent);
            box-shadow: 0 0 30px var(--accent-glow);
            transform: translateY(-2px);
        }
        
        .card h2 {
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .card.full { grid-column: 1 / -1; }
        
        button {
            background: var(--gradient-1);
            color: white;
            border: none;
            padding: 0.65rem 1.5rem;
            border-radius: 10px;
            font-weight: 600;
            font-size: 0.85rem;
            cursor: pointer;
            transition: all 0.2s;
            font-family: 'Inter', sans-serif;
        }
        
        button:hover { opacity: 0.9; transform: scale(1.02); }
        button:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }
        
        .btn-secondary {
            background: var(--bg-card-hover);
            border: 1px solid var(--border);
        }
        
        pre, .output {
            background: #0d0d14;
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 1rem;
            font-size: 0.8rem;
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
            overflow-x: auto;
            max-height: 300px;
            overflow-y: auto;
            color: var(--text-secondary);
            line-height: 1.6;
        }
        
        .stat-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1rem;
        }
        
        .stat {
            text-align: center;
            padding: 1rem;
            background: var(--bg-dark);
            border-radius: 12px;
            border: 1px solid var(--border);
        }
        
        .stat-value {
            font-size: 1.5rem;
            font-weight: 800;
        }
        
        .stat-label {
            font-size: 0.7rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 0.25rem;
        }
        
        .reward-bar {
            height: 8px;
            background: var(--bg-dark);
            border-radius: 99px;
            overflow: hidden;
            margin: 0.5rem 0;
        }
        
        .reward-fill {
            height: 100%;
            border-radius: 99px;
            transition: width 0.5s ease;
        }
        
        .reward-fill.positive { background: var(--gradient-2); }
        .reward-fill.negative { background: var(--gradient-3); }
        
        textarea, input, select {
            background: var(--bg-dark);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 0.65rem 1rem;
            color: var(--text-primary);
            font-family: 'Inter', sans-serif;
            font-size: 0.85rem;
            width: 100%;
            transition: border-color 0.2s;
        }
        
        textarea:focus, input:focus, select:focus {
            outline: none;
            border-color: var(--accent);
        }
        
        label {
            display: block;
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.4rem;
        }
        
        .form-row { display: flex; gap: 1rem; margin-bottom: 1rem; }
        .form-row > div { flex: 1; }
        
        .verdict-badge {
            display: inline-block;
            padding: 0.3rem 1rem;
            border-radius: 99px;
            font-weight: 700;
            font-size: 0.85rem;
        }
        
        .verdict-badge.acquitted { background: rgba(46,204,113,0.15); color: var(--green); }
        .verdict-badge.convicted { background: rgba(231,76,60,0.15); color: var(--red); }
        
        .api-table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
        .api-table th { text-align: left; color: var(--text-muted); font-weight: 600; padding: 0.5rem; border-bottom: 1px solid var(--border); }
        .api-table td { padding: 0.5rem; border-bottom: 1px solid rgba(42,42,58,0.5); }
        .api-table code { background: var(--bg-dark); padding: 0.15rem 0.5rem; border-radius: 5px; font-size: 0.75rem; }
        
        .method { font-weight: 700; padding: 0.15rem 0.5rem; border-radius: 5px; font-size: 0.7rem; }
        .method.post { background: rgba(108,99,255,0.15); color: var(--accent); }
        .method.get { background: rgba(46,204,113,0.15); color: var(--green); }

        @media (max-width: 768px) {
            .grid { grid-template-columns: 1fr; }
            .stat-grid { grid-template-columns: repeat(2, 1fr); }
            .form-row { flex-direction: column; }
            h1 { font-size: 1.5rem; }
        }
    </style>
</head>
<body>
    <div class="bg-pattern"></div>
    <div class="container">
        <header>
            <div class="logo">⚖️</div>
            <h1>CourtLLM Environment</h1>
            <p class="subtitle">Multi-Agent Courtroom for LLM Hallucination Reduction</p>
            <div class="badge-row">
                <span class="badge live">● Live</span>
                <span class="badge">OpenEnv v0.2.3</span>
                <span class="badge">4-Signal Reward</span>
                <span class="badge">GRPO Training</span>
            </div>
        </header>

        <div class="grid">
            <div class="card">
                <h2>🏛️ Environment Controls</h2>
                <div style="display:flex;gap:0.75rem;margin-bottom:1rem;">
                    <button onclick="resetEnv()">Reset Environment</button>
                    <button class="btn-secondary" onclick="checkHealth()">Health Check</button>
                </div>
                <div id="controlOutput" class="output" style="min-height:80px;">Click "Reset Environment" to generate a new case...</div>
            </div>

            <div class="card">
                <h2>📊 Episode Stats</h2>
                <div class="stat-grid">
                    <div class="stat"><div class="stat-value" id="statReward">—</div><div class="stat-label">Reward</div></div>
                    <div class="stat"><div class="stat-value" id="statStep">0</div><div class="stat-label">Step</div></div>
                    <div class="stat"><div class="stat-value" id="statAcquit">0</div><div class="stat-label">Acquittals</div></div>
                    <div class="stat"><div class="stat-value" id="statConvict">0</div><div class="stat-label">Convictions</div></div>
                </div>
            </div>

            <div class="card full">
                <h2>🎤 Defendant Testimony</h2>
                <div class="form-row">
                    <div style="flex:2;">
                        <label>Testimony Content</label>
                        <textarea id="content" rows="3" placeholder="Enter your factual claim with supporting evidence..."></textarea>
                    </div>
                </div>
                <div class="form-row">
                    <div>
                        <label>Action Type</label>
                        <select id="actionType">
                            <option value="generate_testimony">Generate Testimony</option>
                            <option value="cite_source">Cite Source</option>
                            <option value="concede_claim">Concede Claim</option>
                            <option value="request_clarification">Request Clarification</option>
                        </select>
                    </div>
                    <div>
                        <label>Claim IDs (comma-separated)</label>
                        <input id="claimIds" placeholder="claim_000" value="claim_000" />
                    </div>
                    <div>
                        <label>Confidence (0.0 - 1.0)</label>
                        <input id="confidence" type="number" min="0" max="1" step="0.05" value="0.85" />
                    </div>
                    <div>
                        <label>Source IDs (comma-separated)</label>
                        <input id="sourceIds" placeholder="BIOMEDICAL_0001" />
                    </div>
                </div>
                <button onclick="submitAction()" id="submitBtn">Submit Testimony ⚖️</button>
                <div id="verdict" style="margin-top:1rem;"></div>
            </div>

            <div class="card full">
                <h2>📜 Case Details & Evidence</h2>
                <div id="caseDetails" class="output" style="min-height:120px;">No case loaded. Click "Reset Environment" to begin.</div>
            </div>

            <div class="card full">
                <h2>🔌 API Reference</h2>
                <table class="api-table">
                    <thead><tr><th>Method</th><th>Endpoint</th><th>Description</th></tr></thead>
                    <tbody>
                        <tr><td><span class="method get">GET</span></td><td><code>/health</code></td><td>Health check</td></tr>
                        <tr><td><span class="method post">POST</span></td><td><code>/reset</code></td><td>Reset environment, get new case</td></tr>
                        <tr><td><span class="method post">POST</span></td><td><code>/step</code></td><td>Submit action, get observation + reward</td></tr>
                        <tr><td><span class="method get">GET</span></td><td><code>/state</code></td><td>Get current episode state</td></tr>
                        <tr><td><span class="method post">POST</span></td><td><code>/set_stage/{n}</code></td><td>Set curriculum stage (0-3)</td></tr>
                        <tr><td><span class="method get">GET</span></td><td><code>/docs</code></td><td>OpenAPI documentation</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        let currentObs = null;

        async function checkHealth() {
            const out = document.getElementById('controlOutput');
            try {
                const r = await fetch('/health');
                const d = await r.json();
                out.textContent = JSON.stringify(d, null, 2);
            } catch(e) { out.textContent = 'Error: ' + e.message; }
        }

        async function resetEnv() {
            const out = document.getElementById('controlOutput');
            out.textContent = 'Generating new case...';
            try {
                const r = await fetch('/reset', { method: 'POST' });
                currentObs = await r.json();
                out.textContent = 'Case generated!\\nQuery: ' + currentObs.plaintiff_query + '\\nClaims: ' + currentObs.flagged_claims.length + '\\nEvidence sources: ' + currentObs.evidence_corpus.length;
                
                document.getElementById('statReward').textContent = '—';
                document.getElementById('statStep').textContent = '0';
                document.getElementById('statAcquit').textContent = '0';
                document.getElementById('statConvict').textContent = '0';
                document.getElementById('verdict').innerHTML = '';
                
                let details = 'QUERY: ' + currentObs.plaintiff_query + '\\n\\n';
                details += 'FLAGGED CLAIMS:\\n';
                currentObs.flagged_claims.forEach(c => {
                    details += '  [' + c.claim_id + '] ' + c.claim_text + '\\n    Suspicion: ' + c.suspicion_reason + '\\n';
                });
                details += '\\nEVIDENCE CORPUS (' + currentObs.evidence_corpus.length + ' sources):\\n';
                currentObs.evidence_corpus.slice(0, 10).forEach(s => {
                    details += '  [' + s.source_id + '] ' + s.title + '\\n    ' + (s.snippet || '').substring(0, 100) + '\\n';
                });
                if (currentObs.evidence_corpus.length > 10) details += '  ... and ' + (currentObs.evidence_corpus.length - 10) + ' more sources';
                
                document.getElementById('caseDetails').textContent = details;
                
                if (currentObs.evidence_corpus.length > 0) {
                    document.getElementById('sourceIds').value = currentObs.evidence_corpus[0].source_id;
                }
                if (currentObs.flagged_claims.length > 0) {
                    document.getElementById('claimIds').value = currentObs.flagged_claims[0].claim_id;
                }
            } catch(e) { out.textContent = 'Error: ' + e.message; }
        }

        async function submitAction() {
            if (!currentObs) { alert('Reset the environment first!'); return; }
            const btn = document.getElementById('submitBtn');
            btn.disabled = true;
            try {
                const payload = {
                    action_type: document.getElementById('actionType').value,
                    content: document.getElementById('content').value,
                    claim_ids: document.getElementById('claimIds').value.split(',').map(s => s.trim()),
                    confidence: parseFloat(document.getElementById('confidence').value),
                    source_ids: document.getElementById('sourceIds').value ? document.getElementById('sourceIds').value.split(',').map(s => s.trim()) : []
                };
                
                const r = await fetch('/step', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload) });
                const obs = await r.json();
                currentObs = obs;
                
                const reward = obs.reward;
                document.getElementById('statReward').textContent = reward.toFixed(3);
                document.getElementById('statReward').style.color = reward > 0 ? 'var(--green)' : reward < 0 ? 'var(--red)' : 'var(--text-primary)';
                document.getElementById('statStep').textContent = obs.step_count;
                
                const stateR = await fetch('/state');
                const state = await stateR.json();
                document.getElementById('statAcquit').textContent = state.total_acquittals;
                document.getElementById('statConvict').textContent = state.total_convictions;
                
                const isAcquitted = reward > 0;
                document.getElementById('verdict').innerHTML = 
                    '<span class="verdict-badge ' + (isAcquitted ? 'acquitted' : 'convicted') + '">' +
                    (isAcquitted ? '✅ ACQUITTED' : '❌ CONVICTED') + '</span>' +
                    '  <span style="color:var(--text-secondary);font-size:0.85rem;">Reward: ' + reward.toFixed(3) + '</span>' +
                    (obs.done ? ' <span style="color:var(--yellow);font-weight:600;margin-left:1rem;">Episode Complete</span>' : '');
                
            } catch(e) { alert('Error: ' + e.message); }
            btn.disabled = false;
        }
    </script>
</body>
</html>
"""

# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def root():
    """Interactive Web UI"""
    return WEB_UI_HTML

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/reset")
async def reset(body: dict = None) -> ObservationResponse:
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
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
