import httpx

base = "https://mishatul-courtllm-openenv.hf.space"

# Test health
print("=== Health Check ===")
r = httpx.get(f"{base}/health", timeout=30)
print(f"Status: {r.status_code}")
print(f"Response: {r.json()}")

# Test reset
print("\n=== Reset Environment ===")
r = httpx.post(f"{base}/reset", timeout=30)
print(f"Status: {r.status_code}")
obs = r.json()
print(f"Case ID: {obs['case_id']}")
print(f"Query: {obs['plaintiff_query'][:80]}...")
print(f"Evidence sources: {len(obs['evidence_corpus'])}")
print(f"Flagged claims: {len(obs['flagged_claims'])}")

# Test step
print("\n=== Step (Valid Citation) ===")
source_id = obs["evidence_corpus"][0]["source_id"] if obs["evidence_corpus"] else "TEST_001"
payload = {
    "action_type": "generate_testimony",
    "content": "This is a factual claim supported by evidence",
    "claim_ids": ["claim_000"],
    "confidence": 0.85,
    "source_ids": [source_id]
}
r = httpx.post(f"{base}/step", json=payload, timeout=30)
result = r.json()
print(f"Status: {r.status_code}")
print(f"Reward: {result['reward']:.3f}")
verdict = "ACQUITTED" if result["reward"] > 0 else "CONVICTED"
print(f"Verdict: {verdict}")
print(f"Done: {result['done']}")

# Test state
print("\n=== State ===")
r = httpx.get(f"{base}/state", timeout=30)
state = r.json()
print(f"Stage: {state['stage']}")
print(f"Acquittals: {state['total_acquittals']}")
print(f"Convictions: {state['total_convictions']}")

print("\n=== ALL ENDPOINTS WORKING! ===")
print(f"Live URL: {base}")
print(f"Space: https://huggingface.co/spaces/mishatul/CourtLLM_OpenEnv")
