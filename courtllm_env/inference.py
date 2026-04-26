#!/usr/bin/env python3
"""
inference.py — Required by OpenEnv validator.
Runs 3 tasks against the CourtLLM environment to demonstrate end-to-end functionality.
"""

import sys
import os
import json
import time
import httpx

# ─── Configuration ───────────────────────────────────────────────
BASE_URL = os.environ.get("ENV_URL", "http://localhost:7860")
TIMEOUT = 30.0


def reset_env(client):
    """Reset the environment, handling both empty and JSON body."""
    try:
        r = client.post(f"{BASE_URL}/reset", json={})
        r.raise_for_status()
        return r.json()
    except Exception:
        # Fallback: try with no body
        r = client.post(f"{BASE_URL}/reset")
        r.raise_for_status()
        return r.json()


def step_env(client, action):
    """Send an action to the environment."""
    r = client.post(f"{BASE_URL}/step", json=action)
    r.raise_for_status()
    return r.json()


def health_check(client):
    """Check if environment is healthy."""
    r = client.get(f"{BASE_URL}/health")
    r.raise_for_status()
    return r.json()


def run_task(task_id, seed=42):
    """Run a single task: reset → build action from observation → step → return result."""
    print(f"\n{'='*50}")
    print(f"Task {task_id} (seed={seed})")
    print(f"{'='*50}")

    with httpx.Client(timeout=TIMEOUT) as client:
        # 1. Reset environment
        obs = reset_env(client)
        print(f"  Case ID: {obs['case_id']}")
        print(f"  Query: {obs['plaintiff_query'][:100]}...")
        print(f"  Flagged claims: {len(obs['flagged_claims'])}")
        print(f"  Evidence sources: {len(obs['evidence_corpus'])}")

        # 2. Build a defense action from observation
        # Pick a claim to address
        claim_ids = []
        source_ids = []

        if obs["flagged_claims"]:
            claim_ids = [obs["flagged_claims"][0]["claim_id"]]

        if obs["evidence_corpus"]:
            source_ids = [obs["evidence_corpus"][0]["source_id"]]

        # Build action using evidence from the corpus
        content = f"Regarding the query: {obs['plaintiff_query'][:200]}. "
        if obs["evidence_corpus"]:
            first_evidence = obs["evidence_corpus"][0]
            content += (
                f"Based on source {first_evidence['source_id']}: "
                f"{first_evidence.get('snippet', first_evidence.get('title', 'N/A'))[:300]}"
            )

        action = {
            "action_type": "generate_testimony",
            "content": content,
            "claim_ids": claim_ids,
            "confidence": 0.75,
            "source_ids": source_ids,
        }

        # 3. Step environment
        result = step_env(client, action)
        reward = result["reward"]
        verdict = "ACQUITTED" if reward > 0 else "CONVICTED"

        print(f"  Reward: {reward:.3f}")
        print(f"  Verdict: {verdict}")
        print(f"  Done: {result['done']}")

        return {
            "task_id": task_id,
            "seed": seed,
            "case_id": obs["case_id"],
            "reward": reward,
            "verdict": verdict,
            "step_count": result["step_count"],
            "done": result["done"],
        }


def wait_for_server(max_wait=60):
    """Wait for the server to become available."""
    print(f"Waiting for server at {BASE_URL}...")
    start = time.time()
    while time.time() - start < max_wait:
        try:
            with httpx.Client(timeout=5.0) as client:
                r = client.get(f"{BASE_URL}/health")
                if r.status_code == 200:
                    print(f"  Server ready! ({time.time() - start:.1f}s)")
                    return True
        except Exception:
            pass
        time.sleep(2)
    print(f"  Server not available after {max_wait}s")
    return False


def main():
    """Run inference tasks."""
    print("=" * 60)
    print("CourtLLM Inference — OpenEnv Validator")
    print(f"  Target: {BASE_URL}")
    print("=" * 60)

    # Wait for server
    if not wait_for_server():
        print("ERROR: Server not reachable. Exiting.")
        sys.exit(1)

    # Run 3 tasks
    try:
        results = [
            run_task(1, seed=42),
            run_task(2, seed=42),
            run_task(3, seed=42),
        ]
    except Exception as e:
        print(f"\nERROR during task execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(f"  Task {r['task_id']}: reward={r['reward']:.3f}, verdict={r['verdict']}")

    avg_reward = sum(r["reward"] for r in results) / len(results)
    print(f"\n  Average Reward: {avg_reward:.3f}")
    print(f"  Tasks Completed: {len(results)}/3")

    # Save results
    output_path = "inference_results.json"
    with open(output_path, "w") as f:
        json.dump({"results": results, "avg_reward": avg_reward}, f, indent=2)
    print(f"\n  Results saved to: {output_path}")
    print("Done!")


if __name__ == "__main__":
    main()
