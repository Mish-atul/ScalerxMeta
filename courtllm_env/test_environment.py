#!/usr/bin/env python
"""
Quick test script for CourtLLM environment
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models import CourtAction, CourtObservation, CourtState
from server.courtllm_environment import CourtLLMEnvironment

def test_environment():
    """Test basic environment functionality"""
    print("=" * 60)
    print("CourtLLM Environment Test")
    print("=" * 60)

    # Initialize environment
    print("\n1. Initializing environment...")
    env = CourtLLMEnvironment(stage=0)
    print("   ✓ Environment initialized")

    # Test reset
    print("\n2. Testing reset()...")
    obs = env.reset()
    print(f"   ✓ Case ID: {obs.case_id}")
    print(f"   ✓ Query: {obs.plaintiff_query[:80]}...")
    print(f"   ✓ Evidence sources: {len(obs.evidence_corpus)}")
    print(f"   ✓ Flagged claims: {len(obs.flagged_claims)}")

    # Test step with valid citation
    print("\n3. Testing step() with valid citation...")
    if obs.evidence_corpus:
        valid_source = obs.evidence_corpus[0]["source_id"]
        action = CourtAction(
            action_type="generate_testimony",
            content="This is a factual claim supported by evidence",
            claim_ids=["claim_000"],
            confidence=0.85,
            source_ids=[valid_source]
        )
        result = env.step(action)
        print(f"   ✓ Reward: {result.reward:.3f}")
        print(f"   ✓ Verdict: {'Acquitted' if result.reward > 0 else 'Convicted'}")
        print(f"   ✓ Step count: {result.step_count}")

    # Test step with invalid citation
    print("\n4. Testing step() with invalid citation...")
    env.reset()
    action = CourtAction(
        action_type="generate_testimony",
        content="This claim cites a non-existent source",
        claim_ids=["claim_001"],
        confidence=0.9,
        source_ids=["FAKE_SOURCE_999"]
    )
    result = env.step(action)
    print(f"   ✓ Reward: {result.reward:.3f}")
    print(f"   ✓ Verdict: {'Acquitted' if result.reward > 0 else 'Convicted'}")

    # Test state
    print("\n5. Testing state()...")
    state = env.state()
    print(f"   ✓ Episode ID: {state.episode_id[:16]}...")
    print(f"   ✓ Stage: {state.stage}")
    print(f"   ✓ Convictions: {state.total_convictions}")
    print(f"   ✓ Acquittals: {state.total_acquittals}")

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)

if __name__ == "__main__":
    test_environment()
