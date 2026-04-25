#!/usr/bin/env python
"""
Simple example demonstrating CourtLLM environment usage
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models import CourtAction
from server.courtllm_environment import CourtLLMEnvironment

def main():
    print("=" * 70)
    print("CourtLLM Environment Demo")
    print("=" * 70)

    # Initialize environment
    env = CourtLLMEnvironment(stage=0)
    print("\n[1] Environment initialized (Stage 0 - Warm-up)")

    # Reset to get a new case
    obs = env.reset()
    print(f"\n[2] New case generated:")
    print(f"    Case ID: {obs.case_id}")
    print(f"    Query: {obs.plaintiff_query}")
    print(f"    Flagged claims: {len(obs.flagged_claims)}")
    print(f"    Evidence sources available: {len(obs.evidence_corpus)}")

    # Show first flagged claim
    if obs.flagged_claims:
        claim = obs.flagged_claims[0]
        print(f"\n[3] First flagged claim:")
        print(f"    ID: {claim['claim_id']}")
        print(f"    Text: {claim['claim_text']}")
        print(f"    Suspicion: {claim['suspicion_reason']}")

    # Show some evidence sources
    print(f"\n[4] Sample evidence sources:")
    for i, source in enumerate(obs.evidence_corpus[:3]):
        print(f"    [{source['source_id']}] {source['title']}")
        print(f"        {source['snippet'][:80]}...")

    # Example 1: Valid testimony with citation
    print("\n" + "=" * 70)
    print("Example 1: Valid Testimony with Citation")
    print("=" * 70)

    valid_source = obs.evidence_corpus[0]["source_id"]
    action1 = CourtAction(
        action_type="generate_testimony",
        content=f"Based on the evidence, {obs.evidence_corpus[0]['snippet'][:50]}",
        claim_ids=["claim_000"],
        confidence=0.85,
        source_ids=[valid_source]
    )

    result1 = env.step(action1)
    print(f"\nDefendant's testimony: {action1.content}")
    print(f"Cited source: {action1.source_ids[0]}")
    print(f"Confidence: {action1.confidence}")
    print(f"\nVerdict:")
    print(f"  Reward: {result1.reward:.3f}")
    print(f"  Status: {'ACQUITTED' if result1.reward > 0 else 'CONVICTED'}")
    print(f"  Jury questions: {len(result1.jury_questions)}")

    # Example 2: Invalid testimony (hallucinated citation)
    print("\n" + "=" * 70)
    print("Example 2: Invalid Testimony (Hallucinated Citation)")
    print("=" * 70)

    env.reset()  # Start fresh episode
    action2 = CourtAction(
        action_type="generate_testimony",
        content="This claim is supported by a non-existent source",
        claim_ids=["claim_001"],
        confidence=0.9,
        source_ids=["FAKE_SOURCE_999"]
    )

    result2 = env.step(action2)
    print(f"\nDefendant's testimony: {action2.content}")
    print(f"Cited source: {action2.source_ids[0]}")
    print(f"Confidence: {action2.confidence}")
    print(f"\nVerdict:")
    print(f"  Reward: {result2.reward:.3f}")
    print(f"  Status: {'ACQUITTED' if result2.reward > 0 else 'CONVICTED'}")
    print(f"  Penalty for hallucinated citation!")

    # Example 3: Uncertain testimony (no citation)
    print("\n" + "=" * 70)
    print("Example 3: Uncertain Testimony (No Citation)")
    print("=" * 70)

    env.reset()
    action3 = CourtAction(
        action_type="generate_testimony",
        content="I'm not certain about this claim and cannot find supporting evidence",
        claim_ids=["claim_002"],
        confidence=0.3,  # Low confidence
        source_ids=[]  # No citations
    )

    result3 = env.step(action3)
    print(f"\nDefendant's testimony: {action3.content}")
    print(f"Cited sources: {action3.source_ids if action3.source_ids else 'None'}")
    print(f"Confidence: {action3.confidence}")
    print(f"\nVerdict:")
    print(f"  Reward: {result3.reward:.3f}")
    print(f"  Status: {'ACQUITTED' if result3.reward > 0 else 'CONVICTED'}")
    print(f"  Appropriate uncertainty is rewarded!")

    # Show final statistics
    state = env.state()
    print("\n" + "=" * 70)
    print("Episode Statistics")
    print("=" * 70)
    print(f"Total steps: {state.step_count}")
    print(f"Convictions: {state.total_convictions}")
    print(f"Acquittals: {state.total_acquittals}")
    print(f"Stage: {state.stage}")

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. Valid citations with supporting evidence = positive reward")
    print("  2. Hallucinated citations = negative reward")
    print("  3. Appropriate uncertainty (low confidence) = partial credit")
    print("  4. The environment trains LLMs to be factually grounded")

if __name__ == "__main__":
    main()
