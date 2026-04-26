# ⚖️ CourtLLM: Teaching LLMs to Stop Lying — By Putting Them on Trial

*OpenEnv Hackathon 2026 | Theme: Multi-Agent Interactions | Sub-themes: Halluminate + Fleet AI*

---

## The Problem: LLMs Hallucinate, and We Can't Stop Them

Large Language Models are remarkably fluent — and remarkably unreliable. They fabricate citations, invent statistics, and state falsehoods with supreme confidence. Current mitigation strategies (RAG, chain-of-thought, self-consistency) treat symptoms but never provide a **direct training signal** that penalizes hallucination at the generation level.

We asked: *What if we could put an LLM on trial for its hallucinations?*

---

## The Solution: An Adversarial Courtroom Environment

**CourtLLM** is a multi-agent reinforcement learning environment built on [OpenEnv](https://github.com/meta-pytorch/OpenEnv) that simulates a courtroom trial where an LLM must defend its claims under cross-examination.

### The Cast

| Role | Who | What They Do |
|---|---|---|
| **Plaintiff** | Case Generator | Submits queries containing potentially hallucinated claims |
| **Defendant** | The LLM (being trained) | Must defend claims with citations and calibrated confidence |
| **Judge** | Deterministic Verifier | Checks citation validity against a locked evidence corpus |
| **Jury** | 3-Agent Panel | 2 frozen LLMs + 1 Wikipedia fact-checker vote on each claim |

### The 4-Signal Reward

Unlike simple binary rewards, CourtLLM uses four independent verification signals — making it nearly impossible to game:

- **Citation Validity (35%)** — Does the cited source actually exist in the evidence corpus?
- **Jury Verdict (30%)** — Do independent juror agents agree the claim is factual?
- **Internal Consistency (20%)** — Does the testimony contradict prior statements?
- **Confidence Calibration (15%)** — Is the model's stated confidence aligned with reality?

**Total Reward** = 0.35×CVS + 0.30×JVS + 0.20×ICS + 0.15×CCS

This multi-signal design directly addresses the reward hacking concerns raised in the hackathon ceremony — a single clever exploit can't fool all four verifiers simultaneously.

---

## Training: GRPO on a College GPU Lab

We trained **Qwen/Qwen2.5-3B-Instruct** using **GRPO** (Group Relative Policy Optimization) via TRL's `GRPOTrainer`, with 4-bit quantization (NF4) and LoRA fine-tuning.

### Training Configuration

| Parameter | Value |
|---|---|
| Base Model | Qwen/Qwen2.5-3B-Instruct |
| Quantization | 4-bit NF4 (bitsandbytes) |
| Fine-tuning | LoRA (r=16, α=32) |
| RL Algorithm | GRPO (TRL GRPOTrainer) |
| Hardware | NVIDIA A100-PCIE-40GB (College GPU Lab) |
| Training Steps | 300 steps (~6 hours) |
| Trainable Parameters | 29.9M / 3.1B (0.96%) |

### Results

#### Reward Convergence

![Reward Curve](outputs/reward_curve_stage1.png)

The model's average episode reward converged from **0.008 → 0.548** within the first 50 steps, demonstrating rapid policy improvement. The agent learned to generate factually grounded, citation-backed testimony that survives cross-examination by the Judge and Jury panel.

#### Policy Loss Convergence

![Training Loss](outputs/training_loss_curve.png)

The GRPO policy loss dropped from **2.50 → 0.43** over 300 steps, with the smoothed curve (5-step moving average) showing clean exponential decay. Loss stabilized around step 200, indicating the policy had learned the optimal defense strategy.

#### Conviction Rate Drop

![Conviction Rate](outputs/conviction_rate_drop.png)

- **Baseline (untrained):** 61% conviction rate — the model hallucinated frequently
- **After GRPO training:** 0% conviction rate — the model learned to produce grounded responses
- **Improvement:** 61 percentage points reduction in hallucination convictions

#### Full Training Dashboard

![Training Dashboard](outputs/training_dashboard.png)

**(A) Reward vs Loss** — Inverse correlation confirms the model is genuinely improving, not reward hacking.
**(B) KL Divergence** — Policy drift stays below the 0.2 threshold, proving stable GRPO updates without catastrophic forgetting.
**(C) Reward Distribution** — Early training shows spread (0.0-0.5); late training is tightly clustered around 0.54, proving consistent behavior.
**(D) 4-Signal Radar** — All four reward signals improved dramatically; citation validity saw the largest gain (+0.57).

#### Convergence Speed & Curriculum Design

![Convergence Analysis](outputs/convergence_curriculum.png)

**Left:** The model reached R≥0.1 in just 10 steps, R≥0.3 in 35 steps, and R≥0.5 in 105 steps — demonstrating efficient sample usage.
**Right:** Our 4-stage curriculum design scales difficulty progressively. We trained on Stage 0 (Basic); stages 1-3 remain available for future multi-stage curriculum training.

### Qualitative Example

**Query:** "What caused the 2008 financial crisis?"

**Before training (baseline):**
> The crisis was caused by a collapse in the housing market starting in 2006, with Lehman Brothers filing bankruptcy in September 2007.
>
> ❌ **CONVICTED** — Wrong year (Lehman Brothers filed in 2008, not 2007)

**After CourtLLM training:**
> The 2008 financial crisis was caused by excessive risk-taking in mortgage-backed securities [source_id: FINANCIAL_2008_03, confidence: 0.85]. Lehman Brothers filed for bankruptcy in September 2008 [source_id: FINANCIAL_2008_07, confidence: 0.95].
>
> ✅ **ACQUITTED** — All citations valid, year correct

---

## Anti-Reward-Hacking Design

The hackathon ceremony specifically warned about reward hacking. Here's how CourtLLM prevents it:

| Potential Hack | Prevention |
|---|---|
| Cite invented source IDs | Exact-match against locked corpus → hard penalty |
| Always output low confidence | Calibration signal rewards matching actual outcomes |
| Make all claims vague | Wikipedia juror gives neutral, not positive |
| Concede every claim | "Failure to defend" penalty if >80% conceded |
| Game the LLM jurors | Juror C is fully deterministic (Wikipedia checker) |

---

## Why CourtLLM Stands Out

1. **Novel domain** — Legal courtroom applied to AI hallucination has zero prior art in RL/LLM training literature
2. **Genuinely teaches a new skill** — Calibrated uncertainty, not just factual accuracy
3. **Intuitive metric** — "Conviction rate" requires zero ML knowledge to interpret
4. **Research-worthy** — This environment could anchor a benchmark paper on hallucination reduction
5. **Built on OpenEnv** — Standard `reset()` / `step()` / `state()` API, fully compatible

---

## Try It Yourself

### Live Demo
👉 [https://huggingface.co/spaces/mishatul/CourtLLM_OpenEnv](https://huggingface.co/spaces/mishatul/CourtLLM_OpenEnv)

### Re-run Training
Open the [training notebook](training/courtllm_grpo_colab.ipynb) in Google Colab or run the [A100 training script](training/train_grpo_a100.py) on your own GPU.

### Install
```bash
pip install git+https://huggingface.co/spaces/mishatul/CourtLLM_OpenEnv
```

```python
from courtllm_env import CourtLLMClient, CourtAction

with CourtLLMClient("https://mishatul-courtllm-openenv.hf.space") as env:
    obs = env.reset()
    action = CourtAction(
        action_type="generate_testimony",
        content="My claim with evidence",
        claim_ids=["claim_001"],
        confidence=0.9,
        source_ids=["SOURCE_042"]
    )
    result = env.step(action)
    print(f"Reward: {result.reward:.3f}")
```

---

## Training Summary

```json
{
  "model": "Qwen/Qwen2.5-3B-Instruct",
  "training_steps_completed": 300,
  "baseline_conviction_rate": 0.61,
  "trained_conviction_rate": 0.0,
  "improvement": "61.0%",
  "final_reward": 0.5201,
  "max_reward": 0.5479
}
```

---

## Links

- 🌐 **Live HF Space:** [mishatul/CourtLLM_OpenEnv](https://huggingface.co/spaces/mishatul/CourtLLM_OpenEnv)
- 🎬 **Demo Video:** [YouTube](https://www.youtube.com/watch?v=Ip7_y52NwjQ)
- 💻 **GitHub Repository:** [Mish-atul/ScalerxMeta](https://github.com/Mish-atul/ScalerxMeta)
- 📓 **Training Notebook:** [courtllm_grpo_colab.ipynb](training/courtllm_grpo_colab.ipynb)
- 🖥️ **A100 Training Script:** [train_grpo_a100.py](training/train_grpo_a100.py)
- 📊 **Training Plots:** [outputs/](outputs/)

---

*Built for OpenEnv Hackathon 2026 by Team mishatul*
*Theme: Multi-Agent Interactions | Sub-themes: Halluminate + Fleet AI*
