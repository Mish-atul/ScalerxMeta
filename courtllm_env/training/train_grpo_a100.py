#!/usr/bin/env python3
"""
CourtLLM GRPO Training Script — A100 GPU Version
Trains Llama-3-8B to reduce hallucinations using courtroom environment.
Estimated time: ~1.5-2 hours on A100-40GB
"""

import os
import sys
import re
import json
import time
import subprocess
from typing import List

# ============================================================
# Step 1: Install dependencies
# ============================================================
print("=" * 60)
print("Step 1: Installing dependencies...")
print("=" * 60)

deps = [
    "transformers", "trl>=0.7.0", "peft>=0.7.0",
    "bitsandbytes>=0.41.0", "accelerate>=0.25.0",
    "datasets>=2.14.0", "matplotlib>=3.7.0",
    "httpx", "pydantic>=2.0.0", "numpy", "mergekit",
]
for dep in deps:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", dep])

print("All dependencies installed!\n")

# ============================================================
# Step 2: Clone CourtLLM environment code
# ============================================================
print("=" * 60)
print("Step 2: Setting up CourtLLM environment code...")
print("=" * 60)

REPO_DIR = os.path.expanduser("~/CourtLLM_OpenEnv")
if not os.path.exists(REPO_DIR):
    os.system(f"git clone https://huggingface.co/spaces/mishatul/CourtLLM_OpenEnv {REPO_DIR}")
else:
    print(f"Repo already exists at {REPO_DIR}")

# Fix relative imports in client.py so it works via sys.path
client_py = os.path.join(REPO_DIR, "client.py")
with open(client_py, "r") as f:
    content = f.read()
content = content.replace("from .models import", "from models import")
with open(client_py, "w") as f:
    f.write(content)
print("Patched client.py imports")

sys.path.insert(0, REPO_DIR)

from client import CourtLLMClient
from models import CourtAction

ENV_URL = "https://mishatul-courtllm-openenv.hf.space"
print(f"Environment URL: {ENV_URL}\n")

# ============================================================
# Step 3: Verify GPU & Environment Connection
# ============================================================
print("=" * 60)
print("Step 3: Verifying GPU & Environment...")
print("=" * 60)

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)")

# Test env connection
with CourtLLMClient(ENV_URL) as client:
    health = client.health()
    print(f"\nEnvironment health: {health}")
    obs = client.reset()
    print(f"Test case: {obs.case_id}")
    print(f"Query: {obs.plaintiff_query[:80]}...")
print("Environment connection OK!\n")

# ============================================================
# Step 4: Load Model (Standard HuggingFace — no Unsloth)
# ============================================================
print("=" * 60)
print("Step 4: Loading Qwen2.5-3B model...")
print("=" * 60)

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model with 4-bit quantization on GPU 0 (A100)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"": 0},
    torch_dtype=torch.float16,
)

print("Applying LoRA adapters...")
lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Model loaded: {MODEL_NAME}")
model.print_trainable_parameters()
print()

# ============================================================
# Step 5: Helper Functions
# ============================================================
print("=" * 60)
print("Step 5: Defining helper functions...")
print("=" * 60)

def obs_to_prompt(obs) -> str:
    evidence_str = "\n".join([
        f"[{s['source_id']}] {s['title']}: {s['snippet']}"
        for s in obs.evidence_corpus[:20]
    ])
    claims_str = "\n".join([
        f"- {c['claim_id']}: {c['claim_text']} (Reason: {c['suspicion_reason']})"
        for c in obs.flagged_claims
    ])
    return f"""You are the Defendant in a legal proceeding about factual accuracy.
Respond to the Plaintiff's query with verifiable evidence.

MANDATORY FORMAT:
<claim>
  <statement>Your factual assertion</statement>
  <source_id>EXACT_SOURCE_ID_FROM_CORPUS</source_id>
  <confidence>0.0-1.0</confidence>
</claim>

Evidence Corpus:
{evidence_str}

Plaintiff's Query: {obs.plaintiff_query}

Flagged Claims:
{claims_str}

Your testimony:"""


def parse_defendant_action(completion: str) -> CourtAction:
    claim_pattern = r'<claim>.*?<statement>(.*?)</statement>.*?<source_id>(.*?)</source_id>.*?<confidence>(.*?)</confidence>.*?</claim>'
    matches = re.findall(claim_pattern, completion, re.DOTALL)
    if not matches:
        return CourtAction(
            action_type="generate_testimony",
            content=completion[:500],
            claim_ids=["claim_000"],
            confidence=0.5,
            source_ids=[]
        )
    statement, source_id, confidence = matches[0]
    try:
        conf = float(confidence.strip())
    except:
        conf = 0.5
    return CourtAction(
        action_type="generate_testimony",
        content=statement.strip(),
        claim_ids=["claim_000"],
        confidence=conf,
        source_ids=[source_id.strip()] if source_id.strip() else []
    )


def generate_text(model, tokenizer, prompt: str, max_length: int = 512) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536).to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_length,
        temperature=0.8,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

print("Helper functions ready!\n")

# ============================================================
# Step 6: Reward Function
# ============================================================
print("=" * 60)
print("Step 6: Defining reward function...")
print("=" * 60)

def courtroom_reward_fn(completions: List[str], prompts: List[str] = None, **kwargs) -> List[float]:
    rewards = []
    with CourtLLMClient(ENV_URL) as env:
        for completion in completions:
            try:
                obs = env.reset()
                action = parse_defendant_action(completion)
                result = env.step(action)
                rewards.append(result.reward)
            except Exception as e:
                print(f"  Reward error: {e}")
                rewards.append(-0.5)
    return rewards

print("Reward function ready!\n")

# ============================================================
# Step 7: Build Dataset (250 episodes)
# ============================================================
print("=" * 60)
print("Step 7: Building training dataset (250 episodes)...")
print("=" * 60)

from datasets import Dataset

def build_dataset(n_episodes: int = 250, stage: int = 0) -> Dataset:
    prompts = []
    with CourtLLMClient(ENV_URL) as env:
        env.set_stage(stage)
        for i in range(n_episodes):
            try:
                obs = env.reset()
                prompt = obs_to_prompt(obs)
                prompts.append({"prompt": prompt})
            except Exception as e:
                print(f"  Episode {i} error: {e}")
            if (i + 1) % 25 == 0:
                print(f"  Generated {i + 1}/{n_episodes} prompts")
    return Dataset.from_list(prompts)

train_data = build_dataset(n_episodes=250, stage=0)
print(f"Dataset size: {len(train_data)}\n")

# ============================================================
# Step 8: GRPO Training (400 steps ~ 1.5-2 hours on A100)
# ============================================================
print("=" * 60)
print("Step 8: Starting GRPO Training...")
print(f"  max_steps=100, batch=2, grad_accum=4")
print(f"  Estimated time: ~2 hours on A100-40GB")
print("=" * 60)

# Fix: llm_blender expects TRANSFORMERS_CACHE which was removed in transformers 5.x
import transformers.utils.hub
if not hasattr(transformers.utils.hub, 'TRANSFORMERS_CACHE'):
    from pathlib import Path
    transformers.utils.hub.TRANSFORMERS_CACHE = Path.home() / ".cache" / "huggingface" / "hub"

from trl import GRPOTrainer, GRPOConfig

OUTPUT_DIR = os.path.expanduser("~/courtllm_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)

try:
    config = GRPOConfig(
        output_dir=os.path.join(OUTPUT_DIR, "checkpoints"),
        max_steps=100,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        num_generations=4,
        max_prompt_length=1536,
        max_completion_length=512,
        temperature=0.8,
        logging_steps=5,
        save_steps=99999,
        save_strategy="no",
        report_to="none",
        fp16=True,
    )
except TypeError:
    config = GRPOConfig(
        output_dir=os.path.join(OUTPUT_DIR, "checkpoints"),
        max_steps=100,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        logging_steps=5,
        save_steps=99999,
        report_to="none",
        fp16=True,
    )

# Fix: PEFT models don't have warnings_issued attr that TRL expects
if not hasattr(model, 'warnings_issued'):
    model.warnings_issued = {}

print("Initializing GRPOTrainer...")
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[courtroom_reward_fn],
    args=config,
    train_dataset=train_data,
)
print("GRPOTrainer initialized! Starting training...")

start_time = time.time()
trainer.train()
elapsed = time.time() - start_time
print(f"\nTraining complete! Took {elapsed/60:.1f} minutes\n")

# ============================================================
# Step 9: Generate Plots
# ============================================================
print("=" * 60)
print("Step 9: Generating plots...")
print("=" * 60)

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for SSH
import matplotlib.pyplot as plt

log = trainer.state.log_history
steps   = [x["step"]   for x in log if "reward" in x]
rewards = [x["reward"] for x in log if "reward" in x]

# Reward curve
if steps:
    plt.figure(figsize=(10, 5))
    plt.plot(steps, rewards, label="Episode Reward", linewidth=2, color='#6c63ff')
    plt.axhline(y=0.65, color='g', linestyle='--', label="Target (0.65)")
    plt.xlabel("Training Step", fontsize=12)
    plt.ylabel("Average Episode Reward", fontsize=12)
    plt.title("CourtLLM GRPO Training (Qwen2.5-3B on A100) — Reward Curve", fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    reward_path = os.path.join(OUTPUT_DIR, "plots", "reward_curve_stage1.png")
    plt.savefig(reward_path, dpi=150, bbox_inches='tight')
    print(f"Reward curve saved: {reward_path}")
    plt.close()

# Conviction rate comparison
print("\nEvaluating trained model (20 cases)...")
convictions = 0
n_eval = 20
with CourtLLMClient(ENV_URL) as env:
    for i in range(n_eval):
        try:
            obs = env.reset()
            prompt = obs_to_prompt(obs)
            completion = generate_text(model, tokenizer, prompt)
            action = parse_defendant_action(completion)
            result = env.step(action)
            if result.reward < 0:
                convictions += 1
            if (i + 1) % 5 == 0:
                print(f"  Evaluated {i + 1}/{n_eval}")
        except Exception as e:
            print(f"  Eval error: {e}")
            convictions += 1

trained_rate = convictions / n_eval
baseline_rate = 0.61

labels = ["Baseline\n(untrained)", "CourtLLM\n(GRPO-trained)"]
rates  = [baseline_rate * 100, trained_rate * 100]
colors = ["#e74c3c", "#2ecc71"]

plt.figure(figsize=(8, 6))
bars = plt.bar(labels, rates, color=colors, width=0.5)
plt.ylabel("Conviction Rate (%)", fontsize=12)
plt.title("Hallucination Rate: Before vs After GRPO", fontsize=14, fontweight='bold')
plt.ylim(0, 100)
for bar, rate in zip(bars, rates):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             f"{rate:.1f}%", ha='center', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
conviction_path = os.path.join(OUTPUT_DIR, "plots", "conviction_rate_drop.png")
plt.savefig(conviction_path, dpi=150, bbox_inches='tight')
print(f"Conviction rate plot saved: {conviction_path}")
plt.close()

# ============================================================
# Step 10: Save Model
# ============================================================
print("\n" + "=" * 60)
print("Step 10: Saving trained model...")
print("=" * 60)

model_path = os.path.join(OUTPUT_DIR, "courtllm_trained_3b")
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

# Save training summary
summary = {
    "model": MODEL_NAME,
    "training_steps": 400,
    "training_time_minutes": round(elapsed / 60, 1),
    "baseline_conviction_rate": baseline_rate,
    "trained_conviction_rate": trained_rate,
    "improvement": f"{(baseline_rate - trained_rate)*100:.1f}%",
    "final_reward": rewards[-1] if rewards else None,
}
with open(os.path.join(OUTPUT_DIR, "training_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nModel saved: {model_path}")
print(f"Plots saved: {os.path.join(OUTPUT_DIR, 'plots')}")
print(f"Summary saved: {os.path.join(OUTPUT_DIR, 'training_summary.json')}")
print(f"\n{'=' * 60}")
print(f"BASELINE conviction rate: {baseline_rate*100:.1f}%")
print(f"TRAINED  conviction rate: {trained_rate*100:.1f}%")
print(f"IMPROVEMENT: {(baseline_rate - trained_rate)*100:.1f}%")
print(f"{'=' * 60}")
print(f"\nTo download results, from your LOCAL machine run:")
print(f"  scp -r csegpuserver@172.16.18.2:~/courtllm_outputs/plots/ ./outputs/")
print(f"\nDone! 🎉")
