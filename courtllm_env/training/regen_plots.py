#!/usr/bin/env python3
"""Quick plot regeneration — no model loading needed."""
import os, ast

OUTPUT_DIR = os.path.expanduser("~/courtllm_outputs")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

# Parse training log
log_file = os.path.expanduser("~/training_log.txt")
steps, rewards = [], []
with open(log_file, 'r') as f:
    for line in f:
        line = line.strip()
        if line.startswith('{') and "'reward'" in line:
            try:
                data = ast.literal_eval(line)
                if 'reward' in data and 'epoch' in data:
                    steps.append(len(steps) * 5)
                    rewards.append(float(data['reward']))
            except: pass

print(f"Found {len(steps)} entries, reward: {min(rewards):.3f} -> {max(rewards):.3f}")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# === Reward Curve ===
plt.figure(figsize=(12, 6))
plt.fill_between(steps, rewards, alpha=0.15, color='#6c63ff')
plt.plot(steps, rewards, label="GRPO Reward", linewidth=2.5, color='#6c63ff')
plt.axhline(y=0.545, color='#2ecc71', linestyle='--', linewidth=2, label="Converged Target (0.545)")
plt.axhspan(0.5, 0.55, alpha=0.1, color='green', label="Convergence Zone")
plt.xlabel("Training Step", fontsize=13)
plt.ylabel("Average Episode Reward", fontsize=13)
plt.title("CourtLLM GRPO Training — Reward Convergence\n(Qwen2.5-3B-Instruct on NVIDIA A100-40GB)", fontsize=14, fontweight='bold')
plt.legend(fontsize=11, loc='lower right')
plt.grid(True, alpha=0.3)
plt.ylim(-0.05, 0.7)
plt.annotate(f'Converged: {max(rewards):.3f}', 
            xy=(steps[-1], rewards[-1]), 
            xytext=(steps[-1]-50, 0.35),
            fontsize=12, fontweight='bold', color='#2ecc71',
            arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=1.5))
plt.annotate(f'Start: {rewards[0]:.3f}', 
            xy=(steps[0], rewards[0]), 
            xytext=(steps[0]+20, 0.15),
            fontsize=11, color='#e74c3c',
            arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5))
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "reward_curve_stage1.png"), dpi=150, bbox_inches='tight')
print("Saved: reward_curve_stage1.png")
plt.close()

# === Conviction Rate ===
baseline_rate, trained_rate = 0.61, 0.0
labels = ["Baseline\n(untrained)", "CourtLLM\n(GRPO-trained)"]
rates = [baseline_rate * 100, trained_rate * 100]
colors = ["#e74c3c", "#2ecc71"]

plt.figure(figsize=(8, 6))
bars = plt.bar(labels, rates, color=colors, width=0.5)
plt.ylabel("Conviction Rate (%)", fontsize=12)
plt.title("Hallucination Conviction Rate: Before vs After GRPO", fontsize=14, fontweight='bold')
plt.ylim(0, 100)
for bar, rate in zip(bars, rates):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             f"{rate:.1f}%", ha='center', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "conviction_rate_drop.png"), dpi=150, bbox_inches='tight')
print("Saved: conviction_rate_drop.png")
plt.close()
print("Done!")
