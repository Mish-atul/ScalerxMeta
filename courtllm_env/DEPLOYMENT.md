# CourtLLM Deployment Guide

## 🚀 Quick Start

### Local Development

1. **Install dependencies:**
```bash
cd courtllm_env
pip install -r requirements.txt
```

2. **Test the environment:**
```bash
python test_environment.py
```

3. **Start the FastAPI server:**
```bash
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

4. **Test the API:**
```bash
curl http://localhost:8000/health
```

### Docker Deployment

1. **Build the image:**
```bash
cd courtllm_env
docker build -t courtllm-env:latest .
```

2. **Run the container:**
```bash
docker run -d -p 8000:8000 courtllm-env:latest
```

3. **Test:**
```bash
curl http://localhost:8000/health
```

## 📤 HuggingFace Spaces Deployment

### Prerequisites
- HuggingFace account
- `huggingface_hub` installed: `pip install huggingface_hub`

### Steps

1. **Login to HuggingFace:**
```bash
huggingface-cli login
```

2. **Create a new Space:**
- Go to https://huggingface.co/new-space
- Name: `courtllm-env`
- SDK: Docker
- Hardware: CPU Basic (free)

3. **Push to Space:**
```bash
cd courtllm_env
git init
git add .
git commit -m "Initial commit: CourtLLM environment"
git remote add space https://huggingface.co/spaces/<username>/courtllm-env
git push space main
```

4. **Verify deployment:**
- Visit `https://<username>-courtllm-env.hf.space`
- Check `/health` endpoint

## 🎓 Training on HuggingFace

### Option 1: Google Colab (Free T4 GPU)

1. Open `training/courtllm_grpo_colab.ipynb` in Colab
2. Update `ENV_URL` with your Space URL
3. Run all cells
4. Download outputs and commit to repo

### Option 2: HuggingFace Jobs

```bash
# Install HF CLI
pip install huggingface_hub[cli]

# Submit training job
huggingface-cli jobs submit \
  --flavor t4-small \
  --script training/train_grpo.py \
  --env ENV_URL=https://<username>-courtllm-env.hf.space
```

## 📊 Submission Checklist

- [x] OpenEnv v0.2.3 used
- [x] Working Colab training script
- [x] Evidence of training (plots in outputs/)
- [x] README with motivation and results
- [x] Environment pushed to HF Space
- [x] All links in README
- [ ] Mini-blog on HuggingFace (<2 min video or writeup)
- [ ] No large video files in repo

## 🔧 Troubleshooting

### Import Errors
If you get import errors, ensure you're running from the project root:
```bash
cd courtllm_env
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python test_environment.py
```

### Port Already in Use
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9  # macOS/Linux
netstat -ano | findstr :8000   # Windows
```

### Docker Build Fails
```bash
# Clean build
docker system prune -a
docker build --no-cache -t courtllm-env:latest .
```

## 📝 Next Steps

1. **Create HF Blog Post:**
   - Title: "CourtLLM: Training LLMs to Stop Hallucinating with Adversarial RL"
   - Include: motivation, architecture diagram, results, code snippets
   - Link to Space and GitHub

2. **Record Demo Video (<2 min):**
   - Show environment reset
   - Demonstrate step with valid/invalid citations
   - Show reward computation
   - Display before/after conviction rates

3. **Submit to Hackathon:**
   - HF Space URL
   - GitHub repo URL
   - Blog post URL
   - Video URL

## 🎯 Evaluation Criteria Alignment

| Criterion | Weight | How CourtLLM Addresses It |
|---|---|---|
| Environment Innovation | 40% | Novel courtroom metaphor, 4-signal reward, anti-gaming design |
| Storytelling | 30% | Clear narrative, conviction rate metric, visual results |
| Reward Improvement | 20% | Plots committed, 61% → 18% conviction rate drop |
| Technical Quality | 10% | OpenEnv compliant, working Colab, proper structure |

## 📚 Resources

- [OpenEnv Documentation](https://github.com/meta-pytorch/OpenEnv)
- [TRL GRPO Guide](https://huggingface.co/docs/trl/grpo)
- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [HF Spaces Guide](https://huggingface.co/docs/hub/spaces)

---

**Built for OpenEnv Hackathon 2026** | Theme: Multi-Agent Interactions
