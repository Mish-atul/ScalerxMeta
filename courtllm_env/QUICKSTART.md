# 🚀 CourtLLM Quick Start

Get up and running with CourtLLM in 5 minutes.

## Installation

```bash
# Clone from HuggingFace Space
git clone https://huggingface.co/spaces/mishatul/CourtLLM_OpenEnv
cd CourtLLM_OpenEnv

# Install dependencies
pip install -r requirements.txt
```

## Run the Demo

```bash
python demo_example.py
```

This will show you:
- How the environment generates cases
- Valid vs invalid testimonies
- Reward computation in action

## Start the Server

```bash
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```

Visit http://localhost:7860 for the interactive Web UI.
Visit http://localhost:7860/docs for API documentation.

## Use the Client

```python
from courtllm_env import CourtLLMClient, CourtAction

# Connect to live HF Space
with CourtLLMClient("https://mishatul-courtllm-openenv.hf.space") as env:
    obs = env.reset()
    print(f"Query: {obs.plaintiff_query}")
    
    action = CourtAction(
        action_type="generate_testimony",
        content="Your factual claim here",
        claim_ids=["claim_000"],
        confidence=0.8,
        source_ids=["BIOMEDICAL_0001"]
    )
    
    result = env.step(action)
    print(f"Reward: {result.reward}")
```

## Train a Model

Open `training/courtllm_grpo_colab.ipynb` in Google Colab and run all cells.

## Next Steps

- Read [README.md](README.md) for full documentation
- Check [DEPLOYMENT.md](DEPLOYMENT.md) for deployment guide
- Explore the code in `server/` directory

## Need Help?

- Check the [HF Space](https://huggingface.co/spaces/mishatul/CourtLLM_OpenEnv)
- Read the [OpenEnv docs](https://github.com/meta-pytorch/OpenEnv)
